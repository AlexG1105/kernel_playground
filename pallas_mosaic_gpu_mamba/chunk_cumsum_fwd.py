"""
pallas_mosaic_gpu_mamba/chunk_cumsum_fwd.py

Three implementations of _chunk_cumsum_fwd for the Mamba2 SSM forward pass:

1. chunk_cumsum_fwd_mosaic   — Pure XLA (bias + softplus + clip + cumsum + A).
                               Uses standard JAX ops; XLA applies fused elementwise
                               kernels + parallel prefix-sum.  Default path.

2. chunk_cumsum_fwd_pallas   — Pallas *Triton* backend kernel.
                               Sequential pl.loop compiled via Pallas → Triton IR → PTX.
                               Closest to native Triton performance (~1.3× of Triton).
                               NOTE: uses the Triton backend, NOT Mosaic GPU.

3. chunk_cumsum_fwd_naive_jax — Naive JAX: same ops as pure XLA but multiplies
                                dt * A *before* cumsum (like the Triton kernel does).
                                Useful as a correctness reference and for benchmarking.

Algorithm
---------
Given:
  dt       : (batch, seqlen, nheads)   float32
  A        : (nheads,)                 float32, always negative
  dt_bias  : (nheads,)                 float32, optional

Produces:
  dt_out   : (batch, nheads, nchunks, chunk_size)  float32
  dA_cumsum: (batch, nheads, nchunks, chunk_size)  float32

where  nchunks = ceil(seqlen / chunk_size), and:
  dt_out[b,h,c,q]     = clip( softplus( dt[b,c*Q+q,h] + bias[h] ), dt_min, dt_max )
  dA_cumsum[b,h,c,q]  = sum_{k=0}^{q} dt_out[b,h,c,k] * A[h]

Usage
-----
  from pallas_mosaic_gpu_mamba.chunk_cumsum_fwd import (
      chunk_cumsum_fwd_mosaic,      # pure XLA (default)
      chunk_cumsum_fwd_pallas,      # Pallas Triton backend
      chunk_cumsum_fwd_naive_jax,   # naive JAX reference
  )
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl


# ---------------------------------------------------------------------------
# Pallas Triton backend kernel
# ---------------------------------------------------------------------------

def _chunk_cumsum_fwd_kernel(
    dt_ref, A_ref,                        # inputs
    dt_out_ref, dA_cs_ref, acc_ref,       # outputs (acc_ref is scratch)
    *,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
):
    """
    Pallas kernel for chunk_cumsum_fwd (compiled via the Triton backend).

    Ref shapes (per grid cell):
      dt_ref     : (1, chunk_size, block_size_h)
      A_ref      : (block_size_h,)
      dt_out_ref : (1, block_size_h, chunk_size)
      dA_cs_ref  : (1, block_size_h, chunk_size)
      acc_ref    : (1, block_size_h)              — scratch accumulator
    """
    acc_ref[0, :] = jnp.zeros(acc_ref.shape[1], dtype=jnp.float32)
    A_vals = A_ref[:]
    chunk_size = dt_ref.shape[1]

    @pl.loop(0, chunk_size)
    def scan_body(i):
        dt_i = dt_ref[0, i, :]

        if dt_softplus:
            safe_x = jnp.where(dt_i <= 20.0, dt_i, jnp.zeros_like(dt_i))
            dt_i = jnp.where(dt_i <= 20.0, jnp.log1p(jnp.exp(safe_x)), dt_i)

        dt_i = jnp.clip(dt_i, dt_min, dt_max)
        dt_out_ref[0, :, i] = dt_i

        dA_i = dt_i * A_vals
        acc_ref[0, :] = acc_ref[0, :] + dA_i
        dA_cs_ref[0, :, i] = acc_ref[0, :]


# ---------------------------------------------------------------------------
# Pallas Triton backend wrapper
# ---------------------------------------------------------------------------

def chunk_cumsum_fwd_pallas(
    dt,                         # (batch, seqlen, nheads)  float32
    A,                          # (nheads,)                float32, negative
    chunk_size: int,
    dt_bias=None,               # (nheads,) or None
    dt_softplus: bool = False,
    dt_limit=(0.0, float("inf")),
    block_size_h: int = 8,
    num_warps: int = 4,
) -> tuple:
    """
    Pallas Triton-backend implementation of _chunk_cumsum_fwd.

    Returns
    -------
    dA_cumsum : (batch, nheads, nchunks, chunk_size)  float32
    dt_out    : (batch, nheads, nchunks, chunk_size)  float32
    """
    from jax._src.pallas.triton.core import CompilerParams

    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)

    # Pad seqlen to exact multiple of chunk_size
    pad_len = nchunks * chunk_size - seqlen
    if pad_len > 0:
        dt = jnp.pad(dt, ((0, 0), (0, pad_len), (0, 0)))

    # Fuse bias addition before the kernel
    if dt_bias is not None:
        dt = dt + dt_bias[None, None, :]

    # Reshape to (batch*nchunks, chunk_size, nheads)
    dt_flat = dt.reshape(batch * nchunks, chunk_size, nheads)

    # Pad nheads to multiple of block_size_h
    nheads_blocks = math.ceil(nheads / block_size_h)
    nheads_padded = nheads_blocks * block_size_h
    pad_h = nheads_padded - nheads
    if pad_h > 0:
        dt_flat = jnp.pad(dt_flat, ((0, 0), (0, 0), (0, pad_h)))
        A_padded = jnp.pad(A, (0, pad_h))
    else:
        A_padded = A

    # Build pallas_call with Triton backend
    f = pl.pallas_call(
        lambda dt_ref, A_ref, dt_out_ref, dA_cs_ref, acc_ref:
            _chunk_cumsum_fwd_kernel(
                dt_ref, A_ref, dt_out_ref, dA_cs_ref, acc_ref,
                dt_softplus=dt_softplus,
                dt_min=dt_limit[0],
                dt_max=dt_limit[1],
            ),
        out_shape=[
            jax.ShapeDtypeStruct((batch * nchunks, nheads_padded, chunk_size), jnp.float32),
            jax.ShapeDtypeStruct((batch * nchunks, nheads_padded, chunk_size), jnp.float32),
            jax.ShapeDtypeStruct((batch * nchunks, nheads_padded), jnp.float32),
        ],
        grid=(batch * nchunks, nheads_blocks),
        in_specs=[
            pl.BlockSpec((1, chunk_size, block_size_h), lambda bc, bh: (bc, 0, bh)),
            pl.BlockSpec((block_size_h,), lambda bc, bh: (bh,)),
        ],
        out_specs=[
            pl.BlockSpec((1, block_size_h, chunk_size), lambda bc, bh: (bc, bh, 0)),
            pl.BlockSpec((1, block_size_h, chunk_size), lambda bc, bh: (bc, bh, 0)),
            pl.BlockSpec((1, block_size_h), lambda bc, bh: (bc, bh)),
        ],
        compiler_params=CompilerParams(num_warps=num_warps, num_stages=1),
    )

    dt_out_flat, dA_cs_flat, _ = f(dt_flat, A_padded)

    # Reshape: (B*K, H_pad, Q) → (B, H, K, Q)
    dt_out = (
        dt_out_flat
        .reshape(batch, nchunks, nheads_padded, chunk_size)
        .transpose(0, 2, 1, 3)
        [:, :nheads, :, :]
    )
    dA_cs = (
        dA_cs_flat
        .reshape(batch, nchunks, nheads_padded, chunk_size)
        .transpose(0, 2, 1, 3)
        [:, :nheads, :, :]
    )

    return dA_cs, dt_out


# ---------------------------------------------------------------------------
# Pure XLA path (default)
# ---------------------------------------------------------------------------

def chunk_cumsum_fwd_mosaic(
    dt,                                   # (batch, seqlen, nheads)   float32
    A,                                    # (nheads,)                 float32, < 0
    chunk_size: int,
    dt_bias=None,                         # (nheads,)                 float32, optional
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple:
    """
    Pure-XLA implementation of _chunk_cumsum_fwd.

    Uses standard JAX ops (elementwise + jnp.cumsum) so XLA can apply
    fusion and parallel prefix-sum algorithms.  No Pallas kernel.

    Returns
    -------
    dA_cumsum : (batch, nheads, nchunks, chunk_size)  float32
    dt_out    : (batch, nheads, nchunks, chunk_size)  float32
    """
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,), f"A.shape={A.shape} != ({nheads},)"
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)

    nchunks = math.ceil(seqlen / chunk_size)
    dt_min, dt_max = float(dt_limit[0]), float(dt_limit[1])

    if dt_bias is not None:
        dt = dt + dt_bias[None, None, :]
    if dt_softplus:
        safe = jnp.where(dt <= 20.0, dt, jnp.zeros_like(dt))
        dt = jnp.where(dt <= 20.0, jnp.log1p(jnp.exp(safe)), dt)
    dt = jnp.clip(dt, dt_min, dt_max)

    pad_len = nchunks * chunk_size - seqlen
    if pad_len > 0:
        dt = jnp.pad(dt, ((0, 0), (0, pad_len), (0, 0)))
    dt_out = dt.reshape(batch, nchunks, chunk_size, nheads).transpose(0, 3, 1, 2)

    dA_cumsum = jnp.cumsum(dt_out, axis=3) * A[None, :, None, None]

    return dA_cumsum, dt_out


# ---------------------------------------------------------------------------
# Naive JAX reference
# ---------------------------------------------------------------------------

def chunk_cumsum_fwd_naive_jax(
    dt,                                   # (batch, seqlen, nheads)   float32
    A,                                    # (nheads,)                 float32, < 0
    chunk_size: int,
    dt_bias=None,                         # (nheads,)                 float32, optional
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple:
    """
    Naive JAX implementation of _chunk_cumsum_fwd.

    Same logic as the pure XLA path, but multiplies dt * A *before*
    cumsum (matching the Triton kernel order).  Useful as a correctness
    reference and for benchmarking.

    Returns
    -------
    dA_cumsum : (batch, nheads, nchunks, chunk_size)  float32
    dt_out    : (batch, nheads, nchunks, chunk_size)  float32
    """
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)

    if dt_bias is not None:
        dt = dt + dt_bias[None, None, :]
    if dt_softplus:
        safe = jnp.where(dt <= 20.0, dt, jnp.zeros_like(dt))
        dt = jnp.where(dt <= 20.0, jnp.log1p(jnp.exp(safe)), dt)
    dt = jnp.clip(dt, dt_limit[0], dt_limit[1])

    pad_len = nchunks * chunk_size - seqlen
    if pad_len > 0:
        dt = jnp.pad(dt, ((0, 0), (0, pad_len), (0, 0)))
    dt_chunked = dt.reshape(batch, nchunks, chunk_size, nheads)
    dt_out = dt_chunked.transpose(0, 3, 1, 2)

    dA = dt_out * A[None, :, None, None]
    dA_cumsum = jnp.cumsum(dA, axis=3)

    return dA_cumsum, dt_out


# ---------------------------------------------------------------------------
# Triton-compatible alias (default = pure XLA)
# ---------------------------------------------------------------------------

def chunk_cumsum_fwd(
    dt,
    A,
    chunk_size: int,
    dt_bias=None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple:
    """Drop-in replacement for mamba_ssm._chunk_cumsum_fwd."""
    return chunk_cumsum_fwd_mosaic(
        dt, A, chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
