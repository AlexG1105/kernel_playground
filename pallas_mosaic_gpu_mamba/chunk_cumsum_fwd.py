"""
pallas_mosaic_gpu_mamba/chunk_cumsum_fwd.py

Mosaic GPU (H100/H200) Pallas implementation of _chunk_cumsum_fwd.

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
  dA_cumsum[b,h,c,q]  = A[h] * sum_{k=0}^{q} dt_out[b,h,c,k]

Note: cumsum(dt_out * A[h]) == A[h] * cumsum(dt_out) because A[h] is a
per-head constant.  We exploit this to keep A out of the kernel body entirely,
computing the A multiplication in the wrapper after the kernel returns.

Grid and tile design
--------------------
Outer pallas_call grid: (batch * nchunks,)
  One CTA per (batch, chunk) pair.  All CTAs are independent.

Tile per CTA: (1, chunk_size, nheads_padded)
  dt is stored with nheads as the *innermost* (fastest-varying) dimension so
  that each loop step does a contiguous nheads_padded-wide read/write.

Mosaic GPU warpgroup constraint
--------------------------------
The H100 warpgroup has 128 threads.  Any SMEM or register vector accessed
via `load_strided` inside a `pl.loop` must contain a multiple of 128 elements.
We satisfy this by padding:

  nheads_padded = ceil(nheads / 128) * 128   (minimum 128)

This also guarantees the TMA bulk-copy minimum of 128 bytes
(32 float32 values), since:
  tile bytes  = chunk_size * nheads_padded * 4 >= 128 * 4 = 512 bytes.

Kernel body
-----------
  accumulator (SMEM): (nheads_padded,)          — 128+ elements, satisfies warpgroup
  @pl.loop(0, chunk_size):
      dt_i  = dt_ref[0, i, :]                   — nheads_padded contiguous reads ✓
      apply softplus + clip
      dt_out_ref[0, i, :] = dt_i                — nheads_padded contiguous writes ✓
      acc   = acc + dt_i                         — 128-element SMEM RMW ✓
      cs_ref[0, i, :] = acc                     — nheads_padded contiguous writes ✓

A is NOT in the kernel.  After the kernel, the wrapper computes:
  dA_cumsum = cs_out * A[None, :, None, None]   — pure XLA elementwise

No emit_pipeline is used.  The outer pallas_call stages each CTA's tile
via TMA before the kernel body starts; TMA commits the outputs after the
body returns.  Multiple CTAs overlap naturally via SM scheduling.

SMEM usage per CTA (chunk_size=256, nheads_padded=128):
  Input  staging (dt)    : 1 * 256 * 128 * 4 = 128 KB
  Output staging (dt_out): 1 * 256 * 128 * 4 = 128 KB
  Output staging (cs)    : 1 * 256 * 128 * 4 = 128 KB
  Accumulator            : 128 * 4             ~   0 KB
  Total                                        ~ 384 KB

  NOTE: 384 KB exceeds the per-CTA SMEM limit on H100 (228 KB).
  For chunk_size=256 reduce occupancy or use chunk_size <= 128:
    chunk_size=128: 3 * 128 * 128 * 4 = 192 KB  ✓
    chunk_size=64 : 3 *  64 * 128 * 4 =  96 KB  ✓
  Typical Mamba2 chunk_size values are 64–256; choose based on nheads and
  available SMEM.  The kernel does NOT validate SMEM limits at Python level.

Usage
-----
  from pallas_mosaic_gpu_mamba.chunk_cumsum_fwd import chunk_cumsum_fwd_mosaic

  dA_cumsum, dt_out = chunk_cumsum_fwd_mosaic(
      dt, A, chunk_size=64,
      dt_bias=dt_bias, dt_softplus=True,
  )
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


# ---------------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------------

def _chunk_cumsum_mosaic_kernel(
    dt_ref,       # SMEM ref: (1, chunk_size, nheads_padded)  — input dt tile
    dt_out_ref,   # SMEM ref: (1, chunk_size, nheads_padded)  — output dt_out
    cs_ref,       # SMEM ref: (1, chunk_size, nheads_padded)  — output cumsum(dt_out)
    *,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    chunk_size: int,
    nheads_padded: int,
):
    """
    One CTA handles one (batch, chunk) pair.

    Reads dt, applies softplus+clip, accumulates a sequential prefix sum,
    and writes dt_out and cumsum(dt_out).  A is NOT used here; the wrapper
    multiplies by A after the kernel returns.

    All per-step reads/writes are `ref[0, i, :]` — nheads_padded contiguous
    elements with a dynamic loop index i.  This satisfies Mosaic GPU's
    warpgroup constraint (nheads_padded >= 128 >= warpgroup_size).
    """

    def _scan(acc_smem):
        # Initialise the running prefix-sum accumulator.
        acc_smem[:] = jnp.zeros((nheads_padded,), jnp.float32)

        @pl.loop(0, chunk_size)
        def _step(i):
            # Load one time-step slice: shape (nheads_padded,), contiguous.
            dt_i = dt_ref[0, i, :].astype(jnp.float32)

            # Optional softplus: log1p(exp(x)).  Guard for overflow (x > 20).
            if dt_softplus:
                safe = jnp.where(dt_i <= 20.0, dt_i, jnp.zeros_like(dt_i))
                dt_i = jnp.where(dt_i <= 20.0, jnp.log1p(jnp.exp(safe)), dt_i)

            # Clamp to [dt_min, dt_max].
            dt_i = jnp.clip(dt_i, dt_min, dt_max)

            # Write processed dt to output.
            dt_out_ref[0, i, :] = dt_i

            # Prefix sum: acc += dt_i  (A will be multiplied in the wrapper).
            acc_smem[:] = acc_smem[:] + dt_i

            # Write current prefix sum.
            cs_ref[0, i, :] = acc_smem[:]

    pl.run_scoped(
        _scan,
        plgpu.SMEM((nheads_padded,), jnp.float32),  # accumulator: 128+ elements ✓
    )


# ---------------------------------------------------------------------------
# Public wrapper
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
    H100/H200 Pallas Mosaic GPU port of _chunk_cumsum_fwd.

    Parameters
    ----------
    dt         : (batch, seqlen, nheads)   float32
    A          : (nheads,)                 float32, always negative
    chunk_size : int
    dt_bias    : (nheads,)                 float32, optional bias added to dt
    dt_softplus: bool  — apply softplus before clamping
    dt_limit   : (dt_min, dt_max)          default (0.0, +inf)

    Returns
    -------
    dA_cumsum : (batch, nheads, nchunks, chunk_size)  float32
    dt_out    : (batch, nheads, nchunks, chunk_size)  float32

    Same return order as the Triton reference (_chunk_cumsum_fwd).
    """
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,), f"A.shape={A.shape} != ({nheads},)"
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)

    nchunks = math.ceil(seqlen / chunk_size)
    dt_min, dt_max = float(dt_limit[0]), float(dt_limit[1])

    # ------------------------------------------------------------------
    # Step 1: Add bias (elementwise; fused by XLA into the next op).
    # ------------------------------------------------------------------
    if dt_bias is not None:
        dt = dt + dt_bias[None, None, :]

    # ------------------------------------------------------------------
    # Step 2: Pad seqlen to a multiple of chunk_size.
    #   Padded positions get dt=0 → prefix sum continues correctly.
    # ------------------------------------------------------------------
    pad_len = nchunks * chunk_size - seqlen
    if pad_len > 0:
        dt = jnp.pad(dt, ((0, 0), (0, pad_len), (0, 0)))

    # ------------------------------------------------------------------
    # Step 3: Reshape to (batch*nchunks, chunk_size, nheads).
    #   Layout: nheads is the innermost (contiguous) dimension so that
    #   each kernel step reads/writes nheads_padded contiguous elements.
    # ------------------------------------------------------------------
    n_chunks_total = batch * nchunks
    dt_flat = dt.reshape(n_chunks_total, chunk_size, nheads)

    # ------------------------------------------------------------------
    # Step 4: Pad nheads to the next multiple of 128.
    #
    #   Mosaic GPU warpgroup constraint: any vector accessed via
    #   load_strided inside pl.loop must have >= 128 elements.
    #   nheads_padded = ceil(nheads / 128) * 128 satisfies this.
    #
    #   TMA bulk-copy constraint: tile must be >= 128 bytes (32 floats).
    #   tile = chunk_size * nheads_padded * 4 >= 128 * 4 = 512 bytes ✓
    # ------------------------------------------------------------------
    nheads_padded = math.ceil(nheads / 128) * 128
    pad_h = nheads_padded - nheads
    if pad_h > 0:
        dt_flat = jnp.pad(dt_flat, ((0, 0), (0, 0), (0, pad_h)))

    # ------------------------------------------------------------------
    # Step 5: Launch the Pallas Mosaic GPU kernel.
    #
    #   Grid: (n_chunks_total,) — one independent CTA per (batch, chunk).
    #
    #   BlockSpec: each CTA sees (1, chunk_size, nheads_padded) of dt_flat.
    #   The kernel writes (1, chunk_size, nheads_padded) tiles of dt_out
    #   and cs (cumsum of dt_out, without A).
    # ------------------------------------------------------------------
    kernel = partial(
        _chunk_cumsum_mosaic_kernel,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=dt_max,
        chunk_size=chunk_size,
        nheads_padded=nheads_padded,
    )

    tile_spec = pl.BlockSpec(
        (1, chunk_size, nheads_padded),
        lambda bc: (bc, 0, 0),
    )

    dt_out_flat, cs_flat = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct(
                (n_chunks_total, chunk_size, nheads_padded), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (n_chunks_total, chunk_size, nheads_padded), jnp.float32
            ),
        ],
        grid=(n_chunks_total,),
        in_specs=[tile_spec],
        out_specs=[tile_spec, tile_spec],
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel"],
        ),
    )(dt_flat)

    # ------------------------------------------------------------------
    # Step 6: Reshape and transpose outputs to (batch, nheads, nchunks, chunk_size).
    #
    #   flat:     (n_chunks_total, chunk_size, nheads_padded)
    #          =  (batch*nchunks,  chunk_size, nheads_padded)
    #   reshape:  (batch, nchunks, chunk_size, nheads_padded)
    #   transpose (0,3,1,2): (batch, nheads_padded, nchunks, chunk_size)
    #   slice:    (batch, nheads, nchunks, chunk_size)
    # ------------------------------------------------------------------
    def _restore(flat):
        return (
            flat
            .reshape(batch, nchunks, chunk_size, nheads_padded)
            .transpose(0, 3, 1, 2)
            [:, :nheads, :, :]
        )

    dt_out = _restore(dt_out_flat)
    cs     = _restore(cs_flat)

    # ------------------------------------------------------------------
    # Step 7: Multiply prefix sum by A to get dA_cumsum.
    #
    #   cumsum(dt_out * A[h]) == A[h] * cumsum(dt_out)   for scalar A[h].
    #   A: (nheads,) → broadcast to (batch, nheads, nchunks, chunk_size).
    # ------------------------------------------------------------------
    dA_cs = cs * A[None, :, None, None]

    # Same return order as the Triton reference: (dA_cumsum, dt_out)
    return dA_cs, dt_out


# ---------------------------------------------------------------------------
# Triton-compatible alias
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
