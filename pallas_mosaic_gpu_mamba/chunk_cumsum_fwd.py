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

SMEM budget and automatic sub-chunking
--------------------------------------
For small chunk_sizes the three tiles (dt, dt_out, cs) fit in SMEM:
  SMEM = 3 * chunk_size * nheads_padded * 4
    chunk_size=64 , nheads_padded=128 → 96 KB  ✓
    chunk_size=128, nheads_padded=128 → 192 KB ✓

For large chunk_sizes (e.g. 256) the tiles exceed the H100 per-CTA limit
(~228 KB).  The wrapper detects this and automatically switches to a
**sub-chunked kernel** that uses a 2D grid:

  Grid: (batch*nchunks, chunk_size // BQ)
    dim 0 — parallel  (one CTA per batch×chunk)
    dim 1 — sequential (iterate over sub-chunks of size BQ)

  Mosaic GPU lowers sequential dimensions via emit_pipeline, so only
  one sub-chunk tile set resides in SMEM at a time:
    SMEM ≈ 3 * BQ * nheads_padded * 4  (+ scratch accumulator)

  The prefix-sum accumulator is carried across sequential iterations
  via a scratch_shapes buffer (persistent SMEM allocated outside the
  pipeline loop).

  BQ is chosen as the largest power-of-2 that divides chunk_size and
  keeps SMEM within the hardware limit.

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
# SMEM budget helpers
# ---------------------------------------------------------------------------

# H100 per-CTA SMEM limit reported by the Mosaic GPU runtime.
_MAX_SMEM_BYTES = 232_448
# Conservative margin for alignment padding and runtime metadata.
_SMEM_OVERHEAD = 4096
_EFFECTIVE_MAX_SMEM = _MAX_SMEM_BYTES - _SMEM_OVERHEAD


def _tiles_smem(chunk_size: int, nheads_padded: int) -> int:
    """SMEM bytes for the simple kernel (3 full tiles + accumulator)."""
    return 3 * chunk_size * nheads_padded * 4 + nheads_padded * 4


def _choose_BQ(chunk_size: int, nheads_padded: int) -> int:
    """Largest power-of-2 sub-chunk that fits in SMEM and divides chunk_size.

    SMEM per sub-chunk iteration (max_concurrent_steps=1):
        3 * BQ * nheads_padded * 4   (input + 2 outputs)
      + nheads_padded * 4            (scratch accumulator)
    """
    acc_bytes = nheads_padded * 4
    row_bytes = 3 * nheads_padded * 4  # one row across all 3 tiles
    max_rows = (_EFFECTIVE_MAX_SMEM - acc_bytes) // row_bytes
    BQ = 1
    while BQ * 2 <= min(max_rows, chunk_size) and chunk_size % (BQ * 2) == 0:
        BQ *= 2
    return BQ


# ---------------------------------------------------------------------------
# Kernel body — simple path (full tile fits in SMEM)
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
                dt_i = jnp.where(dt_i <= 20.0, jnp.log(1+jnp.exp(safe)), dt_i)

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
# Kernel body — sub-chunk path (for large chunk_size)
# ---------------------------------------------------------------------------

def _chunk_cumsum_sub_kernel(
    dt_ref,        # SMEM ref: (1, BQ, nheads_padded) — input sub-chunk
    acc_in_ref,    # SMEM ref: (1, nheads_padded)     — input accumulator
    dt_out_ref,    # SMEM ref: (1, BQ, nheads_padded) — output dt_out
    cs_ref,        # SMEM ref: (1, BQ, nheads_padded) — output cumsum
    acc_out_ref,   # SMEM ref: (1, nheads_padded)     — output accumulator
    *,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    BQ: int,
    nheads_padded: int,
):
    """
    Processes one sub-chunk of BQ timesteps with an explicit accumulator.

    Called via jax.lax.scan over sub-chunks.  The accumulator is passed
    in/out through GMEM (acc_in_ref / acc_out_ref) to avoid the Mosaic GPU
    sequential-grid output-TMA bug that silently drops writes for tiles
    larger than a single element.
    """
    def _scan(acc_smem):
        acc_smem[:] = acc_in_ref[0, :]

        @pl.loop(0, BQ)
        def _step(i):
            dt_i = dt_ref[0, i, :].astype(jnp.float32)

            if dt_softplus:
                safe = jnp.where(dt_i <= 20.0, dt_i, jnp.zeros_like(dt_i))
                dt_i = jnp.where(dt_i <= 20.0, jnp.log(1 + jnp.exp(safe)), dt_i)

            dt_i = jnp.clip(dt_i, dt_min, dt_max)
            dt_out_ref[0, i, :] = dt_i
            acc_smem[:] = acc_smem[:] + dt_i
            cs_ref[0, i, :] = acc_smem[:]

        acc_out_ref[0, :] = acc_smem[:]

    pl.run_scoped(
        _scan,
        plgpu.SMEM((nheads_padded,), jnp.float32),
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
    #   Two paths depending on whether full tiles fit in SMEM:
    #   a) Simple:    1D grid, full (1, chunk_size, nheads_padded) tiles.
    #   b) Pipelined: 2D grid with sequential sub-chunking along
    #                 chunk_size, tiles of (1, BQ, nheads_padded).
    # ------------------------------------------------------------------
    out_shapes = [
        jax.ShapeDtypeStruct(
            (n_chunks_total, chunk_size, nheads_padded), jnp.float32
        ),
        jax.ShapeDtypeStruct(
            (n_chunks_total, chunk_size, nheads_padded), jnp.float32
        ),
    ]

    if _tiles_smem(chunk_size, nheads_padded) <= _EFFECTIVE_MAX_SMEM:
        # --- simple path: full tile fits in SMEM ---
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
            out_shape=out_shapes,
            grid=(n_chunks_total,),
            in_specs=[tile_spec],
            out_specs=[tile_spec, tile_spec],
            compiler_params=plgpu.CompilerParams(
                dimension_semantics=["parallel"],
            ),
        )(dt_flat)
    else:
        # --- sub-chunk path: lax.scan over sub-chunks ---
        #
        # The Mosaic GPU sequential-grid output TMA has a known issue
        # with multi-element tiles, so we instead iterate over sub-chunks
        # via jax.lax.scan, each processed by a 1D-parallel Pallas kernel
        # that takes the accumulator as an explicit input/output.
        BQ = _choose_BQ(chunk_size, nheads_padded)
        num_sub = chunk_size // BQ

        kernel = partial(
            _chunk_cumsum_sub_kernel,
            dt_softplus=dt_softplus,
            dt_min=dt_min,
            dt_max=dt_max,
            BQ=BQ,
            nheads_padded=nheads_padded,
        )

        dt_tile = pl.BlockSpec((1, BQ, nheads_padded), lambda bc: (bc, 0, 0))
        acc_tile = pl.BlockSpec((1, nheads_padded), lambda bc: (bc, 0))

        # Reshape dt to (num_sub, n_chunks_total, BQ, nheads_padded)
        # so lax.scan iterates over sub-chunks.
        dt_subs = dt_flat.reshape(
            n_chunks_total, num_sub, BQ, nheads_padded
        ).transpose(1, 0, 2, 3)

        def _scan_body(carry, dt_sub):
            # carry: (n_chunks_total, nheads_padded)
            # dt_sub: (n_chunks_total, BQ, nheads_padded)
            dt_out_sub, cs_sub, acc_out = pl.pallas_call(
                kernel,
                out_shape=[
                    jax.ShapeDtypeStruct(
                        (n_chunks_total, BQ, nheads_padded), jnp.float32
                    ),
                    jax.ShapeDtypeStruct(
                        (n_chunks_total, BQ, nheads_padded), jnp.float32
                    ),
                    jax.ShapeDtypeStruct(
                        (n_chunks_total, nheads_padded), jnp.float32
                    ),
                ],
                grid=(n_chunks_total,),
                in_specs=[dt_tile, acc_tile],
                out_specs=[dt_tile, dt_tile, acc_tile],
                compiler_params=plgpu.CompilerParams(
                    dimension_semantics=["parallel"],
                ),
            )(dt_sub, carry)

            return acc_out, (dt_out_sub, cs_sub)

        init_carry = jnp.zeros((n_chunks_total, nheads_padded), jnp.float32)
        _, (dt_out_subs, cs_subs) = jax.lax.scan(
            _scan_body, init_carry, dt_subs,
        )

        # dt_out_subs: (num_sub, n_chunks_total, BQ, nheads_padded)
        # → (n_chunks_total, chunk_size, nheads_padded)
        dt_out_flat = dt_out_subs.transpose(1, 0, 2, 3).reshape(
            n_chunks_total, chunk_size, nheads_padded
        )
        cs_flat = cs_subs.transpose(1, 0, 2, 3).reshape(
            n_chunks_total, chunk_size, nheads_padded
        )

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
