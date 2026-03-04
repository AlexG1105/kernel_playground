"""
pallas_mosaic_gpu_mamba/chunk_state_fwd.py

Mosaic GPU (H100/H200) Pallas implementation of _chunk_state_fwd.

Algorithm
---------
Given:
  x         : (batch, seqlen, nheads, headdim)   float32
  B         : (batch, seqlen, ngroups, dstate)   float32
  dt        : (batch, nheads, nchunks, chunk_size) float32   (post-softplus, post-clip)
  dA_cumsum : (batch, nheads, nchunks, chunk_size) float32
  seq_idx   : (batch, seqlen) int32, optional — sequence index per position

Produces:
  states : (batch, nchunks, nheads, headdim, dstate)  float32

where for each (batch b, chunk c, head h, group g = h // ratio):
  scale[k]       = exp(min(dA_cumsum[b,h,c,-1] - dA_cumsum[b,h,c,k], 0)) * dt[b,h,c,k]
  if seq_idx is not None and seq_idx[k] != seq_idx[chunk_end]:
      scale[k]   = 0   (position belongs to a different sequence)
  states[b,c,h]  = x[b,c,:,h,:].T  @  (B[b,c,:,g,:] * scale[:,None])
                 = (headdim, chunk_size) @ (chunk_size, dstate)
                 = (headdim, dstate)

Grid and tile design
--------------------
pl.kernel with Mesh grid: (BCH, PM, PN)
  BCH = batch * nchunks * nheads   — one CTA per (batch, chunk, head)
  PM  = headdim_padded // BM       — tiles over headdim
  PN  = dstate_padded  // BN       — tiles over dstate

  All three dimensions are "parallel" (independent CTAs).
  pl.kernel / core_map gives the kernel body GMEM refs.

Tile per CTA: (BM, chunk_size) for x_T, (chunk_size, BN) for B_scaled
  BM = BN = BK = 64 (defaults; typical for H100 WGMMA)

Mosaic GPU emit_pipeline + WGMMA
---------------------------------
emit_pipeline receives GMEM refs and manages TMA double-buffered loading
into swizzled SMEM.  The pipeline body passes swizzled SMEM refs directly
to WGMMA — no intermediate register copies needed.

  step k (K = chunk_size // BK):
    TMA loads (BM, BK) bf16 of x_T into swizzled SMEM → a_smem
    TMA loads (BK, BN) bf16 of B   into swizzled SMEM → b_smem
    wgmma(acc, a_smem, b_smem)  — accumulates in f32 ACC registers

  After all K steps:
    states_ref[bch, pm*BM:(pm+1)*BM, pn*BN:(pn+1)*BN] = acc[...]

Note: we CANNOT use pallas_call with BlockSpec here because pallas_call
stages tiles into SMEM before the kernel runs, but emit_pipeline needs
GMEM refs (it does its own TMA).  Writing from registers to
WGMMA-transformed SMEM is not supported (WGStridedFragLayout error).

Scale folded into x, B stays group-indexed
-------------------------------------------
  scale = exp(min(dA_cs_last - dA_cumsum, 0)) * dt
  x_scaled = x_T * scale[None, :]     (scale broadcast over headdim)
  states = x_scaled @ B               (B is per-group, no expansion)

  Math: states = x_T @ diag(scale) @ B = (x_T * scale) @ B

  Scale is folded into x (not B) so that B stays group-indexed (BCG)
  without replication.  The kernel maps bch→bcg to read the correct
  group's B slice.  Multiple heads share B data via L2 cache.

  Both x_scaled and B are cast to bf16 in the wrapper since H100
  WGMMA requires bf16 (or f16/tf32/fp8) operands.  The WGMMA
  accumulator is f32, preserving precision in the reduction.

SMEM budget (BM=BN=BK=64, num_stages=2)
-----------------------------------------
  x_T staging  : 2 * 64*64*2  =  16 KB  (bf16, double-buffered)
  B   staging  : 2 * 64*64*2  =  16 KB  (bf16, double-buffered)
  ACC (regs)   : 64*64*4      =  16 KB  (f32, not SMEM)
  Total SMEM   :              ~  32 KB  ✓  (H100 limit: 228 KB)

Usage
-----
  from pallas_mosaic_gpu_mamba.chunk_state_fwd import chunk_state_fwd_mosaic

  states = chunk_state_fwd_mosaic(x, B, dt, dA_cumsum)
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


# ---------------------------------------------------------------------------
# Kernel body (receives GMEM refs from pl.kernel / core_map)
# ---------------------------------------------------------------------------

def _chunk_state_kernel_body(
    x_t_ref,    # GMEM ref: (BCH, headdim_padded, chunk_size) bf16  (scale folded in)
    B_ref,      # GMEM ref: (BCG, chunk_size, dstate_padded) bf16   (group-indexed)
    states_ref, # GMEM ref: (BCH, headdim_padded, dstate_padded) f32
    *,
    BM: int,
    BK: int,
    BN: int,
    chunk_size: int,
    num_stages: int,
    nheads: int,
    ngroups: int,
):
    """
    One CTA computes one (BM, BN) output tile for a given (bch, pm, pn).

    Uses emit_pipeline for double-buffered TMA loading of bf16 tiles
    into WGMMA-compatible swizzled SMEM.  The pipeline body passes
    swizzled SMEM refs directly to wgmma (no intermediate register copy).

    B is indexed by group (BCG), not head (BCH).  Each CTA computes
    bcg = f(bch) to read the correct group's B slice.  Multiple heads
    within the same group share B data via L2 cache.
    """
    bch = lax.axis_index("bch")
    pm  = lax.axis_index("pm")
    pn  = lax.axis_index("pn")

    K = chunk_size // BK

    # Map bch → bcg for group-indexed B lookup
    ratio = nheads // ngroups
    chunk_batch = bch // nheads              # batch_idx * nchunks + chunk_idx
    head_idx = bch % nheads
    bcg = chunk_batch * ngroups + head_idx // ratio

    # GMEM sub-refs for this CTA's portion of the arrays
    x_t_gmem = x_t_ref.at[bch, pl.ds(pm * BM, BM), :]     # (BM, chunk_size) bf16
    B_gmem   = B_ref.at[bcg, :, pl.ds(pn * BN, BN)]        # (chunk_size, BN) bf16

    # Swizzle/tiling transforms for WGMMA-compatible SMEM layout
    a_swizzle = plgpu.find_swizzle(BK * 16)                 # 16 bits per bf16
    a_transforms = (
        plgpu.TilingTransform((8, a_swizzle // 2)),
        plgpu.SwizzleTransform(a_swizzle),
    )
    b_swizzle = plgpu.find_swizzle(BN * 16)
    b_transforms = (
        plgpu.TilingTransform((8, b_swizzle // 2)),
        plgpu.SwizzleTransform(b_swizzle),
    )

    def _with_acc(acc_ref):
        # Pipeline body: TMA loads into swizzled SMEM, then WGMMA accumulates.
        # acc_ref is captured from closure (allocated outside the pipeline).
        # No init_carry → body receives (indices, *in_smems) only, no carry.
        def pipeline_body(step, a_smem, b_smem):
            plgpu.wgmma(acc_ref, a_smem, b_smem)
            plgpu.wgmma_wait(0)

        plgpu.emit_pipeline(
            pipeline_body,
            grid=(K,),
            in_specs=[
                plgpu.BlockSpec(
                    (BM, BK), lambda k: (0, k),
                    transforms=a_transforms,
                ),
                plgpu.BlockSpec(
                    (BK, BN), lambda k: (k, 0),
                    transforms=b_transforms,
                ),
            ],
            max_concurrent_steps=num_stages,
        )(x_t_gmem, B_gmem)

        # Write accumulated f32 result to output GMEM
        states_ref[bch, pl.ds(pm * BM, BM), pl.ds(pn * BN, BN)] = (
            acc_ref[...].astype(jnp.float32)
        )

    pl.run_scoped(_with_acc, plgpu.ACC((BM, BN), jnp.float32))


# ---------------------------------------------------------------------------
# Preprocessing (JAX/XLA ops — scale, reshape, pad, bf16 cast)
# ---------------------------------------------------------------------------

def chunk_state_preprocess(
    x,          # (batch, seqlen, nheads, headdim)            float32
    B,          # (batch, seqlen, ngroups, dstate)            float32
    dt,         # (batch, nheads, nchunks, chunk_size)        float32
    dA_cumsum,  # (batch, nheads, nchunks, chunk_size)        float32
    seq_idx=None,  # (batch, seqlen) int32, optional
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
):
    """
    Preprocess inputs for the Pallas kernel.

    Scale is folded into x (not B), so B stays group-indexed (BCG)
    instead of being expanded to head-indexed (BCH).  This reduces
    B memory by nheads/ngroups (the group ratio).

    Math: states = x_T @ diag(scale) @ B = (x_T * scale) @ B

    When seq_idx is provided, positions where seq_idx[k] differs from
    seq_idx at the last valid position of the chunk have their scale
    zeroed out, so they don't contribute to the chunk state.  This
    handles within-chunk sequence boundaries for packed variable-length
    sequences.

    Returns
    -------
    x_scaled  : (BCH, headdim_padded, chunk_size) bf16  — scale folded in
    B_flat    : (BCG, chunk_size, dstate_padded) bf16    — group-indexed
    meta      : dict with BCH, BCG, headdim, headdim_padded, dstate,
                dstate_padded, batch, nchunks, nheads, ngroups, chunk_size, BK
    """
    batch, seqlen, nheads, headdim = x.shape
    _, nheads_, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape

    assert nheads % ngroups == 0
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    BK = min(BK, chunk_size)
    assert chunk_size % BK == 0

    # Pad seqlen
    total_len = nchunks * chunk_size
    if seqlen < total_len:
        pad = total_len - seqlen
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        B = jnp.pad(B, ((0, 0), (0, pad), (0, 0), (0, 0)))
        if seq_idx is not None:
            # Pad seq_idx with -1 so padded positions never match seq_idx_last
            seq_idx = jnp.pad(seq_idx, ((0, 0), (0, pad)), constant_values=-1)

    # Scale: (batch, nheads, nchunks, chunk_size)
    dA_cs_last = dA_cumsum[:, :, :, -1:]
    scale = jnp.exp(jnp.minimum(dA_cs_last - dA_cumsum, 0.0)) * dt

    # Handle seq_idx: zero scale where seq_idx[k] != seq_idx[chunk_end].
    # Matches Triton: scale = where((seq_idx_last >= 0) & (seq_idx_k == seq_idx_last), scale, 0)
    if seq_idx is not None:
        # seq_idx at last valid position of each chunk
        chunk_ends = jnp.minimum(
            jnp.arange(1, nchunks + 1) * chunk_size, seqlen
        ) - 1  # (nchunks,)
        seq_idx_last = seq_idx[:, chunk_ends]  # (batch, nchunks)
        # Reshape seq_idx to (batch, nchunks, chunk_size)
        seq_idx_chunked = seq_idx.reshape(batch, nchunks, chunk_size)
        # Mask: True where position belongs to same sequence as chunk end
        same_seq = (seq_idx_last[:, :, None] >= 0) & (
            seq_idx_chunked == seq_idx_last[:, :, None]
        )  # (batch, nchunks, chunk_size)
        # scale is (batch, nheads, nchunks, chunk_size)
        # same_seq is (batch, nchunks, chunk_size) → broadcast over nheads
        scale = jnp.where(same_seq[:, None, :, :], scale, 0.0)

    # Reshape x → (BCH, headdim, chunk_size), fold scale into x
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    x_t = x.transpose(0, 1, 3, 4, 2)           # (batch, nchunks, nheads, headdim, chunk_size)
    BCH = batch * nchunks * nheads
    x_flat = x_t.reshape(BCH, headdim, chunk_size)

    # scale → (BCH, chunk_size), broadcast over headdim
    scale_t = scale.transpose(0, 2, 1, 3)       # (batch, nchunks, nheads, chunk_size)
    scale_flat = scale_t.reshape(BCH, chunk_size)
    x_flat = x_flat * scale_flat[:, None, :]     # (BCH, headdim, chunk_size) * (BCH, 1, chunk_size)

    headdim_padded = math.ceil(headdim / BM) * BM
    if headdim_padded > headdim:
        x_flat = jnp.pad(x_flat, ((0, 0), (0, headdim_padded - headdim), (0, 0)))
    x_flat = x_flat.astype(jnp.bfloat16)

    # Reshape B → (BCG, chunk_size, dstate_padded) bf16  (group-indexed, no expansion)
    B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    B = B.transpose(0, 1, 3, 2, 4)              # (batch, nchunks, ngroups, chunk_size, dstate)
    BCG = batch * nchunks * ngroups
    B_flat = B.reshape(BCG, chunk_size, dstate)

    dstate_padded = math.ceil(dstate / BN) * BN
    if dstate_padded > dstate:
        B_flat = jnp.pad(B_flat, ((0, 0), (0, 0), (0, dstate_padded - dstate)))
    B_flat = B_flat.astype(jnp.bfloat16)

    meta = dict(
        BCH=BCH, BCG=BCG, headdim=headdim, headdim_padded=headdim_padded,
        dstate=dstate, dstate_padded=dstate_padded,
        batch=batch, nchunks=nchunks, nheads=nheads, ngroups=ngroups,
        chunk_size=chunk_size, BK=BK,
    )
    return x_flat, B_flat, meta


# ---------------------------------------------------------------------------
# Kernel-only launch (no preprocessing)
# ---------------------------------------------------------------------------

def chunk_state_kernel_only(
    x_flat,     # (BCH, headdim_padded, chunk_size) bf16  (scale folded in)
    B_flat,     # (BCG, chunk_size, dstate_padded) bf16   (group-indexed)
    *,
    BCH: int,
    BCG: int,
    headdim: int,
    headdim_padded: int,
    dstate: int,
    dstate_padded: int,
    batch: int,
    nchunks: int,
    nheads: int,
    ngroups: int,
    chunk_size: int,
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """
    Launch just the Pallas kernel + output reshape.

    Takes pre-processed bf16 inputs (from chunk_state_preprocess).
    B is group-indexed (BCG); the kernel maps bch→bcg internally.
    """
    PM = headdim_padded // BM
    PN = dstate_padded  // BN

    mesh = plgpu.Mesh(
        grid=(BCH, PM, PN),
        grid_names=("bch", "pm", "pn"),
    )

    kernel_fn = pl.kernel(
        partial(
            _chunk_state_kernel_body,
            BM=BM, BK=BK, BN=BN,
            chunk_size=chunk_size,
            num_stages=num_stages,
            nheads=nheads,
            ngroups=ngroups,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (BCH, headdim_padded, dstate_padded), jnp.float32
        ),
        mesh=mesh,
    )

    states_flat = kernel_fn(x_flat, B_flat)

    states = (
        states_flat[:, :headdim, :dstate]
        .reshape(batch, nchunks, nheads, headdim, dstate)
    )
    return states


# ---------------------------------------------------------------------------
# Public wrapper (end-to-end: preprocess + kernel)
# ---------------------------------------------------------------------------

def chunk_state_fwd_mosaic(
    x,          # (batch, seqlen, nheads, headdim)            float32
    B,          # (batch, seqlen, ngroups, dstate)            float32
    dt,         # (batch, nheads, nchunks, chunk_size)        float32
    dA_cumsum,  # (batch, nheads, nchunks, chunk_size)        float32
    seq_idx=None,  # (batch, seqlen)                          int32, optional
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """
    H100/H200 Pallas Mosaic GPU port of _chunk_state_fwd.

    Parameters
    ----------
    x          : (batch, seqlen, nheads, headdim)   float32
    B          : (batch, seqlen, ngroups, dstate)   float32
    dt         : (batch, nheads, nchunks, chunk_size) float32  (post-processed)
    dA_cumsum  : (batch, nheads, nchunks, chunk_size) float32
    seq_idx    : (batch, seqlen) int32, optional
        Sequence index per position for packed variable-length sequences.
        Positions where seq_idx differs from the chunk's last position
        have their contribution zeroed (scale = 0).
    BM, BK, BN : tile sizes (default 64).  chunk_size must be divisible by BK.
    num_stages : TMA pipeline depth (default 2)

    Returns
    -------
    states : (batch, nchunks, nheads, headdim, dstate)  float32

    Same semantics as the Triton reference _chunk_state_fwd.
    """
    x_flat, B_flat, meta = chunk_state_preprocess(
        x, B, dt, dA_cumsum, seq_idx=seq_idx, BM=BM, BK=BK, BN=BN,
    )
    return chunk_state_kernel_only(
        x_flat, B_flat,
        BM=BM, BK=meta['BK'], BN=BN, num_stages=num_stages,
        **{k: meta[k] for k in (
            'BCH', 'BCG', 'headdim', 'headdim_padded', 'dstate',
            'dstate_padded', 'batch', 'nchunks', 'nheads', 'ngroups',
            'chunk_size',
        )},
    )


# ---------------------------------------------------------------------------
# Triton-compatible alias
# ---------------------------------------------------------------------------

def chunk_state_fwd(
    x,
    B,
    dt,
    dA_cumsum,
    seq_idx=None,
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """Drop-in replacement for mamba_ssm._chunk_state_fwd (JAX/Pallas version)."""
    return chunk_state_fwd_mosaic(
        x, B, dt, dA_cumsum,
        seq_idx=seq_idx,
        BM=BM, BK=BK, BN=BN,
        num_stages=num_stages,
    )


# ===========================================================================
# chunk_state_varlen — variable-length final-state computation
# ===========================================================================

def chunk_state_varlen_mosaic(
    B,              # (total_seqlen, ngroups, dstate)       float32
    x,              # (total_seqlen, nheads, headdim)       float32
    dt,             # (nheads, nchunks, chunk_size)         float32
    dA_cumsum,      # (nheads, nchunks, chunk_size)         float32
    cu_seqlens,     # (batch + 1,)                          int32
    chunk_states,   # (nchunks, nheads, headdim, dstate)    float32
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """
    H100/H200 Pallas Mosaic GPU port of chunk_state_varlen.

    Computes the final SSM state for each variable-length sequence in a
    batch of concatenated (packed) sequences.

    For each sequence b (defined by cu_seqlens[b]:cu_seqlens[b+1]):
      1. Find the last chunk containing this sequence's end
      2. Compute x_T @ diag(scale) @ B for positions in that chunk
         belonging to this sequence
      3. Add chunk_states[last_chunk] * exp(dA_cs_last) if the sequence
         spans earlier chunks

    Input shapes differ from chunk_state_fwd: no batch dimension, all
    sequences are concatenated along the sequence axis.

    Parameters
    ----------
    B            : (total_seqlen, ngroups, dstate) float32
    x            : (total_seqlen, nheads, headdim) float32
    dt           : (nheads, nchunks, chunk_size) float32
    dA_cumsum    : (nheads, nchunks, chunk_size) float32
    cu_seqlens   : (batch + 1,) int32 — cumulative sequence lengths
    chunk_states : (nchunks, nheads, headdim, dstate) float32
        Running states after state_passing_fwd (state at START of each chunk).
    BM, BK, BN  : tile sizes (default 64)
    num_stages   : TMA pipeline depth (default 2)

    Returns
    -------
    states : (batch, nheads, headdim, dstate) float32
        Final state for each sequence.
    """
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1

    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)

    BK = min(BK, chunk_size)
    assert chunk_size % BK == 0

    # --- Pad x and B to nchunks * chunk_size ---
    padded_len = nchunks * chunk_size
    if total_seqlen < padded_len:
        pad = padded_len - total_seqlen
        x = jnp.pad(x, ((0, pad), (0, 0), (0, 0)))
        B = jnp.pad(B, ((0, pad), (0, 0), (0, 0)))

    # --- Per-sequence indexing ---
    end_idxs = cu_seqlens[1:]                             # (batch,)
    start_idxs = cu_seqlens[:-1]                          # (batch,)
    last_chunks = (end_idxs - 1) // chunk_size            # (batch,)
    chunk_starts = last_chunks * chunk_size               # (batch,)
    chunk_size_limits = end_idxs - chunk_starts           # (batch,)
    start_in_chunks = jnp.maximum(start_idxs - chunk_starts, 0)  # (batch,)

    # --- Gather last-chunk data per sequence ---
    # x, B: use advanced indexing with (batch, chunk_size) offsets
    offsets = chunk_starts[:, None] + jnp.arange(chunk_size)[None, :]  # (batch, chunk_size)
    x_last = x[offsets]       # (batch, chunk_size, nheads, headdim)
    B_last = B[offsets]       # (batch, chunk_size, ngroups, dstate)

    # dt, dA_cumsum: index by last_chunks
    dt_last = dt[:, last_chunks, :].transpose(1, 0, 2)        # (batch, nheads, chunk_size)
    dA_last = dA_cumsum[:, last_chunks, :].transpose(1, 0, 2)  # (batch, nheads, chunk_size)

    # dA_cs_last: dA_cumsum at the last valid position per sequence
    cs_lim_idx = jnp.broadcast_to(
        (chunk_size_limits - 1)[:, None, None], (batch, nheads, 1)
    )
    dA_cs_last = jnp.take_along_axis(dA_last, cs_lim_idx, axis=2)[:, :, 0]
    # (batch, nheads)

    # --- Compute scale with valid-position masking ---
    scale = jnp.exp(jnp.minimum(dA_cs_last[:, :, None] - dA_last, 0.0)) * dt_last
    # (batch, nheads, chunk_size)

    positions = jnp.arange(chunk_size)
    valid_mask = (
        (positions[None, :] >= start_in_chunks[:, None])
        & (positions[None, :] < chunk_size_limits[:, None])
    )  # (batch, chunk_size)
    scale = jnp.where(valid_mask[:, None, :], scale, 0.0)

    # --- Fold scale into x, prepare for WGMMA ---
    # x_last: (batch, chunk_size, nheads, headdim) → x_T: (batch, nheads, headdim, chunk_size)
    x_T = x_last.transpose(0, 2, 3, 1)
    x_scaled_T = x_T * scale[:, :, None, :]  # (batch, nheads, headdim, chunk_size)

    BCH = batch * nheads
    BCG = batch * ngroups
    x_flat = x_scaled_T.reshape(BCH, headdim, chunk_size)

    # B_last: (batch, chunk_size, ngroups, dstate) → (BCG, chunk_size, dstate)
    B_flat = B_last.transpose(0, 2, 1, 3).reshape(BCG, chunk_size, dstate)

    # --- Pad and cast to bf16 ---
    headdim_padded = math.ceil(headdim / BM) * BM
    dstate_padded = math.ceil(dstate / BN) * BN

    if headdim_padded > headdim:
        x_flat = jnp.pad(x_flat, ((0, 0), (0, headdim_padded - headdim), (0, 0)))
    if dstate_padded > dstate:
        B_flat = jnp.pad(B_flat, ((0, 0), (0, 0), (0, dstate_padded - dstate)))

    x_flat = x_flat.astype(jnp.bfloat16)
    B_flat = B_flat.astype(jnp.bfloat16)

    # --- Run WGMMA kernel (reuse chunk_state_kernel_only with nchunks=1) ---
    states_raw = chunk_state_kernel_only(
        x_flat, B_flat,
        BCH=BCH, BCG=BCG,
        headdim=headdim, headdim_padded=headdim_padded,
        dstate=dstate, dstate_padded=dstate_padded,
        batch=batch, nchunks=1, nheads=nheads, ngroups=ngroups,
        chunk_size=chunk_size,
        BM=BM, BK=BK, BN=BN, num_stages=num_stages,
    )
    # states_raw: (batch, 1, nheads, headdim, dstate)
    states_raw = states_raw[:, 0]  # (batch, nheads, headdim, dstate)

    # --- Add chunk_states contribution ---
    # Only if the sequence spans earlier chunks (start_idx < chunk_start)
    cs_last = chunk_states[last_chunks]  # (batch, nheads, headdim, dstate)
    has_prev = start_idxs < chunk_starts  # (batch,)
    cs_scale = jnp.exp(dA_cs_last)       # (batch, nheads)
    cs_contrib = cs_last * cs_scale[:, :, None, None]
    states = states_raw + jnp.where(
        has_prev[:, None, None, None], cs_contrib, 0.0
    )

    return states
