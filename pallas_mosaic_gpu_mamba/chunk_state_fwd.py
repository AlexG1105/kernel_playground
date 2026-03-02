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

Produces:
  states : (batch, nchunks, nheads, headdim, dstate)  float32

where for each (batch b, chunk c, head h, group g = h // ratio):
  scale[k]       = exp(min(dA_cumsum[b,h,c,-1] - dA_cumsum[b,h,c,k], 0)) * dt[b,h,c,k]
  states[b,c,h]  = x[b,c,:,h,:].T  @  (B[b,c,:,g,:] * scale[:,None])
                 = (headdim, chunk_size) @ (chunk_size, dstate)
                 = (headdim, dstate)

Grid and tile design
--------------------
Outer pallas_call grid: (BCH, PM, PN)
  BCH = batch * nchunks * nheads   — one CTA per (batch, chunk, head)
  PM  = headdim_padded // BM       — tiles over headdim
  PN  = dstate_padded  // BN       — tiles over dstate

  All three dimensions are "parallel" (independent CTAs).

Tile per CTA: (1, BM, chunk_size) for x_T, (1, chunk_size, BN) for B
  BM = BN = BK = 64 (defaults; typical for H100 WGMMA)

Mosaic GPU WGMMA + emit_pipeline
---------------------------------
Inside each CTA, emit_pipeline runs K = chunk_size // BK pipeline steps:

  step k:
    x_T_tile : (BM, BK) f32   from x_T_ref.at[0] at column k
    B_tile   : (BK, BN) f32   from B_ref.at[0] at row k
    scale_k  : (BK,)   f32   from scale_ref.at[0] at position k

    B_scaled = (B_tile * scale_k[:, None]).astype(bf16)   — in registers
    x_T_bf16 = x_T_tile.astype(bf16)                      — in registers

    pl.run_scoped(do_wgmma, ACC(BM,BN,f32), SMEM(BM,BK,bf16,A_transforms),
                            SMEM(BK,BN,bf16,B_transforms))
      a_scratch[...] = x_T_bf16   → commit_smem() → wgmma → wgmma_wait(0)
      b_scratch[...] = B_scaled

    carry += tile_result

  states_ref[0,:,:] = carry

Scale precomputed in wrapper
----------------------------
  dA_cs_last = dA_cumsum[:,:,:,-1:]   # (batch,nheads,nchunks,1)
  scale = exp(min(dA_cs_last - dA_cumsum, 0)) * dt   # (batch,nheads,nchunks,chunk_size)

  Then flattened to (BCH, chunk_size) for the kernel.

ngroups mapping
---------------
  B is indexed by BCG = batch * nchunks * ngroups.
  Given bch = batch_idx*(nchunks*nheads) + chunk_idx*nheads + head_idx:
    chunk_batch = bch // nheads        (batch_idx*nchunks + chunk_idx)
    group_idx   = (bch % nheads) // ratio
    bcg         = chunk_batch * ngroups + group_idx

SMEM budget (BM=BN=BK=64, num_stages=2)
-----------------------------------------
  x_T staging  : 2 * 64*64*4 =  32 KB
  B   staging  : 2 * 64*64*4 =  32 KB
  scale staging: 2 * 64*4    ~   0.5 KB
  A scratch    : 64*64*2     =   8 KB  (bf16)
  B scratch    : 64*64*2     =   8 KB  (bf16)
  ACC (regs)   : 64*64*4     =  16 KB  (not SMEM)
  Total SMEM   :             ~ 80.5 KB  ✓  (H100 limit: 228 KB)

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
import jax.experimental.pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


# ---------------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------------

def _chunk_state_mosaic_kernel(
    x_t_ref,    # GMEM ref: (1, BM, chunk_size) f32  — A-matrix (x transposed)
    B_ref,      # GMEM ref: (1, chunk_size, BN) f32  — B-matrix (to be scaled)
    scale_ref,  # GMEM ref: (1, chunk_size)    f32  — precomputed scale
    states_ref, # GMEM ref: (1, BM, BN)        f32  — output
    *,
    BM: int,
    BK: int,
    BN: int,
    chunk_size: int,
    num_stages: int,
):
    """
    One CTA handles one (batch, chunk, head, BM-tile, BN-tile) output block.

    emit_pipeline runs K = chunk_size // BK steps.  Each step:
      - TMA-loads (BM, BK) of x_T and (BK, BN) of B into SMEM
      - Scales B by scale in registers
      - Writes both to bf16 SMEM scratch with WGMMA-compatible layout
      - Issues WGMMA and accumulates into a register ACC

    The final ACC is written back to states_ref.
    """
    K = chunk_size // BK

    # Swizzle/tiling transforms for WGMMA A-matrix scratch (BM, BK) bf16
    a_swizzle = plgpu.find_swizzle(BK * 16)          # 16 bits per bf16 element
    a_transforms = (
        plgpu.TilingTransform((8, a_swizzle // 2)),
        plgpu.SwizzleTransform(a_swizzle),
    )

    # Swizzle/tiling transforms for WGMMA B-matrix scratch (BK, BN) bf16
    b_swizzle = plgpu.find_swizzle(BN * 16)
    b_transforms = (
        plgpu.TilingTransform((8, b_swizzle // 2)),
        plgpu.SwizzleTransform(b_swizzle),
    )

    def pipeline_body(step_indices, x_t_smem, B_smem, scale_smem, carry):
        # x_t_smem : SMEM (BM, BK) f32  — A-matrix tile (raw, no transforms)
        # B_smem   : SMEM (BK, BN) f32  — B-matrix tile (raw, no transforms)
        # scale_smem: SMEM (BK,)  f32  — scale slice

        # Load from SMEM into registers, cast for WGMMA
        x_t_tile = x_t_smem[...].astype(jnp.bfloat16)          # (BM, BK) bf16
        B_tile   = B_smem[...]                                   # (BK, BN) f32
        scale_k  = scale_smem[...]                               # (BK,)    f32

        # Scale B in registers, cast to bf16 for WGMMA
        B_scaled = (B_tile * scale_k[:, None]).astype(jnp.bfloat16)  # (BK, BN) bf16

        def do_wgmma(acc_ref, a_scratch_ref, b_scratch_ref):
            # Write A and B to SMEM scratch with WGMMA-compatible layout
            a_scratch_ref[...] = x_t_tile   # (BM, BK) bf16 → A scratch
            b_scratch_ref[...] = B_scaled   # (BK, BN) bf16 → B scratch
            plgpu.commit_smem()
            plgpu.wgmma(acc_ref, a_scratch_ref, b_scratch_ref)
            plgpu.wgmma_wait(0)
            return acc_ref[...]

        tile_result = pl.run_scoped(
            do_wgmma,
            plgpu.ACC((BM, BN), jnp.float32),
            plgpu.SMEM((BM, BK), jnp.bfloat16, transforms=a_transforms),
            plgpu.SMEM((BK, BN), jnp.bfloat16, transforms=b_transforms),
        )
        return carry + tile_result

    # Emit double-buffered TMA pipeline over K reduction steps
    acc = plgpu.emit_pipeline(
        pipeline_body,
        grid=(K,),
        in_specs=[
            # x_T: (BM, chunk_size) GMEM → load (BM, BK) tile at column k
            plgpu.BlockSpec((BM, BK), lambda k: (0, k)),
            # B:   (chunk_size, BN) GMEM → load (BK, BN) tile at row k
            plgpu.BlockSpec((BK, BN), lambda k: (k, 0)),
            # scale: (chunk_size,) GMEM → load (BK,) slice at position k
            plgpu.BlockSpec((BK,), lambda k: (k,)),
        ],
        max_concurrent_steps=num_stages,
        init_carry=jnp.zeros((BM, BN), jnp.float32),
    )(x_t_ref.at[0], B_ref.at[0], scale_ref.at[0])

    # Write accumulated result to output
    states_ref[0, :, :] = acc


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

def chunk_state_fwd_mosaic(
    x,          # (batch, seqlen, nheads, headdim)            float32
    B,          # (batch, seqlen, ngroups, dstate)            float32
    dt,         # (batch, nheads, nchunks, chunk_size)        float32
    dA_cumsum,  # (batch, nheads, nchunks, chunk_size)        float32
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
    BM, BK, BN : tile sizes (default 64).  chunk_size must be divisible by BK.
    num_stages : TMA pipeline depth (default 2)

    Returns
    -------
    states : (batch, nchunks, nheads, headdim, dstate)  float32

    Same semantics as the Triton reference _chunk_state_fwd.
    """
    batch, seqlen, nheads, headdim = x.shape
    _, nheads_, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups  # Python int — constant

    assert nheads % ngroups == 0, f"nheads={nheads} must be divisible by ngroups={ngroups}"
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape

    # Adapt BK to chunk_size if necessary
    BK = min(BK, chunk_size)
    assert chunk_size % BK == 0, f"chunk_size={chunk_size} must be divisible by BK={BK}"

    # ------------------------------------------------------------------
    # Step 1: Pad seqlen to nchunks * chunk_size.
    # ------------------------------------------------------------------
    total_len = nchunks * chunk_size
    if seqlen < total_len:
        pad = total_len - seqlen
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        B = jnp.pad(B, ((0, 0), (0, pad), (0, 0), (0, 0)))

    # ------------------------------------------------------------------
    # Step 2: Precompute scale in JAX (avoids scalar GMEM reads in kernel).
    #
    #   dA_cs_last : (batch, nheads, nchunks, 1)
    #   scale      : (batch, nheads, nchunks, chunk_size)
    # ------------------------------------------------------------------
    dA_cs_last = dA_cumsum[:, :, :, -1:]
    scale = jnp.exp(jnp.minimum(dA_cs_last - dA_cumsum, 0.0)) * dt

    # ------------------------------------------------------------------
    # Step 3: Reshape x → (BCH, headdim_padded, chunk_size).
    #
    #   x: (batch, seqlen, nheads, headdim)
    #   → (batch, nchunks, chunk_size, nheads, headdim)
    #   → (batch, nchunks, nheads, headdim, chunk_size)   [transpose]
    #   → (BCH, headdim, chunk_size)                      [reshape]
    #   → (BCH, headdim_padded, chunk_size)               [pad]
    # ------------------------------------------------------------------
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    x_t = x.transpose(0, 1, 3, 4, 2)           # (batch, nchunks, nheads, headdim, chunk_size)
    BCH = batch * nchunks * nheads
    x_flat = x_t.reshape(BCH, headdim, chunk_size)

    headdim_padded = math.ceil(headdim / BM) * BM
    if headdim_padded > headdim:
        x_flat = jnp.pad(x_flat, ((0, 0), (0, headdim_padded - headdim), (0, 0)))

    # ------------------------------------------------------------------
    # Step 4: Reshape B → (BCG, chunk_size, dstate_padded).
    #
    #   B: (batch, seqlen, ngroups, dstate)
    #   → (batch, nchunks, chunk_size, ngroups, dstate)
    #   → (batch, nchunks, ngroups, chunk_size, dstate)   [transpose]
    #   → (BCG, chunk_size, dstate)                       [reshape]
    #   → (BCG, chunk_size, dstate_padded)                [pad]
    # ------------------------------------------------------------------
    B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    B = B.transpose(0, 1, 3, 2, 4)             # (batch, nchunks, ngroups, chunk_size, dstate)
    BCG = batch * nchunks * ngroups
    B_flat = B.reshape(BCG, chunk_size, dstate)

    dstate_padded = math.ceil(dstate / BN) * BN
    if dstate_padded > dstate:
        B_flat = jnp.pad(B_flat, ((0, 0), (0, 0), (0, dstate_padded - dstate)))

    # ------------------------------------------------------------------
    # Step 5: Flatten scale → (BCH, chunk_size).
    #
    #   scale: (batch, nheads, nchunks, chunk_size)
    #   → (batch, nchunks, nheads, chunk_size)   [transpose]
    #   → (BCH, chunk_size)                      [reshape]
    # ------------------------------------------------------------------
    scale_t = scale.transpose(0, 2, 1, 3)       # (batch, nchunks, nheads, chunk_size)
    scale_flat = scale_t.reshape(BCH, chunk_size)

    # ------------------------------------------------------------------
    # Step 6: Launch Pallas Mosaic GPU kernel.
    #
    #   Grid: (BCH, PM, PN) — all parallel.
    #   Each CTA computes states[bch, pm*BM:(pm+1)*BM, pn*BN:(pn+1)*BN].
    #
    #   B-matrix group mapping:
    #     bch       = batch_idx*(nchunks*nheads) + chunk_idx*nheads + head_idx
    #     head_idx  = bch % nheads
    #     chunk_batch = bch // nheads   (= batch_idx*nchunks + chunk_idx)
    #     group_idx = head_idx // ratio
    #     bcg       = chunk_batch * ngroups + group_idx
    # ------------------------------------------------------------------
    PM = headdim_padded // BM
    PN = dstate_padded  // BN

    kernel = partial(
        _chunk_state_mosaic_kernel,
        BM=BM, BK=BK, BN=BN,
        chunk_size=chunk_size,
        num_stages=num_stages,
    )

    def b_index_map(bch, pm, pn):
        head_idx    = bch % nheads
        chunk_batch = bch // nheads            # batch_idx*nchunks + chunk_idx
        group_idx   = head_idx // ratio        # ratio is a Python constant
        bcg         = chunk_batch * ngroups + group_idx
        return (bcg, 0, pn)

    states_flat, = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct(
                (BCH, headdim_padded, dstate_padded), jnp.float32
            )
        ],
        grid=(BCH, PM, PN),
        in_specs=[
            # x_T: each CTA sees (1, BM, chunk_size) tile
            pl.BlockSpec((1, BM, chunk_size), lambda bch, pm, pn: (bch, pm, 0)),
            # B:   each CTA sees (1, chunk_size, BN) tile from group bcg
            pl.BlockSpec((1, chunk_size, BN), b_index_map),
            # scale: each CTA sees (1, chunk_size) slice
            pl.BlockSpec((1, chunk_size), lambda bch, pm, pn: (bch, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, BM, BN), lambda bch, pm, pn: (bch, pm, pn)),
        ],
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel", "parallel", "parallel"],
        ),
    )(x_flat, B_flat, scale_flat)

    # ------------------------------------------------------------------
    # Step 7: Reshape output → (batch, nchunks, nheads, headdim, dstate).
    #
    #   states_flat: (BCH, headdim_padded, dstate_padded)
    #   slice:       (BCH, headdim, dstate)
    #   reshape:     (batch, nchunks, nheads, headdim, dstate)
    # ------------------------------------------------------------------
    states = (
        states_flat[:, :headdim, :dstate]
        .reshape(batch, nchunks, nheads, headdim, dstate)
    )

    return states


# ---------------------------------------------------------------------------
# Triton-compatible alias
# ---------------------------------------------------------------------------

def chunk_state_fwd(
    x,
    B,
    dt,
    dA_cumsum,
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """Drop-in replacement for mamba_ssm._chunk_state_fwd (JAX/Pallas version)."""
    return chunk_state_fwd_mosaic(
        x, B, dt, dA_cumsum,
        BM=BM, BK=BK, BN=BN,
        num_stages=num_stages,
    )
