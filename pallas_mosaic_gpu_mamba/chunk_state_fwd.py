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

Scale and group mapping precomputed in wrapper
----------------------------------------------
  scale = exp(min(dA_cs_last - dA_cumsum, 0)) * dt
  B_scaled = B_expanded[head→group] * scale[:, :, None]

  This is done in JAX/XLA before the kernel, so the kernel only sees
  B_scaled indexed by BCH (no group mapping, no scale input).

  Both x_T and B_scaled are cast to bf16 in the wrapper since H100
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
    x_t_ref,    # GMEM ref: (BCH, headdim_padded, chunk_size) bf16
    B_ref,      # GMEM ref: (BCH, chunk_size, dstate_padded) bf16  (pre-scaled)
    states_ref, # GMEM ref: (BCH, headdim_padded, dstate_padded) f32
    *,
    BM: int,
    BK: int,
    BN: int,
    chunk_size: int,
    num_stages: int,
):
    """
    One CTA computes one (BM, BN) output tile for a given (bch, pm, pn).

    Uses emit_pipeline for double-buffered TMA loading of bf16 tiles
    into WGMMA-compatible swizzled SMEM.  The pipeline body passes
    swizzled SMEM refs directly to wgmma (no intermediate register copy).
    """
    bch = lax.axis_index("bch")
    pm  = lax.axis_index("pm")
    pn  = lax.axis_index("pn")

    K = chunk_size // BK

    # GMEM sub-refs for this CTA's portion of the arrays
    x_t_gmem = x_t_ref.at[bch, pl.ds(pm * BM, BM), :]     # (BM, chunk_size) bf16
    B_gmem   = B_ref.at[bch, :, pl.ds(pn * BN, BN)]        # (chunk_size, BN) bf16

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
        def pipeline_body(step, a_smem, b_smem, carry):
            plgpu.wgmma(acc_ref, a_smem, b_smem)
            plgpu.wgmma_wait(0)
            return ()

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
    # Step 3: Reshape x → (BCH, headdim_padded, chunk_size) bf16.
    #
    #   x: (batch, seqlen, nheads, headdim)
    #   → (batch, nchunks, chunk_size, nheads, headdim)
    #   → (batch, nchunks, nheads, headdim, chunk_size)   [transpose]
    #   → (BCH, headdim, chunk_size)                      [reshape]
    #   → (BCH, headdim_padded, chunk_size)               [pad]
    #   → bf16
    # ------------------------------------------------------------------
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    x_t = x.transpose(0, 1, 3, 4, 2)           # (batch, nchunks, nheads, headdim, chunk_size)
    BCH = batch * nchunks * nheads
    x_flat = x_t.reshape(BCH, headdim, chunk_size)

    headdim_padded = math.ceil(headdim / BM) * BM
    if headdim_padded > headdim:
        x_flat = jnp.pad(x_flat, ((0, 0), (0, headdim_padded - headdim), (0, 0)))

    x_flat = x_flat.astype(jnp.bfloat16)       # WGMMA requires bf16

    # ------------------------------------------------------------------
    # Step 4: Pre-scale B and reshape to (BCH, chunk_size, dstate_padded) bf16.
    #
    #   B: (batch, seqlen, ngroups, dstate)
    #   → (batch, nchunks, ngroups, chunk_size, dstate)
    #   scale: (batch, nheads, nchunks, chunk_size)
    #
    #   We expand B from (BCG,) to (BCH,) by repeating each group's B
    #   across its heads, then multiply by scale.  This avoids passing
    #   scale as a separate kernel input and eliminates the group index
    #   mapping inside the kernel.
    # ------------------------------------------------------------------
    B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    B = B.transpose(0, 1, 3, 2, 4)             # (batch, nchunks, ngroups, chunk_size, dstate)
    BCG = batch * nchunks * ngroups
    B_flat = B.reshape(BCG, chunk_size, dstate)

    dstate_padded = math.ceil(dstate / BN) * BN
    if dstate_padded > dstate:
        B_flat = jnp.pad(B_flat, ((0, 0), (0, 0), (0, dstate_padded - dstate)))

    # Map each bch index to its corresponding bcg index for group expansion.
    #   bch = batch_idx*(nchunks*nheads) + chunk_idx*nheads + head_idx
    #   bcg = (batch_idx*nchunks + chunk_idx)*ngroups + head_idx // ratio
    bch_indices = jnp.arange(BCH)
    chunk_batch = bch_indices // nheads          # batch_idx*nchunks + chunk_idx
    head_idx = bch_indices % nheads
    group_idx = head_idx // ratio
    bcg_indices = chunk_batch * ngroups + group_idx

    B_expanded = B_flat[bcg_indices]             # (BCH, chunk_size, dstate_padded)

    # Flatten scale: (batch, nheads, nchunks, chunk_size)
    #   → (batch, nchunks, nheads, chunk_size) → (BCH, chunk_size)
    scale_t = scale.transpose(0, 2, 1, 3)
    scale_flat = scale_t.reshape(BCH, chunk_size)

    # Pre-multiply: B_scaled[bch, t, :] = B[bcg, t, :] * scale[bch, t]
    B_scaled = (B_expanded * scale_flat[:, :, None]).astype(jnp.bfloat16)

    # ------------------------------------------------------------------
    # Step 5: Launch Pallas Mosaic GPU kernel.
    #
    #   pl.kernel uses core_map to dispatch over a Mesh.  The kernel
    #   body gets GMEM refs (not SMEM), which emit_pipeline can TMA-load
    #   into swizzled SMEM for WGMMA.
    # ------------------------------------------------------------------
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
        ),
        out_shape=jax.ShapeDtypeStruct(
            (BCH, headdim_padded, dstate_padded), jnp.float32
        ),
        mesh=mesh,
    )

    states_flat = kernel_fn(x_flat, B_scaled)

    # ------------------------------------------------------------------
    # Step 6: Reshape output → (batch, nchunks, nheads, headdim, dstate).
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
