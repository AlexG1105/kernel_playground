"""
pallas_mosaic_gpu_mamba/bmm_chunk_fwd.py

Mosaic GPU (H100/H200) Pallas implementation of _bmm_chunk_fwd.

Algorithm
---------
Given:
  a : (batch, seqlen, ngroups, k)  float32   (typically C from SSM)
  b : (batch, seqlen, ngroups, k)  float32   (typically B from SSM)

Produces:
  out : (batch, nchunks, ngroups, chunk_size, chunk_size)  float32

For each (batch b, chunk c, group g):
  out[b,c,g] = a[b, c*Q:(c+1)*Q, g, :] @ b[b, c*Q:(c+1)*Q, g, :].T
             = (chunk_size, k) @ (k, chunk_size)
             = (chunk_size, chunk_size)

Optional flags (handled in post-processing, no kernel changes):
  causal  : if True, zero out upper triangle of each chunk's output
  seq_idx : (batch, seqlen) int32.  After matmul, zero out entries where
            seq_idx[m] != seq_idx[n] within each chunk.

Grid and tile design
--------------------
pl.kernel with Mesh grid: (BCG, PM, PN)
  BCG = batch * nchunks * ngroups  — one CTA per (batch, chunk, group)
  PM  = chunk_size_padded // BM    — tiles over output rows (chunk_size)
  PN  = chunk_size_padded // BN    — tiles over output cols (chunk_size)

  All three dimensions are "parallel" (independent CTAs).
  pl.kernel / core_map gives the kernel body GMEM refs.

Matmul: (BM, BK) @ (BK, BN) accumulated in f32 via WGMMA.
  BM = BN = BK = 64 (defaults; typical for H100 WGMMA)

Mosaic GPU emit_pipeline + WGMMA
---------------------------------
emit_pipeline receives GMEM refs and manages TMA double-buffered loading
into swizzled SMEM.  The pipeline body passes swizzled SMEM refs directly
to WGMMA — no intermediate register copies needed.

  step k (K_tiles = dstate // BK):
    TMA loads (BM, BK) bf16 of a into swizzled SMEM → a_smem
    TMA loads (BK, BN) bf16 of b_T into swizzled SMEM → b_smem
    wgmma(acc, a_smem, b_smem) — accumulates in f32 ACC registers

  After all K_tiles steps:
    out_ref[bcg, pm*BM:(pm+1)*BM, pn*BN:(pn+1)*BN] = acc[...]

Note: b is transposed in preprocessing: b_T has shape (BCG, k, chunk_size)
so the matmul becomes a @ b_T = (chunk_size, k) @ (k, chunk_size).

SMEM budget (BM=BN=BK=64, num_stages=2)
-----------------------------------------
  a staging   : 2 * 64*64*2 =  16 KB  (bf16, double-buffered)
  b_T staging : 2 * 64*64*2 =  16 KB  (bf16, double-buffered)
  ACC (regs)  : 64*64*4     =  16 KB  (f32, not SMEM)
  Total SMEM  :             ~  32 KB  (H100 limit: 228 KB)

Usage
-----
  from pallas_mosaic_gpu_mamba.bmm_chunk_fwd import bmm_chunk_fwd_mosaic

  CB = bmm_chunk_fwd_mosaic(C, B, chunk_size)
  CB = bmm_chunk_fwd_mosaic(C, B, chunk_size, causal=True)
  CB = bmm_chunk_fwd_mosaic(C, B, chunk_size, seq_idx=seq_idx, causal=True)
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

def _bmm_chunk_fwd_kernel_body(
    a_ref,      # GMEM ref: (BCG, chunk_size_padded, k_padded) bf16
    b_T_ref,    # GMEM ref: (BCG, k_padded, chunk_size_padded) bf16
    out_ref,    # GMEM ref: (BCG, chunk_size_padded, chunk_size_padded) f32
    *,
    BM: int,
    BK: int,
    BN: int,
    k_padded: int,
    num_stages: int,
):
    """
    One CTA computes one (BM, BN) output tile for a given (bcg, pm, pn).

    Uses emit_pipeline for double-buffered TMA loading of bf16 tiles
    into WGMMA-compatible swizzled SMEM.
    """
    bcg = lax.axis_index("bcg")
    pm  = lax.axis_index("pm")
    pn  = lax.axis_index("pn")

    K_tiles = k_padded // BK

    # GMEM sub-refs for this CTA's portion
    a_gmem   = a_ref.at[bcg, pl.ds(pm * BM, BM), :]      # (BM, k_padded) bf16
    b_T_gmem = b_T_ref.at[bcg, :, pl.ds(pn * BN, BN)]    # (k_padded, BN) bf16

    # Swizzle/tiling transforms for WGMMA-compatible SMEM layout
    a_swizzle = plgpu.find_swizzle(BK * 16)               # 16 bits per bf16
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
        def pipeline_body(step, a_smem, b_smem):
            plgpu.wgmma(acc_ref, a_smem, b_smem)
            plgpu.wgmma_wait(0)

        plgpu.emit_pipeline(
            pipeline_body,
            grid=(K_tiles,),
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
        )(a_gmem, b_T_gmem)

        # Write accumulated f32 result to output GMEM
        out_ref[bcg, pl.ds(pm * BM, BM), pl.ds(pn * BN, BN)] = (
            acc_ref[...].astype(jnp.float32)
        )

    pl.run_scoped(_with_acc, plgpu.ACC((BM, BN), jnp.float32))


# ---------------------------------------------------------------------------
# Preprocessing (JAX/XLA ops — reshape, pad, bf16 cast)
# ---------------------------------------------------------------------------

def bmm_chunk_preprocess(
    a,          # (batch, seqlen, ngroups, k) float32
    b,          # (batch, seqlen, ngroups, k) float32
    chunk_size: int,
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
):
    """
    Preprocess inputs for the Pallas kernel.

    Reshapes a → (BCG, chunk_size, k) and b → (BCG, k, chunk_size) (transposed),
    pads to tile multiples, and casts to bf16.

    Returns
    -------
    a_flat   : (BCG, chunk_size_padded, k_padded) bf16
    b_T_flat : (BCG, k_padded, chunk_size_padded) bf16
    meta     : dict with BCG, chunk_size, chunk_size_padded, k, k_padded,
               batch, nchunks, ngroups
    """
    batch, seqlen, ngroups, k = a.shape
    assert b.shape == a.shape

    nchunks = math.ceil(seqlen / chunk_size)

    # Pad seqlen to multiple of chunk_size
    total_len = nchunks * chunk_size
    if seqlen < total_len:
        pad = total_len - seqlen
        a = jnp.pad(a, ((0, 0), (0, pad), (0, 0), (0, 0)))
        b = jnp.pad(b, ((0, 0), (0, pad), (0, 0), (0, 0)))

    # Reshape: (batch, nchunks, chunk_size, ngroups, k)
    BCG = batch * nchunks * ngroups

    a_chunked = a.reshape(batch, nchunks, chunk_size, ngroups, k)
    a_chunked = a_chunked.transpose(0, 1, 3, 2, 4)  # (batch, nchunks, ngroups, chunk_size, k)
    a_flat = a_chunked.reshape(BCG, chunk_size, k)

    b_chunked = b.reshape(batch, nchunks, chunk_size, ngroups, k)
    b_chunked = b_chunked.transpose(0, 1, 3, 4, 2)  # (batch, nchunks, ngroups, k, chunk_size)
    b_T_flat = b_chunked.reshape(BCG, k, chunk_size)

    # Pad chunk_size to multiples of BM and BN
    chunk_size_padded = math.ceil(chunk_size / max(BM, BN)) * max(BM, BN)
    # Pad k to multiple of BK; shrink BK if k is small
    BK = min(BK, k)
    k_padded = math.ceil(k / BK) * BK

    if chunk_size_padded > chunk_size:
        pad_cs = chunk_size_padded - chunk_size
        a_flat = jnp.pad(a_flat, ((0, 0), (0, pad_cs), (0, 0)))
        b_T_flat = jnp.pad(b_T_flat, ((0, 0), (0, 0), (0, pad_cs)))
    if k_padded > k:
        pad_k = k_padded - k
        a_flat = jnp.pad(a_flat, ((0, 0), (0, 0), (0, pad_k)))
        b_T_flat = jnp.pad(b_T_flat, ((0, 0), (0, pad_k), (0, 0)))

    a_flat = a_flat.astype(jnp.bfloat16)
    b_T_flat = b_T_flat.astype(jnp.bfloat16)

    meta = dict(
        BCG=BCG, chunk_size=chunk_size, chunk_size_padded=chunk_size_padded,
        k=k, k_padded=k_padded,
        batch=batch, nchunks=nchunks, ngroups=ngroups,
        BK=BK,
    )
    return a_flat, b_T_flat, meta


# ---------------------------------------------------------------------------
# Kernel-only launch (no preprocessing)
# ---------------------------------------------------------------------------

def bmm_chunk_kernel_only(
    a_flat,      # (BCG, chunk_size_padded, k_padded) bf16
    b_T_flat,    # (BCG, k_padded, chunk_size_padded) bf16
    *,
    BCG: int,
    chunk_size: int,
    chunk_size_padded: int,
    k: int,
    k_padded: int,
    batch: int,
    nchunks: int,
    ngroups: int,
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """
    Launch just the Pallas kernel + output reshape.

    Takes pre-processed bf16 inputs.
    """
    PM = chunk_size_padded // BM
    PN = chunk_size_padded // BN

    mesh = plgpu.Mesh(
        grid=(BCG, PM, PN),
        grid_names=("bcg", "pm", "pn"),
    )

    kernel_fn = pl.kernel(
        partial(
            _bmm_chunk_fwd_kernel_body,
            BM=BM, BK=BK, BN=BN,
            k_padded=k_padded,
            num_stages=num_stages,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (BCG, chunk_size_padded, chunk_size_padded), jnp.float32
        ),
        mesh=mesh,
    )

    out_flat = kernel_fn(a_flat, b_T_flat)

    # Slice away padding and reshape to (batch, nchunks, ngroups, chunk_size, chunk_size)
    out = (
        out_flat[:, :chunk_size, :chunk_size]
        .reshape(batch, nchunks, ngroups, chunk_size, chunk_size)
    )
    return out


# ---------------------------------------------------------------------------
# Public wrapper (end-to-end: preprocess + kernel)
# ---------------------------------------------------------------------------

def bmm_chunk_fwd_mosaic(
    a,              # (batch, seqlen, ngroups, k) float32
    b,              # (batch, seqlen, ngroups, k) float32
    chunk_size: int,
    seq_idx=None,   # (batch, seqlen) int32, optional
    causal=False,   # if True, zero upper triangle
    output_dtype=None,
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """
    H100/H200 Pallas Mosaic GPU port of _bmm_chunk_fwd.

    Computes the per-chunk batched matrix product:
      out[b,c,g] = a[b, c*Q:(c+1)*Q, g, :] @ b[b, c*Q:(c+1)*Q, g, :].T

    Parameters
    ----------
    a          : (batch, seqlen, ngroups, k) float32
    b          : (batch, seqlen, ngroups, k) float32
    chunk_size : int
    seq_idx    : (batch, seqlen) int32, optional
        Sequence index per position.  After matmul, entries where
        seq_idx[m] != seq_idx[n] within each chunk are zeroed.
    causal     : bool, default False
        If True, zero the upper triangle of each chunk output.
    output_dtype : optional, defaults to float32
    BM, BK, BN : tile sizes (default 64)
    num_stages : TMA pipeline depth (default 2)

    Returns
    -------
    out : (batch, nchunks, ngroups, chunk_size, chunk_size) float32
    """
    batch, seqlen = a.shape[0], a.shape[1]
    nchunks = math.ceil(seqlen / chunk_size)
    output_dtype = jnp.float32 if output_dtype is None else output_dtype

    a_flat, b_T_flat, meta = bmm_chunk_preprocess(
        a, b, chunk_size, BM=BM, BK=BK, BN=BN,
    )
    out = bmm_chunk_kernel_only(
        a_flat, b_T_flat,
        BM=BM, BK=meta['BK'], BN=BN, num_stages=num_stages,
        **{kk: meta[kk] for kk in (
            'BCG', 'chunk_size', 'chunk_size_padded', 'k', 'k_padded',
            'batch', 'nchunks', 'ngroups',
        )},
    )

    # ------------------------------------------------------------------
    # Post-processing: causal mask and seq_idx mask
    # ------------------------------------------------------------------
    if causal:
        # Zero upper triangle of each (chunk_size, chunk_size) block
        causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
        out = jnp.where(causal_mask[None, None, None, :, :], out, 0.0)

    if seq_idx is not None:
        # Pad seqlen to nchunks * chunk_size (match kernel padding)
        total_len = nchunks * chunk_size
        if seqlen < total_len:
            # Pad with -1 so padded positions never match anything
            seq_idx = jnp.pad(
                seq_idx,
                ((0, 0), (0, total_len - seqlen)),
                constant_values=-1,
            )
        # Reshape to (batch, nchunks, chunk_size)
        seq_idx_chunked = seq_idx.reshape(batch, nchunks, chunk_size)
        # Build mask: seq_idx[m] == seq_idx[n]  →  (batch, nchunks, Q, Q)
        # Use -1 vs -2 for m vs n OOB so they never match (like Triton)
        seq_mask = (
            seq_idx_chunked[:, :, :, None]
            == seq_idx_chunked[:, :, None, :]
        )  # (batch, nchunks, chunk_size, chunk_size)
        # Broadcast over ngroups: out is (batch, nchunks, ngroups, Q, Q)
        out = jnp.where(seq_mask[:, :, None, :, :], out, 0.0)

    return out.astype(output_dtype)


# ---------------------------------------------------------------------------
# Triton-compatible alias
# ---------------------------------------------------------------------------

def bmm_chunk_fwd(
    a,
    b,
    chunk_size: int,
    seq_idx=None,
    causal=False,
    output_dtype=None,
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """Drop-in replacement for mamba_ssm._bmm_chunk_fwd (JAX/Pallas version)."""
    return bmm_chunk_fwd_mosaic(
        a, b, chunk_size,
        seq_idx=seq_idx, causal=causal,
        output_dtype=output_dtype,
        BM=BM, BK=BK, BN=BN,
        num_stages=num_stages,
    )
