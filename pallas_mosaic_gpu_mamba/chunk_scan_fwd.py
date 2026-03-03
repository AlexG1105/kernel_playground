"""
pallas_mosaic_gpu_mamba/chunk_scan_fwd.py

Mosaic GPU (H100/H200) Pallas implementation of _chunk_scan_fwd.

Algorithm
---------
Given:
  cb        : (batch, nchunks, ngroups, chunk_size, chunk_size)  float32
  x         : (batch, seqlen, nheads, hdim)                     float32
  dt        : (batch, nheads, nchunks, chunk_size)               float32
  dA_cumsum : (batch, nheads, nchunks, chunk_size)               float32
  C         : (batch, seqlen, ngroups, dstate)                   float32
  states    : (batch, nchunks, nheads, hdim, dstate)             float32  (prev_states)
  D         : (nheads,) or (nheads, hdim)                        float32, optional
  z         : (batch, seqlen, nheads, hdim)                      float32, optional

Produces:
  out       : (batch, seqlen, nheads, hdim)                      float32
  out_x     : (batch, seqlen, nheads, hdim)                      float32, or None

Full computation per output element:
  out[m,n] = exp(dA_cs[m]) * (C[m,:] @ states[:,n])
           + exp(dA_cs[m]) * sum_k CB_causal[m,k] * dt[k] * exp(-dA_cs[k]) * x[k,n]
           + D * x[m,n]                                      (if D)
           * z * sigmoid(z)                                   (if z)

seq_idx handling (no kernel changes):
  When seq_idx is provided, positions where the sequence changed from the
  previous chunk must have their state contribution zeroed (the scan
  contribution is already correct because CB was seq_idx-masked upstream).
  We achieve this by zeroing C at those positions in preprocessing:
    C[m, :] = 0 where seq_idx[m] != seq_idx_prev_chunk
  This zeros the C @ states_T product for those rows while leaving the
  CB @ x_scaled scan contribution untouched.

Factoring for WGMMA compatibility
----------------------------------
WGMMA cannot do elementwise ops between TMA load and matmul.  The Triton
kernel scales CB in registers per K-tile — we can't do this.

Key insight: exp(dA_cs[m] - dA_cs[k]) = exp(dA_cs[m]) * exp(-dA_cs[k]).
Since A < 0, dA_cumsum is monotonically decreasing, so for causal m >= k:
dA_cs[m] - dA_cs[k] <= 0, and min(...,0) clamp is redundant.

Define: x_scaled[k,n] = dt[k] * exp(-dA_cs[k]) * x[k,n]  (precomputed)

Then the kernel computes (two WGMMA matmuls sharing one ACC):
  acc = CB_causal @ x_scaled + C @ states_T

Post-processing applies: out = exp(dA_cs[m]) * acc + D*x

CB and C stay group-indexed — no per-head memory expansion.

Grid and tile design
--------------------
pl.kernel with Mesh grid: (BCH, PM, PN)
  BCH = batch * nchunks * nheads  — one CTA per (batch, chunk, head)
  PM  = chunk_size_padded // BM   — tiles over output rows (chunk positions)
  PN  = hdim_padded // BN         — tiles over output cols (hdim)

  All three dimensions are parallel.

Matmul 1 (scan contribution): CB_causal @ x_scaled
  A: (BM, BK) from CB_causal  (group-indexed via bcg)
  B: (BK, BN) from x_scaled   (per-head via bch)
  K-loop: K1_max = min((pm+1)*BM/BK, Q_padded/BK)  — causal K_MAX optimization

Matmul 2 (state contribution): C @ states_T
  A: (BM, BK_ds) from C       (group-indexed via bcg)
  B: (BK_ds, BN) from states_T (per-head via bch)
  K-loop: K2 = dstate_padded / BK_ds  — full reduction

Both matmuls accumulate into the same f32 ACC register file.

SMEM budget (BM=BN=BK=64, num_stages=2)
-----------------------------------------
  Pipeline 1 : 2 * 64*64*2 * 2 =  32 KB  (bf16, double-buffered, sequential)
  Pipeline 2 : 2 * 64*64*2 * 2 =  32 KB  (bf16, double-buffered, sequential)
  ACC (regs) : 64*64*4          =  16 KB  (f32, not SMEM)
  Peak SMEM  :                  ~  32 KB  (pipelines don't overlap)

Usage
-----
  from pallas_mosaic_gpu_mamba.chunk_scan_fwd import chunk_scan_fwd_mosaic

  out, out_x = chunk_scan_fwd_mosaic(cb, x, dt, dA_cumsum, C, states)
  out, out_x = chunk_scan_fwd_mosaic(cb, x, dt, dA_cumsum, C, states, D=D, z=z)
  out, out_x = chunk_scan_fwd_mosaic(cb, x, dt, dA_cumsum, C, states, seq_idx=seq_idx)
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

def _chunk_scan_fwd_kernel_body(
    cb_ref,        # GMEM ref: (BCG, chunk_size_padded, chunk_size_padded) bf16
    x_scaled_ref,  # GMEM ref: (BCH, chunk_size_padded, hdim_padded) bf16
    C_ref,         # GMEM ref: (BCG, chunk_size_padded, dstate_padded) bf16
    states_T_ref,  # GMEM ref: (BCH, dstate_padded, hdim_padded) bf16
    out_ref,       # GMEM ref: (BCH, chunk_size_padded, hdim_padded) f32
    *,
    BM: int,
    BK_cs: int,
    BK_ds: int,
    BN: int,
    chunk_size_padded: int,
    dstate_padded: int,
    num_stages: int,
    nheads: int,
    ngroups: int,
):
    """
    One CTA computes one (BM, BN) output tile for a given (bch, pm, pn).

    Two sequential emit_pipeline calls share one ACC:
      1. CB_causal @ x_scaled  (scan contribution, with causal K_MAX)
      2. C @ states_T           (state contribution, full dstate)

    CB and C are group-indexed (BCG); x_scaled and states_T are per-head (BCH).
    Group mapping: bcg = (bch // nheads) * ngroups + (bch % nheads) // ratio
    """
    bch = lax.axis_index("bch")
    pm  = lax.axis_index("pm")
    pn  = lax.axis_index("pn")

    # K-tiles for each matmul
    K1_tiles = chunk_size_padded // BK_cs     # total tiles for CB @ x_scaled
    K2_tiles = dstate_padded // BK_ds         # total tiles for C @ states_T

    # Causal K_MAX: only process columns 0..(pm+1)*BM of CB (lower-triangular)
    K1_max = jnp.minimum((pm + 1) * (BM // BK_cs), K1_tiles)

    # Map bch → bcg for group-indexed CB and C lookup
    ratio = nheads // ngroups
    chunk_batch = bch // nheads               # batch_idx * nchunks + chunk_idx
    head_idx = bch % nheads
    bcg = chunk_batch * ngroups + head_idx // ratio

    # GMEM sub-refs for this CTA
    cb_gmem       = cb_ref.at[bcg, pl.ds(pm * BM, BM), :]        # (BM, Q_padded)
    x_scaled_gmem = x_scaled_ref.at[bch, :, pl.ds(pn * BN, BN)]  # (Q_padded, BN)
    C_gmem        = C_ref.at[bcg, pl.ds(pm * BM, BM), :]          # (BM, dstate_padded)
    states_T_gmem = states_T_ref.at[bch, :, pl.ds(pn * BN, BN)]   # (dstate_padded, BN)

    # Swizzle/tiling transforms for WGMMA-compatible SMEM layout
    # Matmul 1: A-side = CB (BM, BK_cs), B-side = x_scaled (BK_cs, BN)
    a1_swizzle = plgpu.find_swizzle(BK_cs * 16)
    a1_transforms = (
        plgpu.TilingTransform((8, a1_swizzle // 2)),
        plgpu.SwizzleTransform(a1_swizzle),
    )
    b_swizzle = plgpu.find_swizzle(BN * 16)
    b_transforms = (
        plgpu.TilingTransform((8, b_swizzle // 2)),
        plgpu.SwizzleTransform(b_swizzle),
    )
    # Matmul 2: A-side = C (BM, BK_ds), B-side = states_T (BK_ds, BN)
    a2_swizzle = plgpu.find_swizzle(BK_ds * 16)
    a2_transforms = (
        plgpu.TilingTransform((8, a2_swizzle // 2)),
        plgpu.SwizzleTransform(a2_swizzle),
    )

    def _with_acc(acc_ref):
        # --- Matmul 1: CB_causal @ x_scaled (causal K_MAX) ---
        def pipeline_body_1(step, a_smem, b_smem):
            plgpu.wgmma(acc_ref, a_smem, b_smem)
            plgpu.wgmma_wait(0)

        plgpu.emit_pipeline(
            pipeline_body_1,
            grid=(K1_max,),
            in_specs=[
                plgpu.BlockSpec(
                    (BM, BK_cs), lambda k: (0, k),
                    transforms=a1_transforms,
                ),
                plgpu.BlockSpec(
                    (BK_cs, BN), lambda k: (k, 0),
                    transforms=b_transforms,
                ),
            ],
            max_concurrent_steps=num_stages,
        )(cb_gmem, x_scaled_gmem)

        # --- Matmul 2: C @ states_T (full dstate reduction) ---
        def pipeline_body_2(step, a_smem, b_smem):
            plgpu.wgmma(acc_ref, a_smem, b_smem)
            plgpu.wgmma_wait(0)

        plgpu.emit_pipeline(
            pipeline_body_2,
            grid=(K2_tiles,),
            in_specs=[
                plgpu.BlockSpec(
                    (BM, BK_ds), lambda k: (0, k),
                    transforms=a2_transforms,
                ),
                plgpu.BlockSpec(
                    (BK_ds, BN), lambda k: (k, 0),
                    transforms=b_transforms,
                ),
            ],
            max_concurrent_steps=num_stages,
        )(C_gmem, states_T_gmem)

        # Write accumulated f32 result to output GMEM
        out_ref[bch, pl.ds(pm * BM, BM), pl.ds(pn * BN, BN)] = (
            acc_ref[...].astype(jnp.float32)
        )

    pl.run_scoped(_with_acc, plgpu.ACC((BM, BN), jnp.float32))


# ---------------------------------------------------------------------------
# Preprocessing (JAX/XLA ops — reshape, scale, pad, bf16 cast)
# ---------------------------------------------------------------------------

def chunk_scan_preprocess(
    cb,         # (batch, nchunks, ngroups, chunk_size, chunk_size) float32
    x,          # (batch, seqlen, nheads, hdim) float32
    dt,         # (batch, nheads, nchunks, chunk_size) float32
    dA_cumsum,  # (batch, nheads, nchunks, chunk_size) float32
    C,          # (batch, seqlen, ngroups, dstate) float32
    states,     # (batch, nchunks, nheads, hdim, dstate) float32
    seq_idx=None,  # (batch, seqlen) int32, optional
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
):
    """
    Preprocess inputs for the chunk_scan Pallas kernel.

    Key factoring: exp(dA_cs[m] - dA_cs[k]) = exp(dA_cs[m]) * exp(-dA_cs[k]).
    We fold dt[k] * exp(-dA_cs[k]) into x (→ x_scaled), and defer
    exp(dA_cs[m]) to post-processing.  CB stays group-indexed.

    When seq_idx is provided, C is zeroed at positions where the sequence
    changed from the previous chunk, which zeros the state contribution
    (C @ states_T) for those rows while leaving the scan contribution
    (CB @ x_scaled) unaffected.

    Returns
    -------
    cb_flat      : (BCG, Q_padded, Q_padded) bf16           — causal-masked, group-indexed
    x_scaled_flat: (BCH, Q_padded, hdim_padded) bf16        — per-head
    C_flat       : (BCG, Q_padded, dstate_padded) bf16      — group-indexed
    states_T_flat: (BCH, dstate_padded, hdim_padded) bf16   — per-head
    meta         : dict with dimensions and padding info
    """
    batch, seqlen, nheads, hdim = x.shape
    _, nheads_, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape

    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert C.shape[:2] == (batch, seqlen)
    assert states.shape == (batch, nchunks, nheads, hdim, dstate)
    assert nheads % ngroups == 0

    BCH = batch * nchunks * nheads
    BCG = batch * nchunks * ngroups

    # --- Pad seqlen to nchunks * chunk_size ---
    total_len = nchunks * chunk_size
    if seqlen < total_len:
        pad = total_len - seqlen
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        C = jnp.pad(C, ((0, 0), (0, pad), (0, 0), (0, 0)))
        if seq_idx is not None:
            seq_idx = jnp.pad(seq_idx, ((0, 0), (0, pad)), constant_values=-1)

    # --- seq_idx handling: zero C where sequence changed from prev chunk ---
    if seq_idx is not None:
        # seq_idx_prev: for chunk 0 use 0 (matches Triton's seq_idx=0 init),
        # for chunk c use seq_idx at position c*chunk_size - 1.
        #   prev positions: [-1, chunk_size-1, 2*chunk_size-1, ...]
        chunk_starts = jnp.arange(nchunks) * chunk_size  # [0, Q, 2Q, ...]
        prev_positions = chunk_starts - 1                  # [-1, Q-1, 2Q-1, ...]
        # For chunk 0, prev_positions=-1 → clamp to 0 and set seq_idx_prev=0
        seq_idx_prev = jnp.where(
            prev_positions >= 0,
            seq_idx[:, jnp.maximum(prev_positions, 0)],  # (batch, nchunks)
            0,
        )  # (batch, nchunks)

        # seq_idx for each position in each chunk
        seq_idx_full = seq_idx.reshape(batch, nchunks, chunk_size)  # (batch, nchunks, Q)

        # Mask: True where seq_idx[m] == seq_idx_prev for this chunk
        same_seq = seq_idx_full == seq_idx_prev[:, :, None]  # (batch, nchunks, Q)

        # C is (batch, total_len, ngroups, dstate) → reshape to chunks
        C_chunked = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        # Zero C at positions where sequence changed
        C_chunked = jnp.where(
            same_seq[:, :, :, None, None],  # (batch, nchunks, Q, 1, 1)
            C_chunked,
            0.0,
        )
        C = C_chunked.reshape(batch, total_len, ngroups, dstate)

    # --- 1. x_scaled: fold dt * exp(-dA_cs) into x ---
    # x: (batch, nchunks*Q, nheads, hdim) → (batch, nchunks, Q, nheads, hdim)
    x_chunked = x.reshape(batch, nchunks, chunk_size, nheads, hdim)

    # scale_x = dt * exp(-dA_cs): (batch, nheads, nchunks, Q)
    scale_x = dt * jnp.exp(-dA_cumsum)

    # Rearrange to (batch, nchunks, Q, nheads) for broadcasting with x_chunked
    scale_for_x = scale_x.transpose(0, 2, 3, 1)  # (batch, nchunks, Q, nheads)

    # x_scaled: (batch, nchunks, Q, nheads, hdim)
    x_scaled = x_chunked * scale_for_x[:, :, :, :, None]

    # Flatten to (BCH, Q, hdim): head becomes part of batch dim
    x_scaled_flat = (
        x_scaled.transpose(0, 1, 3, 2, 4)    # (batch, nchunks, nheads, Q, hdim)
        .reshape(BCH, chunk_size, hdim)
    )

    # --- 2. CB_causal: apply lower-triangular causal mask ---
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    cb_causal = jnp.where(causal_mask[None, None, None, :, :], cb, 0.0)

    # Flatten to (BCG, Q, Q)
    cb_flat = cb_causal.reshape(BCG, chunk_size, chunk_size)

    # --- 3. C: reshape to (BCG, Q, dstate) ---
    C_chunked = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    C_flat = (
        C_chunked.transpose(0, 1, 3, 2, 4)   # (batch, nchunks, ngroups, Q, dstate)
        .reshape(BCG, chunk_size, dstate)
    )

    # --- 4. states_T: transpose to (BCH, dstate, hdim) ---
    states_T_flat = (
        states.transpose(0, 1, 2, 4, 3)      # (batch, nchunks, nheads, dstate, hdim)
        .reshape(BCH, dstate, hdim)
    )

    # --- 5. Padding ---
    BK_cs = min(BK, chunk_size)
    BK_ds = min(BK, dstate)

    chunk_size_padded = math.ceil(chunk_size / max(BM, BK_cs)) * max(BM, BK_cs)
    hdim_padded = math.ceil(hdim / BN) * BN
    dstate_padded = math.ceil(dstate / BK_ds) * BK_ds

    # Pad cb: (BCG, Q, Q) → (BCG, Q_padded, Q_padded)
    if chunk_size_padded > chunk_size:
        pad_q = chunk_size_padded - chunk_size
        cb_flat = jnp.pad(cb_flat, ((0, 0), (0, pad_q), (0, pad_q)))

    # Pad x_scaled: (BCH, Q, hdim) → (BCH, Q_padded, hdim_padded)
    pad_q = chunk_size_padded - chunk_size
    pad_n = hdim_padded - hdim
    if pad_q > 0 or pad_n > 0:
        x_scaled_flat = jnp.pad(x_scaled_flat, ((0, 0), (0, pad_q), (0, pad_n)))

    # Pad C: (BCG, Q, dstate) → (BCG, Q_padded, dstate_padded)
    pad_q = chunk_size_padded - chunk_size
    pad_ds = dstate_padded - dstate
    if pad_q > 0 or pad_ds > 0:
        C_flat = jnp.pad(C_flat, ((0, 0), (0, pad_q), (0, pad_ds)))

    # Pad states_T: (BCH, dstate, hdim) → (BCH, dstate_padded, hdim_padded)
    if pad_ds > 0 or pad_n > 0:
        states_T_flat = jnp.pad(states_T_flat, ((0, 0), (0, pad_ds), (0, pad_n)))

    # --- 6. Cast to bf16 ---
    cb_flat = cb_flat.astype(jnp.bfloat16)
    x_scaled_flat = x_scaled_flat.astype(jnp.bfloat16)
    C_flat = C_flat.astype(jnp.bfloat16)
    states_T_flat = states_T_flat.astype(jnp.bfloat16)

    meta = dict(
        BCH=BCH, BCG=BCG,
        chunk_size=chunk_size, chunk_size_padded=chunk_size_padded,
        hdim=hdim, hdim_padded=hdim_padded,
        dstate=dstate, dstate_padded=dstate_padded,
        batch=batch, nchunks=nchunks, nheads=nheads, ngroups=ngroups,
        BK_cs=BK_cs, BK_ds=BK_ds,
    )
    return cb_flat, x_scaled_flat, C_flat, states_T_flat, meta


# ---------------------------------------------------------------------------
# Kernel-only launch (no preprocessing)
# ---------------------------------------------------------------------------

def chunk_scan_kernel_only(
    cb_flat,        # (BCG, Q_padded, Q_padded) bf16
    x_scaled_flat,  # (BCH, Q_padded, hdim_padded) bf16
    C_flat,         # (BCG, Q_padded, dstate_padded) bf16
    states_T_flat,  # (BCH, dstate_padded, hdim_padded) bf16
    *,
    BCH: int,
    BCG: int,
    chunk_size: int,
    chunk_size_padded: int,
    hdim: int,
    hdim_padded: int,
    dstate: int,
    dstate_padded: int,
    batch: int,
    nchunks: int,
    nheads: int,
    ngroups: int,
    BM: int = 64,
    BK_cs: int = 64,
    BK_ds: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> jnp.ndarray:
    """
    Launch just the Pallas kernel + output slicing.

    Takes pre-processed bf16 inputs (from chunk_scan_preprocess).
    Returns raw matmul result: (batch, nchunks, nheads, chunk_size, hdim) f32.
    The caller must still apply exp(dA_cs) scaling, D residual, and z gating.
    """
    PM = chunk_size_padded // BM
    PN = hdim_padded // BN

    mesh = plgpu.Mesh(
        grid=(BCH, PM, PN),
        grid_names=("bch", "pm", "pn"),
    )

    kernel_fn = pl.kernel(
        partial(
            _chunk_scan_fwd_kernel_body,
            BM=BM, BK_cs=BK_cs, BK_ds=BK_ds, BN=BN,
            chunk_size_padded=chunk_size_padded,
            dstate_padded=dstate_padded,
            num_stages=num_stages,
            nheads=nheads,
            ngroups=ngroups,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (BCH, chunk_size_padded, hdim_padded), jnp.float32
        ),
        mesh=mesh,
    )

    out_flat = kernel_fn(cb_flat, x_scaled_flat, C_flat, states_T_flat)

    # Slice away padding, reshape to (batch, nchunks, nheads, chunk_size, hdim)
    out = (
        out_flat[:, :chunk_size, :hdim]
        .reshape(batch, nchunks, nheads, chunk_size, hdim)
    )
    return out


# ---------------------------------------------------------------------------
# Public wrapper (end-to-end: preprocess + kernel + postprocess)
# ---------------------------------------------------------------------------

def chunk_scan_fwd_mosaic(
    cb,             # (batch, nchunks, ngroups, chunk_size, chunk_size) float32
    x,              # (batch, seqlen, nheads, hdim) float32
    dt,             # (batch, nheads, nchunks, chunk_size) float32
    dA_cumsum,      # (batch, nheads, nchunks, chunk_size) float32
    C,              # (batch, seqlen, ngroups, dstate) float32
    states,         # (batch, nchunks, nheads, hdim, dstate) float32
    D=None,         # (nheads,) or (nheads, hdim) float32, optional
    z=None,         # (batch, seqlen, nheads, hdim) float32, optional
    seq_idx=None,   # (batch, seqlen) int32, optional
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> tuple:
    """
    H100/H200 Pallas Mosaic GPU port of _chunk_scan_fwd.

    Parameters
    ----------
    cb         : (batch, nchunks, ngroups, chunk_size, chunk_size) float32
    x          : (batch, seqlen, nheads, hdim)  float32
    dt         : (batch, nheads, nchunks, chunk_size) float32
    dA_cumsum  : (batch, nheads, nchunks, chunk_size) float32
    C          : (batch, seqlen, ngroups, dstate)  float32
    states     : (batch, nchunks, nheads, hdim, dstate) float32 (prev_states)
    D          : (nheads,) or (nheads, hdim), optional
    z          : (batch, seqlen, nheads, hdim), optional gating
    seq_idx    : (batch, seqlen) int32, optional
        Sequence index per position.  When packed variable-length sequences
        share a single batch row, seq_idx marks which sequence each position
        belongs to.  At chunk boundaries where the sequence changed, the
        state contribution is zeroed (scan contribution is unaffected
        because CB was already seq_idx-masked upstream).
    BM, BK, BN : tile sizes (default 64)
    num_stages : TMA pipeline depth (default 2)

    Returns
    -------
    out   : (batch, seqlen, nheads, hdim)  float32
    out_x : (batch, seqlen, nheads, hdim)  float32, or None (pre-gate output if z)
    """
    batch, seqlen, nheads, hdim = x.shape
    _, nheads_, nchunks, chunk_size = dt.shape

    # --- Preprocess ---
    cb_flat, x_scaled_flat, C_flat, states_T_flat, meta = chunk_scan_preprocess(
        cb, x, dt, dA_cumsum, C, states, seq_idx=seq_idx, BM=BM, BK=BK, BN=BN,
    )

    # --- Kernel: acc = CB_causal @ x_scaled + C @ states_T ---
    out_raw = chunk_scan_kernel_only(
        cb_flat, x_scaled_flat, C_flat, states_T_flat,
        BM=BM, BK_cs=meta['BK_cs'], BK_ds=meta['BK_ds'], BN=BN,
        num_stages=num_stages,
        **{k: meta[k] for k in (
            'BCH', 'BCG', 'chunk_size', 'chunk_size_padded',
            'hdim', 'hdim_padded', 'dstate', 'dstate_padded',
            'batch', 'nchunks', 'nheads', 'ngroups',
        )},
    )
    # out_raw: (batch, nchunks, nheads, chunk_size, hdim) f32

    # --- Post-process: apply exp(dA_cs[m]) scaling ---
    # dA_cumsum: (batch, nheads, nchunks, chunk_size)
    exp_dA = jnp.exp(dA_cumsum)
    # Rearrange to (batch, nchunks, nheads, chunk_size, 1) for broadcasting
    exp_dA_bcast = exp_dA.transpose(0, 2, 1, 3)[:, :, :, :, None]
    out = out_raw * exp_dA_bcast

    # Reshape to (batch, seqlen, nheads, hdim)
    out = (
        out.transpose(0, 1, 3, 2, 4)         # (batch, nchunks, chunk_size, nheads, hdim)
        .reshape(batch, nchunks * chunk_size, nheads, hdim)
        [:, :seqlen, :, :]
    )

    # --- D residual ---
    if D is not None:
        if D.ndim == 2:  # (nheads, hdim)
            out = out + x[:, :seqlen, :, :] * D[None, None, :, :]
        else:  # (nheads,)
            out = out + x[:, :seqlen, :, :] * D[None, None, :, None]

    # --- z gating ---
    out_x = None
    if z is not None:
        out_x = out
        out = out * z * jax.nn.sigmoid(z)

    return out, out_x


# ---------------------------------------------------------------------------
# Triton-compatible alias
# ---------------------------------------------------------------------------

def chunk_scan_fwd(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    D=None,
    z=None,
    seq_idx=None,
    BM: int = 64,
    BK: int = 64,
    BN: int = 64,
    num_stages: int = 2,
) -> tuple:
    """Drop-in replacement for mamba_ssm._chunk_scan_fwd (JAX/Pallas version)."""
    return chunk_scan_fwd_mosaic(
        cb, x, dt, dA_cumsum, C, states,
        D=D, z=z, seq_idx=seq_idx,
        BM=BM, BK=BK, BN=BN,
        num_stages=num_stages,
    )
