"""
test_mosaic3.py

Correctness checks and benchmarks for:
  - chunk_scan_fwd_mosaic

Run on H100/H200 (Hopper) with the Mosaic GPU Pallas backend:
  python test_mosaic3.py

Tests numerical correctness against a naive JAX reference (and optionally
against the Triton reference from mamba_ssm), then benchmarks throughput.
"""

import os
import sys
import math
import types
import time

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Make the package importable when running from inside the directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pallas_mosaic_gpu_mamba.chunk_scan_fwd import (
    chunk_scan_fwd_mosaic,
    chunk_scan_preprocess,
    chunk_scan_kernel_only,
)

# ---------------------------------------------------------------------------
# Optional Triton reference (may not be available on all machines).
# ---------------------------------------------------------------------------
_HAS_TRITON = False
try:
    import torch
    MAMBA_ROOT = os.path.expanduser("/workspace/mamba")
    sys.path.insert(0, MAMBA_ROOT)
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__    = [os.path.join(MAMBA_ROOT, "mamba_ssm")]
    pkg.__package__ = "mamba_ssm"
    sys.modules.setdefault("mamba_ssm", pkg)
    from mamba_ssm.ops.triton.ssd_chunk_scan import (
        _chunk_scan_fwd as _triton_chunk_scan_fwd,
    )
    _HAS_TRITON = True
except Exception as e:
    print(f"[info] Triton reference unavailable ({e}); skipping Triton comparison.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_torch(x):
    import torch
    return torch.tensor(np.array(x), device="cuda", dtype=torch.float32)


def check(name, jax_arr, ref_arr, atol=1e-3):
    d = float(np.abs(np.array(jax_arr) - np.array(ref_arr)).max())
    ok = "\u2713" if d < atol else "\u2717"
    print(f"  {ok}  {name:40s}  max|diff|={d:.2e}  (atol={atol:.0e})")
    return d < atol


# ===========================================================================
# chunk_scan_fwd — naive JAX reference
# ===========================================================================

def _naive_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D=None, z=None,
                          seq_idx=None):
    """
    Naive JAX reference for _chunk_scan_fwd.

    cb        : (batch, nchunks, ngroups, chunk_size, chunk_size)
    x         : (batch, seqlen, nheads, hdim)
    dt        : (batch, nheads, nchunks, chunk_size)
    dA_cumsum : (batch, nheads, nchunks, chunk_size)
    C         : (batch, seqlen, ngroups, dstate)
    states    : (batch, nchunks, nheads, hdim, dstate)
    D         : (nheads,) or (nheads, hdim), optional
    z         : (batch, seqlen, nheads, hdim), optional
    seq_idx   : (batch, seqlen) int32, optional

    Returns:
      out   : (batch, seqlen, nheads, hdim)
      out_x : (batch, seqlen, nheads, hdim) or None
    """
    batch, seqlen, nheads, hdim = x.shape
    _, nheads_, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    ratio = nheads // ngroups

    # Pad seqlen
    total_len = nchunks * chunk_size
    if seqlen < total_len:
        pad = total_len - seqlen
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        C = jnp.pad(C, ((0, 0), (0, pad), (0, 0), (0, 0)))
        if seq_idx is not None:
            seq_idx = jnp.pad(seq_idx, ((0, 0), (0, pad)), constant_values=-1)

    # Reshape to chunks
    x_c = x.reshape(batch, nchunks, chunk_size, nheads, hdim)
    C_c = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)

    # --- seq_idx: compute state scale mask ---
    # For state contribution: zero where seq changed from previous chunk
    if seq_idx is not None:
        seq_idx_full = seq_idx.reshape(batch, nchunks, chunk_size)
        chunk_starts = jnp.arange(nchunks) * chunk_size
        prev_positions = chunk_starts - 1
        seq_idx_prev = jnp.where(
            prev_positions >= 0,
            seq_idx[:, jnp.maximum(prev_positions, 0)],
            0,
        )  # (batch, nchunks)
        # same_seq: (batch, nchunks, chunk_size)
        same_seq_state = seq_idx_full == seq_idx_prev[:, :, None]
    else:
        same_seq_state = None

    # Expand groups to heads for cb and C
    cb_exp = jnp.repeat(cb, ratio, axis=2)       # (batch, nchunks, nheads, Q, Q)
    C_exp = jnp.repeat(C_c, ratio, axis=3)       # (batch, nchunks, Q, nheads, dstate)

    # dA_cumsum: (batch, nheads, nchunks, Q) → (batch, nchunks, nheads, Q)
    dA_t = dA_cumsum.transpose(0, 2, 1, 3)
    dt_t = dt.transpose(0, 2, 1, 3)

    # --- Part 1: state contribution ---
    # C_exp[m,:] @ states[:, n] * scale_m
    states_t = states.transpose(0, 1, 2, 4, 3)
    state_contrib = jnp.einsum('bcmhd,bchdn->bcmhn', C_exp, states_t)
    # Scale by exp(dA_cs[m]) with seq_idx masking
    exp_dA = jnp.exp(dA_t)  # (batch, nchunks, nheads, Q)
    if same_seq_state is not None:
        # Zero scale where sequence changed
        scale_m = jnp.where(
            same_seq_state[:, :, None, :],  # (batch, nchunks, 1, Q)
            exp_dA,
            0.0,
        )  # (batch, nchunks, nheads, Q)
    else:
        scale_m = exp_dA
    scale_m_bcast = scale_m.transpose(0, 1, 3, 2)[:, :, :, :, None]  # (batch, nchunks, Q, nheads, 1)
    state_contrib = state_contrib * scale_m_bcast

    # --- Part 2: scan contribution ---
    # CB_scaled[m,k] = cb[m,k] * exp(min(dA[m]-dA[k], 0)) * dt[k] * causal(m,k)
    dA_m = dA_t[:, :, :, :, None]   # (batch, nchunks, nheads, Q, 1)
    dA_k = dA_t[:, :, :, None, :]   # (batch, nchunks, nheads, 1, Q)
    cb_scaled = cb_exp * jnp.exp(jnp.minimum(dA_m - dA_k, 0.0))
    cb_scaled = cb_scaled * dt_t[:, :, :, None, :]

    # Causal mask
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    cb_scaled = jnp.where(causal_mask[None, None, None, :, :], cb_scaled, 0.0)

    # seq_idx mask on CB (scan part): zero where seq_idx[m] != seq_idx[k]
    # (Triton assumes caller already masked CB, but for the naive ref we do it here)
    if seq_idx is not None:
        seq_idx_chunked = seq_idx.reshape(batch, nchunks, chunk_size)
        seq_mask_cb = (
            seq_idx_chunked[:, :, :, None] == seq_idx_chunked[:, :, None, :]
        )  # (batch, nchunks, Q, Q)
        cb_scaled = jnp.where(seq_mask_cb[:, :, None, :, :], cb_scaled, 0.0)

    # scan_contrib = cb_scaled @ x_c
    x_ct = x_c.transpose(0, 1, 3, 2, 4)
    scan_contrib = jnp.einsum('bchmk,bchkn->bchmn', cb_scaled, x_ct)
    scan_contrib = scan_contrib.transpose(0, 1, 3, 2, 4)

    # Combine
    out = state_contrib + scan_contrib
    out = out.reshape(batch, nchunks * chunk_size, nheads, hdim)[:, :seqlen]

    # D residual
    if D is not None:
        x_orig = x[:, :seqlen]
        if D.ndim == 2:
            out = out + x_orig * D[None, None, :, :]
        else:
            out = out + x_orig * D[None, None, :, None]

    # z gating
    out_x = None
    if z is not None:
        out_x = out
        out = out * z * jax.nn.sigmoid(z)

    return out, out_x


# ===========================================================================
# chunk_scan_fwd — correctness test
# ===========================================================================

def _build_seq_idx(batch, seqlen, seq_lengths_per_batch):
    """Build seq_idx array from per-batch sequence lengths."""
    seq_idx_list = []
    for b_idx in range(batch):
        lens = seq_lengths_per_batch[b_idx]
        idx = []
        for seq_id, l in enumerate(lens):
            idx.extend([seq_id] * l)
        idx = idx[:seqlen]
        if len(idx) < seqlen:
            idx.extend([idx[-1]] * (seqlen - len(idx)))
        seq_idx_list.append(idx)
    return jnp.array(seq_idx_list, dtype=jnp.int32)


def test_chunk_scan_correctness(
    batch=2,
    seqlen=512,
    nheads=8,
    hdim=64,
    dstate=64,
    ngroups=1,
    chunk_size=64,
    has_D=False,
    has_z=False,
    seq_idx_config=None,
    atol=2.0,
):
    """
    Correctness test for chunk_scan_fwd_mosaic.

    Uses atol=2.0 because WGMMA uses bf16 matmul while naive uses f32.

    seq_idx_config: None or dict with keys:
        'seq_lengths': list[list[int]] per batch element
    """
    nchunks = math.ceil(seqlen / chunk_size)
    seq_str = f"seq_idx={seq_idx_config is not None}"
    print(f"\n\u2500\u2500 [chunk_scan_fwd] correctness  "
          f"B={batch} L={seqlen} H={nheads} N={hdim} D={dstate} G={ngroups} "
          f"Q={chunk_size} D?={has_D} z?={has_z} {seq_str} \u2500\u2500")

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)

    cb_jax = jax.random.normal(keys[0], (batch, nchunks, ngroups, chunk_size, chunk_size)) * 0.02
    x_jax = jax.random.normal(keys[1], (batch, seqlen, nheads, hdim)) * 0.1
    dt_jax = jax.random.uniform(keys[2], (batch, nheads, nchunks, chunk_size), minval=0.01, maxval=0.1)
    dA_cs_jax = -jax.random.uniform(keys[3], (batch, nheads, nchunks, chunk_size)) * 0.5
    # Make dA_cumsum monotonically decreasing within each chunk (realistic)
    dA_cs_jax = jnp.cumsum(dA_cs_jax, axis=-1)
    C_jax = jax.random.normal(keys[4], (batch, seqlen, ngroups, dstate)) * 0.1
    states_jax = jax.random.normal(keys[5], (batch, nchunks, nheads, hdim, dstate)) * 0.01

    D_jax = None
    if has_D:
        D_jax = jax.random.normal(keys[6], (nheads,)) * 0.1

    z_jax = None
    if has_z:
        z_jax = jax.random.normal(keys[7], (batch, seqlen, nheads, hdim))

    # Build seq_idx if requested
    seq_idx_jax = None
    if seq_idx_config is not None:
        seq_idx_jax = _build_seq_idx(batch, seqlen, seq_idx_config['seq_lengths'])
        print(f"  seq_idx shape: {tuple(seq_idx_jax.shape)}")

        # Apply seq_idx masking to CB (as bmm_chunk_fwd would do upstream)
        seq_idx_padded = seq_idx_jax
        total_len = nchunks * chunk_size
        if seqlen < total_len:
            seq_idx_padded = jnp.pad(seq_idx_jax, ((0, 0), (0, total_len - seqlen)),
                                     constant_values=-1)
        seq_idx_chunked = seq_idx_padded.reshape(batch, nchunks, chunk_size)
        seq_mask_cb = (
            seq_idx_chunked[:, :, :, None] == seq_idx_chunked[:, :, None, :]
        )  # (batch, nchunks, Q, Q)
        cb_jax = jnp.where(seq_mask_cb[:, :, None, :, :], cb_jax, 0.0)

    # --- Mosaic GPU ---
    out_pal, out_x_pal = chunk_scan_fwd_mosaic(
        cb_jax, x_jax, dt_jax, dA_cs_jax, C_jax, states_jax,
        D=D_jax, z=z_jax, seq_idx=seq_idx_jax,
    )
    jax.block_until_ready(out_pal)
    print(f"  out shape: {tuple(out_pal.shape)}")
    if out_x_pal is not None:
        print(f"  out_x shape: {tuple(out_x_pal.shape)}")

    # --- Naive reference ---
    out_ref, out_x_ref = _naive_chunk_scan_fwd(
        cb_jax, x_jax, dt_jax, dA_cs_jax, C_jax, states_jax,
        D=D_jax, z=z_jax, seq_idx=seq_idx_jax,
    )

    all_ok = True
    all_ok &= check("out vs naive", out_pal, out_ref, atol=atol)
    if has_z:
        all_ok &= check("out_x vs naive", out_x_pal, out_x_ref, atol=atol)

    # --- Optional Triton comparison ---
    if _HAS_TRITON:
        import torch as _torch
        cb_t = _to_torch(cb_jax)
        x_t = _to_torch(x_jax)
        dt_t = _to_torch(dt_jax)
        dA_cs_t = _to_torch(dA_cs_jax)
        C_t = _to_torch(C_jax)
        states_t = _to_torch(states_jax)
        D_t = _to_torch(D_jax) if D_jax is not None else None
        z_t = _to_torch(z_jax) if z_jax is not None else None
        seq_idx_t = None
        if seq_idx_jax is not None:
            seq_idx_t = _torch.tensor(
                np.array(seq_idx_jax), device="cuda", dtype=_torch.int32,
            )

        out_tri, out_x_tri = _triton_chunk_scan_fwd(
            cb_t, x_t, dt_t, dA_cs_t, C_t, states_t,
            D=D_t, z=z_t, seq_idx=seq_idx_t,
        )
        out_tri_j = jnp.array(out_tri.cpu().numpy())
        all_ok &= check("out vs Triton", out_pal, out_tri_j, atol=atol)
        if has_z and out_x_tri is not None:
            out_x_tri_j = jnp.array(out_x_tri.cpu().numpy())
            all_ok &= check("out_x vs Triton", out_x_pal, out_x_tri_j, atol=atol)

    print(f"  {'ALL PASS \u2713' if all_ok else 'FAILURES DETECTED \u2717'}")
    return all_ok


# ===========================================================================
# chunk_scan_fwd — benchmark
# ===========================================================================

def benchmark_chunk_scan(
    batch=2,
    seqlen=2048,
    nheads=64,
    hdim=64,
    dstate=64,
    ngroups=1,
    chunk_size=256,
    warmup=25,
    rep=200,
):
    """Benchmark chunk_scan_fwd_mosaic vs naive and Triton."""
    nchunks = math.ceil(seqlen / chunk_size)
    print(f"\n\u2500\u2500 [chunk_scan_fwd] benchmark  "
          f"B={batch} L={seqlen} H={nheads} N={hdim} D={dstate} G={ngroups} "
          f"Q={chunk_size} \u2500\u2500")

    key = jax.random.PRNGKey(7)
    keys = jax.random.split(key, 6)

    cb_j = jax.random.normal(keys[0], (batch, nchunks, ngroups, chunk_size, chunk_size)) * 0.02
    x_j = jax.random.normal(keys[1], (batch, seqlen, nheads, hdim)) * 0.1
    dt_j = jax.random.uniform(keys[2], (batch, nheads, nchunks, chunk_size), minval=0.01, maxval=0.1)
    dA_cs_j = jnp.cumsum(-jax.random.uniform(keys[3], (batch, nheads, nchunks, chunk_size)) * 0.5, axis=-1)
    C_j = jax.random.normal(keys[4], (batch, seqlen, ngroups, dstate)) * 0.1
    states_j = jax.random.normal(keys[5], (batch, nchunks, nheads, hdim, dstate)) * 0.01

    # --- Naive JAX ---
    fn_naive = jax.jit(lambda: _naive_chunk_scan_fwd(cb_j, x_j, dt_j, dA_cs_j, C_j, states_j))
    jax.block_until_ready(fn_naive())

    if _HAS_TRITON:
        from triton.testing import do_bench
        ms_naive = do_bench(lambda: jax.block_until_ready(fn_naive()), warmup=warmup, rep=rep)
    else:
        times_naive = []
        for _ in range(warmup + rep):
            t0 = time.perf_counter()
            jax.block_until_ready(fn_naive())
            times_naive.append(time.perf_counter() - t0)
        ms_naive = float(np.median(times_naive[warmup:])) * 1e3

    # --- Mosaic GPU (end-to-end) ---
    fn_mosaic = jax.jit(lambda: chunk_scan_fwd_mosaic(
        cb_j, x_j, dt_j, dA_cs_j, C_j, states_j,
    ))
    jax.block_until_ready(fn_mosaic())

    if _HAS_TRITON:
        from triton.testing import do_bench
        ms_mosaic = do_bench(lambda: jax.block_until_ready(fn_mosaic()), warmup=warmup, rep=rep)
    else:
        times_mosaic = []
        for _ in range(warmup + rep):
            t0 = time.perf_counter()
            jax.block_until_ready(fn_mosaic())
            times_mosaic.append(time.perf_counter() - t0)
        ms_mosaic = float(np.median(times_mosaic[warmup:])) * 1e3

    # --- Mosaic GPU (kernel-only) ---
    cb_flat, x_scaled_flat, C_flat, states_T_flat, meta = chunk_scan_preprocess(
        cb_j, x_j, dt_j, dA_cs_j, C_j, states_j,
    )
    fn_kernel = jax.jit(lambda: chunk_scan_kernel_only(
        cb_flat, x_scaled_flat, C_flat, states_T_flat,
        BM=64, BK_cs=meta['BK_cs'], BK_ds=meta['BK_ds'], BN=64,
        num_stages=2,
        **{k: meta[k] for k in (
            'BCH', 'BCG', 'chunk_size', 'chunk_size_padded',
            'hdim', 'hdim_padded', 'dstate', 'dstate_padded',
            'batch', 'nchunks', 'nheads', 'ngroups',
        )},
    ))
    jax.block_until_ready(fn_kernel())

    if _HAS_TRITON:
        from triton.testing import do_bench
        ms_kernel = do_bench(lambda: jax.block_until_ready(fn_kernel()), warmup=warmup, rep=rep)
    else:
        times_kernel = []
        for _ in range(warmup + rep):
            t0 = time.perf_counter()
            jax.block_until_ready(fn_kernel())
            times_kernel.append(time.perf_counter() - t0)
        ms_kernel = float(np.median(times_kernel[warmup:])) * 1e3

    print(f"  Naive JAX       : {ms_naive:.3f} ms")
    print(f"  Mosaic GPU (e2e): {ms_mosaic:.3f} ms")
    print(f"  Mosaic kernel   : {ms_kernel:.3f} ms")

    if _HAS_TRITON:
        import torch
        from triton.testing import do_bench
        cb_t = _to_torch(cb_j)
        x_t = _to_torch(x_j)
        dt_t = _to_torch(dt_j)
        dA_cs_t = _to_torch(dA_cs_j)
        C_t = _to_torch(C_j)
        states_t = _to_torch(states_j)

        def _triton_fn():
            _triton_chunk_scan_fwd(cb_t, x_t, dt_t, dA_cs_t, C_t, states_t)
            torch.cuda.synchronize()

        ms_triton = do_bench(_triton_fn, warmup=warmup, rep=rep)
        print(f"  Triton          : {ms_triton:.3f} ms")

    print(f"  \u2500\u2500 Ratios \u2500\u2500")
    print(f"  Mosaic e2e / Naive   : {ms_mosaic/ms_naive:.2f}x")
    print(f"  Mosaic kernel/ Naive : {ms_kernel/ms_naive:.2f}x")
    if _HAS_TRITON:
        print(f"  Mosaic e2e / Triton  : {ms_mosaic/ms_triton:.2f}x")
        print(f"  Mosaic kernel/ Triton: {ms_kernel/ms_triton:.2f}x")
        print(f"  Naive / Triton       : {ms_naive/ms_triton:.2f}x")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("chunk_scan_fwd_mosaic — Correctness Tests")
    print("=" * 70)

    # Basic configs
    test_chunk_scan_correctness(batch=1, seqlen=256, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=64)
    test_chunk_scan_correctness(batch=2, seqlen=512, nheads=8, hdim=64, dstate=64, ngroups=1, chunk_size=128)
    test_chunk_scan_correctness(batch=1, seqlen=256, nheads=8, hdim=128, dstate=64, ngroups=1, chunk_size=64)

    # Multi-group
    test_chunk_scan_correctness(batch=2, seqlen=512, nheads=16, hdim=64, dstate=64, ngroups=2, chunk_size=128)
    test_chunk_scan_correctness(batch=1, seqlen=256, nheads=8, hdim=64, dstate=64, ngroups=8, chunk_size=64)

    # Large chunk_size
    test_chunk_scan_correctness(batch=1, seqlen=512, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=256)

    # With D
    test_chunk_scan_correctness(batch=2, seqlen=256, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=64, has_D=True)

    # With z
    test_chunk_scan_correctness(batch=2, seqlen=256, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=64, has_z=True)

    # With D and z
    test_chunk_scan_correctness(batch=1, seqlen=256, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=64, has_D=True, has_z=True)

    # Larger dstate
    test_chunk_scan_correctness(batch=1, seqlen=256, nheads=4, hdim=64, dstate=128, ngroups=1, chunk_size=64)

    print("\n" + "=" * 70)
    print("chunk_scan_fwd_mosaic — SEQ_IDX Correctness Tests")
    print("=" * 70)

    # Single sequence (no-op)
    test_chunk_scan_correctness(
        batch=1, seqlen=256, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=64,
        seq_idx_config={'seq_lengths': [[256]]},
    )

    # Two sequences, boundary at chunk edge
    test_chunk_scan_correctness(
        batch=1, seqlen=256, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=64,
        seq_idx_config={'seq_lengths': [[128, 128]]},
    )

    # Two sequences, mid-chunk boundary
    test_chunk_scan_correctness(
        batch=2, seqlen=512, nheads=8, hdim=64, dstate=64, ngroups=1, chunk_size=64,
        seq_idx_config={'seq_lengths': [
            [200, 312],
            [100, 412],
        ]},
    )

    # Three sequences, multi-group
    test_chunk_scan_correctness(
        batch=2, seqlen=512, nheads=8, hdim=64, dstate=64, ngroups=4, chunk_size=64,
        seq_idx_config={'seq_lengths': [
            [64, 192, 256],
            [128, 128, 256],
        ]},
    )

    # Every chunk different sequence
    test_chunk_scan_correctness(
        batch=1, seqlen=256, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=64,
        seq_idx_config={'seq_lengths': [[64, 64, 64, 64]]},
    )

    # seq_idx + D + z combined
    test_chunk_scan_correctness(
        batch=2, seqlen=512, nheads=8, hdim=64, dstate=64, ngroups=1, chunk_size=128,
        has_D=True, has_z=True,
        seq_idx_config={'seq_lengths': [
            [256, 256],
            [128, 384],
        ]},
    )

    # seq_idx with larger dstate and multi-group
    test_chunk_scan_correctness(
        batch=1, seqlen=256, nheads=8, hdim=64, dstate=128, ngroups=8, chunk_size=64,
        seq_idx_config={'seq_lengths': [[100, 156]]},
    )

    print("\n" + "=" * 70)
    print("chunk_scan_fwd_mosaic — Benchmarks")
    print("=" * 70)

    # Standard config
    benchmark_chunk_scan(batch=2, seqlen=2048, nheads=64, hdim=64, dstate=64, ngroups=1, chunk_size=256)

    # Nemotron-style
    benchmark_chunk_scan(batch=2, seqlen=2048, nheads=64, hdim=64, dstate=64, ngroups=8, chunk_size=256)

    # Small chunk
    benchmark_chunk_scan(batch=2, seqlen=2048, nheads=64, hdim=64, dstate=64, ngroups=1, chunk_size=64)

    print("\nDone.")
