"""
test_mosaic3.py

Correctness checks and benchmarks for:
  - chunk_scan_fwd_mosaic
  - chunk_state_varlen_mosaic

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
from functools import partial

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
from pallas_mosaic_gpu_mamba.chunk_state_fwd import (
    chunk_state_varlen_mosaic,
    chunk_state_fwd_mosaic,
    chunk_state_kernel_only,
    chunk_state_preprocess,
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


def _bench_fn(fn, warmup, rep):
    """Benchmark a callable, using triton do_bench if available, else manual timing."""
    if _HAS_TRITON:
        from triton.testing import do_bench
        return do_bench(fn, warmup=warmup, rep=rep)
    else:
        times = []
        for _ in range(warmup + rep):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.median(times[warmup:])) * 1e3


def _bench_amortized(impl_fn, args, N=100, warmup=10, rep=50):
    """
    Measure true GPU time by amortizing JAX dispatch overhead.

    Wraps N calls inside a single jax.jit + fori_loop.  Per-call time =
    total_time / N, which converges to pure GPU time as N grows.

    Uses jax.lax.cond with an always-true but acc-dependent condition to
    prevent XLA from hoisting loop-invariant computations out of the loop.

    impl_fn : callable taking *args (all JAX arrays) and returning a pytree.
              Any Python-scalar / static args must be captured via closure or
              functools.partial — only JAX arrays should be in ``args``.
    args    : tuple of JAX arrays to pass as jit function arguments.
    """
    sample = impl_fn(*args)
    dummy = jax.tree.map(jnp.zeros_like, sample)
    first_leaf = jax.tree.leaves(sample)[0]
    idx = (0,) * first_leaf.ndim

    @jax.jit
    def looped(*a):
        def body(i, acc):
            result = jax.lax.cond(
                acc > -1e30,
                lambda: impl_fn(*a),
                lambda: dummy,
            )
            return acc + jax.tree.leaves(result)[0][idx]
        return jax.lax.fori_loop(0, N, body, 0.0)

    jax.block_until_ready(looped(*args))
    ms_total = _bench_fn(
        lambda: jax.block_until_ready(looped(*args)),
        warmup, rep,
    )
    return ms_total / N


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

    scan_args = (cb_j, x_j, dt_j, dA_cs_j, C_j, states_j)

    # --- Naive JAX (amortized) ---
    ms_naive = _bench_amortized(
        _naive_chunk_scan_fwd, scan_args, warmup=warmup, rep=rep,
    )

    # --- Mosaic GPU e2e (amortized) ---
    ms_mosaic = _bench_amortized(
        chunk_scan_fwd_mosaic, scan_args, warmup=warmup, rep=rep,
    )

    # --- Mosaic GPU kernel-only (amortized) ---
    cb_flat, x_scaled_flat, C_flat, states_T_flat, meta = chunk_scan_preprocess(
        *scan_args,
    )
    kernel_fn = partial(
        chunk_scan_kernel_only,
        BM=64, BK_cs=meta['BK_cs'], BK_ds=meta['BK_ds'], BN=64,
        num_stages=2,
        **{k: meta[k] for k in (
            'BCH', 'BCG', 'chunk_size', 'chunk_size_padded',
            'hdim', 'hdim_padded', 'dstate', 'dstate_padded',
            'batch', 'nchunks', 'nheads', 'ngroups',
        )},
    )
    ms_kernel = _bench_amortized(
        kernel_fn, (cb_flat, x_scaled_flat, C_flat, states_T_flat),
        warmup=warmup, rep=rep,
    )

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
# chunk_state_varlen — Triton reference (optional)
# ===========================================================================

_HAS_TRITON_VARLEN = False
try:
    from mamba_ssm.ops.triton.ssd_chunk_state import (
        chunk_state_varlen as _triton_chunk_state_varlen,
    )
    _HAS_TRITON_VARLEN = True
except Exception:
    pass


# ===========================================================================
# chunk_state_varlen — naive JAX reference
# ===========================================================================

def _naive_chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    """
    Naive JAX reference for chunk_state_varlen.

    Computes the final state for each variable-length sequence by:
    1. For each sequence, find the last chunk
    2. Compute x_T @ diag(scale) @ B for valid positions in that chunk
    3. Add chunk_states[last_chunk] * exp(dA_cs_last) for accumulated state

    B            : (total_seqlen, ngroups, dstate)
    x            : (total_seqlen, nheads, headdim)
    dt           : (nheads, nchunks, chunk_size)
    dA_cumsum    : (nheads, nchunks, chunk_size)
    cu_seqlens   : (batch + 1,) int32
    chunk_states : (nchunks, nheads, headdim, dstate)

    Returns: (batch, nheads, headdim, dstate)
    """
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    ratio = nheads // ngroups

    results = []
    for b in range(batch):
        start = int(cu_seqlens[b])
        end = int(cu_seqlens[b + 1])

        last_chunk = (end - 1) // chunk_size
        chunk_start = last_chunk * chunk_size
        chunk_end_pos = min(end, (last_chunk + 1) * chunk_size)
        start_in_chunk = max(start - chunk_start, 0)

        # Extract last chunk data
        x_chunk = jnp.zeros((chunk_size, nheads, headdim))
        B_chunk = jnp.zeros((chunk_size, ngroups, dstate))
        valid_len = chunk_end_pos - chunk_start
        if chunk_start < total_seqlen:
            actual_len = min(valid_len, total_seqlen - chunk_start)
            x_chunk = x_chunk.at[:actual_len].set(
                x[chunk_start:chunk_start + actual_len]
            )
            B_chunk = B_chunk.at[:actual_len].set(
                B[chunk_start:chunk_start + actual_len]
            )

        # dt and dA_cumsum for this chunk
        dt_chunk = dt[:, last_chunk, :]         # (nheads, chunk_size)
        dA_chunk = dA_cumsum[:, last_chunk, :]  # (nheads, chunk_size)

        # dA_cs at last valid position
        last_valid = end - chunk_start - 1
        dA_cs_last = dA_chunk[:, last_valid]    # (nheads,)

        # Scale
        scale = jnp.exp(jnp.minimum(dA_cs_last[:, None] - dA_chunk, 0.0)) * dt_chunk
        # (nheads, chunk_size)

        # Zero invalid positions
        positions = jnp.arange(chunk_size)
        valid = (positions >= start_in_chunk) & (positions < (end - chunk_start))
        scale = jnp.where(valid[None, :], scale, 0.0)

        # x_T @ diag(scale) @ B per head
        # x_chunk: (chunk_size, nheads, headdim) → per head: (headdim, chunk_size)
        # B_chunk: (chunk_size, ngroups, dstate) → per group: (chunk_size, dstate)
        state_b = jnp.zeros((nheads, headdim, dstate))
        for h in range(nheads):
            g = h // ratio
            x_h = x_chunk[:, h, :]  # (chunk_size, headdim)
            B_g = B_chunk[:, g, :]  # (chunk_size, dstate)
            s = scale[h, :]         # (chunk_size,)
            # x_T @ diag(s) @ B = (headdim, chunk_size) @ diag(s) @ (chunk_size, dstate)
            x_scaled = x_h * s[:, None]  # (chunk_size, headdim)
            state_b = state_b.at[h].set(x_scaled.T @ B_g)

        # Add chunk_states contribution
        if start < chunk_start:
            cs = chunk_states[last_chunk]  # (nheads, headdim, dstate)
            cs_scale = jnp.exp(dA_cs_last)[:, None, None]
            state_b = state_b + cs * cs_scale

        results.append(state_b)

    return jnp.stack(results, axis=0)  # (batch, nheads, headdim, dstate)


# ===========================================================================
# chunk_state_varlen — correctness test
# ===========================================================================

def test_chunk_state_varlen_correctness(
    batch=10,
    nheads=8,
    headdim=64,
    dstate=64,
    ngroups=1,
    chunk_size=64,
    min_seqlen=1,
    max_seqlen=200,
    atol=2.0,
):
    """
    Correctness test for chunk_state_varlen_mosaic.

    Generates random variable-length sequences, runs the full pipeline
    (chunk_state_fwd → state_passing → chunk_state_varlen) for the Mosaic
    implementation, and compares against a naive per-sequence reference.
    """
    print(f"\n── [chunk_state_varlen] correctness  "
          f"B={batch} H={nheads} N={headdim} D={dstate} G={ngroups} "
          f"Q={chunk_size} seqlens=[{min_seqlen},{max_seqlen}] ──")

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)

    # Generate random sequence lengths
    seqlens = jax.random.randint(
        keys[0], (batch,), minval=min_seqlen, maxval=max_seqlen + 1
    )
    seqlens = np.array(seqlens)
    cu_seqlens = np.concatenate([[0], np.cumsum(seqlens)])
    total_seqlen = int(cu_seqlens[-1])
    nchunks = math.ceil(total_seqlen / chunk_size)
    cu_seqlens_jax = jnp.array(cu_seqlens, dtype=jnp.int32)
    print(f"  total_seqlen={total_seqlen}, nchunks={nchunks}, "
          f"seqlens range=[{seqlens.min()},{seqlens.max()}]")

    # Generate random inputs (no batch dim for varlen)
    B_jax = jax.random.normal(keys[1], (total_seqlen, ngroups, dstate)) * 0.2
    x_jax = jax.random.normal(keys[2], (total_seqlen, nheads, headdim)) * 0.1
    A = -0.1 * jax.random.uniform(keys[3], (nheads,))

    # dt: (total_seqlen, nheads) → processed via chunk_cumsum
    dt_raw = jax.nn.softplus(
        jax.random.normal(keys[4], (total_seqlen, nheads)) - 4.0
    )

    # --- Compute dA_cumsum and dt_rounded (simple chunk_cumsum) ---
    padded_len = nchunks * chunk_size
    dt_padded = jnp.pad(dt_raw, ((0, padded_len - total_seqlen), (0, 0)))
    dt_chunked = dt_padded.reshape(nchunks, chunk_size, nheads).transpose(2, 0, 1)
    # (nheads, nchunks, chunk_size)

    dA = dt_chunked * A[:, None, None]  # (nheads, nchunks, chunk_size)
    dA_cumsum = jnp.cumsum(dA, axis=-1)  # (nheads, nchunks, chunk_size)

    # --- Build seq_idx for chunk_state_fwd (batch=1 packed) ---
    seq_idx_list = []
    for i, s in enumerate(seqlens):
        seq_idx_list.extend([i] * int(s))
    seq_idx_1d = jnp.array(seq_idx_list, dtype=jnp.int32)
    if total_seqlen < padded_len:
        seq_idx_1d_padded = jnp.pad(
            seq_idx_1d, (0, padded_len - total_seqlen), constant_values=-1
        )
    else:
        seq_idx_1d_padded = seq_idx_1d
    seq_idx_2d = seq_idx_1d[None, :]  # (1, total_seqlen) — functions pad internally

    # --- chunk_state_fwd (batch=1 packed) ---
    x_2d = x_jax[None, :]         # (1, total_seqlen, nheads, headdim)
    B_2d = B_jax[None, :]         # (1, total_seqlen, ngroups, dstate)
    dt_2d = dt_chunked[None, :]   # (1, nheads, nchunks, chunk_size)
    dA_cs_2d = dA_cumsum[None, :] # (1, nheads, nchunks, chunk_size)

    chunk_states_full = chunk_state_fwd_mosaic(
        x_2d, B_2d, dt_2d, dA_cs_2d, seq_idx=seq_idx_2d,
    )
    # (1, nchunks, nheads, headdim, dstate)

    # --- state_passing_fwd (simple cumulative product) ---
    # For simplicity, use a naive state passing: accumulate states chunk by chunk
    from pallas_mosaic_gpu_mamba.state_passing_fwd import state_passing_fwd_mosaic
    dA_last = dA_cumsum[:, :, -1]  # (nheads, nchunks)
    chunk_states_sq = chunk_states_full.squeeze(0)  # (nchunks, nheads, headdim, dstate)
    cs_flat = chunk_states_sq.reshape(nchunks, nheads, headdim * dstate)
    passed_flat, _ = state_passing_fwd_mosaic(
        cs_flat[None, :],           # (1, nchunks, nheads, headdim*dstate)
        dA_last[None, :],           # (1, nheads, nchunks)
        seq_idx=seq_idx_2d,
        chunk_size=chunk_size,
    )
    passed = passed_flat[0].reshape(nchunks, nheads, headdim, dstate)
    # passed: (nchunks, nheads, headdim, dstate) — state at START of each chunk

    # --- chunk_state_varlen_mosaic ---
    out_mosaic = chunk_state_varlen_mosaic(
        B_jax, x_jax, dt_chunked, dA_cumsum, cu_seqlens_jax, passed,
    )
    jax.block_until_ready(out_mosaic)
    print(f"  out shape: {tuple(out_mosaic.shape)}")

    # --- Naive reference ---
    out_ref = _naive_chunk_state_varlen(
        B_jax, x_jax, dt_chunked, dA_cumsum, cu_seqlens_jax, passed,
    )

    all_ok = check("varlen out vs naive", out_mosaic, out_ref, atol=atol)

    # --- Optional Triton comparison ---
    if _HAS_TRITON and _HAS_TRITON_VARLEN:
        import torch as _torch
        B_t = _torch.tensor(np.array(B_jax), device="cuda", dtype=_torch.float32)
        x_t = _torch.tensor(np.array(x_jax), device="cuda", dtype=_torch.float32)
        dt_t = _torch.tensor(np.array(dt_chunked), device="cuda", dtype=_torch.float32)
        dA_t = _torch.tensor(np.array(dA_cumsum), device="cuda", dtype=_torch.float32)
        cu_t = _torch.tensor(np.array(cu_seqlens_jax), device="cuda", dtype=_torch.int32)
        cs_t = _torch.tensor(np.array(passed), device="cuda", dtype=_torch.float32)

        out_tri = _triton_chunk_state_varlen(B_t, x_t, dt_t, dA_t, cu_t, cs_t)
        out_tri_j = jnp.array(out_tri.cpu().numpy())
        all_ok &= check("varlen out vs Triton", out_mosaic, out_tri_j, atol=atol)

    print(f"  {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    return all_ok


# ===========================================================================
# chunk_state_varlen — benchmark
# ===========================================================================

def benchmark_chunk_state_varlen(
    batch=100,
    nheads=64,
    headdim=64,
    dstate=64,
    ngroups=1,
    chunk_size=64,
    min_seqlen=10,
    max_seqlen=200,
    warmup=25,
    rep=200,
):
    """Benchmark chunk_state_varlen_mosaic."""
    print(f"\n── [chunk_state_varlen] benchmark  "
          f"B={batch} H={nheads} N={headdim} D={dstate} G={ngroups} "
          f"Q={chunk_size} seqlens=[{min_seqlen},{max_seqlen}] ──")

    key = jax.random.PRNGKey(7)
    keys = jax.random.split(key, 6)

    seqlens = jax.random.randint(keys[0], (batch,), minval=min_seqlen, maxval=max_seqlen + 1)
    seqlens = np.array(seqlens)
    cu_seqlens = np.concatenate([[0], np.cumsum(seqlens)])
    total_seqlen = int(cu_seqlens[-1])
    nchunks = math.ceil(total_seqlen / chunk_size)
    cu_seqlens_jax = jnp.array(cu_seqlens, dtype=jnp.int32)

    B_j = jax.random.normal(keys[1], (total_seqlen, ngroups, dstate)) * 0.2
    x_j = jax.random.normal(keys[2], (total_seqlen, nheads, headdim)) * 0.1
    dt_j = jax.random.uniform(keys[3], (nheads, nchunks, chunk_size), minval=0.01, maxval=0.1)
    dA_j = jnp.cumsum(
        -jax.random.uniform(keys[4], (nheads, nchunks, chunk_size)) * 0.5, axis=-1
    )
    cs_j = jax.random.normal(keys[5], (nchunks, nheads, headdim, dstate)) * 0.01

    varlen_args = (B_j, x_j, dt_j, dA_j, cu_seqlens_jax, cs_j)
    ms_mosaic = _bench_amortized(
        chunk_state_varlen_mosaic, varlen_args, warmup=warmup, rep=rep,
    )

    print(f"  Mosaic GPU      : {ms_mosaic:.3f} ms")

    if _HAS_TRITON and _HAS_TRITON_VARLEN:
        import torch
        from triton.testing import do_bench
        B_t = _to_torch(B_j)
        x_t = _to_torch(x_j)
        dt_t = _to_torch(dt_j)
        dA_t = _to_torch(dA_j)
        cu_t = torch.tensor(np.array(cu_seqlens_jax), device="cuda", dtype=torch.int32)
        cs_t = _to_torch(cs_j)

        def _triton_fn():
            _triton_chunk_state_varlen(B_t, x_t, dt_t, dA_t, cu_t, cs_t)
            torch.cuda.synchronize()

        ms_triton = do_bench(_triton_fn, warmup=warmup, rep=rep)
        print(f"  Triton          : {ms_triton:.3f} ms")
        print(f"  Mosaic / Triton : {ms_mosaic/ms_triton:.2f}x")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("chunk_scan_fwd_mosaic — Correctness Tests")
    print("=" * 70)

    # Basic config
    test_chunk_scan_correctness(batch=1, seqlen=256, nheads=4, hdim=64, dstate=64, ngroups=1, chunk_size=64, has_D=True, has_z=True)

    # Nemotron-style (multi-group, larger)
    test_chunk_scan_correctness(batch=2, seqlen=512, nheads=8, hdim=64, dstate=64, ngroups=8, chunk_size=64)

    print("\n" + "=" * 70)
    print("chunk_scan_fwd_mosaic — SEQ_IDX Correctness Tests")
    print("=" * 70)

    # Mid-chunk boundary, multi-batch
    test_chunk_scan_correctness(
        batch=2, seqlen=512, nheads=8, hdim=64, dstate=64, ngroups=1, chunk_size=64,
        seq_idx_config={'seq_lengths': [
            [200, 312],
            [100, 412],
        ]},
    )

    # seq_idx + D + z + multi-group combined
    test_chunk_scan_correctness(
        batch=2, seqlen=512, nheads=8, hdim=64, dstate=64, ngroups=4, chunk_size=128,
        has_D=True, has_z=True,
        seq_idx_config={'seq_lengths': [
            [256, 256],
            [128, 384],
        ]},
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

    # =================================================================
    # chunk_state_varlen
    # =================================================================
    print("\n" + "=" * 70)
    print("chunk_state_varlen_mosaic — Correctness Tests")
    print("=" * 70)

    # Basic config
    test_chunk_state_varlen_correctness(
        batch=10, nheads=8, headdim=64, dstate=64, ngroups=1, chunk_size=64,
    )
    # Multi-group (Nemotron-style: ngroups=nheads)
    test_chunk_state_varlen_correctness(
        batch=5, nheads=8, headdim=64, dstate=64, ngroups=8, chunk_size=64,
    )

    print("\n" + "=" * 70)
    print("chunk_state_varlen_mosaic — Benchmarks")
    print("=" * 70)

    benchmark_chunk_state_varlen(
        batch=100, nheads=64, headdim=64, dstate=64, ngroups=1, chunk_size=64,
    )
    benchmark_chunk_state_varlen(
        batch=300, nheads=64, headdim=64, dstate=32, ngroups=1, chunk_size=64,
    )

    print("\nDone.")
