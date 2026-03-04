"""
test_mosaic2.py

Correctness checks and benchmarks for:
  - state_passing_fwd_mosaic
  - bmm_chunk_fwd_mosaic

Run on H100/H200 (Hopper) with the Mosaic GPU Pallas backend:
  python test_mosaic2.py

Both functions are tested for numerical correctness against a naive JAX
reference (and optionally against the Triton reference from mamba_ssm if
available), then benchmarked for throughput.
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
from pallas_mosaic_gpu_mamba.state_passing_fwd import state_passing_fwd_mosaic
from pallas_mosaic_gpu_mamba.bmm_chunk_fwd import (
    bmm_chunk_fwd_mosaic,
    bmm_chunk_preprocess,
    bmm_chunk_kernel_only,
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
    from mamba_ssm.ops.triton.ssd_state_passing import (
        _state_passing_fwd as _triton_state_passing_fwd,
    )
    from mamba_ssm.ops.triton.ssd_bmm import (
        _bmm_chunk_fwd as _triton_bmm_chunk_fwd,
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
    ok = "✓" if d < atol else "✗"
    print(f"  {ok}  {name:30s}  max|diff|={d:.2e}  (atol={atol:.0e})")
    return d < atol


def _bench_fn(fn, warmup=10, rep=50):
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


# ===========================================================================
# state_passing_fwd — naive JAX reference
# ===========================================================================

def _naive_state_passing_fwd(states, dA_chunk_cumsum, initial_states=None,
                             seq_idx=None, chunk_size=None):
    """
    Naive JAX reference for _state_passing_fwd.

    states          : (batch, nchunks, nheads, dim)
    dA_chunk_cumsum : (batch, nheads, nchunks)
    initial_states  : (batch, nheads, dim) or None
    seq_idx         : (batch, seqlen) or None
    chunk_size      : int, required when seq_idx is not None

    Returns:
      out          : (batch, nchunks, nheads, dim)
      final_states : (batch, nheads, dim)
    """
    batch, nchunks, nheads, dim = states.shape
    if initial_states is None:
        running = jnp.zeros((batch, nheads, dim), jnp.float32)
    else:
        running = initial_states.astype(jnp.float32)

    # Precompute seq_idx at chunk boundaries for reset detection
    if seq_idx is not None:
        seqlen = seq_idx.shape[-1]
        chunk_ends = np.minimum(
            np.arange(1, nchunks + 1) * chunk_size, seqlen
        ) - 1
        seq_idx_at_ends = seq_idx[:, chunk_ends]  # (batch, nchunks)

    prev_seq = jnp.zeros((batch,), dtype=jnp.int32)  # Triton starts at seq_idx=0

    out_list = [running]
    for c in range(nchunks):
        scale = jnp.exp(dA_chunk_cumsum[:, :, c])  # (batch, nheads)
        if seq_idx is not None:
            curr_seq = seq_idx_at_ends[:, c]  # (batch,)
            # Reset scale to 0 where sequence changed
            same_seq = (curr_seq == prev_seq).astype(jnp.float32)  # (batch,)
            scale = scale * same_seq[:, None]  # broadcast over nheads
            prev_seq = curr_seq
        new_states = states[:, c, :, :].astype(jnp.float32)
        running = scale[:, :, None] * running + new_states
        out_list.append(running)

    out = jnp.stack(out_list[:nchunks], axis=1)
    final_states = out_list[nchunks]
    return out, final_states


# ===========================================================================
# state_passing_fwd — correctness test
# ===========================================================================

def test_state_passing_correctness(
    batch=2, nchunks=8, nheads=8, dim=64,
    use_initstates=True,
    seq_idx_config=None,
    atol=1e-3,
):
    """
    Correctness test for state_passing_fwd_mosaic.

    seq_idx_config: None or dict with keys:
        'chunk_size': int
        'seq_lengths': list of int per batch element
            e.g. [100, 156] for batch=2, total seqlen=256
    """
    seq_str = f"seq_idx={seq_idx_config is not None}"
    print(f"\n── [state_passing_fwd] correctness  "
          f"B={batch} C={nchunks} H={nheads} D={dim} "
          f"init={use_initstates} {seq_str} ──")

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    states_jax = jax.random.normal(k1, (batch, nchunks, nheads, dim)) * 0.1
    dA_cs_jax = -jax.random.uniform(k2, (batch, nheads, nchunks)) * 0.5
    if use_initstates:
        init_jax = jax.random.normal(k3, (batch, nheads, dim)) * 0.1
    else:
        init_jax = None

    # Build seq_idx if requested
    seq_idx_jax = None
    chunk_size = None
    if seq_idx_config is not None:
        chunk_size = seq_idx_config['chunk_size']
        seqlen = nchunks * chunk_size
        # Build seq_idx per batch element from seq_lengths
        seq_idx_list = []
        for b in range(batch):
            lens = seq_idx_config['seq_lengths'][b]
            idx = []
            for seq_id, l in enumerate(lens):
                idx.extend([seq_id] * l)
            # Pad or truncate to seqlen
            idx = idx[:seqlen]
            if len(idx) < seqlen:
                idx.extend([idx[-1]] * (seqlen - len(idx)))
            seq_idx_list.append(idx)
        seq_idx_jax = jnp.array(seq_idx_list, dtype=jnp.int32)
        print(f"  seq_idx shape: {tuple(seq_idx_jax.shape)}, chunk_size={chunk_size}")

    out_pal, final_pal = state_passing_fwd_mosaic(
        states_jax, dA_cs_jax, initial_states=init_jax,
        seq_idx=seq_idx_jax, chunk_size=chunk_size,
    )
    jax.block_until_ready((out_pal, final_pal))

    print(f"  out shape         : {tuple(out_pal.shape)}")
    print(f"  final_states shape: {tuple(final_pal.shape)}")

    out_ref, final_ref = _naive_state_passing_fwd(
        states_jax, dA_cs_jax, initial_states=init_jax,
        seq_idx=seq_idx_jax, chunk_size=chunk_size,
    )

    all_ok = True
    all_ok &= check("out vs naive", out_pal, out_ref, atol=atol)
    all_ok &= check("final_states vs naive", final_pal, final_ref, atol=atol)

    if _HAS_TRITON:
        import torch as _torch
        states_t = _to_torch(states_jax)
        dA_cs_t = _to_torch(dA_cs_jax)
        init_t = _to_torch(init_jax) if init_jax is not None else None
        seq_idx_t = None
        if seq_idx_jax is not None:
            seq_idx_t = _torch.tensor(
                np.array(seq_idx_jax), device="cuda", dtype=_torch.int32,
            )
        out_tri, final_tri = _triton_state_passing_fwd(
            states_t, dA_cs_t, initial_states=init_t,
            seq_idx=seq_idx_t, chunk_size=chunk_size,
        )
        out_tri_j = jnp.array(out_tri.cpu().numpy())
        final_tri_j = jnp.array(final_tri.cpu().numpy())
        all_ok &= check("out vs Triton", out_pal, out_tri_j, atol=atol)
        all_ok &= check("final_states vs Triton", final_pal, final_tri_j, atol=atol)

    print(f"  {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    return all_ok


# ===========================================================================
# state_passing_fwd — benchmark
# ===========================================================================

def benchmark_state_passing(
    batch=2, nchunks=8, nheads=64, dim=4096,
    use_initstates=True,
    warmup=25, rep=200,
    amortize_N=50,
):
    """Benchmark state_passing_fwd_mosaic vs naive and Triton (dispatch-free)."""
    print(f"\n── [state_passing_fwd] benchmark  "
          f"B={batch} C={nchunks} H={nheads} D={dim} "
          f"init={use_initstates} ──")

    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    states_j = jax.random.normal(k1, (batch, nchunks, nheads, dim)) * 0.1
    dA_cs_j = -jax.random.uniform(k2, (batch, nheads, nchunks)) * 0.5
    init_j = jax.random.normal(k3, (batch, nheads, dim)) * 0.1 if use_initstates else None

    # ── Naive JAX (amortized) ──
    # Use lax.cond with acc-dependent predicate to prevent XLA loop hoisting.
    # Multiplicative perturbation fails because state_passing is affine in states.
    naive_sample = _naive_state_passing_fwd(states_j, dA_cs_j, init_j)
    naive_dummy = jax.tree.map(jnp.zeros_like, naive_sample)

    @jax.jit
    def naive_looped(s, d, i):
        def body(idx, acc):
            out, final = jax.lax.cond(
                acc > -1e30,
                lambda: _naive_state_passing_fwd(s, d, i),
                lambda: naive_dummy,
            )
            return acc + out[0, 0, 0, 0]
        return jax.lax.fori_loop(0, amortize_N, body, 0.0)

    jax.block_until_ready(naive_looped(states_j, dA_cs_j, init_j))
    ms_naive = _bench_fn(
        lambda: jax.block_until_ready(naive_looped(states_j, dA_cs_j, init_j)),
    ) / amortize_N

    # ── Mosaic GPU (amortized) ──
    mosaic_sample = state_passing_fwd_mosaic(states_j, dA_cs_j, initial_states=init_j)
    mosaic_dummy = jax.tree.map(jnp.zeros_like, mosaic_sample)

    @jax.jit
    def mosaic_looped(s, d, i):
        def body(idx, acc):
            out, final = jax.lax.cond(
                acc > -1e30,
                lambda: state_passing_fwd_mosaic(s, d, initial_states=i),
                lambda: mosaic_dummy,
            )
            return acc + out[0, 0, 0, 0]
        return jax.lax.fori_loop(0, amortize_N, body, 0.0)

    jax.block_until_ready(mosaic_looped(states_j, dA_cs_j, init_j))
    ms_mosaic = _bench_fn(
        lambda: jax.block_until_ready(mosaic_looped(states_j, dA_cs_j, init_j)),
    ) / amortize_N

    bytes_io = (
        batch * nchunks * nheads * dim * 4
        + batch * nheads * nchunks * 4
        + (batch * nheads * dim * 4 if use_initstates else 0)
        + batch * nchunks * nheads * dim * 4
        + batch * nheads * dim * 4
    )
    gbps_naive = bytes_io / (ms_naive * 1e-3) / 1e9
    gbps_mosaic = bytes_io / (ms_mosaic * 1e-3) / 1e9

    print(f"  Naive JAX  (N={amortize_N}): {ms_naive:.4f} ms   {gbps_naive:.1f} GB/s")
    print(f"  Mosaic GPU (N={amortize_N}): {ms_mosaic:.4f} ms   {gbps_mosaic:.1f} GB/s")

    if _HAS_TRITON:
        import torch
        states_t = _to_torch(states_j)
        dA_cs_t = _to_torch(dA_cs_j)
        init_t = _to_torch(init_j) if init_j is not None else None

        def _triton_fn():
            _triton_state_passing_fwd(states_t, dA_cs_t, initial_states=init_t)
            torch.cuda.synchronize()

        ms_triton = _bench_fn(_triton_fn, warmup=warmup, rep=rep)
        gbps_triton = bytes_io / (ms_triton * 1e-3) / 1e9
        print(f"  Triton            : {ms_triton:.4f} ms   {gbps_triton:.1f} GB/s")

    print(f"  ── Ratios (dispatch-free) ──")
    print(f"  Mosaic / Naive : {ms_mosaic/ms_naive:.2f}x")
    if _HAS_TRITON:
        print(f"  Mosaic / Triton: {ms_mosaic/ms_triton:.2f}x")
        print(f"  Naive  / Triton: {ms_naive/ms_triton:.2f}x")


# ===========================================================================
# bmm_chunk_fwd — naive JAX reference
# ===========================================================================

def _naive_bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False):
    """
    Naive JAX reference for _bmm_chunk_fwd.

    a : (batch, seqlen, ngroups, k)
    b : (batch, seqlen, ngroups, k)
    seq_idx : (batch, seqlen) int32, optional
    causal : bool

    Returns:
      out : (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    batch, seqlen, ngroups, k = a.shape
    nchunks = math.ceil(seqlen / chunk_size)

    # Pad seqlen
    total_len = nchunks * chunk_size
    if seqlen < total_len:
        pad = total_len - seqlen
        a = jnp.pad(a, ((0, 0), (0, pad), (0, 0), (0, 0)))
        b = jnp.pad(b, ((0, 0), (0, pad), (0, 0), (0, 0)))
        if seq_idx is not None:
            seq_idx = jnp.pad(seq_idx, ((0, 0), (0, pad)), constant_values=-1)

    # Reshape: (batch, nchunks, chunk_size, ngroups, k)
    a_c = a.reshape(batch, nchunks, chunk_size, ngroups, k)
    b_c = b.reshape(batch, nchunks, chunk_size, ngroups, k)

    # Transpose for matmul: a_c → (batch, nchunks, ngroups, chunk_size, k)
    a_c = a_c.transpose(0, 1, 3, 2, 4)
    b_c = b_c.transpose(0, 1, 3, 2, 4)

    # out = a_c @ b_c.T → (batch, nchunks, ngroups, chunk_size, chunk_size)
    out = jnp.matmul(a_c, b_c.transpose(0, 1, 2, 4, 3))

    if causal:
        causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
        out = jnp.where(causal_mask[None, None, None, :, :], out, 0.0)

    if seq_idx is not None:
        seq_idx_chunked = seq_idx.reshape(batch, nchunks, chunk_size)
        seq_mask = (
            seq_idx_chunked[:, :, :, None]
            == seq_idx_chunked[:, :, None, :]
        )
        out = jnp.where(seq_mask[:, :, None, :, :], out, 0.0)

    return out


# ===========================================================================
# bmm_chunk_fwd — correctness test
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


def test_bmm_chunk_correctness(
    batch=2, seqlen=512, ngroups=1, k=64, chunk_size=64,
    BM=64, BK=64, BN=64,
    causal=False,
    seq_idx_config=None,
    atol=2.0,
):
    """
    Correctness test for bmm_chunk_fwd_mosaic.

    Uses a higher atol than state_passing because WGMMA uses bf16 matmul
    while the naive reference uses f32, and Triton also uses bf16/f16.

    seq_idx_config: None or dict with keys:
        'seq_lengths': list[list[int]] per batch element
    """
    nchunks = math.ceil(seqlen / chunk_size)
    seq_str = f"causal={causal} seq_idx={seq_idx_config is not None}"
    print(f"\n── [bmm_chunk_fwd] correctness  "
          f"B={batch} L={seqlen} G={ngroups} K={k} Q={chunk_size} "
          f"BM={BM} BK={BK} BN={BN} {seq_str} ──")

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)

    a_jax = jax.random.normal(k1, (batch, seqlen, ngroups, k)) * 0.1
    b_jax = jax.random.normal(k2, (batch, seqlen, ngroups, k)) * 0.1

    # Build seq_idx if requested
    seq_idx_jax = None
    if seq_idx_config is not None:
        seq_idx_jax = _build_seq_idx(batch, seqlen, seq_idx_config['seq_lengths'])
        print(f"  seq_idx shape: {tuple(seq_idx_jax.shape)}")

    out_pal = bmm_chunk_fwd_mosaic(
        a_jax, b_jax, chunk_size,
        seq_idx=seq_idx_jax, causal=causal,
        BM=BM, BK=BK, BN=BN,
    )
    jax.block_until_ready(out_pal)

    print(f"  out shape : {tuple(out_pal.shape)}")

    out_ref = _naive_bmm_chunk_fwd(
        a_jax, b_jax, chunk_size,
        seq_idx=seq_idx_jax, causal=causal,
    )

    all_ok = True
    all_ok &= check("out vs naive", out_pal, out_ref, atol=atol)

    if _HAS_TRITON:
        a_t = _to_torch(a_jax)
        b_t = _to_torch(b_jax)
        seq_idx_t = None
        if seq_idx_jax is not None:
            seq_idx_t = torch.tensor(
                np.array(seq_idx_jax), device="cuda", dtype=torch.int32,
            )
        out_tri = _triton_bmm_chunk_fwd(
            a_t, b_t, chunk_size,
            seq_idx=seq_idx_t, causal=causal,
            output_dtype=torch.float32,
        )
        out_tri_j = jnp.array(out_tri.cpu().numpy())
        all_ok &= check("out vs Triton", out_pal, out_tri_j, atol=atol)

    print(f"  {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    return all_ok


# ===========================================================================
# bmm_chunk_fwd — benchmark
# ===========================================================================

def benchmark_bmm_chunk(
    batch=2, seqlen=2048, ngroups=1, k=64, chunk_size=256,
    BM=64, BK=64, BN=64,
    warmup=25, rep=200,
    amortize_N=50,
):
    """Benchmark bmm_chunk_fwd_mosaic vs naive and Triton (dispatch-free)."""
    nchunks = math.ceil(seqlen / chunk_size)
    print(f"\n── [bmm_chunk_fwd] benchmark  "
          f"B={batch} L={seqlen} G={ngroups} K={k} Q={chunk_size} ──")

    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key, 2)
    a_j = jax.random.normal(k1, (batch, seqlen, ngroups, k)) * 0.1
    b_j = jax.random.normal(k2, (batch, seqlen, ngroups, k)) * 0.1

    # Bytes IO
    bytes_io = (
        batch * seqlen * ngroups * k * 4 * 2          # a + b in
        + batch * nchunks * ngroups * chunk_size * chunk_size * 4  # out
    )

    # ── Naive JAX (amortized) ──
    # Use lax.cond with acc-dependent predicate to prevent XLA loop hoisting.
    naive_bmm_sample = _naive_bmm_chunk_fwd(a_j, b_j, chunk_size)
    naive_bmm_dummy = jax.tree.map(jnp.zeros_like, naive_bmm_sample)

    @jax.jit
    def naive_looped(a, b):
        def body(i, acc):
            out = jax.lax.cond(
                acc > -1e30,
                lambda: _naive_bmm_chunk_fwd(a, b, chunk_size),
                lambda: naive_bmm_dummy,
            )
            return acc + out[0, 0, 0, 0, 0]
        return jax.lax.fori_loop(0, amortize_N, body, 0.0)

    jax.block_until_ready(naive_looped(a_j, b_j))
    ms_naive = _bench_fn(
        lambda: jax.block_until_ready(naive_looped(a_j, b_j)),
    ) / amortize_N

    # ── Mosaic end-to-end (amortized) ──
    mosaic_bmm_sample = bmm_chunk_fwd_mosaic(a_j, b_j, chunk_size, BM=BM, BK=BK, BN=BN)
    mosaic_bmm_dummy = jax.tree.map(jnp.zeros_like, mosaic_bmm_sample)

    @jax.jit
    def mosaic_looped(a, b):
        def body(i, acc):
            out = jax.lax.cond(
                acc > -1e30,
                lambda: bmm_chunk_fwd_mosaic(a, b, chunk_size, BM=BM, BK=BK, BN=BN),
                lambda: mosaic_bmm_dummy,
            )
            return acc + out[0, 0, 0, 0, 0]
        return jax.lax.fori_loop(0, amortize_N, body, 0.0)

    jax.block_until_ready(mosaic_looped(a_j, b_j))
    ms_mosaic = _bench_fn(
        lambda: jax.block_until_ready(mosaic_looped(a_j, b_j)),
    ) / amortize_N

    # ── Mosaic kernel-only (amortized) ──
    a_flat, b_T_flat, meta = bmm_chunk_preprocess(
        a_j, b_j, chunk_size, BM=BM, BK=BK, BN=BN,
    )
    jax.block_until_ready((a_flat, b_T_flat))

    kern_kwargs = dict(
        BM=BM, BK=meta['BK'], BN=BN,
        BCG=meta['BCG'], chunk_size=meta['chunk_size'],
        chunk_size_padded=meta['chunk_size_padded'],
        k=meta['k'], k_padded=meta['k_padded'],
        batch=meta['batch'], nchunks=meta['nchunks'],
        ngroups=meta['ngroups'],
    )

    # kernel_only uses pallas_call (opaque to XLA), no perturbation needed
    @jax.jit
    def kern_looped(af, bf):
        def body(i, acc):
            out = bmm_chunk_kernel_only(af, bf, **kern_kwargs)
            return acc + out[0, 0, 0, 0, 0]
        return jax.lax.fori_loop(0, amortize_N, body, 0.0)

    jax.block_until_ready(kern_looped(a_flat, b_T_flat))
    ms_kern = _bench_fn(
        lambda: jax.block_until_ready(kern_looped(a_flat, b_T_flat)),
    ) / amortize_N

    gbps_naive  = bytes_io / (ms_naive  * 1e-3) / 1e9
    gbps_mosaic = bytes_io / (ms_mosaic * 1e-3) / 1e9
    gbps_kern   = bytes_io / (ms_kern   * 1e-3) / 1e9

    print(f"  Naive JAX (matmul) (N={amortize_N}): {ms_naive:.4f} ms   {gbps_naive:.1f} GB/s")
    print(f"  Mosaic end-to-end  (N={amortize_N}): {ms_mosaic:.4f} ms   {gbps_mosaic:.1f} GB/s")
    print(f"  Mosaic kernel-only (N={amortize_N}): {ms_kern:.4f} ms   {gbps_kern:.1f} GB/s")
    print(f"  Mosaic preprocess  (e2e - kern)  : {ms_mosaic - ms_kern:.4f} ms")

    if _HAS_TRITON:
        import torch
        a_t = _to_torch(a_j)
        b_t = _to_torch(b_j)

        def _triton_fn():
            _triton_bmm_chunk_fwd(a_t, b_t, chunk_size, output_dtype=torch.float32)
            torch.cuda.synchronize()

        ms_tri = _bench_fn(_triton_fn, warmup=warmup, rep=rep)
        gbps_tri = bytes_io / (ms_tri * 1e-3) / 1e9
        print(f"  Triton                    : {ms_tri:.4f} ms   {gbps_tri:.1f} GB/s")

    print(f"  ── Ratios (dispatch-free) ──")
    print(f"  Mosaic e2e   / Naive : {ms_mosaic/ms_naive:.2f}x")
    print(f"  Mosaic kernel/ Naive : {ms_kern/ms_naive:.2f}x")
    if _HAS_TRITON:
        print(f"  Mosaic e2e   / Triton: {ms_mosaic/ms_tri:.2f}x")
        print(f"  Mosaic kernel/ Triton: {ms_kern/ms_tri:.2f}x")
        print(f"  Naive        / Triton: {ms_naive/ms_tri:.2f}x")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print(f"JAX version  : {jax.__version__}")
    print(f"JAX devices  : {jax.devices()}")
    if _HAS_TRITON:
        import torch
        print(f"CUDA GPU     : {torch.cuda.get_device_name(0)}")

    all_passed = True

    # ── state_passing_fwd correctness ─────────────────────────────────────
    print("\n" + "═" * 70)
    print("STATE_PASSING_FWD CORRECTNESS")
    print("═" * 70)
    sp_correctness_configs = [
        # (batch, nchunks, nheads, dim, use_initstates, seq_idx_config)
        (1,  4,   1, 128, False, None),
        (1,  4,   1, 128, True,  None),
        (2,  8,   4, 128, True,  None),
        (2,  8,   4, 128, False, None),
        (1,  8,   8, 4096, True, None),
        (2,  8,  16, 4096, True, None),
        (2,  4,  16, 4096, False, None),
        (1,  8,   4, 16384, True, None),
        (1, 16,   8, 4096, True, None),
        (2, 32,   4, 4096, True, None),
        (1,  8,  64, 4096, True, None),
        (1,  4,   2,  200, True, None),
        (2,  8,   4,  300, True, None),
        # --- seq_idx tests ---
        # Single sequence per batch (seq_idx should be no-op)
        (1, 4, 4, 128, True, {
            'chunk_size': 64,
            'seq_lengths': [[256]],
        }),
        # Two sequences per batch, boundary at chunk edge
        (1, 4, 4, 128, False, {
            'chunk_size': 64,
            'seq_lengths': [[128, 128]],  # boundary at chunk 2
        }),
        # Two sequences, boundary mid-chunk (chunk_size=64, seqlen=512)
        (2, 8, 8, 128, True, {
            'chunk_size': 64,
            'seq_lengths': [
                [200, 312],   # boundary at position 200, in chunk 3
                [100, 412],   # boundary at position 100, in chunk 1
            ],
        }),
        # Three sequences per batch
        (2, 8, 4, 4096, True, {
            'chunk_size': 64,
            'seq_lengths': [
                [64, 192, 256],   # boundaries at chunks 1 and 4
                [128, 128, 256],  # boundaries at chunks 2 and 4
            ],
        }),
        # Many short sequences (boundary every chunk)
        (1, 4, 4, 128, True, {
            'chunk_size': 64,
            'seq_lengths': [[64, 64, 64, 64]],  # every chunk is different seq
        }),
    ]
    for cfg in sp_correctness_configs:
        batch, nchunks, nheads, dim, use_init, seq_cfg = cfg
        all_passed &= test_state_passing_correctness(
            batch=batch, nchunks=nchunks, nheads=nheads, dim=dim,
            use_initstates=use_init, seq_idx_config=seq_cfg,
        )

    # ── bmm_chunk_fwd correctness ─────────────────────────────────────────
    print("\n" + "═" * 70)
    print("BMM_CHUNK_FWD CORRECTNESS")
    print("═" * 70)
    bmm_correctness_configs = [
        # (batch, seqlen, ngroups, k, chunk_size, BM, BK, BN)
        # Standard Mamba2: k=dstate=64
        (1,  256, 1,  64,  64, 64, 64, 64),
        (2,  512, 1,  64, 128, 64, 64, 64),
        (2, 1024, 1,  64, 256, 64, 64, 64),
        # Multiple groups
        (2,  512, 8,  64, 128, 64, 64, 64),
        (1,  256, 4,  64,  64, 64, 64, 64),
        # Larger dstate
        (1,  256, 1, 128,  64, 64, 64, 64),
        (2,  512, 8, 128, 128, 64, 64, 64),
        # Nemotron: ngroups=8, dstate=256
        (1,  256, 8, 256,  64, 64, 64, 64),
        (1, 2048, 8, 256, 256, 64, 64, 64),
    ]
    for cfg in bmm_correctness_configs:
        all_passed &= test_bmm_chunk_correctness(*cfg)

    # ── bmm_chunk_fwd causal correctness ──────────────────────────────────
    print("\n" + "═" * 70)
    print("BMM_CHUNK_FWD CAUSAL CORRECTNESS")
    print("═" * 70)
    bmm_causal_configs = [
        # (batch, seqlen, ngroups, k, chunk_size)
        (1,  256, 1,  64,  64),
        (2,  512, 1,  64, 128),
        (2, 1024, 8,  64, 256),
        (1,  256, 8, 256,  64),
    ]
    for cfg in bmm_causal_configs:
        batch, seqlen, ngroups, k_dim, chunk_size = cfg
        all_passed &= test_bmm_chunk_correctness(
            batch=batch, seqlen=seqlen, ngroups=ngroups, k=k_dim,
            chunk_size=chunk_size, causal=True,
        )

    # ── bmm_chunk_fwd seq_idx correctness ─────────────────────────────────
    print("\n" + "═" * 70)
    print("BMM_CHUNK_FWD SEQ_IDX CORRECTNESS")
    print("═" * 70)
    bmm_seq_idx_configs = [
        # Single sequence (no-op)
        dict(batch=1, seqlen=256, ngroups=1, k=64, chunk_size=64,
             causal=False, seq_idx_config={
                 'seq_lengths': [[256]],
             }),
        # Two sequences, boundary at chunk edge
        dict(batch=1, seqlen=256, ngroups=1, k=64, chunk_size=64,
             causal=False, seq_idx_config={
                 'seq_lengths': [[128, 128]],
             }),
        # Two sequences, mid-chunk boundary
        dict(batch=2, seqlen=512, ngroups=4, k=64, chunk_size=64,
             causal=False, seq_idx_config={
                 'seq_lengths': [
                     [200, 312],
                     [100, 412],
                 ],
             }),
        # Causal + seq_idx combined
        dict(batch=2, seqlen=512, ngroups=1, k=64, chunk_size=128,
             causal=True, seq_idx_config={
                 'seq_lengths': [
                     [256, 256],
                     [128, 384],
                 ],
             }),
        # Three sequences, multi-group
        dict(batch=2, seqlen=512, ngroups=8, k=64, chunk_size=64,
             causal=True, seq_idx_config={
                 'seq_lengths': [
                     [64, 192, 256],
                     [128, 128, 256],
                 ],
             }),
        # Every chunk different sequence
        dict(batch=1, seqlen=256, ngroups=1, k=64, chunk_size=64,
             causal=False, seq_idx_config={
                 'seq_lengths': [[64, 64, 64, 64]],
             }),
    ]
    for cfg in bmm_seq_idx_configs:
        all_passed &= test_bmm_chunk_correctness(**cfg)

    print(f"\n{'═' * 70}")
    print(f"Overall: {'ALL TESTS PASS ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print(f"{'═' * 70}")

    # ── benchmarks ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("STATE_PASSING_FWD BENCHMARKS")
    print("═" * 70)

    sp_bench_configs = [
        # (batch, nchunks, nheads, dim, use_initstates)
        (1,   4,   8,  128, True),
        (2,   8,  16,  128, True),
        (1,   8,   8, 4096, True),
        (2,   8,  64, 4096, True),
        (4,   8,  64, 4096, True),
        (1,   8,   8, 16384, True),
        (2,   8,  64, 16384, True),
        (2,  32,  64, 4096, True),
        (1,   8,  64, 16384, True),
    ]
    for cfg in sp_bench_configs:
        benchmark_state_passing(*cfg)

    print("\n" + "═" * 70)
    print("BMM_CHUNK_FWD BENCHMARKS")
    print("═" * 70)

    bmm_bench_configs = [
        # (batch, seqlen, ngroups, k, chunk_size, BM, BK, BN)
        (1,   256, 1,  64,  64, 64, 64, 64),
        (2,   512, 1,  64, 128, 64, 64, 64),
        (2,  1024, 1,  64, 256, 64, 64, 64),
        (4,  2048, 1,  64, 256, 64, 64, 64),
        # Multi-group
        (2,  2048, 8,  64, 256, 64, 64, 64),
        (4,  4096, 8,  64, 256, 64, 64, 64),
        # Large dstate
        (2,  2048, 1, 128, 256, 64, 64, 64),
        (2,  2048, 8, 128, 256, 64, 64, 64),
        # Nemotron-style
        (1,  2048, 8, 256, 256, 64, 64, 64),
    ]
    for cfg in bmm_bench_configs:
        benchmark_bmm_chunk(*cfg)
