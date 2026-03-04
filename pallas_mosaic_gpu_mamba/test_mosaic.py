"""
test_mosaic.py

Correctness checks and benchmarks for:
  - chunk_cumsum_fwd_mosaic
  - chunk_state_fwd_mosaic

Run on H100/H200 (Hopper) with the Mosaic GPU Pallas backend:
  python test_mosaic.py

Both functions are tested for numerical correctness against a naive JAX
reference (and optionally against the Triton reference from mamba_ssm if
available), then benchmarked for throughput.
"""

import os
import sys
import math
import types

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Make the package importable when running from inside the directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pallas_mosaic_gpu_mamba.chunk_cumsum_fwd import (
    chunk_cumsum_fwd_mosaic,
    chunk_cumsum_fwd_pallas,
    chunk_cumsum_fwd_naive_jax,
)
from pallas_mosaic_gpu_mamba.chunk_state_fwd import (
    chunk_state_fwd_mosaic,
    chunk_state_preprocess,
    chunk_state_kernel_only,
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
    from mamba_ssm.ops.triton.ssd_chunk_state import (
        _chunk_cumsum_fwd as _triton_cumsum_fwd,
        _chunk_state_fwd  as _triton_state_fwd,
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
    print(f"  {ok}  {name:20s}  max|diff|={d:.2e}  (atol={atol:.0e})")
    return d < atol


# ===========================================================================
# chunk_cumsum_fwd
# ===========================================================================

def _naive_cumsum(dt, A, bias=None, softplus=False, chunk_size=64):
    """Naive JAX reference for chunk_cumsum_fwd."""
    if bias is not None:
        dt = dt + bias[None, None, :]
    if softplus:
        safe = jnp.where(dt <= 20.0, dt, jnp.zeros_like(dt))
        dt   = jnp.where(dt <= 20.0, jnp.log1p(jnp.exp(safe)), dt)
    dt = jnp.clip(dt, 0.0, float("inf"))
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)
    pad = nchunks * chunk_size - seqlen
    if pad:
        dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
    dt = dt.reshape(batch, nchunks, chunk_size, nheads).transpose(0, 3, 1, 2)
    # dt now: (batch, nheads, nchunks, chunk_size)
    dA = dt * A[None, :, None, None]
    return jnp.cumsum(dA, axis=3), dt


def test_cumsum_correctness(
    batch=2, seqlen=512, nheads=24, chunk_size=256,
    dt_softplus=True, use_bias=True,
    atol=1e-3,
):
    print(f"\n── [chunk_cumsum_fwd] correctness  "
          f"B={batch} L={seqlen} H={nheads} Q={chunk_size} "
          f"softplus={dt_softplus} bias={use_bias} ──")

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    dt_jax   = jax.random.normal(k1, (batch, seqlen, nheads))
    A_jax    = -jax.random.uniform(k2, (nheads,)) * 0.1
    bias_jax = jax.random.normal(k3, (nheads,)) if use_bias else None

    dA_pal, dt_pal = chunk_cumsum_fwd_mosaic(
        dt_jax, A_jax, chunk_size,
        dt_bias=bias_jax, dt_softplus=dt_softplus,
    )
    jax.block_until_ready((dA_pal, dt_pal))

    dA_ref, dt_ref = _naive_cumsum(dt_jax, A_jax, bias_jax, dt_softplus, chunk_size)

    all_ok = True
    all_ok &= check("dt_out  vs naive", dt_pal, dt_ref, atol=atol)
    all_ok &= check("dA_cs   vs naive", dA_pal, dA_ref, atol=atol)

    if _HAS_TRITON:
        dt_t   = _to_torch(dt_jax)
        A_t    = _to_torch(A_jax)
        bias_t = _to_torch(bias_jax) if bias_jax is not None else None
        dA_tri, dt_tri = _triton_cumsum_fwd(
            dt_t, A_t, chunk_size,
            dt_bias=bias_t, dt_softplus=dt_softplus, dt_limit=(0.0, float("inf")),
        )
        all_ok &= check("dt_out  vs Triton", dt_pal,
                        jnp.array(dt_tri.cpu().numpy()), atol=atol)
        all_ok &= check("dA_cs   vs Triton", dA_pal,
                        jnp.array(dA_tri.cpu().numpy()), atol=atol)

    print(f"  {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    return all_ok


def _bench_fn(fn, warmup, rep):
    """Benchmark a callable, using triton do_bench if available, else manual timing."""
    if _HAS_TRITON:
        from triton.testing import do_bench
        return do_bench(fn, warmup=warmup, rep=rep)
    else:
        import time
        times = []
        for _ in range(warmup + rep):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.median(times[warmup:])) * 1e3


def _bench_amortized(impl_fn, dt, A, bias, chunk_size, dt_softplus,
                     N=100, warmup=10, rep=50):
    """
    Measure true GPU time by amortizing JAX dispatch overhead.

    Wraps N calls inside a single jax.jit + fori_loop.  The accumulator
    dependency prevents XLA from dead-code-eliminating the kernel.
    Per-call time = total_time / N, which converges to pure GPU time as
    N grows.

    Uses jax.lax.cond with an always-true but acc-dependent condition to
    prevent XLA from hoisting loop-invariant pure-JAX computations out of
    the loop.  Multiplicative perturbation (e.g. ``* (1 + acc * eps)``)
    does NOT work because functions like cumsum are affine in their input,
    allowing XLA to decompose and hoist regardless of scale.
    """
    # Run once to get output shapes for the dummy (false) branch of lax.cond.
    # (jax.eval_shape can't be used here because impl_fn may do
    # shape-dependent control flow with concrete integer args.)
    sample = impl_fn(dt, A, chunk_size, dt_bias=bias, dt_softplus=dt_softplus)
    dummy = jax.tree.map(jnp.zeros_like, sample)

    @jax.jit
    def looped(dt, A, b):
        def body(i, acc):
            # lax.cond with acc-dependent predicate prevents XLA loop hoisting
            dA, dt_o = jax.lax.cond(
                acc > -1e30,  # always True, but depends on acc
                lambda: impl_fn(dt, A, chunk_size, dt_bias=b,
                                dt_softplus=dt_softplus),
                lambda: dummy,
            )
            return acc + dA[0, 0, 0, 0]  # prevent DCE
        return jax.lax.fori_loop(0, N, body, 0.0)

    # Warm up (compile)
    jax.block_until_ready(looped(dt, A, bias))

    ms_total = _bench_fn(
        lambda: jax.block_until_ready(looped(dt, A, bias)),
        warmup, rep,
    )
    return ms_total / N


def benchmark_cumsum(
    batch=2, seqlen=2048, nheads=64, chunk_size=256,
    dt_softplus=True, warmup=25, rep=200,
    amortize_N=100,
):
    print(f"\n── [chunk_cumsum_fwd] benchmark  "
          f"B={batch} L={seqlen} H={nheads} Q={chunk_size} ──")

    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    dt_j   = jax.random.normal(k1, (batch, seqlen, nheads))
    A_j    = -jax.random.uniform(k2, (nheads,)) * 0.1
    bias_j = jax.random.normal(k3, (nheads,))

    nchunks = math.ceil(seqlen / chunk_size)
    bytes_io = (
        batch * seqlen * nheads * 4
        + nheads * 4
        + nheads * 4
        + 2 * batch * nheads * nchunks * chunk_size * 4
    )

    # ── 1. Pure XLA (amortized — no dispatch overhead) ──
    ms_xla = _bench_amortized(
        chunk_cumsum_fwd_mosaic, dt_j, A_j, bias_j,
        chunk_size, dt_softplus, N=amortize_N,
    )

    # ── 2. Naive JAX (amortized) ──
    ms_naive = _bench_amortized(
        chunk_cumsum_fwd_naive_jax, dt_j, A_j, bias_j,
        chunk_size, dt_softplus, N=amortize_N,
    )

    # ── 3. Pallas Triton backend (amortized) ──
    ms_pal = _bench_amortized(
        chunk_cumsum_fwd_pallas, dt_j, A_j, bias_j,
        chunk_size, dt_softplus, N=amortize_N,
    )

    gbps_xla   = bytes_io / (ms_xla   * 1e-3) / 1e9
    gbps_naive = bytes_io / (ms_naive  * 1e-3) / 1e9
    gbps_pal   = bytes_io / (ms_pal    * 1e-3) / 1e9
    print(f"  Pure XLA      (N={amortize_N}): {ms_xla:.4f} ms   {gbps_xla:.1f} GB/s")
    print(f"  Naive JAX     (N={amortize_N}): {ms_naive:.4f} ms   {gbps_naive:.1f} GB/s")
    print(f"  Pallas Triton (N={amortize_N}): {ms_pal:.4f} ms   {gbps_pal:.1f} GB/s")

    # ── 4. Triton reference (if available) ──
    if _HAS_TRITON:
        import torch
        dt_t   = _to_torch(dt_j)
        A_t    = _to_torch(A_j)
        bias_t = _to_torch(bias_j)

        def _triton_fn():
            _triton_cumsum_fwd(dt_t, A_t, chunk_size, dt_bias=bias_t,
                               dt_softplus=dt_softplus, dt_limit=(0.0, float("inf")))
            torch.cuda.synchronize()

        ms_tri = _bench_fn(_triton_fn, warmup, rep)
        gbps_tri = bytes_io / (ms_tri * 1e-3) / 1e9
        print(f"  Triton             : {ms_tri:.4f} ms   {gbps_tri:.1f} GB/s")

    # ── Ratios ──
    print(f"  ── Ratios (dispatch-free) ──")
    print(f"  XLA          / Naive : {ms_xla/ms_naive:.2f}x")
    print(f"  Pallas Triton/ Naive : {ms_pal/ms_naive:.2f}x")
    if _HAS_TRITON:
        print(f"  XLA          / Triton: {ms_xla/ms_tri:.2f}x")
        print(f"  Naive JAX    / Triton: {ms_naive/ms_tri:.2f}x")
        print(f"  Pallas Triton/ Triton: {ms_pal/ms_tri:.2f}x")


# ===========================================================================
# chunk_state_fwd
# ===========================================================================

def _naive_chunk_state(x, B, dt, dA_cumsum, seq_idx=None):
    """
    Naive JAX reference for chunk_state_fwd.

    x         : (batch, seqlen, nheads, headdim)
    B         : (batch, seqlen, ngroups, dstate)
    dt        : (batch, nheads, nchunks, chunk_size)   (post-processed)
    dA_cumsum : (batch, nheads, nchunks, chunk_size)
    seq_idx   : (batch, seqlen) or None

    Returns states : (batch, nchunks, nheads, headdim, dstate)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups

    # Pad seqlen if needed
    total_len = nchunks * chunk_size
    if seqlen < total_len:
        pad = total_len - seqlen
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        B = jnp.pad(B, ((0, 0), (0, pad), (0, 0), (0, 0)))
        if seq_idx is not None:
            seq_idx = jnp.pad(seq_idx, ((0, 0), (0, pad)), constant_values=-1)

    # scale: (batch, nheads, nchunks, chunk_size)
    dA_cs_last = dA_cumsum[:, :, :, -1:]
    scale = jnp.exp(jnp.minimum(dA_cs_last - dA_cumsum, 0.0)) * dt

    # Handle seq_idx: zero scale where seq_idx[k] != seq_idx[chunk_end]
    if seq_idx is not None:
        chunk_ends = np.minimum(
            np.arange(1, nchunks + 1) * chunk_size, seqlen
        ) - 1
        seq_idx_last = seq_idx[:, chunk_ends]  # (batch, nchunks)
        seq_idx_chunked = seq_idx.reshape(batch, nchunks, chunk_size)
        same_seq = (seq_idx_last[:, :, None] >= 0) & (
            seq_idx_chunked == seq_idx_last[:, :, None]
        )  # (batch, nchunks, chunk_size)
        # scale is (batch, nheads, nchunks, chunk_size)
        # same_seq is (batch, nchunks, chunk_size) → need (batch, 1, nchunks, chunk_size)
        scale = jnp.where(same_seq[:, None, :, :], scale, 0.0)

    # x → (batch, nchunks, nheads, headdim, chunk_size)
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim).transpose(0, 1, 3, 4, 2)
    # B → (batch, nchunks, ngroups, chunk_size, dstate)
    B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).transpose(0, 1, 3, 2, 4)

    # Expand B along heads axis: (batch, nchunks, nheads, chunk_size, dstate)
    B_exp = jnp.repeat(B, ratio, axis=2)

    # scale → (batch, nchunks, nheads, chunk_size)
    scale_t = scale.transpose(0, 2, 1, 3)

    # B_pre: (batch, nchunks, nheads, chunk_size, dstate)
    B_pre = B_exp * scale_t[:, :, :, :, None]

    # states = x @ B_pre → (batch, nchunks, nheads, headdim, dstate)
    states = jnp.matmul(x, B_pre)
    return states


def _build_seq_idx(batch, seqlen, seq_lengths_per_batch):
    """
    Build a seq_idx array from per-batch sequence lengths.

    seq_lengths_per_batch: list of list of ints, one per batch element.
        e.g. [[100, 156], [200, 56]] for batch=2, seqlen=256
    Returns: jnp.array of shape (batch, seqlen), dtype int32
    """
    seq_idx_list = []
    for b in range(batch):
        lens = seq_lengths_per_batch[b]
        idx = []
        for seq_id, l in enumerate(lens):
            idx.extend([seq_id] * l)
        idx = idx[:seqlen]
        if len(idx) < seqlen:
            idx.extend([idx[-1]] * (seqlen - len(idx)))
        seq_idx_list.append(idx)
    return jnp.array(seq_idx_list, dtype=jnp.int32)


def test_state_correctness(
    batch=2, seqlen=512, nheads=8, headdim=64, dstate=64,
    ngroups=1, chunk_size=64,
    BM=64, BK=64, BN=64,
    seq_idx_config=None,
    atol=1e-2,
):
    """
    Correctness test for chunk_state_fwd_mosaic.

    dt and dA_cumsum are constructed directly in (batch, nheads, nchunks, chunk_size)
    format (i.e. they represent already-processed values, as produced by
    chunk_cumsum_fwd).

    seq_idx_config: None or dict with key 'seq_lengths' (list of list of ints).
    """
    nchunks = math.ceil(seqlen / chunk_size)
    seq_str = f"seq_idx={seq_idx_config is not None}"
    print(f"\n── [chunk_state_fwd] correctness  "
          f"B={batch} L={seqlen} H={nheads} D={headdim} S={dstate} "
          f"G={ngroups} Q={chunk_size} BM={BM} BK={BK} BN={BN} {seq_str} ──")

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x_jax  = jax.random.normal(k1, (batch, seqlen, nheads, headdim))
    B_jax  = jax.random.normal(k2, (batch, seqlen, ngroups, dstate))
    # dt in (batch, nheads, nchunks, chunk_size), positive (post-softplus)
    dt_jax = jax.random.uniform(k3, (batch, nheads, nchunks, chunk_size)) * 0.1 + 0.01
    # dA_cumsum: negative, monotonically decreasing in last dim (approx)
    dA_raw = -jax.random.uniform(k4, (batch, nheads, nchunks, chunk_size)) * 0.1
    dA_cumsum_jax = jnp.cumsum(dA_raw, axis=3)

    # Build seq_idx if requested
    seq_idx_jax = None
    if seq_idx_config is not None:
        seq_idx_jax = _build_seq_idx(batch, seqlen, seq_idx_config['seq_lengths'])
        print(f"  seq_idx shape: {tuple(seq_idx_jax.shape)}")

    states_pal = chunk_state_fwd_mosaic(
        x_jax, B_jax, dt_jax, dA_cumsum_jax,
        seq_idx=seq_idx_jax,
        BM=BM, BK=BK, BN=BN,
    )
    jax.block_until_ready(states_pal)

    print(f"  states shape : {tuple(states_pal.shape)}")

    states_ref = _naive_chunk_state(x_jax, B_jax, dt_jax, dA_cumsum_jax,
                                    seq_idx=seq_idx_jax)

    all_ok = True
    all_ok &= check("states vs naive", states_pal, states_ref, atol=atol)

    if _HAS_TRITON:
        import torch as _torch
        # Triton reference: _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=...)
        B_t          = _to_torch(B_jax)
        x_t          = _to_torch(x_jax)
        dt_t         = _to_torch(dt_jax)
        dA_cumsum_t  = _to_torch(dA_cumsum_jax)
        seq_idx_t = None
        if seq_idx_jax is not None:
            seq_idx_t = _torch.tensor(
                np.array(seq_idx_jax), device="cuda", dtype=_torch.int32,
            )
        states_tri   = _triton_state_fwd(B_t, x_t, dt_t, dA_cumsum_t,
                                         seq_idx=seq_idx_t)
        states_tri_j = jnp.array(states_tri.cpu().numpy())
        all_ok &= check("states vs Triton", states_pal, states_tri_j, atol=atol)

    print(f"  {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    return all_ok


def _bench_state_amortized(impl_fn, x, B, dt, dA, N=50, warmup=10, rep=50,
                           **kwargs):
    """
    Amortize JAX dispatch for chunk_state benchmarks.

    Wraps N calls in fori_loop; per-call time = total / N.
    Uses jax.lax.cond with an acc-dependent predicate to prevent XLA
    from hoisting loop-invariant pure-JAX computations.
    """
    sample = impl_fn(x, B, dt, dA, **kwargs)
    dummy = jax.tree.map(jnp.zeros_like, sample)

    @jax.jit
    def looped(x, B, dt, dA):
        def body(i, acc):
            states = jax.lax.cond(
                acc > -1e30,
                lambda: impl_fn(x, B, dt, dA, **kwargs),
                lambda: dummy,
            )
            return acc + states[0, 0, 0, 0, 0]  # prevent DCE
        return jax.lax.fori_loop(0, N, body, 0.0)

    jax.block_until_ready(looped(x, B, dt, dA))  # compile

    ms_total = _bench_fn(
        lambda: jax.block_until_ready(looped(x, B, dt, dA)),
        warmup, rep,
    )
    return ms_total / N


def benchmark_state(
    batch=2, seqlen=2048, nheads=64, headdim=64, dstate=64,
    ngroups=1, chunk_size=256,
    BM=64, BK=64, BN=64,
    warmup=25, rep=200,
    amortize_N=50,
):
    nchunks = math.ceil(seqlen / chunk_size)
    print(f"\n── [chunk_state_fwd] benchmark  "
          f"B={batch} L={seqlen} H={nheads} D={headdim} S={dstate} "
          f"G={ngroups} Q={chunk_size} ──")

    key = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x_j   = jax.random.normal(k1, (batch, seqlen, nheads, headdim))
    B_j   = jax.random.normal(k2, (batch, seqlen, ngroups, dstate))
    dt_j  = jax.random.uniform(k3, (batch, nheads, nchunks, chunk_size)) * 0.1 + 0.01
    dA_j  = jnp.cumsum(
        -jax.random.uniform(k4, (batch, nheads, nchunks, chunk_size)) * 0.1,
        axis=3,
    )

    # Bytes: read x + B + dt + dA_cumsum, write states  (all float32)
    bytes_io = (
        batch * seqlen * nheads * headdim * 4          # x
        + batch * seqlen * ngroups * dstate * 4        # B
        + batch * nheads * nchunks * chunk_size * 4    # dt
        + batch * nheads * nchunks * chunk_size * 4    # dA_cumsum
        + batch * nchunks * nheads * headdim * dstate * 4  # states out
    )

    # ── Naive JAX (amortized) ──
    ms_naive = _bench_state_amortized(
        _naive_chunk_state, x_j, B_j, dt_j, dA_j, N=amortize_N,
    )

    # ── Mosaic end-to-end (amortized) ──
    ms_e2e = _bench_state_amortized(
        lambda x, B, dt, dA, **kw: chunk_state_fwd_mosaic(
            x, B, dt, dA, BM=BM, BK=BK, BN=BN,
        ),
        x_j, B_j, dt_j, dA_j, N=amortize_N,
    )

    # ── Mosaic kernel-only (amortized via fori_loop) ──
    # Preprocess eagerly, then amortize just the kernel
    x_flat, B_pre, meta = chunk_state_preprocess(
        x_j, B_j, dt_j, dA_j, BM=BM, BK=BK, BN=BN,
    )
    jax.block_until_ready((x_flat, B_pre))

    kern_kwargs = dict(
        BM=BM, BK=meta['BK'], BN=BN,
        BCH=meta['BCH'], BCG=meta['BCG'],
        headdim=meta['headdim'], headdim_padded=meta['headdim_padded'],
        dstate=meta['dstate'], dstate_padded=meta['dstate_padded'],
        batch=meta['batch'], nchunks=meta['nchunks'],
        nheads=meta['nheads'], ngroups=meta['ngroups'],
        chunk_size=meta['chunk_size'],
    )

    @jax.jit
    def kern_looped(xf, bf):
        def body(i, acc):
            states = chunk_state_kernel_only(xf, bf, **kern_kwargs)
            return acc + states[0, 0, 0, 0, 0]
        return jax.lax.fori_loop(0, amortize_N, body, 0.0)

    jax.block_until_ready(kern_looped(x_flat, B_pre))  # compile
    ms_kern = _bench_fn(
        lambda: jax.block_until_ready(kern_looped(x_flat, B_pre)),
        warmup=10, rep=50,
    ) / amortize_N

    gbps_e2e   = bytes_io / (ms_e2e   * 1e-3) / 1e9
    gbps_naive = bytes_io / (ms_naive  * 1e-3) / 1e9
    gbps_kern  = bytes_io / (ms_kern   * 1e-3) / 1e9
    print(f"  Naive JAX (matmul) (N={amortize_N}): {ms_naive:.4f} ms   {gbps_naive:.1f} GB/s")
    print(f"  Mosaic end-to-end  (N={amortize_N}): {ms_e2e:.4f} ms   {gbps_e2e:.1f} GB/s")
    print(f"  Mosaic kernel-only (N={amortize_N}): {ms_kern:.4f} ms   {gbps_kern:.1f} GB/s")
    print(f"  Mosaic preprocess  (e2e - kern)  : {ms_e2e - ms_kern:.4f} ms")

    if _HAS_TRITON:
        import torch
        B_t   = _to_torch(B_j)
        x_t   = _to_torch(x_j)
        dt_t  = _to_torch(dt_j)
        dA_t  = _to_torch(dA_j)

        def _triton_fn():
            _triton_state_fwd(B_t, x_t, dt_t, dA_t)
            torch.cuda.synchronize()

        ms_tri  = _bench_fn(_triton_fn, warmup=warmup, rep=rep)
        gbps_tri = bytes_io / (ms_tri * 1e-3) / 1e9
        print(f"  Triton                    : {ms_tri:.4f} ms   {gbps_tri:.1f} GB/s")

    # ── Summary ratios (all dispatch-free) ──
    print(f"  ── Ratios (dispatch-free) ──")
    print(f"  Mosaic e2e   / Naive : {ms_e2e/ms_naive:.2f}x")
    print(f"  Mosaic kernel/ Naive : {ms_kern/ms_naive:.2f}x")
    if _HAS_TRITON:
        print(f"  Mosaic e2e   / Triton: {ms_e2e/ms_tri:.2f}x")
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

    # ── chunk_cumsum_fwd correctness ───────────────────────────────────────
    print("\n" + "═" * 70)
    print("CHUNK_CUMSUM_FWD CORRECTNESS")
    print("═" * 70)
    # Minimal correctness configs – fast smoke tests.
    cumsum_configs = [
        # (batch, seqlen, nheads, chunk_size, softplus, bias)
        (1, 128,  24,  64, True,  True),
        (1, 256,  64, 256, True,  True),
    ]
    for cfg in cumsum_configs:
        all_passed &= test_cumsum_correctness(*cfg)

    # ── chunk_state_fwd correctness ────────────────────────────────────────
    print("\n" + "═" * 70)
    print("CHUNK_STATE_FWD CORRECTNESS")
    print("═" * 70)
    # (batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size, BM, BK, BN)
    # Minimal correctness configs – just enough to verify each kernel shape.
    # Full-size runs belong in the benchmark section below.
    state_configs = [
        (1, 128,  8,  64,  64, 1,  64, 64, 64, 64),
        (1, 128, 16,  64,  64, 2, 128, 64, 64, 64),
        (1, 256, 64,  64,  64, 8, 256, 64, 64, 64),
        (1, 128,  4, 128,  64, 1,  64, 64, 64, 64),
        # Nemotron-H-56B kernel shape (reduced B/L for speed)
        (1, 256, 256, 64, 256, 8, 256, 64, 64, 64),
    ]
    for cfg in state_configs:
        all_passed &= test_state_correctness(*cfg)

    # ── chunk_state_fwd seq_idx correctness ───────────────────────────────
    print("\n" + "═" * 70)
    print("CHUNK_STATE_FWD SEQ_IDX CORRECTNESS")
    print("═" * 70)
    # Minimal seq_idx correctness cases
    state_seq_idx_configs = [
        (1, 128, 8, 64, 64, 1, 64, 64, 64, 64, {'seq_lengths': [[64, 64]]}),
        (1, 256, 16, 64, 64, 2, 128, 64, 64, 64, {'seq_lengths': [[128, 128]]}),
    ]
    for cfg in state_seq_idx_configs:
        batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size, BM, BK, BN, seq_cfg = cfg
        all_passed &= test_state_correctness(
            batch=batch, seqlen=seqlen, nheads=nheads, headdim=headdim,
            dstate=dstate, ngroups=ngroups, chunk_size=chunk_size,
            BM=BM, BK=BK, BN=BN, seq_idx_config=seq_cfg,
        )

    print(f"\n{'═' * 70}")
    print(f"Overall: {'ALL TESTS PASS ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print(f"{'═' * 70}")

    # ── benchmarks ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("BENCHMARKS")
    print("═" * 70)

    cumsum_bench_configs = [
        # (batch, seqlen, nheads, chunk_size, softplus)
        (1,   256,  24,  64, True),
        (2,   512,  24, 128, True),
        (2,  1024,  24, 256, True),
        (4,  2048,  64, 256, True),
        (8,  4096,  64, 256, True),
        (1,  2048, 128, 256, True),
    ]
    for cfg in cumsum_bench_configs:
        benchmark_cumsum(*cfg)

    # (batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size, BM, BK, BN)
    state_bench_configs = [
        (1,   256,  8,  64,  64, 1,  64, 64, 64, 64),
        (2,   512,  8,  64,  64, 1, 128, 64, 64, 64),
        (2,  1024,  8,  64,  64, 1, 256, 64, 64, 64),
        (4,  2048, 64,  64,  64, 8, 256, 64, 64, 64),
        (8,  4096, 64,  64,  64, 8, 256, 64, 64, 64),
        (2,  2048, 64, 128,  64, 8, 256, 64, 64, 64),
        (1,  2048, 64,  64, 128, 8, 256, 64, 64, 64),
        # Nemotron-H-56B
        (1,  2048, 256, 64, 256, 8, 256, 64, 64, 64),
    ]
    for cfg in state_bench_configs:
        benchmark_state(*cfg)
