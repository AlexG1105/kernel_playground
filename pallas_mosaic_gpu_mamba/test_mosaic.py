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
from pallas_mosaic_gpu_mamba.chunk_cumsum_fwd import chunk_cumsum_fwd_mosaic
from pallas_mosaic_gpu_mamba.chunk_state_fwd import chunk_state_fwd_mosaic

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


def benchmark_cumsum(
    batch=2, seqlen=2048, nheads=64, chunk_size=256,
    dt_softplus=True, warmup=25, rep=200,
):
    print(f"\n── [chunk_cumsum_fwd] benchmark  "
          f"B={batch} L={seqlen} H={nheads} Q={chunk_size} ──")

    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    dt_j   = jax.random.normal(k1, (batch, seqlen, nheads))
    A_j    = -jax.random.uniform(k2, (nheads,)) * 0.1
    bias_j = jax.random.normal(k3, (nheads,))

    fn = jax.jit(lambda dt, A, b: chunk_cumsum_fwd_mosaic(
        dt, A, chunk_size, dt_bias=b, dt_softplus=dt_softplus,
    ))
    out = fn(dt_j, A_j, bias_j)
    jax.block_until_ready(out)

    if _HAS_TRITON:
        from triton.testing import do_bench
        ms = do_bench(
            lambda: jax.block_until_ready(fn(dt_j, A_j, bias_j)),
            warmup=warmup, rep=rep,
        )
    else:
        import time
        times = []
        for _ in range(warmup + rep):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(dt_j, A_j, bias_j))
            times.append(time.perf_counter() - t0)
        ms = float(np.median(times[warmup:])) * 1e3

    nchunks = math.ceil(seqlen / chunk_size)
    bytes_io = (
        batch * seqlen * nheads * 4
        + nheads * 4
        + nheads * 4
        + 2 * batch * nheads * nchunks * chunk_size * 4
    )
    gbps = bytes_io / (ms * 1e-3) / 1e9
    print(f"  Mosaic GPU : {ms:.3f} ms   {gbps:.1f} GB/s")

    if _HAS_TRITON:
        import torch
        dt_t   = _to_torch(dt_j)
        A_t    = _to_torch(A_j)
        bias_t = _to_torch(bias_j)
        from triton.testing import do_bench

        def _triton_fn():
            _triton_cumsum_fwd(dt_t, A_t, chunk_size, dt_bias=bias_t,
                               dt_softplus=dt_softplus, dt_limit=(0.0, float("inf")))
            torch.cuda.synchronize()

        ms_tri = do_bench(_triton_fn, warmup=warmup, rep=rep)
        gbps_tri = bytes_io / (ms_tri * 1e-3) / 1e9
        print(f"  Triton     : {ms_tri:.3f} ms   {gbps_tri:.1f} GB/s")
        print(f"  Ratio (Mosaic/Triton): {ms/ms_tri:.2f}x")


# ===========================================================================
# chunk_state_fwd
# ===========================================================================

def _naive_chunk_state(x, B, dt, dA_cumsum):
    """
    Naive JAX reference for chunk_state_fwd.

    x         : (batch, seqlen, nheads, headdim)
    B         : (batch, seqlen, ngroups, dstate)
    dt        : (batch, nheads, nchunks, chunk_size)   (post-processed)
    dA_cumsum : (batch, nheads, nchunks, chunk_size)

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

    # scale: (batch, nheads, nchunks, chunk_size)
    dA_cs_last = dA_cumsum[:, :, :, -1:]
    scale = jnp.exp(jnp.minimum(dA_cs_last - dA_cumsum, 0.0)) * dt

    # x → (batch, nchunks, nheads, headdim, chunk_size)
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim).transpose(0, 1, 3, 4, 2)
    # B → (batch, nchunks, ngroups, chunk_size, dstate)
    B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).transpose(0, 1, 3, 2, 4)

    # Expand B along heads axis: (batch, nchunks, nheads, chunk_size, dstate)
    B_exp = jnp.repeat(B, ratio, axis=2)

    # scale → (batch, nchunks, nheads, chunk_size)
    scale_t = scale.transpose(0, 2, 1, 3)

    # B_scaled: (batch, nchunks, nheads, chunk_size, dstate)
    B_scaled = B_exp * scale_t[:, :, :, :, None]

    # states = x @ B_scaled → (batch, nchunks, nheads, headdim, dstate)
    states = jnp.matmul(x, B_scaled)
    return states


def test_state_correctness(
    batch=2, seqlen=512, nheads=8, headdim=64, dstate=64,
    ngroups=1, chunk_size=64,
    BM=64, BK=64, BN=64,
    atol=1e-2,
):
    """
    Correctness test for chunk_state_fwd_mosaic.

    dt and dA_cumsum are constructed directly in (batch, nheads, nchunks, chunk_size)
    format (i.e. they represent already-processed values, as produced by
    chunk_cumsum_fwd).
    """
    nchunks = math.ceil(seqlen / chunk_size)
    print(f"\n── [chunk_state_fwd] correctness  "
          f"B={batch} L={seqlen} H={nheads} D={headdim} S={dstate} "
          f"G={ngroups} Q={chunk_size} BM={BM} BK={BK} BN={BN} ──")

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x_jax  = jax.random.normal(k1, (batch, seqlen, nheads, headdim))
    B_jax  = jax.random.normal(k2, (batch, seqlen, ngroups, dstate))
    # dt in (batch, nheads, nchunks, chunk_size), positive (post-softplus)
    dt_jax = jax.random.uniform(k3, (batch, nheads, nchunks, chunk_size)) * 0.1 + 0.01
    # dA_cumsum: negative, monotonically decreasing in last dim (approx)
    dA_raw = -jax.random.uniform(k4, (batch, nheads, nchunks, chunk_size)) * 0.1
    dA_cumsum_jax = jnp.cumsum(dA_raw, axis=3)

    states_pal = chunk_state_fwd_mosaic(
        x_jax, B_jax, dt_jax, dA_cumsum_jax,
        BM=BM, BK=BK, BN=BN,
    )
    jax.block_until_ready(states_pal)

    print(f"  states shape : {tuple(states_pal.shape)}")

    states_ref = _naive_chunk_state(x_jax, B_jax, dt_jax, dA_cumsum_jax)

    all_ok = True
    all_ok &= check("states vs naive", states_pal, states_ref, atol=atol)

    if _HAS_TRITON:
        # Triton reference: _chunk_state_fwd(B, x, dt, dA_cumsum)
        # Its dt argument is (batch, nheads, nchunks, chunk_size).
        # Its x argument is  (batch, seqlen, nheads, headdim).
        B_t          = _to_torch(B_jax)
        x_t          = _to_torch(x_jax)
        dt_t         = _to_torch(dt_jax)
        dA_cumsum_t  = _to_torch(dA_cumsum_jax)
        states_tri   = _triton_state_fwd(B_t, x_t, dt_t, dA_cumsum_t)
        states_tri_j = jnp.array(states_tri.cpu().numpy())
        all_ok &= check("states vs Triton", states_pal, states_tri_j, atol=atol)

    print(f"  {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    return all_ok


def benchmark_state(
    batch=2, seqlen=2048, nheads=64, headdim=64, dstate=64,
    ngroups=1, chunk_size=256,
    BM=64, BK=64, BN=64,
    warmup=25, rep=200,
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

    fn = jax.jit(lambda x, B, dt, dA: chunk_state_fwd_mosaic(
        x, B, dt, dA, BM=BM, BK=BK, BN=BN,
    ))
    out = fn(x_j, B_j, dt_j, dA_j)
    jax.block_until_ready(out)

    if _HAS_TRITON:
        from triton.testing import do_bench
        ms = do_bench(
            lambda: jax.block_until_ready(fn(x_j, B_j, dt_j, dA_j)),
            warmup=warmup, rep=rep,
        )
    else:
        import time
        times = []
        for _ in range(warmup + rep):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(x_j, B_j, dt_j, dA_j))
            times.append(time.perf_counter() - t0)
        ms = float(np.median(times[warmup:])) * 1e3

    # Bytes: read x + B + dt + dA_cumsum, write states  (all float32)
    bytes_io = (
        batch * seqlen * nheads * headdim * 4          # x
        + batch * seqlen * ngroups * dstate * 4        # B
        + batch * nheads * nchunks * chunk_size * 4    # dt
        + batch * nheads * nchunks * chunk_size * 4    # dA_cumsum
        + batch * nchunks * nheads * headdim * dstate * 4  # states out
    )
    gbps = bytes_io / (ms * 1e-3) / 1e9
    print(f"  Mosaic GPU : {ms:.3f} ms   {gbps:.1f} GB/s")

    if _HAS_TRITON:
        import torch
        from triton.testing import do_bench
        B_t   = _to_torch(B_j)
        x_t   = _to_torch(x_j)
        dt_t  = _to_torch(dt_j)
        dA_t  = _to_torch(dA_j)

        def _triton_fn():
            _triton_state_fwd(B_t, x_t, dt_t, dA_t)
            torch.cuda.synchronize()

        ms_tri  = do_bench(_triton_fn, warmup=warmup, rep=rep)
        gbps_tri = bytes_io / (ms_tri * 1e-3) / 1e9
        print(f"  Triton     : {ms_tri:.3f} ms   {gbps_tri:.1f} GB/s")
        print(f"  Ratio (Mosaic/Triton): {ms/ms_tri:.2f}x")


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
    cumsum_configs = [
        # (batch, seqlen, nheads, chunk_size, softplus, bias)
        (1,  256,  24,  64,  True,  True),
        (2,  512,  24, 128,  True,  True),
        (2, 1024,  24, 256,  True,  True),
        (4, 2048,  64, 256, False, False),
        (2,  512,  24, 256,  True, False),
        (1,  256,  64,  64,  True,  True),
    ]
    for cfg in cumsum_configs:
        all_passed &= test_cumsum_correctness(*cfg)

    # ── chunk_state_fwd correctness ────────────────────────────────────────
    print("\n" + "═" * 70)
    print("CHUNK_STATE_FWD CORRECTNESS")
    print("═" * 70)
    # (batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size, BM, BK, BN)
    state_configs = [
        # Standard Mamba2 configs
        (1,  256,  8,  64,  64,  1,  64, 64, 64, 64),
        (2,  512,  8,  64,  64,  1, 128, 64, 64, 64),
        (2, 1024,  8,  64,  64,  1, 256, 64, 64, 64),
        # Multi-head, grouped (ngroups < nheads)
        (2,  512, 16,  64,  64,  2, 128, 64, 64, 64),
        (2,  512, 16,  64, 128,  2, 128, 64, 64, 64),
        # Larger headdim
        (1,  256,  4, 128,  64,  1,  64, 64, 64, 64),
        # Nemotron-style: many heads, many groups
        (1,  256, 64,  64,  64,  8,  64, 64, 64, 64),
        (2,  512, 64,  64,  64,  8, 128, 64, 64, 64),
    ]
    for cfg in state_configs:
        all_passed &= test_state_correctness(*cfg)

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
    ]
    for cfg in state_bench_configs:
        benchmark_state(*cfg)
