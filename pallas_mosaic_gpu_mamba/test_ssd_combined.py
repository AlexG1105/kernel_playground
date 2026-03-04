"""
test_ssd_combined.py

Correctness and benchmark tests for ssd_combined_pallas (Mamba2 SSD forward).

Compares:
  1. Pallas Mosaic GPU  vs  Triton (mamba_ssm)   — correctness + speed
  2. Pallas Mosaic GPU  vs  Naive JAX (mamba2-jax) — correctness + speed

Run on H100/H200:
  python test_ssd_combined.py
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
# Make imports work
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pallas_mosaic_gpu_mamba.ssd_combined_pallas import (
    mamba_chunk_scan_combined_fwd,
)

# ---------------------------------------------------------------------------
# Optional Triton reference
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
    from mamba_ssm.ops.triton.ssd_combined import (
        _mamba_chunk_scan_combined_fwd as _triton_combined_fwd,
    )
    _HAS_TRITON = True
except Exception as e:
    print(f"[info] Triton reference unavailable ({e}); skipping Triton comparison.")

# ---------------------------------------------------------------------------
# Naive JAX reference (from mamba2-jax)
# ---------------------------------------------------------------------------
_HAS_NAIVE_JAX = False
try:
    MAMBA2_JAX_ROOT = os.path.expanduser("/workspace/mamba2-jax")
    sys.path.insert(0, MAMBA2_JAX_ROOT)
    from mamba2_jax.modeling import ssd_forward as _naive_ssd_forward
    _HAS_NAIVE_JAX = True
except Exception as e:
    print(f"[info] mamba2-jax reference unavailable ({e}); skipping naive JAX comparison.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_torch(x):
    return torch.tensor(np.array(x), device="cuda", dtype=torch.float32)

def _to_torch_int(x):
    return torch.tensor(np.array(x), device="cuda", dtype=torch.int32)

def check(name, jax_arr, ref_arr, atol=1e-2):
    d = float(np.abs(np.array(jax_arr) - np.array(ref_arr)).max())
    ok = "✓" if d < atol else "✗"
    print(f"  {ok}  {name:45s}  max|diff|={d:.2e}  (atol={atol:.0e})")
    return d < atol


def _bench_fn(fn, warmup, rep):
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


def _bench_amortized(impl_fn, args, N=50, warmup=10, rep=50):
    """Amortized benchmark (dispatch overhead removed via fori_loop + lax.cond).

    Always uses wall-clock timing (not do_bench CUDA events) since JAX
    runs on its own CUDA stream.  block_until_ready ensures we measure
    actual GPU time, and dividing by N removes the one-time sync cost.
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

    # Warmup
    for _ in range(warmup):
        jax.block_until_ready(looped(*args))

    # Timed reps
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        jax.block_until_ready(looped(*args))
        times.append(time.perf_counter() - t0)
    ms_total = float(np.median(times)) * 1e3
    return ms_total / N


# ===========================================================================
# Correctness test
# ===========================================================================

def test_correctness(
    batch=2,
    seqlen=2048,
    nheads=8,
    headdim=64,
    dstate=64,
    ngroups=1,
    chunk_size=256,
    has_D=True,
    has_z=True,
    has_initial_states=False,
    dt_softplus=True,
    atol=2.0,
):
    """Compare Pallas combined forward vs Triton and naive JAX references."""
    print(f"\n── [ssd_combined_fwd] correctness  "
          f"B={batch} L={seqlen} H={nheads} P={headdim} N={dstate} G={ngroups} "
          f"Q={chunk_size} D?={has_D} z?={has_z} init?={has_initial_states} ──")

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    x_j = jax.random.normal(keys[0], (batch, seqlen, nheads, headdim)) * 0.1
    dt_j = jax.random.normal(keys[1], (batch, seqlen, nheads)) * 0.5
    A_j = -jax.random.uniform(keys[2], (nheads,), minval=0.01, maxval=0.1)
    B_j = jax.random.normal(keys[3], (batch, seqlen, ngroups, dstate)) * 0.1
    C_j = jax.random.normal(keys[4], (batch, seqlen, ngroups, dstate)) * 0.1
    dt_bias_j = jax.random.normal(keys[5], (nheads,)) * 0.1

    D_j = None
    if has_D:
        D_j = jax.random.normal(keys[6], (nheads,)) * 0.1

    z_j = None
    if has_z:
        z_j = jax.random.normal(keys[7], (batch, seqlen, nheads, headdim)) * 0.5

    init_j = None
    if has_initial_states:
        init_j = jax.random.normal(keys[8], (batch, nheads, headdim, dstate)) * 0.01

    dt_limit = (0.0, float("inf"))

    # --- Pallas Mosaic GPU ---
    pallas_out, pallas_final = mamba_chunk_scan_combined_fwd(
        x_j, dt_j, A_j, B_j, C_j, chunk_size,
        D=D_j, z=z_j, dt_bias=dt_bias_j,
        initial_states=init_j,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        return_final_states=True,
    )
    jax.block_until_ready(pallas_out)
    print(f"  Pallas out shape: {tuple(pallas_out.shape)}")

    all_ok = True

    # --- Triton comparison ---
    if _HAS_TRITON:
        x_t = _to_torch(x_j)
        dt_t = _to_torch(dt_j)
        A_t = _to_torch(A_j)
        B_t = _to_torch(B_j)
        C_t = _to_torch(C_j)
        dt_bias_t = _to_torch(dt_bias_j)
        D_t = _to_torch(D_j) if D_j is not None else None
        z_t = _to_torch(z_j) if z_j is not None else None
        init_t = _to_torch(init_j) if init_j is not None else None

        triton_result = _triton_combined_fwd(
            x_t, dt_t, A_t, B_t, C_t, chunk_size,
            D=D_t, z=z_t, dt_bias=dt_bias_t,
            initial_states=init_t,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )
        triton_out = jnp.array(triton_result[0].cpu().numpy())
        triton_final = jnp.array(triton_result[5].cpu().numpy())

        all_ok &= check("out vs Triton", pallas_out, triton_out, atol=atol)
        all_ok &= check("final_states vs Triton", pallas_final, triton_final, atol=atol)

    # --- Naive JAX comparison (mamba2-jax) ---
    # mamba2-jax has no ngroups concept — B/C always have full nheads dim.
    # When ngroups < nheads, broadcast B/C to nheads before calling.
    if _HAS_NAIVE_JAX:
        heads_per_group = nheads // ngroups
        B_naive = jnp.repeat(B_j, heads_per_group, axis=2) if ngroups != nheads else B_j
        C_naive = jnp.repeat(C_j, heads_per_group, axis=2) if ngroups != nheads else C_j
        naive_out, naive_final = _naive_ssd_forward(
            x_j, dt_j, A_j, B_naive, C_naive, chunk_size,
            D=D_j if D_j is not None else jnp.zeros(nheads),
            dt_bias=dt_bias_j,
            dt_min=dt_limit[0],
            dt_max=dt_limit[1],
            initial_states=init_j[:, None, :, :, :] if init_j is not None else None,
            return_final_states=True,
        )
        # naive z gating is not in mamba2-jax, so compare without z if possible
        if z_j is None:
            all_ok &= check("out vs naive JAX", pallas_out, naive_out, atol=atol)
        if naive_final is not None:
            all_ok &= check("final_states vs naive JAX", pallas_final, naive_final, atol=atol)

    print(f"  {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    return all_ok


# ===========================================================================
# Benchmark
# ===========================================================================

def benchmark_combined(
    batch=2,
    seqlen=2048,
    nheads=64,
    headdim=64,
    dstate=64,
    ngroups=1,
    chunk_size=256,
    has_D=True,
    has_z=True,
    dt_softplus=True,
    warmup=10,
    rep=50,
):
    """Benchmark ssd_combined: Pallas vs Triton vs naive JAX."""
    print(f"\n── [ssd_combined_fwd] benchmark  "
          f"B={batch} L={seqlen} H={nheads} P={headdim} N={dstate} G={ngroups} "
          f"Q={chunk_size} D?={has_D} z?={has_z} ──")

    key = jax.random.PRNGKey(7)
    keys = jax.random.split(key, 8)

    x_j = jax.random.normal(keys[0], (batch, seqlen, nheads, headdim)) * 0.1
    dt_j = jax.random.normal(keys[1], (batch, seqlen, nheads)) * 0.5
    A_j = -jax.random.uniform(keys[2], (nheads,), minval=0.01, maxval=0.1)
    B_j = jax.random.normal(keys[3], (batch, seqlen, ngroups, dstate)) * 0.1
    C_j = jax.random.normal(keys[4], (batch, seqlen, ngroups, dstate)) * 0.1
    dt_bias_j = jax.random.normal(keys[5], (nheads,)) * 0.1
    D_j = jax.random.normal(keys[6], (nheads,)) * 0.1 if has_D else None
    z_j = jax.random.normal(keys[7], (batch, seqlen, nheads, headdim)) * 0.5 if has_z else None
    dt_limit = (0.0, float("inf"))

    # --- Pallas (amortized) ---
    pallas_fn = partial(
        mamba_chunk_scan_combined_fwd,
        chunk_size=chunk_size,
        D=D_j, z=z_j, dt_bias=dt_bias_j,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        return_final_states=True,
    )
    pallas_args = (x_j, dt_j, A_j, B_j, C_j)
    ms_pallas = _bench_amortized(pallas_fn, pallas_args, warmup=warmup, rep=rep)
    print(f"  Pallas Mosaic    : {ms_pallas:.3f} ms")

    # --- Naive JAX (amortized) ---
    # mamba2-jax has no ngroups — broadcast B/C to nheads for fair comparison.
    if _HAS_NAIVE_JAX:
        heads_per_group = nheads // ngroups
        B_naive = jnp.repeat(B_j, heads_per_group, axis=2) if ngroups != nheads else B_j
        C_naive = jnp.repeat(C_j, heads_per_group, axis=2) if ngroups != nheads else C_j
        naive_fn = partial(
            _naive_ssd_forward,
            chunk_size=chunk_size,
            D=D_j if D_j is not None else jnp.zeros(nheads),
            dt_bias=dt_bias_j,
            dt_min=dt_limit[0],
            dt_max=dt_limit[1],
            return_final_states=True,
        )
        naive_args = (x_j, dt_j, A_j, B_naive, C_naive)
        ms_naive = _bench_amortized(naive_fn, naive_args, warmup=warmup, rep=rep)
        print(f"  Naive JAX        : {ms_naive:.3f} ms")
    else:
        ms_naive = None

    # --- Triton ---
    if _HAS_TRITON:
        import torch
        from triton.testing import do_bench
        x_t = _to_torch(x_j)
        dt_t = _to_torch(dt_j)
        A_t = _to_torch(A_j)
        B_t = _to_torch(B_j)
        C_t = _to_torch(C_j)
        dt_bias_t = _to_torch(dt_bias_j)
        D_t = _to_torch(D_j) if D_j is not None else None
        z_t = _to_torch(z_j) if z_j is not None else None

        def _triton_fn():
            _triton_combined_fwd(
                x_t, dt_t, A_t, B_t, C_t, chunk_size,
                D=D_t, z=z_t, dt_bias=dt_bias_t,
                dt_softplus=dt_softplus, dt_limit=dt_limit,
            )

        ms_triton = do_bench(_triton_fn, warmup=warmup, rep=rep)
        print(f"  Triton           : {ms_triton:.3f} ms")

    # --- Ratios ---
    print(f"  ── Ratios ──")
    if _HAS_TRITON:
        print(f"  Pallas / Triton  : {ms_pallas/ms_triton:.2f}x")
    if ms_naive is not None:
        print(f"  Pallas / Naive   : {ms_pallas/ms_naive:.2f}x")
    if ms_naive is not None and _HAS_TRITON:
        print(f"  Naive  / Triton  : {ms_naive/ms_triton:.2f}x")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ssd_combined_fwd — Correctness Tests")
    print("=" * 70)

    # Basic: ngroups=1, with D+z
    test_correctness(
        batch=2, seqlen=512, nheads=8, headdim=64, dstate=64,
        ngroups=1, chunk_size=64,
        has_D=True, has_z=True, has_initial_states=False,
    )

    # Nemotron-style: ngroups=nheads, with initial states
    test_correctness(
        batch=2, seqlen=512, nheads=8, headdim=64, dstate=64,
        ngroups=8, chunk_size=64,
        has_D=True, has_z=False, has_initial_states=True,
    )

    print("\n" + "=" * 70)
    print("ssd_combined_fwd — Benchmarks")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Configs with ngroups==nheads (enables all three: Pallas, Triton, naive JAX)
    # -------------------------------------------------------------------

    # Small: H=8
    benchmark_combined(
        batch=2, seqlen=2048, nheads=8, headdim=64, dstate=64,
        ngroups=8, chunk_size=256,
    )

    # Medium: H=32
    benchmark_combined(
        batch=2, seqlen=2048, nheads=32, headdim=64, dstate=64,
        ngroups=32, chunk_size=256,
    )

    # Large: H=64
    benchmark_combined(
        batch=2, seqlen=2048, nheads=64, headdim=64, dstate=64,
        ngroups=64, chunk_size=256,
    )

    # Longer sequence: L=4096
    benchmark_combined(
        batch=2, seqlen=4096, nheads=64, headdim=64, dstate=64,
        ngroups=64, chunk_size=256,
    )

    # Larger batch: B=8
    benchmark_combined(
        batch=8, seqlen=2048, nheads=64, headdim=64, dstate=64,
        ngroups=64, chunk_size=256,
    )

    # headdim=128
    benchmark_combined(
        batch=2, seqlen=2048, nheads=64, headdim=128, dstate=64,
        ngroups=64, chunk_size=256,
    )

    # dstate=128
    benchmark_combined(
        batch=2, seqlen=2048, nheads=64, headdim=64, dstate=128,
        ngroups=64, chunk_size=256,
    )

    # -------------------------------------------------------------------
    # Configs with ngroups < nheads (Pallas + Triton only)
    # -------------------------------------------------------------------

    # Standard: G=1
    benchmark_combined(
        batch=2, seqlen=2048, nheads=64, headdim=64, dstate=64,
        ngroups=1, chunk_size=256,
    )

    # G=8
    benchmark_combined(
        batch=2, seqlen=2048, nheads=64, headdim=64, dstate=64,
        ngroups=8, chunk_size=256,
    )

    # Large seq + G=1
    benchmark_combined(
        batch=2, seqlen=8192, nheads=64, headdim=64, dstate=64,
        ngroups=1, chunk_size=256,
    )

    print("\nDone.")
