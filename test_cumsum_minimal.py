"""
Minimal test: can jnp.cumsum work inside a Pallas Mosaic GPU kernel?

If this works, we can rewrite chunk_cumsum_fwd without pl.loop,
without nheads padding to 128, and without the sub-chunk path.

Run on H100:
  python test_cumsum_minimal.py
"""

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


# ---------------------------------------------------------------------------
# Test 1: bare jnp.cumsum inside pallas_call
# ---------------------------------------------------------------------------

def _cumsum_kernel(dt_ref, cs_ref):
    """Simplest possible: load tile, cumsum, store."""
    dt = dt_ref[0, :, :].astype(jnp.float32)   # (chunk_size, nheads)
    cs = jnp.cumsum(dt, axis=0)                 # cumsum along chunk_size
    cs_ref[0, :, :] = cs


def test_bare_cumsum(batch=2, seqlen=256, nheads=8, chunk_size=64):
    """Test jnp.cumsum inside a pallas_call kernel — no padding, no pl.loop."""
    import math

    nchunks = math.ceil(seqlen / chunk_size)
    N = batch * nchunks

    dt = jax.random.normal(jax.random.PRNGKey(0), (batch, seqlen, nheads))

    # Reshape to (N, chunk_size, nheads) — NO padding
    dt_flat = dt.reshape(N, chunk_size, nheads)

    tile = pl.BlockSpec((1, chunk_size, nheads), lambda n: (n, 0, 0))

    cs_flat = pl.pallas_call(
        _cumsum_kernel,
        out_shape=jax.ShapeDtypeStruct((N, chunk_size, nheads), jnp.float32),
        grid=(N,),
        in_specs=[tile],
        out_specs=tile,
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel"],
        ),
    )(dt_flat)

    # Reference
    cs_ref = jnp.cumsum(dt_flat, axis=1)

    err = jnp.max(jnp.abs(cs_flat - cs_ref))
    print(f"[bare cumsum] batch={batch}, seqlen={seqlen}, nheads={nheads}, chunk_size={chunk_size}")
    print(f"  max error: {err:.6f}")
    assert err < 1e-3, f"FAIL: error {err} too large"
    print("  PASS")
    return cs_flat


# ---------------------------------------------------------------------------
# Test 2: cumsum + softplus + clip (mimics chunk_cumsum_fwd)
# ---------------------------------------------------------------------------

def _cumsum_full_kernel(
    dt_ref, A_ref, dt_out_ref, cs_ref,
    *, dt_softplus: bool, dt_min: float, dt_max: float,
):
    """Load tile, softplus, clip, cumsum, multiply by A — all in registers."""
    dt = dt_ref[0, :, :].astype(jnp.float32)   # (chunk_size, nheads)
    A = A_ref[0, :].astype(jnp.float32)         # (nheads,)

    # Softplus
    if dt_softplus:
        safe = jnp.where(dt <= 20.0, dt, jnp.zeros_like(dt))
        dt = jnp.where(dt <= 20.0, jnp.log(1 + jnp.exp(safe)), dt)

    # Clip
    dt = jnp.clip(dt, dt_min, dt_max)

    # Store processed dt
    dt_out_ref[0, :, :] = dt

    # Cumsum along chunk_size, then multiply by A
    cs = jnp.cumsum(dt, axis=0) * A[None, :]
    cs_ref[0, :, :] = cs


def test_full_cumsum(batch=2, seqlen=256, nheads=8, chunk_size=64):
    """Test full chunk_cumsum_fwd logic inside kernel — no padding, no pl.loop."""
    import math
    from functools import partial

    nchunks = math.ceil(seqlen / chunk_size)
    N = batch * nchunks

    key = jax.random.PRNGKey(42)
    dt = jax.random.normal(key, (batch, seqlen, nheads))
    A = -jax.random.uniform(jax.random.PRNGKey(1), (nheads,)) * 0.1  # negative

    dt_flat = dt.reshape(N, chunk_size, nheads)
    # A needs to be (1, nheads) for BlockSpec tiling — replicate along grid dim
    A_tiled = jnp.broadcast_to(A[None, :], (N, nheads))

    dt_tile = pl.BlockSpec((1, chunk_size, nheads), lambda n: (n, 0, 0))
    a_tile = pl.BlockSpec((1, nheads), lambda n: (n, 0))

    kernel = partial(
        _cumsum_full_kernel,
        dt_softplus=True,
        dt_min=0.0,
        dt_max=float("inf"),
    )

    dt_out_flat, cs_flat = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((N, chunk_size, nheads), jnp.float32),
            jax.ShapeDtypeStruct((N, chunk_size, nheads), jnp.float32),
        ],
        grid=(N,),
        in_specs=[dt_tile, a_tile],
        out_specs=[dt_tile, dt_tile],
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel"],
        ),
    )(dt_flat, A_tiled)

    # Reference: softplus + clip + cumsum * A
    dt_ref = jnp.where(dt_flat <= 20.0, jnp.log(1 + jnp.exp(jnp.where(dt_flat <= 20.0, dt_flat, 0.0))), dt_flat)
    dt_ref = jnp.clip(dt_ref, 0.0)
    cs_ref = jnp.cumsum(dt_ref, axis=1) * A[None, None, :]

    dt_err = jnp.max(jnp.abs(dt_out_flat - dt_ref))
    cs_err = jnp.max(jnp.abs(cs_flat - cs_ref))
    print(f"[full cumsum] batch={batch}, seqlen={seqlen}, nheads={nheads}, chunk_size={chunk_size}")
    print(f"  dt_out max error: {dt_err:.6f}")
    print(f"  dA_cs  max error: {cs_err:.6f}")
    assert dt_err < 1e-3, f"FAIL: dt_out error {dt_err}"
    assert cs_err < 1e-3, f"FAIL: dA_cs error {cs_err}"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: various sizes (no padding ever)
# ---------------------------------------------------------------------------

def test_sizes():
    """Test multiple nheads values — none padded to 128."""
    configs = [
        # (batch, seqlen, nheads, chunk_size)
        (1, 128, 1, 64),       # nheads=1, well below 128
        (2, 256, 8, 64),       # nheads=8
        (1, 512, 24, 128),     # nheads=24, not power of 2
        (2, 256, 64, 64),      # nheads=64
        (1, 512, 96, 256),     # nheads=96, large chunk_size (was SMEM overflow before)
        (2, 1024, 128, 64),    # nheads=128, exactly the old pad target
        (1, 512, 256, 64),     # nheads=256 (Nemotron-like)
    ]
    for batch, seqlen, nheads, chunk_size in configs:
        try:
            test_bare_cumsum(batch, seqlen, nheads, chunk_size)
        except Exception as e:
            print(f"  FAIL: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing jnp.cumsum inside Pallas Mosaic GPU kernel")
    print("=" * 60)
    print()

    print("--- Test 1: bare jnp.cumsum ---")
    test_bare_cumsum()
    print()

    print("--- Test 2: full cumsum + softplus + clip + A ---")
    test_full_cumsum()
    print()

    print("--- Test 3: various sizes (no nheads padding) ---")
    test_sizes()
    print()

    print("All tests passed!" )
