# Mosaic GPU SMEM Overflow Fix: chunk_cumsum_fwd

## The Problem

When running `chunk_cumsum_fwd_mosaic` with `chunk_size=256` and `nheads=24`, the kernel crashes:

```
RESOURCE_EXHAUSTED: smem_bytes=394248 > max_smem_bytes=232448
```

### Root Cause

The Pallas Mosaic GPU kernel stages input/output tiles into shared memory (SMEM) via TMA before the kernel body executes. For `chunk_cumsum_fwd`, there are 3 tiles (1 input `dt`, 2 outputs `dt_out` and `cs`), each of shape `(1, chunk_size, nheads_padded)`:

```
nheads_padded = ceil(nheads / 128) * 128   # warpgroup constraint
              = ceil(24 / 128) * 128 = 128

SMEM per tile = 1 * chunk_size * nheads_padded * 4 bytes
              = 1 * 256 * 128 * 4 = 128 KB

Total SMEM    = 3 tiles * 128 KB = 384 KB
```

The H100 per-CTA SMEM limit is **~228 KB** (232,448 bytes). 384 KB far exceeds this.

Configs that overflow:
- `chunk_size=256, nheads_padded=128` → 384 KB (any nheads 1-128)

Configs that fit:
- `chunk_size=64, nheads_padded=128` → 96 KB
- `chunk_size=128, nheads_padded=128` → 192 KB

### Why nheads gets padded to 128

Mosaic GPU's H100 warpgroup has 128 threads. Any vector accessed via `load_strided` inside `pl.loop` must have at least 128 elements. Since `nheads` is the innermost dimension of the tile (for contiguous memory access), it must be padded:

```python
nheads_padded = math.ceil(nheads / 128) * 128  # minimum 128
```

This means even `nheads=1` results in `nheads_padded=128`.

## Approaches Tried and Why They Failed

### Attempt 1: Sequential Grid with scratch_shapes

The first approach used a 2D grid with a sequential dimension to iterate over sub-chunks:

```python
grid = (batch * nchunks, chunk_size // BQ)
dimension_semantics = ["parallel", "sequential"]
```

With `scratch_shapes` for the accumulator (persistent SMEM across sequential iterations) and `pl.when(pl.program_id(1) == 0)` for conditional initialization.

**Result:** NaN values and incorrect output.

**Why it failed:** We discovered a **Mosaic GPU sequential-grid output TMA bug**. When the sequential dimension has more than 1 element, output tiles are silently dropped or corrupted for iterations after the first. This was confirmed through systematic minimal tests:

- `BQ=1` (1 element per sequential step): works correctly
- `BQ=2, 4, 8, ...` (multiple elements): second iteration's outputs are zeros

This appears to be a bug in how `emit_pipeline` (the internal mechanism Mosaic GPU uses for sequential dimensions) handles output TMA commits. Input tiles load correctly, scratch memory persists correctly, but **output tile writes are silently lost** for sequential iterations beyond the first when tile sizes > 1.

### Attempt 2: pl.loop inside Sequential Grid (no output tiles)

Tried keeping outputs in scratch_shapes and writing them manually — but this hits the "Only SMEM <-> GMEM copies supported" limitation for large arrays that don't fit in scratch.

## The Fix: jax.lax.scan over Sub-Chunk Kernels

The working solution avoids the sequential grid entirely. Instead:

1. **Detect overflow at trace time** using `_tiles_smem()` and `_EFFECTIVE_MAX_SMEM`
2. **Choose a sub-chunk size** `BQ` that fits in SMEM via `_choose_BQ()`
3. **Reshape** the input into `(num_sub, n_chunks_total, BQ, nheads_padded)`
4. **Use `jax.lax.scan`** to iterate over sub-chunks, where each iteration calls a 1D-parallel `pallas_call` with:
   - Input: one sub-chunk of `dt` + accumulator from previous iteration
   - Output: sub-chunk of `dt_out` + sub-chunk of `cs` + updated accumulator
5. **Reassemble** outputs by transposing and reshaping back

### Why This Works

- Each `pallas_call` is a simple 1D-parallel kernel — no sequential dimensions, no `emit_pipeline`, no output TMA bug
- SMEM per kernel: `3 * BQ * nheads_padded * 4 + nheads_padded * 4` (for BQ=128: ~193 KB, fits)
- The accumulator (prefix-sum carry state) flows between iterations through GMEM via `lax.scan`'s carry mechanism
- XLA's compiler handles the scan loop efficiently

### Key Code Structure

```python
# In the wrapper:
if _tiles_smem(chunk_size, nheads_padded) <= _EFFECTIVE_MAX_SMEM:
    # Simple path: full tile fits, use single pallas_call
    ...
else:
    # Sub-chunk path: lax.scan over sub-chunks
    BQ = _choose_BQ(chunk_size, nheads_padded)
    num_sub = chunk_size // BQ

    def _scan_body(carry, dt_sub):
        # carry = accumulator: (n_chunks_total, nheads_padded)
        dt_out_sub, cs_sub, acc_out = pl.pallas_call(
            kernel,
            ...
            grid=(n_chunks_total,),              # 1D parallel only
            in_specs=[dt_tile, acc_tile],         # sub-chunk + accumulator
            out_specs=[dt_tile, dt_tile, acc_tile],
        )(dt_sub, carry)
        return acc_out, (dt_out_sub, cs_sub)

    init_carry = jnp.zeros(...)
    _, (dt_out_subs, cs_subs) = jax.lax.scan(_scan_body, init_carry, dt_subs)
```

### The Sub-Chunk Kernel

```python
def _chunk_cumsum_sub_kernel(
    dt_ref,        # (1, BQ, nheads_padded) — input
    acc_in_ref,    # (1, nheads_padded)     — accumulator input
    dt_out_ref,    # (1, BQ, nheads_padded) — output
    cs_ref,        # (1, BQ, nheads_padded) — cumsum output
    acc_out_ref,   # (1, nheads_padded)     — accumulator output
):
    # Load accumulator from GMEM input
    acc_smem[:] = acc_in_ref[0, :]
    # Process BQ timesteps
    for i in range(BQ):
        dt_i = softplus_clip(dt_ref[0, i, :])
        dt_out_ref[0, i, :] = dt_i
        acc_smem[:] += dt_i
        cs_ref[0, i, :] = acc_smem[:]
    # Write accumulator to GMEM output
    acc_out_ref[0, :] = acc_smem[:]
```

## SMEM Budget Constants

```python
_MAX_SMEM_BYTES = 232_448      # H100 per-CTA limit (from runtime error)
_SMEM_OVERHEAD  = 4_096        # Conservative margin for alignment/metadata
_EFFECTIVE_MAX_SMEM = 228_352  # Usable budget
```

## Mosaic GPU Lessons Learned

1. **Sequential grid output TMA is unreliable for multi-element tiles.** Output writes for iterations beyond the first are silently dropped. Only `BQ=1` works. Use `jax.lax.scan` with 1D-parallel kernels instead.

2. **scratch_shapes DO persist across sequential iterations.** They are allocated via `run_scoped` outside `emit_pipeline` (confirmed in `lowering.py:753-847`). Input TMA and scratch work fine — only output TMA is broken.

3. **`pl.program_id()` works for sequential dimensions.** It returns the loop variable for sequential dims, the blockIdx for parallel dims.

4. **`jnp.where` evaluates both branches.** Inside Pallas kernels, `jnp.where(cond, a, b)` evaluates both `a` and `b`. Reading uninitialized SMEM in the "unused" branch propagates NaN. Use `pl.when()` for conditional execution.

5. **`pl.when()` uses `lax.cond` internally.** It truly skips the false branch, unlike `jnp.where`.

6. **Warpgroup constraint (128 elements minimum)** applies to any vector in `pl.loop`. Pad the innermost dimension to `ceil(n / 128) * 128`.

7. **SMEM budget = num_tiles × tile_size × 4 bytes.** For BlockSpec-based kernels, every `in_specs` and `out_specs` tile is staged in SMEM simultaneously (for `max_concurrent_steps=1`).
