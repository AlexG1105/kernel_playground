# Mosaic GPU Fix: chunk_state_fwd

## The Problem

When running `chunk_state_fwd` with `emit_pipeline` inside a `pallas_call` kernel, it crashes:

```
ValueError: Only SMEM <-> GMEM copies supported
```

Full traceback points to `copy_gmem_to_smem` being called on SMEM refs inside `emit_pipeline`.

### Root Cause

`pallas_call` with `BlockSpec` **stages input/output tiles into SMEM** via TMA *before* the kernel body runs. The kernel body receives **SMEM refs**, not GMEM refs.

`emit_pipeline` performs its own TMA loads from GMEM into swizzled SMEM. It calls `copy_gmem_to_smem` internally, which **requires GMEM source refs**. When it receives SMEM refs (from `pallas_call`'s staging), it fails because TMA only supports GMEM↔SMEM, not SMEM→SMEM.

```
pallas_call + BlockSpec
  → tiles staged into SMEM (TMA)
  → kernel body gets SMEM refs
  → emit_pipeline tries copy_gmem_to_smem(SMEM_ref)  ← ERROR
```

This is a **fundamental architectural incompatibility**: `pallas_call` with `BlockSpec` and `emit_pipeline` cannot coexist in the same kernel because they both want to manage SMEM staging.

## Approaches Tried and Why They Failed

### Attempt 1: Manual pl.loop + WGMMA (replacing emit_pipeline)

The idea was to keep `pallas_call` with `BlockSpec` (so the kernel gets SMEM tiles) and replace `emit_pipeline` with a manual `pl.loop` that reads sub-tiles from SMEM and feeds them to `wgmma`.

This required:
1. Allocating swizzled SMEM scratch via `scratch_shapes` with `TilingTransform + SwizzleTransform`
2. In the loop body, copying a (BM, BK) slice from the input SMEM tile into the swizzled scratch
3. Passing the swizzled scratch to `wgmma`

**Result:**
```
NotImplementedError: WGStridedFragLayout(shape=(64, 64), vec_size=2)
```

**Why it failed:** Writing a register array (from reading the SMEM input tile) into WGMMA-transformed SMEM scratch is **not implemented** in the Mosaic GPU lowering. The store path from `WGStridedFragLayout` (the register layout for a (64, 64) bf16 array) to tiled+swizzled SMEM is missing. There is no way to manually populate WGMMA-compatible swizzled SMEM from registers — only TMA can load directly into that layout.

### Why emit_pipeline is the Only Path to WGMMA

On H100, WGMMA reads operands from **swizzled SMEM** (TilingTransform + SwizzleTransform). There are only two ways to get data into swizzled SMEM:

1. **TMA** (Tensor Memory Accelerator) — loads from GMEM directly into swizzled SMEM. This is what `emit_pipeline` uses.
2. **Register stores** — but the store from `WGStridedFragLayout` to transformed SMEM is not implemented.

So `emit_pipeline` with TMA from GMEM is effectively the **only** way to feed WGMMA on Mosaic GPU.

## The Fix: pl.kernel (core_map) Instead of pallas_call

The working solution replaces `pallas_call` with `pl.kernel`, which uses `core_map` internally. The key difference:

| | `pallas_call` + BlockSpec | `pl.kernel` (core_map) |
|---|---|---|
| Kernel body receives | **SMEM refs** (pre-staged) | **GMEM refs** (raw) |
| TMA staging | Done *before* kernel body | Kernel manages its own |
| Compatible with emit_pipeline | **No** | **Yes** |

With `pl.kernel`, the kernel body gets GMEM refs, which `emit_pipeline` can TMA-load into swizzled SMEM for WGMMA.

```
pl.kernel (core_map)
  → kernel body gets GMEM refs
  → emit_pipeline TMA-loads GMEM → swizzled SMEM  ✓
  → wgmma reads from swizzled SMEM                ✓
  → accumulator written to GMEM output             ✓
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  JAX Wrapper (chunk_state_fwd_mosaic)               │
│                                                     │
│  1. Precompute scale = exp(min(...)) * dt           │
│  2. Pre-multiply B_scaled = B_expanded * scale      │
│  3. Cast x_flat, B_scaled to bf16                   │
│  4. Pad headdim, dstate to multiples of BM, BN      │
│  5. Reshape to (BCH, ...) flat arrays               │
│                                                     │
│  ┌─────────────────────────────────────────────────┐ │
│  │  pl.kernel (core_map over Mesh)                 │ │
│  │  Grid: (BCH, PM, PN) — all parallel             │ │
│  │                                                 │ │
│  │  Kernel body receives GMEM refs:                │ │
│  │    x_t_ref:    (BCH, headdim_pad, chunk_size)   │ │
│  │    B_ref:      (BCH, chunk_size, dstate_pad)    │ │
│  │    states_ref: (BCH, headdim_pad, dstate_pad)   │ │
│  │                                                 │ │
│  │  ┌───────────────────────────────────────────┐  │ │
│  │  │  emit_pipeline (K = chunk_size // BK)     │  │ │
│  │  │                                           │  │ │
│  │  │  For each step k:                         │  │ │
│  │  │    TMA: GMEM → swizzled SMEM (bf16)       │  │ │
│  │  │      a_smem: (BM, BK) from x_T            │  │ │
│  │  │      b_smem: (BK, BN) from B_scaled       │  │ │
│  │  │    wgmma(acc, a_smem, b_smem)  → f32 acc  │  │ │
│  │  │                                           │  │ │
│  │  │  After all steps:                         │  │ │
│  │  │    states_ref[...] = acc (f32)            │  │ │
│  │  └───────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  6. Slice padded output → (batch, nchunks, nheads,  │
│     headdim, dstate)                                │
└─────────────────────────────────────────────────────┘
```

### Key Code: Kernel Body

```python
def _chunk_state_kernel_body(
    x_t_ref,    # GMEM ref: (BCH, headdim_padded, chunk_size) bf16
    B_ref,      # GMEM ref: (BCH, chunk_size, dstate_padded) bf16
    states_ref, # GMEM ref: (BCH, headdim_padded, dstate_padded) f32
    *, BM, BK, BN, chunk_size, num_stages,
):
    bch = lax.axis_index("bch")
    pm  = lax.axis_index("pm")
    pn  = lax.axis_index("pn")

    K = chunk_size // BK

    # Sub-ref this CTA's portion (still GMEM)
    x_t_gmem = x_t_ref.at[bch, pl.ds(pm * BM, BM), :]
    B_gmem   = B_ref.at[bch, :, pl.ds(pn * BN, BN)]

    # Swizzle transforms for WGMMA-compatible SMEM layout
    a_swizzle = plgpu.find_swizzle(BK * 16)  # 16 bits per bf16
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
            grid=(K,),
            in_specs=[
                plgpu.BlockSpec((BM, BK), lambda k: (0, k), transforms=a_transforms),
                plgpu.BlockSpec((BK, BN), lambda k: (k, 0), transforms=b_transforms),
            ],
            max_concurrent_steps=num_stages,
        )(x_t_gmem, B_gmem)

        states_ref[bch, pl.ds(pm * BM, BM), pl.ds(pn * BN, BN)] = (
            acc_ref[...].astype(jnp.float32)
        )

    pl.run_scoped(_with_acc, plgpu.ACC((BM, BN), jnp.float32))
```

### Key Code: Kernel Launch

```python
mesh = plgpu.Mesh(
    grid=(BCH, PM, PN),
    grid_names=("bch", "pm", "pn"),
)

kernel_fn = pl.kernel(
    partial(_chunk_state_kernel_body, BM=BM, BK=BK, BN=BN,
            chunk_size=chunk_size, num_stages=num_stages),
    out_shape=jax.ShapeDtypeStruct(
        (BCH, headdim_padded, dstate_padded), jnp.float32
    ),
    mesh=mesh,
)

states_flat = kernel_fn(x_flat, B_scaled)
```

### Key Code: Wrapper Preprocessing

The wrapper does significant preprocessing to simplify the kernel:

```python
# 1. Precompute scale (avoids scalar reads in kernel)
dA_cs_last = dA_cumsum[:, :, :, -1:]
scale = jnp.exp(jnp.minimum(dA_cs_last - dA_cumsum, 0.0)) * dt

# 2. Expand B from group-indexed to head-indexed, pre-multiply scale
B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
B = B.transpose(0, 1, 3, 2, 4)  # → (batch, nchunks, ngroups, chunk_size, dstate)
# ... index mapping from BCH → BCG ...
B_expanded = B_flat[bcg_indices]  # (BCH, chunk_size, dstate_padded)
B_scaled = (B_expanded * scale_flat[:, :, None]).astype(jnp.bfloat16)

# 3. Reshape x → (BCH, headdim_padded, chunk_size) bf16
x_t = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
x_t = x_t.transpose(0, 1, 3, 4, 2)  # → (..., headdim, chunk_size)
x_flat = x_t.reshape(BCH, headdim, chunk_size)
# ... pad headdim to multiple of BM ...
x_flat = x_flat.astype(jnp.bfloat16)
```

This eliminates from the kernel:
- Scale computation and application
- Group→head index mapping (ngroups→nheads expansion)
- bf16 casting (done before kernel launch)

## The emit_pipeline init_carry Gotcha

During development, an intermediate error occurred:

```
TypeError: pipeline_body() missing 1 required positional argument: 'carry'
```

**Root cause:** `emit_pipeline`'s `init_carry` parameter defaults to `None`, not `()`. When `init_carry is None`, **no carry argument** is passed to the body function. The body signature was initially:

```python
def pipeline_body(step, a_smem, b_smem, carry):  # 4 params, but only 3 args passed
```

**Fix:** Remove the `carry` parameter since we don't need loop carry state:

```python
def pipeline_body(step, a_smem, b_smem):  # 3 params, 3 args ✓
```

The relevant code in `pipeline.py`:
```python
# Line 206
def emit_pipeline(..., init_carry: T | None = None, ...):

# Lines 351-358
body_args = (
    *(prev_body_carry,) if init_carry is not None else (),
    *smem_refs,
)
```

## SMEM Budget

With BM=BN=BK=64 and num_stages=2 (double-buffered):

```
x_T staging  : 2 × 64×64×2  =  16 KB  (bf16, double-buffered)
B   staging  : 2 × 64×64×2  =  16 KB  (bf16, double-buffered)
ACC (regs)   : 64×64×4       =  16 KB  (f32, NOT SMEM — in registers)
Total SMEM   :               ~  32 KB  ✓  (H100 limit: 228 KB)
```

This is very comfortable — `emit_pipeline` manages only small bf16 tiles in SMEM, while the f32 accumulator lives in registers via `plgpu.ACC`.

## Mosaic GPU Lessons Learned

1. **`pallas_call` + `emit_pipeline` is fundamentally incompatible.** `pallas_call` with `BlockSpec` gives SMEM refs; `emit_pipeline` needs GMEM refs for TMA. Use `pl.kernel` (core_map) instead, which provides GMEM refs.

2. **Register → transformed SMEM writes are not implemented.** You cannot manually populate WGMMA-compatible swizzled SMEM from register values (`WGStridedFragLayout` → tiled+swizzled SMEM store is unimplemented). Only TMA can load into swizzled SMEM.

3. **`emit_pipeline` is the only practical path to WGMMA.** Since TMA is the only way to fill swizzled SMEM, and `emit_pipeline` is the only high-level API for TMA pipelines, you must use `emit_pipeline` for any WGMMA-based matmul.

4. **The correct Mosaic GPU matmul pattern is:**
   ```
   pl.kernel (core_map) → GMEM refs
     → emit_pipeline with TilingTransform + SwizzleTransform on in_specs
       → TMA loads bf16 into swizzled SMEM
       → wgmma(acc_ref, a_smem, b_smem) accumulates in f32
     → write acc to GMEM output
   ```

5. **Pre-process in the JAX wrapper, not the kernel.** Scale computation, group→head expansion, bf16 casting, and padding are all simpler and more efficient in JAX/XLA. The kernel should only do the matmul.

6. **`emit_pipeline` init_carry defaults to `None`, not `()`.** When `init_carry is None`, no carry argument is passed to the body function. Only add a carry parameter when you explicitly provide `init_carry`.

7. **`plgpu.Mesh` + `lax.axis_index`** replaces `pl.program_id` when using `pl.kernel`/`core_map`. Named mesh axes (e.g., `"bch"`, `"pm"`, `"pn"`) provide the CTA coordinates.

8. **`plgpu.ACC((M, N), dtype)` allocates WGMMA accumulators in registers**, not SMEM. This is allocated via `pl.run_scoped` and does not count toward the SMEM budget.

9. **`plgpu.find_swizzle(bits)`** computes the correct swizzle size for a given element width in bits. For bf16 (16 bits): `find_swizzle(BK * 16)`. The tiling transform uses `(8, swizzle // 2)` — 8 rows per tile, swizzle/2 elements per row.
