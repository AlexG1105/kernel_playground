# kernel_playground
Playground from custom kernels (cutedsl, triton, pallas), optimized for various hardware.

cutedsl-tutorial.ipynb - Tutorial on cutedsl shapes and operations.

Note about chunk_state_pallas and bmm_chunk_pallas, for k loop in triton is removed, investigate here.

chunk_state_pallas.ipynb - _chunk_state_fwd
state_passing_pallas.ipynb - _state_passing_fwd
bmm_chunk_pallas.ipynb - _bmm_chunk_fwd <-- naive matmul 

chunk_scan_pallas.ipynb <-v1 and v2 are bad. v1 is loading everything not taking advantage of is_causal, v2 is also loading everything but outside of the kernel (it splits up the k blocks before running the kernel, but the problem is that it's still loading it). we will attempt v3, which will split  by the row chunk, and before the kernel, only load the relevant chunk of CB, etc. so we don't load everything.

Improvements for chunk_state_fwd, chunk_cumsum_fwd

1. B group expansion materializes 8x more memory

This is likely the biggest factor. With ratio = nheads/ngroups = 64/8 = 8, the Mosaic wrapper pre-expands B from BCG to BCH — replicating each group's data 8 times in GMEM. The Triton kernel instead does an index computation (group_idx = head_idx // ratio) inside the kernel and reads B once per group, reusing it across heads.

For B=4 L=2048 H=64 G=8:

Triton reads B: BCG * chunk_size * dstate * 4 = ~16 MB (f32, shared across heads)
Mosaic reads B_scaled: BCH * chunk_size * dstate * 2 = ~64 MB (bf16, but 8x replicated)
Even in bf16, the expanded copy is 4x more data. This directly hurts memory bandwidth.

2. Scale is not fused into the kernel

Triton computes exp(min(dA_cs_last - dA_cumsum, 0)) * dt on-the-fly in registers while loading B tiles. The scalar dA_cs_last value is loaded once per CTA. Mosaic can't easily do this because emit_pipeline controls TMA loading — there's no hook to do per-element scalar math between TMA load and WGMMA. The scale is pre-multiplied into B in GMEM, which means that multiplication happened in a separate XLA kernel with its own GMEM read/write round-trip.

3. Output slice is a separate XLA kernel

The Pallas kernel writes (BCH, headdim_padded, dstate_padded) f32. The slice to [:, :headdim, :dstate] + reshape to (batch, nchunks, nheads, headdim, dstate) is a separate XLA memcpy kernel that's still included in kernel-only timing.

4. JAX dispatch overhead

Each jax.jit call has ~30-50us of fixed dispatch overhead (Python → XLA runtime → GPU). For small problems (B=1 S=128 at 0.143ms), this is ~25-35% of the total. Triton's CUDA launch path is leaner.

5. No autotuning

We use fixed BM=BK=BN=64, num_stages=2. The Triton kernel uses an autotuner that searches tile sizes per problem shape. Different configs might benefit from BM=128, BN=32, or num_stages=3.

6. emit_pipeline may not fully hide TMA latency

With num_stages=2 (double-buffered), the pipeline needs the next TMA load to complete before WGMMA can consume it. If WGMMA is faster than TMA for these tile sizes, the pipeline stalls. Triton may use more pipeline stages or different tile shapes that better balance compute vs memory.

Where the biggest wins would come from:

Fusing B group expansion into the kernel (read B by group, broadcast in registers) — would cut B memory traffic by ratiox
Fusing scale computation into the kernel — eliminate the pre-multiply GMEM round-trip
Both require moving away from the clean emit_pipeline → WGMMA pattern toward a more manual approach where we do math between TMA loads and WGMMA calls, which the current Mosaic GPU API makes difficult

TODO:

chunk_cumsum check jnp cumsum is supported in kernel? test_cumsum_minimal.py