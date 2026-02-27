# kernel_playground
Playground from custom kernels (cutedsl, triton, pallas), optimized for various hardware.

cutedsl-tutorial.ipynb - Tutorial on cutedsl shapes and operations.

Note about chunk_state_pallas and bmm_chunk_pallas, for k loop in triton is removed, investigate here.

chunk_state_pallas.ipynb - _chunk_state_fwd
state_passing_pallas.ipynb - _state_passing_fwd
bmm_chunk_pallas.ipynb - _bmm_chunk_fwd <-- naive matmul 

chunk_scan_pallas.ipynb <-v1 and v2 are bad. v1 is loading everything not taking advantage of is_causal, v2 is also loading everything but outside of the kernel (it splits up the k blocks before running the kernel, but the problem is that it's still loading it). we will attempt v3, which will split  by the row chunk, and before the kernel, only load the relevant chunk of CB, etc. so we don't load everything.