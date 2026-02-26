# kernel_playground
Playground from custom kernels (cutedsl, triton, pallas), optimized for various hardware.

cutedsl-tutorial.ipynb - Tutorial on cutedsl shapes and operations.

Note about chunk_state_pallas and bmm_chunk_pallas, for k loop in triton is removed, investigate here.

chunk_state_pallas.ipynb - _chunk_state_fwd
state_passing_pallas.ipynb - _state_passing_fwd
bmm_chunk_pallas.ipynb - _bmm_chunk_fwd <-- naive matmul 