"""
pallas_mosaic_gpu_mamba/ssd_combined_pallas.py

Pallas Mosaic GPU implementation of mamba_chunk_scan_combined (forward only).

This composes the individual Pallas kernels into the full Mamba2 SSD
forward pass, matching the interface of the Triton
``MambaChunkScanCombinedFn.forward`` in mamba_ssm.

Pipeline (mirrors _mamba_chunk_scan_combined_fwd):
  1. chunk_cumsum_fwd   — dt processing + dA_cumsum
  2. chunk_state_fwd    — per-chunk SSM states
  3. state_passing_fwd  — inter-chunk state propagation
  4. bmm_chunk_fwd      — CB = C @ B^T per chunk
  5. chunk_scan_fwd     — final output (scan + state contrib + D + z)
  6. (optional) chunk_state_varlen — final states for variable-length seqs

Usage
-----
  from pallas_mosaic_gpu_mamba.ssd_combined_pallas import (
      mamba_chunk_scan_combined_fwd,
  )
  out, out_x, dt_out, dA_cumsum, states, final_states = \\
      mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pallas_mosaic_gpu_mamba.chunk_cumsum_fwd import chunk_cumsum_fwd
from pallas_mosaic_gpu_mamba.chunk_state_fwd import (
    chunk_state_fwd,
    chunk_state_varlen_mosaic,
)
from pallas_mosaic_gpu_mamba.state_passing_fwd import state_passing_fwd
from pallas_mosaic_gpu_mamba.bmm_chunk_fwd import bmm_chunk_fwd
from pallas_mosaic_gpu_mamba.chunk_scan_fwd import chunk_scan_fwd


# ---------------------------------------------------------------------------
# Core forward pass
# ---------------------------------------------------------------------------

def mamba_chunk_scan_combined_fwd(
    x,                          # (batch, seqlen, nheads, headdim)   float32
    dt,                         # (batch, seqlen, nheads)            float32
    A,                          # (nheads,)                          float32
    B,                          # (batch, seqlen, ngroups, dstate)   float32
    C,                          # (batch, seqlen, ngroups, dstate)   float32
    chunk_size: int,
    D=None,                     # (nheads,) or (nheads, headdim)     float32, optional
    z=None,                     # (batch, seqlen, nheads, headdim)   float32, optional
    dt_bias=None,               # (nheads,)                          float32, optional
    initial_states=None,        # (batch, nheads, headdim, dstate)   float32, optional
    seq_idx=None,               # (batch, seqlen)                    int32, optional
    cu_seqlens=None,            # (num_sequences + 1,)               int32, optional
    dt_softplus: bool = False,
    dt_limit: tuple = (0.0, float("inf")),
    return_final_states: bool = False,
    return_varlen_states: bool = False,
):
    """
    Pallas Mosaic GPU implementation of the full Mamba2 SSD combined forward.

    Matches the interface of mamba_ssm's ``_mamba_chunk_scan_combined_fwd``.

    Parameters
    ----------
    x              : (batch, seqlen, nheads, headdim)
    dt             : (batch, seqlen, nheads)
    A              : (nheads,)
    B              : (batch, seqlen, ngroups, dstate)
    C              : (batch, seqlen, ngroups, dstate)
    chunk_size     : int
    D              : (nheads,) or (nheads, headdim), optional
    z              : (batch, seqlen, nheads, headdim), optional
    dt_bias        : (nheads,), optional
    initial_states : (batch, nheads, headdim, dstate), optional
    seq_idx        : (batch, seqlen), optional
    cu_seqlens     : (num_sequences + 1,), optional
    dt_softplus    : bool
    dt_limit       : (float, float)
    return_final_states : bool
    return_varlen_states : bool

    Returns
    -------
    out            : (batch, seqlen, nheads, headdim)
    final_states   : (batch, nheads, headdim, dstate) or None
    varlen_states  : (num_sequences, nheads, headdim, dstate) or None
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape

    if not return_varlen_states:
        cu_seqlens = None

    # ------------------------------------------------------------------
    # 1. chunk_cumsum_fwd — dt processing + dA_cumsum
    # ------------------------------------------------------------------
    dA_cumsum, dt_out = chunk_cumsum_fwd(
        dt, A, chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    # dA_cumsum: (batch, nheads, nchunks, chunk_size)
    # dt_out:    (batch, nheads, nchunks, chunk_size)

    # ------------------------------------------------------------------
    # 2. chunk_state_fwd — per-chunk SSM states
    # ------------------------------------------------------------------
    states = chunk_state_fwd(
        x, B, dt_out, dA_cumsum,
        seq_idx=seq_idx,
    )
    # states: (batch, nchunks, nheads, headdim, dstate)

    # ------------------------------------------------------------------
    # 3. state_passing_fwd — inter-chunk state propagation
    #
    # Flatten (headdim, dstate) → dim for state_passing, then reshape back.
    # This matches the Triton code's rearrange("... p n -> ... (p n)").
    # ------------------------------------------------------------------
    nchunks = states.shape[1]
    states_flat = states.reshape(batch, nchunks, nheads, headdim * dstate)

    init_flat = None
    if initial_states is not None:
        init_flat = initial_states.reshape(batch, nheads, headdim * dstate)

    states_passed, final_states_flat = state_passing_fwd(
        states_flat,
        dA_cumsum[:, :, :, -1],            # (batch, nheads, nchunks)
        initial_states=init_flat,
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=C.dtype,
    )
    # states_passed:     (batch, nchunks, nheads, headdim * dstate)
    # final_states_flat: (batch, nheads, headdim * dstate)

    states_passed = states_passed.reshape(batch, nchunks, nheads, headdim, dstate)
    final_states = final_states_flat.reshape(batch, nheads, headdim, dstate)

    # ------------------------------------------------------------------
    # 4. bmm_chunk_fwd — CB = C @ B^T per chunk
    # ------------------------------------------------------------------
    CB = bmm_chunk_fwd(
        C, B, chunk_size,
        seq_idx=seq_idx,
        output_dtype=jnp.float32,
    )
    # CB: (batch, nchunks, ngroups, chunk_size, chunk_size)

    # ------------------------------------------------------------------
    # 5. chunk_scan_fwd — final output
    # ------------------------------------------------------------------
    out, out_x = chunk_scan_fwd(
        CB, x, dt_out, dA_cumsum, C, states_passed,
        D=D, z=z, seq_idx=seq_idx,
    )
    # out:   (batch, seqlen, nheads, headdim)
    # out_x: (batch, seqlen, nheads, headdim) or None

    # ------------------------------------------------------------------
    # 6. (optional) chunk_state_varlen
    # ------------------------------------------------------------------
    varlen_states = None
    if cu_seqlens is not None:
        assert batch == 1, (
            "cu_seqlens for varlen states is only supported with batch=1"
        )
        varlen_states = chunk_state_varlen_mosaic(
            B.squeeze(0),                   # (seqlen, ngroups, dstate)
            x.squeeze(0),                   # (seqlen, nheads, headdim)
            dt_out.squeeze(0),              # (nheads, nchunks, chunk_size)
            dA_cumsum.squeeze(0),           # (nheads, nchunks, chunk_size)
            cu_seqlens,
            states_passed.squeeze(0),       # (nchunks, nheads, headdim, dstate)
        )
        # varlen_states: (num_sequences, nheads, headdim, dstate)

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    if z is not None:
        result_out = out
    else:
        result_out = out

    if not return_varlen_states:
        if return_final_states:
            return result_out, final_states
        else:
            return result_out
    else:
        if return_final_states:
            return result_out, final_states, varlen_states
        else:
            return result_out, varlen_states
