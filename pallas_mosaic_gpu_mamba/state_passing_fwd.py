"""
pallas_mosaic_gpu_mamba/state_passing_fwd.py

Mosaic GPU (H100/H200) Pallas implementation of _state_passing_fwd.

Algorithm
---------
Given:
  states           : (batch, nchunks, nheads, dim)  float32
  dA_chunk_cumsum  : (batch, nheads, nchunks)       float32
  initial_states   : (batch, nheads, dim)            float32, optional
  seq_idx          : (batch, seqlen)                 int32,   optional
  chunk_size       : int, required when seq_idx is not None

Produces:
  out              : (batch, nchunks, nheads, dim)   float32
  final_states     : (batch, nheads, dim)            float32

The kernel performs a sequential prefix scan over chunks:
  s[0] = initial_states (or zeros)
  for c in range(nchunks):
      scale = exp(dA_chunk_cumsum[:, :, c])
      if seq_idx is not None and sequence changed at chunk c:
          scale = 0   (reset running state)
      s[c+1] = scale * s[c] + states[:, c, :, :]

  out[:, c, :, :]  = s[c]       for c = 0..nchunks-1
  final_states     = s[nchunks]

seq_idx handling is folded into dA_chunk_cumsum during preprocessing:
  where the sequence changes, dA_chunk_cumsum is set to -inf so that
  exp(-inf) = 0.0, resetting the running state.  No kernel changes needed.

Grid and tile design
--------------------
pallas_call grid: (N,)  where N = BH * PD
  BH = batch * nheads   — flattened batch & head dimensions
  PD = dim_padded // BD  — tiles over the dim dimension (state vector)
  Grid is 1D, all parallel (each CTA independent).

Tile per CTA:
  states input   : (1, nchunks, BD)      — all chunks for one dim tile
  dA_cs input    : (1, nchunks, BD)      — replicated scalar per chunk
  init input     : (1, BD)               — initial state (or zeros)
  buf output     : (1, nchunks + 1, BD)  — output buffer:
                     buf[0]     = init state (s[0])
                     buf[c+1]   = s[c+1] for c = 0..nchunks-1
                   Wrapper slices: out = buf[:nchunks], final = buf[nchunks]

BD (block dim) = 128, the minimum for the H100 warpgroup constraint.

Inside the kernel body, pl.loop iterates over chunks sequentially.
The running-state accumulator lives in SMEM (pl.run_scoped).

SMEM budget (BD=128, nchunks=8)
--------------------------------
  states input   : 8 * 128 * 4   =   4 KB
  dA_cs input    : 8 * 128 * 4   =   4 KB   (replicated to BD for warpgroup)
  init input     : 128 * 4       = 512 B
  buf output     : 9 * 128 * 4   =   4.5 KB
  accumulator    : 128 * 4       = 512 B    (SMEM scratch)
  Total          :               ~  14 KB   (H100 limit: 228 KB)

Usage
-----
  from pallas_mosaic_gpu_mamba.state_passing_fwd import state_passing_fwd_mosaic

  out, final_states = state_passing_fwd_mosaic(states, dA_chunk_cumsum)
  out, final_states = state_passing_fwd_mosaic(states, dA_chunk_cumsum, initial_states=h0)
  out, final_states = state_passing_fwd_mosaic(states, dA_chunk_cumsum, seq_idx=seq_idx, chunk_size=64)
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


# ---------------------------------------------------------------------------
# SMEM budget helpers
# ---------------------------------------------------------------------------

_MAX_SMEM_BYTES = 232_448
_SMEM_OVERHEAD = 4096
_EFFECTIVE_MAX_SMEM = _MAX_SMEM_BYTES - _SMEM_OVERHEAD


def _smem_bytes(nchunks: int, BD: int) -> int:
    """Estimate SMEM usage for the state_passing_fwd kernel."""
    smem = 0
    smem += nchunks * BD * 4           # states input tile
    smem += nchunks * BD * 4           # dA_cs input tile (replicated)
    smem += BD * 4                     # init input tile
    smem += (nchunks + 1) * BD * 4     # buf output tile
    smem += BD * 4                     # SMEM scratch accumulator
    return smem


# ---------------------------------------------------------------------------
# Kernel body — single pallas_call with pl.loop over chunks
# ---------------------------------------------------------------------------

def _state_passing_fwd_kernel(
    states_ref,   # SMEM ref: (1, nchunks, BD)     — input states per chunk
    dA_cs_ref,    # SMEM ref: (1, nchunks, BD)     — exp-decay per chunk (replicated)
    init_ref,     # SMEM ref: (1, BD)              — initial state
    buf_ref,      # SMEM ref: (1, nchunks + 1, BD) — output buffer
    *,
    nchunks: int,
    BD: int,
):
    """
    One CTA computes the sequential scan for one (batch, head, dim_tile).

    Reads initial state, then loops over nchunks:
      acc = exp(dA_cs[c]) * acc + states[c]
    Writes nchunks+1 entries to buf:
      buf[0] = init,  buf[c+1] = state after chunk c.

    dA_cs is replicated to (nchunks, BD) so that access pattern is
    dA_cs_ref[0, c, :] → (BD,) contiguous, satisfying warpgroup constraint.
    All BD elements have the same value; the multiply is elementwise.
    """
    def _scan(acc_smem):
        # Load initial state
        acc_smem[:] = init_ref[0, :].astype(jnp.float32)

        # buf[0] = initial state
        buf_ref[0, 0, :] = acc_smem[:]

        # Sequential scan over chunks
        @pl.loop(0, nchunks)
        def _step(c):
            new_states = states_ref[0, c, :].astype(jnp.float32)
            dA_cs_vec = dA_cs_ref[0, c, :].astype(jnp.float32)
            scale = jnp.exp(dA_cs_vec)

            acc_smem[:] = scale * acc_smem[:] + new_states

            # Write to buf[c+1]
            buf_ref[0, c + 1, :] = acc_smem[:]

    pl.run_scoped(
        _scan,
        plgpu.SMEM((BD,), jnp.float32),
    )


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

def state_passing_fwd_mosaic(
    states,                          # (batch, nchunks, nheads, dim) float32
    dA_chunk_cumsum,                 # (batch, nheads, nchunks)      float32
    initial_states=None,             # (batch, nheads, dim)          float32, optional
    seq_idx=None,                    # (batch, seqlen)               int32, optional
    chunk_size=None,                 # int, required when seq_idx is not None
    out_dtype=None,
) -> tuple:
    """
    H100/H200 Pallas Mosaic GPU port of _state_passing_fwd.

    Parameters
    ----------
    states           : (batch, nchunks, nheads, dim) float32
    dA_chunk_cumsum  : (batch, nheads, nchunks)      float32
    initial_states   : (batch, nheads, dim)           float32, optional
    seq_idx          : (batch, seqlen)                int32, optional
        Sequence index for each position.  When packed variable-length
        sequences share a single batch row, seq_idx marks which sequence
        each position belongs to.  At chunk boundaries where the sequence
        changes, the running state is reset to zero (scale = 0).
    chunk_size       : int, required when seq_idx is not None
    out_dtype        : output dtype, defaults to states.dtype

    Returns
    -------
    out          : (batch, nchunks, nheads, dim)  float32 (or out_dtype)
    final_states : (batch, nheads, dim)           float32
    """
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype

    # ------------------------------------------------------------------
    # Handle seq_idx: fold sequence-boundary resets into dA_chunk_cumsum.
    #
    # At each chunk c the Triton kernel checks whether the sequence index
    # at the last position of chunk c differs from the previous chunk.
    # If so, scale = 0  →  running state is reset.
    #
    # We achieve the same by setting dA_chunk_cumsum = -inf at those
    # positions: exp(-inf) = 0.0 (exact in IEEE 754), so the kernel
    # needs no modification.
    # ------------------------------------------------------------------
    if seq_idx is not None:
        chunk_ends = jnp.minimum(
            jnp.arange(1, nchunks + 1) * chunk_size, seqlen
        ) - 1  # (nchunks,)
        seq_idx_at_ends = seq_idx[:, chunk_ends]  # (batch, nchunks)
        # Previous sequence index: 0 for the first chunk (matches Triton's
        # `seq_idx = 0` initialisation), then the end-of-previous-chunk value.
        seq_idx_prev = jnp.concatenate([
            jnp.zeros((batch, 1), dtype=seq_idx.dtype),
            seq_idx_at_ends[:, :-1],
        ], axis=1)  # (batch, nchunks)
        same_seq = seq_idx_at_ends == seq_idx_prev  # (batch, nchunks)
        # Broadcast to (batch, nheads, nchunks) and apply
        dA_chunk_cumsum = jnp.where(
            same_seq[:, None, :],
            dA_chunk_cumsum,
            jnp.float32(-jnp.inf),
        )

    # ------------------------------------------------------------------
    # Flatten (batch, nheads) → BH for parallel grid
    # ------------------------------------------------------------------
    BH = batch * nheads

    # states: (batch, nchunks, nheads, dim) → (BH, nchunks, dim)
    states_flat = states.transpose(0, 2, 1, 3).reshape(BH, nchunks, dim)

    # dA_chunk_cumsum: (batch, nheads, nchunks) → (BH, nchunks)
    dA_cs_flat = dA_chunk_cumsum.reshape(BH, nchunks)

    # initial_states: (batch, nheads, dim) → (BH, dim), or zeros
    if initial_states is not None:
        init_flat = initial_states.reshape(BH, dim)
    else:
        init_flat = jnp.zeros((BH, dim), jnp.float32)

    # ------------------------------------------------------------------
    # Tile dim: BD = 128 for warpgroup constraint
    # ------------------------------------------------------------------
    BD = 128
    dim_padded = math.ceil(dim / BD) * BD
    PD = dim_padded // BD

    if dim_padded > dim:
        pad_d = dim_padded - dim
        states_flat = jnp.pad(states_flat, ((0, 0), (0, 0), (0, pad_d)))
        init_flat = jnp.pad(init_flat, ((0, 0), (0, pad_d)))

    # Reshape to tile dim: (BH, nchunks, dim_padded) → (BH*PD, nchunks, BD)
    states_tiled = (
        states_flat.reshape(BH, nchunks, PD, BD)
        .transpose(0, 2, 1, 3)
        .reshape(BH * PD, nchunks, BD)
    )
    init_tiled = init_flat.reshape(BH, PD, BD).reshape(BH * PD, BD)

    # Replicate dA_cs to (BH*PD, nchunks, BD) so each CTA gets a
    # (1, nchunks, BD) tile matching the warpgroup-safe access pattern.
    # dA_cs is a scalar per (batch, head, chunk); replicated across PD and BD.
    # Memory: BH*PD * nchunks * BD * 4.  For standard configs this is
    # proportional to the states array size.
    dA_cs_tiled = jnp.repeat(
        jnp.repeat(dA_cs_flat, PD, axis=0)[:, :, None],
        BD, axis=2,
    )  # (BH*PD, nchunks, BD), contiguous

    N = BH * PD  # total parallel CTAs

    # ------------------------------------------------------------------
    # SMEM budget check
    # ------------------------------------------------------------------
    smem_needed = _smem_bytes(nchunks, BD)
    assert smem_needed <= _EFFECTIVE_MAX_SMEM, (
        f"SMEM overflow: {smem_needed} > {_EFFECTIVE_MAX_SMEM}. "
        f"nchunks={nchunks}, BD={BD}"
    )

    # ------------------------------------------------------------------
    # Launch pallas_call — 1D parallel grid
    # ------------------------------------------------------------------
    buf_len = nchunks + 1

    kernel = partial(
        _state_passing_fwd_kernel,
        nchunks=nchunks,
        BD=BD,
    )

    states_tile = pl.BlockSpec((1, nchunks, BD), lambda n: (n, 0, 0))
    dA_cs_tile  = pl.BlockSpec((1, nchunks, BD), lambda n: (n, 0, 0))
    init_tile   = pl.BlockSpec((1, BD),          lambda n: (n, 0))
    buf_tile    = pl.BlockSpec((1, buf_len, BD), lambda n: (n, 0, 0))

    buf_flat = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((N, buf_len, BD), jnp.float32),
        grid=(N,),
        in_specs=[states_tile, dA_cs_tile, init_tile],
        out_specs=buf_tile,
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel"],
        ),
    )(states_tiled, dA_cs_tiled, init_tiled)

    # ------------------------------------------------------------------
    # Extract out (first nchunks entries) and final_states (last entry)
    # ------------------------------------------------------------------
    # buf_flat: (N, nchunks+1, BD) = (BH*PD, nchunks+1, BD)
    out_tiled = buf_flat[:, :nchunks, :]          # (BH*PD, nchunks, BD)
    final_tiled = buf_flat[:, nchunks, :]          # (BH*PD, BD)

    # Untile dim: (BH*PD, nchunks, BD) → (BH, PD, nchunks, BD) → (BH, nchunks, dim_padded)
    out_flat = (
        out_tiled.reshape(BH, PD, nchunks, BD)
        .transpose(0, 2, 1, 3)
        .reshape(BH, nchunks, dim_padded)
        [:, :, :dim]
    )
    # → (batch, nheads, nchunks, dim) → (batch, nchunks, nheads, dim)
    out = (
        out_flat.reshape(batch, nheads, nchunks, dim)
        .transpose(0, 2, 1, 3)
        .astype(out_dtype)
    )

    # final: (BH*PD, BD) → (BH, PD, BD) → (BH, dim_padded) → (BH, dim)
    final_states = (
        final_tiled.reshape(BH, PD, BD)
        .reshape(BH, dim_padded)
        [:, :dim]
        .reshape(batch, nheads, dim)
    )

    return out, final_states


# ---------------------------------------------------------------------------
# Triton-compatible alias
# ---------------------------------------------------------------------------

def state_passing_fwd(
    states,
    dA_chunk_cumsum,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
) -> tuple:
    """Drop-in replacement for mamba_ssm._state_passing_fwd (JAX/Pallas version)."""
    return state_passing_fwd_mosaic(
        states, dA_chunk_cumsum,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=out_dtype,
    )
