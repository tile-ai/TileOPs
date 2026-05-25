"""
Mamba-2 State-Space Dual (SSD) state passing forward kernel.

Inputs:
  states:            (batch, num_chunks, n_heads, d_state)
                     -- chunk-local State Space Model (SSM) states u_c
  dA_chunk_cumsum:   (batch, n_heads, num_chunks)
                     -- per-chunk cumulative decay scalar (float32)
  initial_states:    (batch, n_heads, d_state)
                     -- optional initial state s_{-1}; treated as zero if absent

Outputs:
  out:               (batch, num_chunks, n_heads, d_state)
                     -- running state s_c after each chunk
  final_states:      (batch, n_heads, d_state)
                     -- final running state s_{C-1}

For each (b, h, m), the kernel computes the serial scan:

  s_c[m] = exp(dA_chunk_cumsum[b, h, c]) * s_{c-1}[m] + states[b, c, h, m]

with s_{-1} = initial_states[b, h, :] (or 0 if not provided).

  out[b, c, h, m]      = s_c[m]
  final_states[b, h, m] = s_{C-1}[m]

Parallelization:
  - axis-0: tile over d_state (D)
  - axis-1: batch (B)
  - axis-2: head (H)
  - chunk dimension (C) is scanned serially inside the kernel

Notation:
  B = batch, C = num_chunks, H = n_heads, D = d_state
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["SSDStatePassingFwdKernel"]


@functools.lru_cache(maxsize=32)
def _ssd_state_passing_fwd_kernel(
    batch: int,
    num_chunks: int,
    n_heads: int,
    d_state: int,
    has_initial_states: bool = True,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    H = n_heads
    D = d_state

    @tilelang.jit(out_idx=[-2, -1])
    def kernel_func(
        block_d: int,
        threads: int,
    ):
        # Each thread handles 2 d-state elements (lo and hi halves of block_d).
        # threads = block_d // 2; lo covers [d0 .. d0+threads-1],
        # hi covers [d0+threads .. d0+block_d-1].
        states_shape = (B, C, H, D)
        dA_shape = (B, H, C)
        init_shape = (B, H, D)
        out_shape = (B, C, H, D)
        final_shape = (B, H, D)

        @T.prim_func
        def main(
            states: T.Tensor(states_shape, dtype),              # type: ignore
            dA_chunk_cumsum: T.Tensor(dA_shape, accum_dtype),   # type: ignore
            initial_states: T.Tensor(init_shape, accum_dtype),  # type: ignore
            out: T.Tensor(out_shape, accum_dtype),              # type: ignore
            final_states: T.Tensor(final_shape, accum_dtype),   # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(D, block_d),  # tile over d_state
                B,                      # batch
                H,                      # head
                threads=threads,
            ) as (bd, bb, bh):

                d0 = bd * block_d

                # ------------------------------------------------------------
                # Two fragments per half of block_d so TileLang layout inference
                # sees a consistent index pattern (i only, not i+offset).
                # threads = block_d // 2; lo -> [d0 .. d0+threads-1],
                #                          hi -> [d0+threads .. d0+block_d-1]
                # ------------------------------------------------------------
                s_lo = T.alloc_fragment((threads,), accum_dtype)
                s_hi = T.alloc_fragment((threads,), accum_dtype)
                u_lo = T.alloc_fragment((threads,), accum_dtype)
                u_hi = T.alloc_fragment((threads,), accum_dtype)

                # ------------------------------------------------------------
                # Preload all dA scalars into shared memory in one parallel pass.
                # ------------------------------------------------------------
                dA_shared = T.alloc_shared((C,), accum_dtype)
                for c in T.Parallel(C):
                    dA_shared[c] = dA_chunk_cumsum[bb, bh, c]

                # ------------------------------------------------------------
                # Initialize running state from initial_states (or zero).
                # ------------------------------------------------------------
                for i in T.Parallel(threads):
                    di_lo = d0 + i
                    di_hi = d0 + i + threads
                    if has_initial_states:
                        s_lo[i] = T.if_then_else(
                            di_lo < D, initial_states[bb, bh, di_lo], T.float32(0.0))
                        s_hi[i] = T.if_then_else(
                            di_hi < D, initial_states[bb, bh, di_hi], T.float32(0.0))
                    else:
                        s_lo[i] = T.float32(0.0)
                        s_hi[i] = T.float32(0.0)

                # Write s_{-1} to out[:,0,:,:]
                for i in T.Parallel(threads):
                    di_lo = d0 + i
                    di_hi = d0 + i + threads
                    if di_lo < D:
                        out[bb, 0, bh, di_lo] = s_lo[i]
                    if di_hi < D:
                        out[bb, 0, bh, di_hi] = s_hi[i]

                # ------------------------------------------------------------
                # Serial scan over chunks; dA read from shared memory.
                # ------------------------------------------------------------
                for c in T.serial(C):
                    scale = T.exp(dA_shared[c])

                    for i in T.Parallel(threads):
                        di_lo = d0 + i
                        di_hi = d0 + i + threads
                        u_lo[i] = T.if_then_else(
                            di_lo < D,
                            T.cast(states[bb, c, bh, di_lo], accum_dtype),
                            T.float32(0.0),
                        )
                        u_hi[i] = T.if_then_else(
                            di_hi < D,
                            T.cast(states[bb, c, bh, di_hi], accum_dtype),
                            T.float32(0.0),
                        )

                    for i in T.Parallel(threads):
                        s_lo[i] = scale * s_lo[i] + u_lo[i]
                        s_hi[i] = scale * s_hi[i] + u_hi[i]

                    if c < C - 1:
                        for i in T.Parallel(threads):
                            di_lo = d0 + i
                            di_hi = d0 + i + threads
                            if di_lo < D:
                                out[bb, c + 1, bh, di_lo] = s_lo[i]
                            if di_hi < D:
                                out[bb, c + 1, bh, di_hi] = s_hi[i]

                # ------------------------------------------------------------
                # Write final state s_{C-1}.
                # ------------------------------------------------------------
                for i in T.Parallel(threads):
                    di_lo = d0 + i
                    di_hi = d0 + i + threads
                    if di_lo < D:
                        final_states[bb, bh, di_lo] = s_lo[i]
                    if di_hi < D:
                        final_states[bb, bh, di_hi] = s_hi[i]

        return main

    return kernel_func



@torch.library.custom_op("top::ssd_state_passing_fwd", mutates_args=())
def _ssd_state_passing_fwd_wrapped(
    batch: int,
    num_chunks: int,
    n_heads: int,
    d_state: int,
    has_initial_states: bool,
    dtype: str,
    block_d: int,
    threads: int,
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ssd_state_passing_fwd_kernel(
        batch, num_chunks, n_heads, d_state, has_initial_states, dtype,
    )(block_d, threads)(states, dA_chunk_cumsum, initial_states)


@_ssd_state_passing_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    n_heads: int,
    d_state: int,
    has_initial_states: bool,
    dtype: str,
    block_d: int,
    threads: int,
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = states.new_empty((batch, num_chunks, n_heads, d_state), dtype=torch.float32)
    final = states.new_empty((batch, n_heads, d_state), dtype=torch.float32)
    return out, final


class SSDStatePassingFwdKernel(Kernel):
    """Mamba-2 SSD state passing forward kernel.

    Performs the inter-chunk recurrent scan:

      s_c[m] = exp(dA_chunk_cumsum[b, h, c]) * s_{c-1}[m] + states[b, c, h, m]

    with s_{-1} = initial_states (or 0 if has_initial_states=False).

    Inputs:  states, dA_chunk_cumsum, initial_states
    Outputs: out  (batch, num_chunks, n_heads, d_state), float32
             final_states  (batch, n_heads, d_state), float32
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_heads: int,
        d_state: int,
        has_initial_states: bool = True,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_state = d_state
        self.has_initial_states = has_initial_states
        self.dtype = dtype
        self.kernel = _ssd_state_passing_fwd_kernel(
            batch, num_chunks, n_heads, d_state, has_initial_states, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_d": 64,
            "threads": 32,  # block_d // 2
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # threads = block_d // 2 (each thread handles 2 d-elements)
        return [
            {"block_d": 32,  "threads": 16},
            {"block_d": 64,  "threads": 32},
            {"block_d": 128, "threads": 64},
            {"block_d": 256, "threads": 128},
        ]

    def forward(
        self,
        states: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        initial_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _ssd_state_passing_fwd_wrapped(
            self.batch, self.num_chunks, self.n_heads, self.d_state,
            self.has_initial_states, self.dtype_str,
            self.config["block_d"], self.config["threads"],
            states, dA_chunk_cumsum, initial_states,
        )
