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

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SSDStatePassingFwdKernel"]


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
                # 1) Local running state tile: s_frag = current s_{c-1} / s_c
                # ------------------------------------------------------------
                s_frag = T.alloc_fragment((block_d,), accum_dtype)
                u_frag = T.alloc_fragment((block_d,), accum_dtype)

                # ------------------------------------------------------------
                # 2) Initialize running state from initial_states (or zero)
                # ------------------------------------------------------------
                for i in T.Parallel(block_d):
                    di = d0 + i
                    if has_initial_states:
                        s_frag[i] = T.if_then_else(
                            di < D,
                            initial_states[bb, bh, di],
                            T.float32(0.0),
                        )
                    else:
                        s_frag[i] = T.float32(0.0)

                # ------------------------------------------------------------
                # 3) Scan over chunks serially
                #    s_c = exp(dA_c) * s_{c-1} + u_c
                # ------------------------------------------------------------
                for c in T.serial(C):
                    # load scalar dA_c and compute scale
                    dA_c = dA_chunk_cumsum[bb, bh, c]
                    scale = T.exp(dA_c)

                    # load current chunk local state u_c = states[b, c, h, :]
                    for i in T.Parallel(block_d):
                        di = d0 + i
                        u_frag[i] = T.if_then_else(
                            di < D,
                            T.cast(states[bb, c, bh, di], accum_dtype),
                            T.float32(0.0),
                        )

                    # recurrent update
                    for i in T.Parallel(block_d):
                        s_frag[i] = scale * s_frag[i] + u_frag[i]

                    # write per-chunk output
                    for i in T.Parallel(block_d):
                        di = d0 + i
                        if di < D:
                            out[bb, c, bh, di] = s_frag[i]

                # ------------------------------------------------------------
                # 4) Write final state
                # ------------------------------------------------------------
                for i in T.Parallel(block_d):
                    di = d0 + i
                    if di < D:
                        final_states[bb, bh, di] = s_frag[i]

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
        batch, num_chunks, n_heads, d_state, has_initial_states, dtype)(
        block_d, threads,
    )(states, dA_chunk_cumsum, initial_states)


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
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_d = [32, 64, 128]
        threads = [128, 256]
        _configs = list(itertools.product(block_d, threads))
        return [{"block_d": c[0], "threads": c[1]} for c in _configs]

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
