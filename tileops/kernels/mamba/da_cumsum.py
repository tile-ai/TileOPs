"""
Mamba-2 dA_cumsum forward kernel.

Inputs:
  dt:       (batch, seq_len, n_heads)                  -- per-position discretization factor (float32)
  A:        (n_heads,)                                  -- State Space Model (SSM) decay parameter (float32)

Output:
  dA_cumsum: (batch, n_heads, num_chunks, chunk_len)   -- float32

For each (b, h, c, l), the kernel computes the inclusive prefix sum within each chunk:

  dA_cumsum[b, h, c, l] = sum_{i=0}^{l} dt[b, c*Q + i, h] * A[h]

This matches A_cumsum in the Mamba-2 ssd_minimal_discrete reference:

  A_cumsum = torch.cumsum(dt * A, dim=-1)   # per-chunk, inclusive prefix sum

The output is consumed by:
  - ssd_chunk_scan_fwd:   exp(dA_cumsum[l] - dA_cumsum[s]) scales the intra-chunk causal path
  - ssd_chunk_state_fwd:  exp(dA_cumsum[Q-1] - dA_cumsum[l]) * dt[l] gives per-position decay
  - ssd_state_passing_fwd: dA_cumsum[..., Q-1] is the per-chunk scalar inter-chunk decay

Alignment with Mamba-2 paper:
  In ssd_minimal_discrete, A already absorbs dt (A = dt * A_log), so A_cumsum = cumsum(A).
  Here dt and A are kept separate; dA = dt * A achieves the same result.
  Since A <= 0 in Mamba-2, dA_cumsum is monotonically non-increasing within each chunk,
  and exp(dA_cumsum[l] - dA_cumsum[s]) is a decaying factor in (0, 1] for s <= l.

Notation:
  B = batch, S = seq_len = C * Q, H = n_heads, C = num_chunks, Q = chunk_len
"""

from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["DaCumsumFwdKernel"]


def _da_cumsum_fwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    Q = chunk_len
    H = n_heads
    S = seq_len

    @tilelang.jit(out_idx=[-1])
    def kernel_func(threads: int):
        @T.prim_func
        def main(
            dt: T.Tensor((B, S, H), accum_dtype),            # type: ignore
            A: T.Tensor((H,), accum_dtype),                   # type: ignore
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),  # type: ignore
        ):
            # Grid: one block per (batch, head, chunk)
            # The serial scan over Q positions runs within each block.
            with T.Kernel(B, H, C, threads=threads) as (bb, bh, bc):
                # Load the per-head decay parameter once (scalar, constant across chunk).
                dA_head = A[bh]

                # Running prefix accumulator for the inclusive cumsum.
                running = T.alloc_local((1,), accum_dtype)
                running[0] = T.float32(0.0)

                for l in T.serial(Q):
                    seq_idx = bc * Q + l
                    # Zero-pad positions that fall beyond the actual sequence length
                    # (handles the tail chunk when S is not a multiple of Q).
                    in_bounds = seq_idx < S
                    dt_val = T.if_then_else(
                        in_bounds,
                        dt[bb, seq_idx, bh],
                        T.float32(0.0),
                    )
                    running[0] = running[0] + dt_val * dA_head
                    dA_cumsum[bb, bh, bc, l] = running[0]

        return main

    return kernel_func


@torch.library.custom_op("top::da_cumsum_fwd", mutates_args=())
def _da_cumsum_fwd_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    threads: int,
    dt: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    return _da_cumsum_fwd_kernel(batch, num_chunks, chunk_len, n_heads, seq_len)(
        threads,
    )(dt, A)


@_da_cumsum_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    threads: int,
    dt: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    return dt.new_empty((batch, n_heads, num_chunks, chunk_len), dtype=torch.float32)


class DaCumsumFwdKernel(Kernel):
    """Mamba-2 dA_cumsum forward kernel.

    Computes the chunk-local inclusive prefix sum of dA = dt * A:

      dA_cumsum[b, h, c, l] = sum_{i=0}^{l} dt[b, c*Q+i, h] * A[h]

    This matches A_cumsum from the Mamba-2 ssd_minimal_discrete reference.

    Inputs:  dt  (batch, seq_len, n_heads), float32
             A   (n_heads,), float32
    Output:  dA_cumsum  (batch, n_heads, num_chunks, chunk_len), float32
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        seq_len: int,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.seq_len = seq_len
        # All inputs and output are always float32; no separate dtype parameter needed.
        self.dtype = torch.float32
        self.kernel = _da_cumsum_fwd_kernel(batch, num_chunks, chunk_len, n_heads, seq_len)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # The inner loop is a serial prefix scan with no inner parallelism,
        # so a single thread per block is the natural choice.
        return {"threads": 1}

    @property
    def autotune_configs(self) -> list[dict]:
        # For small batch/head configs, intra-block parallelism can improve occupancy.
        # Warp-level scan with __shfl_up could reduce Q=64 from 64 serial steps to ~6.
        return [
            {"threads": 1},
            {"threads": 32},
            {"threads": 64},
            {"threads": 128},
        ]

    def forward(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        return _da_cumsum_fwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.seq_len,
            self.config["threads"],
            dt, A,
        )
