"""
Mamba-2 dA_cumsum forward kernel.

Inputs:
  dt:       (batch, seq_len, n_heads)                  -- raw per-position dt (float32)
  A:        (n_heads,)                                  -- State Space Model (SSM) decay parameter (float32)
  dt_bias:  (n_heads,)                                  -- optional per-head dt bias (float32)

Outputs:
  dt_out:    (batch, n_heads, num_chunks, chunk_len)   -- float32, processed dt after bias/softplus/clamp
  dA_cumsum: (batch, n_heads, num_chunks, chunk_len)   -- float32, inclusive prefix sum of dA = dt_out * A

For each (b, h, c, l), the kernel computes:

  dt_val            = dt[b, c*Q + l, h]
  if has_dt_bias:   dt_val += dt_bias[h]
  if dt_softplus:   dt_val = softplus(dt_val)   # with bypass for dt_val > 20
                    dt_val = clamp(dt_val, dt_min, dt_max)
  dt_out[b,h,c,l]  = dt_val
  dA_cumsum[b,h,c,l] = sum_{i=0}^{l} dt_out[b,h,c,i] * A[h]

This matches _chunk_cumsum_fwd_kernel in the Mamba-2 Triton reference
(mamba_ssm/ops/triton/ssd_chunk_state.py).

Alignment with Mamba-2 paper:
  In ssd_minimal_discrete, A already absorbs dt (A = dt * A_log), so A_cumsum = cumsum(A).
  Here dt and A are kept separate; dA = dt * A achieves the same result.
  Since A <= 0 in Mamba-2, dA_cumsum is monotonically non-increasing within each chunk,
  and exp(dA_cumsum[l] - dA_cumsum[s]) is a decaying factor in (0, 1] for s <= l.

Notation:
  B = batch, S = seq_len = C * Q, H = n_heads, C = num_chunks, Q = chunk_len
"""

from typing import Callable, Optional, Tuple

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
    dt_softplus: bool = False,
    has_dt_bias: bool = False,
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    Q = chunk_len
    H = n_heads
    S = seq_len

    @tilelang.jit(out_idx=[-2, -1])
    def kernel_func(threads: int):
        @T.prim_func
        def main(
            dt:        T.Tensor((B, S, H), accum_dtype),        # type: ignore  # raw dt input
            A:         T.Tensor((H,),      accum_dtype),         # type: ignore
            dt_bias:   T.Tensor((H,),      accum_dtype),         # type: ignore  # may be dummy zeros if not has_dt_bias
            dt_out:    T.Tensor((B, H, C, Q), accum_dtype),     # type: ignore  # output: processed dt
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),     # type: ignore  # output: inclusive cumsum
        ):
            # Grid: one block per (batch, head, chunk).
            # The serial scan over Q positions runs within each block.
            with T.Kernel(B, H, C, threads=threads) as (bb, bh, bc):
                # Load the per-head decay parameter once (scalar, constant across chunk).
                dA_head = A[bh]

                # Running prefix accumulator for the inclusive cumsum.
                running = T.alloc_local((1,), accum_dtype)
                running[0] = T.float32(0.0)

                for l in T.serial(Q):
                    seq_idx = bc * Q + l
                    in_bounds = seq_idx < S

                    # Step 1: load raw dt; zero-pad out-of-bounds tail positions.
                    dt_val = T.if_then_else(
                        in_bounds,
                        dt[bb, seq_idx, bh],
                        T.float32(0.0),
                    )

                    # Step 2: add per-head bias (compile-time conditional).
                    if has_dt_bias:
                        dt_val = dt_val + dt_bias[bh]

                    # Step 3: softplus with large-value bypass (compile-time conditional).
                    # Uses log(1 + exp(x)) for x <= 20; identity for x > 20 to avoid overflow.
                    if dt_softplus:
                        dt_val = T.if_then_else(
                            dt_val <= T.float32(20.0),
                            T.log(T.float32(1.0) + T.exp(dt_val)),
                            dt_val,
                        )

                    # Step 4: clamp to [dt_min, dt_max].
                    dt_val = T.min(T.max(dt_val, T.float32(dt_min)), T.float32(dt_max))

                    # Step 5: re-apply out-of-bounds zero mask after bias/softplus/clamp.
                    dt_val = T.if_then_else(in_bounds, dt_val, T.float32(0.0))

                    # Step 6: store processed dt and accumulate dA_cumsum.
                    dt_out[bb, bh, bc, l] = dt_val
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
    dt_softplus: bool,
    has_dt_bias: bool,
    dt_min: float,
    dt_max: float,
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _da_cumsum_fwd_kernel(
        batch, num_chunks, chunk_len, n_heads, seq_len,
        dt_softplus, has_dt_bias, dt_min, dt_max,
    )(threads)(dt, A, dt_bias)


@_da_cumsum_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    threads: int,
    dt_softplus: bool,
    has_dt_bias: bool,
    dt_min: float,
    dt_max: float,
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt_out = dt.new_empty((batch, n_heads, num_chunks, chunk_len), dtype=torch.float32)
    dA_cumsum = dt.new_empty((batch, n_heads, num_chunks, chunk_len), dtype=torch.float32)
    return dt_out, dA_cumsum


class DaCumsumFwdKernel(Kernel):
    """Mamba-2 dA_cumsum forward kernel.

    Applies optional per-head bias, optional softplus activation, and clamping to
    raw dt values, then computes the chunk-local inclusive prefix sum of dA = dt * A.

    Inputs:
        dt      (batch, seq_len, n_heads) float32 — raw dt values.
        A       (n_heads,) float32 — State Space Model (SSM) decay parameters.
        dt_bias (n_heads,) float32 — per-head dt bias; required when has_dt_bias=True.

    Outputs:
        dt_out    (batch, n_heads, num_chunks, chunk_len) float32 — processed dt.
        dA_cumsum (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        seq_len: int,
        dt_softplus: bool = False,
        has_dt_bias: bool = False,
        dt_min: float = 0.0,
        dt_max: float = float("inf"),
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.dt_softplus = dt_softplus
        self.has_dt_bias = has_dt_bias
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dtype = torch.float32
        self.kernel = _da_cumsum_fwd_kernel(
            batch, num_chunks, chunk_len, n_heads, seq_len,
            dt_softplus, has_dt_bias, dt_min, dt_max,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # The inner loop is a serial prefix scan with no inner parallelism,
        # so a single thread per block is the natural choice.
        return {"threads": 1}

    @property
    def autotune_configs(self) -> list[dict]:
        # For small batch/head configs, intra-block parallelism can improve occupancy.
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
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the dA_cumsum forward pass.

        Args:
            dt: (batch, seq_len, n_heads) float32 — raw dt values.
            A:  (n_heads,) float32 — SSM decay parameters.
            dt_bias: (n_heads,) float32, optional — per-head dt bias.
                Required when the kernel was constructed with has_dt_bias=True.

        Returns:
            dt_out: (batch, n_heads, num_chunks, chunk_len) float32 — processed dt.
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum.
        """
        if self.has_dt_bias and dt_bias is None:
            raise ValueError("dt_bias is required when has_dt_bias=True")
        if not dt.is_cuda:
            raise ValueError("dt must be a CUDA tensor")
        if dt.dtype != torch.float32:
            raise ValueError(f"Expected float32 dt, got {dt.dtype}")

        dt = dt.contiguous()
        A = A.contiguous()
        # Allocate a dummy zero bias when has_dt_bias=False so the kernel
        # signature stays fixed regardless of the compile-time flag.
        dt_bias = dt.new_zeros(self.n_heads) if dt_bias is None else dt_bias.contiguous()

        return _da_cumsum_fwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.seq_len,
            self.config["threads"],
            self.dt_softplus, self.has_dt_bias, self.dt_min, self.dt_max,
            dt, A, dt_bias,
        )
