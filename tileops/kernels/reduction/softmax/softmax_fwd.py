"""Softmax / log-softmax forward kernel using TileLang.

Implements a 2-pass online softmax algorithm for two operations:
  - softmax:     y[i,j] = exp(x[i,j] - max_i) / sum_i(exp(x[i,j] - max_i))
  - log_softmax: y[i,j] = x[i,j] - max_i - log(sum_i(exp(x[i,j] - max_i)))

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions. Padding zeros are handled by using -infinity fill so
they contribute 0 to the softmax denominator.
"""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

__all__ = ["SoftmaxKernel"]


@functools.lru_cache(maxsize=32)
def _softmax_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a TileLang softmax/log_softmax kernel.

    Uses a 2-pass approach:
      Pass 1: Compute row-wise max for numerical stability.
      Pass 2: Compute exp(x - max) and row-wise sum, then normalize.

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Hidden dimension (last dim).
        op_kind: One of "softmax", "log_softmax".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)

    if op_kind == "softmax":

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")

                    # Load input via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Cast to fp32
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # Pass 1: row-wise max
                    T.fill(row_max, -T.infinity("float32"))
                    T.reduce_max(x_f32, row_max, dim=1, clear=False)

                    # Pass 2: exp(x - max) and row-wise sum
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                    T.reduce_sum(x_f32, row_sum, dim=1)

                    # Epilogue: normalize (softmax)
                    out_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    for i, j in T.Parallel(block_m, N_padded):
                        out_f32[i, j] = x_f32[i, j] / row_sum[i]

                    # Cast back to original dtype and write via shared memory
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = out_f32[i, j]
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

    else:  # log_softmax

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")

                    # Load input via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Cast to fp32
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # Pass 1: row-wise max
                    T.fill(row_max, -T.infinity("float32"))
                    T.reduce_max(x_f32, row_max, dim=1, clear=False)

                    # Pass 2: exp(x - max) and row-wise sum
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                    T.reduce_sum(x_f32, row_sum, dim=1)

                    # Epilogue: log-normalize (log_softmax)
                    out_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    for i, j in T.Parallel(block_m, N_padded):
                        out_f32[i, j] = T.log(x_f32[i, j] / row_sum[i])

                    # Cast back to original dtype and write via shared memory
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = out_f32[i, j]
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

    return _func


# ---------------------------------------------------------------------------
# custom_op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("top::softmax_fwd", mutates_args=())
def _softmax_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _softmax_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)


@_softmax_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------


class SoftmaxKernel(Kernel):
    """Softmax / log-softmax forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Implements a 2-pass online softmax algorithm.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "softmax", "log_softmax".
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if op_kind not in ("softmax", "log_softmax"):
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of 'softmax', 'log_softmax'."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self.kernel = _softmax_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget."""
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_m = 1
        for bm in [1, 2, 4, 8, 16]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax/log_softmax kernel.

        Args:
            x: Input tensor of shape (M, N_padded).

        Returns:
            Output tensor of shape (M, N_padded).
        """
        return _softmax_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
