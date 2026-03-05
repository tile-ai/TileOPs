import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SoftmaxKernel"]


def _softmax_kernel(M: int, N: int, actual_n: int,
                    dtype: str = 'float16') -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _softmax_func(block_m: int, threads: int) -> Callable:

        @T.macro
        def softmax_compute(
            x: T.Buffer((M, N), dtype),
            y: T.Buffer((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as bx:
                x_shared = T.alloc_shared((block_m, N), dtype)
                x_local = T.alloc_fragment((block_m, N), accum_dtype)
                # For max reduction
                max_local = T.alloc_fragment((block_m,), accum_dtype)
                # For exp and sum
                exp_local = T.alloc_fragment((block_m, N), accum_dtype)
                sum_local = T.alloc_fragment((block_m,), accum_dtype)
                # Output
                y_local = T.alloc_fragment((block_m, N), dtype)
                y_shared = T.alloc_shared((block_m, N), dtype)

                # Load input rows
                T.copy(x[bx * block_m, 0], x_shared)
                T.copy(x_shared, x_local)

                # Pass 1: Find max per row
                T.fill(max_local, -T.infinity(accum_dtype))
                T.reduce_max(x_local, max_local, dim=1, clear=False)

                # Pass 2: Compute exp(x - max) and sum
                for i, j in T.Parallel(block_m, N):
                    exp_local[i, j] = T.exp(x_local[i, j] - max_local[i])

                T.fill(sum_local, 0.0)
                T.reduce_sum(exp_local, sum_local, dim=1)

                # Compute y = exp(x - max) / sum
                # Cast to dtype at the end (matches reference: compute in fp32,
                # then cast)
                for i, j in T.Parallel(block_m, N):
                    y_local[i, j] = T.cast(
                        exp_local[i, j] / sum_local[i], dtype)

                T.copy(y_local, y_shared)
                T.copy(y_shared, y[bx * block_m, 0])

        @T.prim_func
        def _softmax_main(
            x: T.Tensor((M, N), dtype),
            y: T.Tensor((M, N), dtype),
        ) -> None:
            softmax_compute(x, y)

        return _softmax_main

    return _softmax_func


@torch.library.custom_op("top::softmax_wrapped_kernel", mutates_args=())
def _softmax_wrapped_kernel(
    M: int,
    N: int,
    actual_n: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _softmax_kernel(M, N, actual_n, dtype)(block_m, threads)(x)


@_softmax_wrapped_kernel.register_fake
def _(M: int, N: int, actual_n: int, dtype: str, block_m: int, threads: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((M, N), dtype=inputs[0].dtype, device=inputs[0].device)


class SoftmaxKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 M: int,
                 N: int,
                 actual_n: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.actual_n = actual_n
        self.dtype = dtype

        self.kernel = _softmax_kernel(M, N, actual_n, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 4,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [1, 2, 4]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, threads))
        return [{"block_m": c[0], "threads": c[1]} for c in _configs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _softmax_wrapped_kernel(
            self.M, self.N, self.actual_n, self.dtype_str,
            self.config["block_m"], self.config["threads"],
            x,
        )
