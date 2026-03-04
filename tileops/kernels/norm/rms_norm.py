import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["RmsNormKernel"]


def _rms_norm_kernel(M: int, N: int, actual_n: int, eps: float, dtype: str = 'float16') -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _rms_norm_func(block_m: int, threads: int) -> Callable:

        @T.macro
        def rms_norm_compute(
            x: T.Buffer((M, N), dtype),
            weight: T.Buffer((N,), dtype),
            y: T.Buffer((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as bx:
                x_shared = T.alloc_shared((block_m, N), dtype)
                x_local = T.alloc_fragment((block_m, N), accum_dtype)
                sq_local = T.alloc_fragment((block_m, N), accum_dtype)
                sq_sum_local = T.alloc_fragment((block_m,), accum_dtype)
                rrms_local = T.alloc_fragment((block_m,), accum_dtype)
                w_shared = T.alloc_shared((N,), dtype)
                w_local = T.alloc_fragment((N,), accum_dtype)
                y_local = T.alloc_fragment((block_m, N), dtype)
                y_shared = T.alloc_shared((block_m, N), dtype)

                # Load input rows
                T.copy(x[bx * block_m, 0], x_shared)
                T.copy(x_shared, x_local)

                # Load weight
                T.copy(weight[0], w_shared)
                T.copy(w_shared, w_local)

                # Compute sum of squares per row
                for i, j in T.Parallel(block_m, N):
                    sq_local[i, j] = x_local[i, j] * x_local[i, j]

                T.fill(sq_sum_local, 0.0)
                T.reduce_sum(sq_local, sq_sum_local, dim=1)

                # Compute rrms = rsqrt(mean(x^2) + eps)
                # Divide by actual_n (not padded N) for correct mean
                for i in T.Parallel(block_m):
                    rrms_local[i] = T.rsqrt(
                        sq_sum_local[i] / T.cast(actual_n, accum_dtype) + T.cast(eps, accum_dtype))

                # Compute y = x * rrms * weight
                for i, j in T.Parallel(block_m, N):
                    y_local[i, j] = T.cast(x_local[i, j] * rrms_local[i] * w_local[j], dtype)

                T.copy(y_local, y_shared)
                T.copy(y_shared, y[bx * block_m, 0])

        @T.prim_func
        def _rms_norm_main(
            x: T.Tensor((M, N), dtype),
            weight: T.Tensor((N,), dtype),
            y: T.Tensor((M, N), dtype),
        ) -> None:
            rms_norm_compute(x, weight, y)

        return _rms_norm_main

    return _rms_norm_func


@torch.library.custom_op("top::rms_norm_wrapped_kernel", mutates_args=())
def _rms_norm_wrapped_kernel(
    M: int,
    N: int,
    actual_n: int,
    eps: float,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return _rms_norm_kernel(M, N, actual_n, eps, dtype)(block_m, threads)(x, weight)


@_rms_norm_wrapped_kernel.register_fake
def _(M: int, N: int, actual_n: int, eps: float, dtype: str, block_m: int, threads: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((M, N), dtype=inputs[0].dtype, device=inputs[0].device)


class RmsNormKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 M: int,
                 N: int,
                 actual_n: int,
                 eps: float,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.actual_n = actual_n
        self.eps = eps
        self.dtype = dtype

        self.kernel = _rms_norm_kernel(M, N, actual_n, eps, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 4,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [1, 2, 4, 8]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, threads))
        return [{"block_m": c[0], "threads": c[1]} for c in _configs]

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _rms_norm_wrapped_kernel(
            self.M, self.N, self.actual_n, self.eps, self.dtype_str,
            self.config["block_m"], self.config["threads"],
            x, weight,
        )
