import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["LayerNormKernel"]


def _layer_norm_kernel(M: int, N: int, actual_n: int, eps: float,
                       dtype: str = 'float16') -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _layer_norm_func(block_m: int, threads: int) -> Callable:

        @T.macro
        def layer_norm_compute(
            x: T.Buffer((M, N), dtype),
            weight: T.Buffer((N,), dtype),
            bias: T.Buffer((N,), dtype),
            y: T.Buffer((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as bx:
                x_shared = T.alloc_shared((block_m, N), dtype)
                x_local = T.alloc_fragment((block_m, N), accum_dtype)
                # For mean reduction
                sum_local = T.alloc_fragment((block_m, N), accum_dtype)
                mean_local = T.alloc_fragment((block_m,), accum_dtype)
                # For variance reduction
                diff_local = T.alloc_fragment((block_m, N), accum_dtype)
                diff_sq_local = T.alloc_fragment((block_m, N), accum_dtype)
                var_sum_local = T.alloc_fragment((block_m,), accum_dtype)
                rstd_local = T.alloc_fragment((block_m,), accum_dtype)
                # Weight and bias (keep in original dtype to match reference cast order)
                w_shared = T.alloc_shared((N,), dtype)
                w_local = T.alloc_fragment((N,), dtype)
                b_shared = T.alloc_shared((N,), dtype)
                b_local = T.alloc_fragment((N,), dtype)
                # Normalized intermediate (cast to dtype before weight/bias)
                norm_local = T.alloc_fragment((block_m, N), dtype)
                # Output
                y_local = T.alloc_fragment((block_m, N), dtype)
                y_shared = T.alloc_shared((block_m, N), dtype)

                # Load input rows
                T.copy(x[bx * block_m, 0], x_shared)
                T.copy(x_shared, x_local)

                # Load weight and bias
                T.copy(weight[0], w_shared)
                T.copy(w_shared, w_local)
                T.copy(bias[0], b_shared)
                T.copy(b_shared, b_local)

                # Pass 1: Compute mean per row
                T.copy(x_shared, sum_local)
                T.fill(mean_local, 0.0)
                T.reduce_sum(sum_local, mean_local, dim=1)
                for i in T.Parallel(block_m):
                    mean_local[i] = mean_local[i] / T.cast(actual_n, accum_dtype)

                # Compute (x - mean)
                for i, j in T.Parallel(block_m, N):
                    diff_local[i, j] = x_local[i, j] - mean_local[i]

                # Pass 2: Compute variance per row = mean((x - mean)^2)
                for i, j in T.Parallel(block_m, N):
                    diff_sq_local[i, j] = diff_local[i, j] * diff_local[i, j]

                T.fill(var_sum_local, 0.0)
                T.reduce_sum(diff_sq_local, var_sum_local, dim=1)

                # rstd = rsqrt(var + eps)
                for i in T.Parallel(block_m):
                    rstd_local[i] = T.rsqrt(
                        var_sum_local[i] / T.cast(actual_n, accum_dtype)
                        + T.cast(eps, accum_dtype))

                # y = ((x - mean) * rstd).to(dtype) * weight + bias
                # Cast normalized value to dtype first, then apply weight/bias
                # in dtype precision (matches PyTorch reference computation order)
                for i, j in T.Parallel(block_m, N):
                    norm_local[i, j] = T.cast(
                        diff_local[i, j] * rstd_local[i], dtype)

                for i, j in T.Parallel(block_m, N):
                    y_local[i, j] = norm_local[i, j] * w_local[j] + b_local[j]

                T.copy(y_local, y_shared)
                T.copy(y_shared, y[bx * block_m, 0])

        @T.prim_func
        def _layer_norm_main(
            x: T.Tensor((M, N), dtype),
            weight: T.Tensor((N,), dtype),
            bias: T.Tensor((N,), dtype),
            y: T.Tensor((M, N), dtype),
        ) -> None:
            layer_norm_compute(x, weight, bias, y)

        return _layer_norm_main

    return _layer_norm_func


@torch.library.custom_op("top::layer_norm_wrapped_kernel", mutates_args=())
def _layer_norm_wrapped_kernel(
    M: int,
    N: int,
    actual_n: int,
    eps: float,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _layer_norm_kernel(M, N, actual_n, eps, dtype)(block_m, threads)(
        x, weight, bias)


@_layer_norm_wrapped_kernel.register_fake
def _(M: int, N: int, actual_n: int, eps: float, dtype: str, block_m: int,
      threads: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((M, N), dtype=inputs[0].dtype, device=inputs[0].device)


class LayerNormKernel(Kernel):
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

        self.kernel = _layer_norm_kernel(M, N, actual_n, eps, self.dtype_str)
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

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        return _layer_norm_wrapped_kernel(
            self.M, self.N, self.actual_n, self.eps, self.dtype_str,
            self.config["block_m"], self.config["threads"],
            x, weight, bias,
        )
