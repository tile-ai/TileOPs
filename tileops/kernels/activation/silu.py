import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SiluKernel"]


def _silu_kernel(M: int, N: int, dtype: str = 'float16') -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _silu_func(block_m: int, threads: int) -> Callable:

        @T.macro
        def silu_compute(
            x: T.Buffer((M, N), dtype),
            y: T.Buffer((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as bx:
                x_shared = T.alloc_shared((block_m, N), dtype)
                x_local = T.alloc_fragment((block_m, N), accum_dtype)
                y_local = T.alloc_fragment((block_m, N), dtype)
                y_shared = T.alloc_shared((block_m, N), dtype)

                # Load input rows
                T.copy(x[bx * block_m, 0], x_shared)
                T.copy(x_shared, x_local)

                # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                for i, j in T.Parallel(block_m, N):
                    y_local[i, j] = T.cast(
                        x_local[i, j] / (1.0 + T.exp(-x_local[i, j])),
                        dtype)

                T.copy(y_local, y_shared)
                T.copy(y_shared, y[bx * block_m, 0])

        @T.prim_func
        def _silu_main(
            x: T.Tensor((M, N), dtype),
            y: T.Tensor((M, N), dtype),
        ) -> None:
            silu_compute(x, y)

        return _silu_main

    return _silu_func


@torch.library.custom_op("top::silu_wrapped_kernel", mutates_args=())
def _silu_wrapped_kernel(
    M: int,
    N: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _silu_kernel(M, N, dtype)(block_m, threads)(x)


@_silu_wrapped_kernel.register_fake
def _(M: int, N: int, dtype: str, block_m: int, threads: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((M, N), dtype=inputs[0].dtype, device=inputs[0].device)


class SiluKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 M: int,
                 N: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.dtype = dtype

        self.kernel = _silu_kernel(M, N, self.dtype_str)
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
        return _silu_wrapped_kernel(
            self.M, self.N, self.dtype_str,
            self.config["block_m"], self.config["threads"],
            x,
        )
