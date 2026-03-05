import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SiluAndMulKernel"]


def _silu_and_mul_kernel(M: int, N: int, dtype: str = 'float16') -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _silu_and_mul_func(block_m: int, threads: int) -> Callable:

        @T.macro
        def silu_and_mul_compute(
            x: T.Buffer((M, N), dtype),
            gate: T.Buffer((M, N), dtype),
            y: T.Buffer((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as bx:
                x_shared = T.alloc_shared((block_m, N), dtype)
                x_local = T.alloc_fragment((block_m, N), accum_dtype)
                g_shared = T.alloc_shared((block_m, N), dtype)
                g_local = T.alloc_fragment((block_m, N), accum_dtype)
                y_local = T.alloc_fragment((block_m, N), dtype)
                y_shared = T.alloc_shared((block_m, N), dtype)

                # Load input rows
                T.copy(x[bx * block_m, 0], x_shared)
                T.copy(x_shared, x_local)
                T.copy(gate[bx * block_m, 0], g_shared)
                T.copy(g_shared, g_local)

                # y = silu(x) * gate = (x / (1 + exp(-x))) * gate
                for i, j in T.Parallel(block_m, N):
                    y_local[i, j] = T.cast(
                        x_local[i, j] / (1.0 + T.exp(-x_local[i, j]))
                        * g_local[i, j],
                        dtype)

                T.copy(y_local, y_shared)
                T.copy(y_shared, y[bx * block_m, 0])

        @T.prim_func
        def _silu_and_mul_main(
            x: T.Tensor((M, N), dtype),
            gate: T.Tensor((M, N), dtype),
            y: T.Tensor((M, N), dtype),
        ) -> None:
            silu_and_mul_compute(x, gate, y)

        return _silu_and_mul_main

    return _silu_and_mul_func


@torch.library.custom_op("top::silu_and_mul_wrapped_kernel", mutates_args=())
def _silu_and_mul_wrapped_kernel(
    M: int,
    N: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return _silu_and_mul_kernel(M, N, dtype)(block_m, threads)(x, gate)


@_silu_and_mul_wrapped_kernel.register_fake
def _(M: int, N: int, dtype: str, block_m: int, threads: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((M, N), dtype=inputs[0].dtype, device=inputs[0].device)


class SiluAndMulKernel(Kernel):
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

        self.kernel = _silu_and_mul_kernel(M, N, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # 3 shared buffers (x, gate, y) × block_m × N × dtype_size must fit
        # in SM shared memory. Use block_m=1 for large N to avoid spilling.
        block_m = 1 if self.N > 4096 else 4
        return {
            "block_m": block_m,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [1, 2, 4]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, threads))
        return [{"block_m": c[0], "threads": c[1]} for c in _configs]

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return _silu_and_mul_wrapped_kernel(
            self.M, self.N, self.dtype_str,
            self.config["block_m"], self.config["threads"],
            x, gate,
        )
