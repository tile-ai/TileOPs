import itertools
import math
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["GeluAndMulKernel"]

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # ~0.7978845608
GELU_COEFF = 0.044715
INV_SQRT_2 = 1.0 / math.sqrt(2.0)  # ~0.7071067812


def _gelu_and_mul_tanh_kernel(M: int, N: int, dtype: str = 'float16') -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gelu_and_mul_tanh_func(block_m: int, threads: int) -> Callable:

        @T.macro
        def gelu_and_mul_tanh_compute(
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

                T.copy(x[bx * block_m, 0], x_shared)
                T.copy(x_shared, x_local)
                T.copy(gate[bx * block_m, 0], g_shared)
                T.copy(g_shared, g_local)

                # y = gelu_tanh(x) * gate
                # gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                for i, j in T.Parallel(block_m, N):
                    val = x_local[i, j]
                    inner = T.cast(SQRT_2_OVER_PI, accum_dtype) * (
                        val + T.cast(GELU_COEFF, accum_dtype) * val * val * val)
                    y_local[i, j] = T.cast(
                        0.5 * val * (1.0 + T.tanh(inner)) * g_local[i, j],
                        dtype)

                T.copy(y_local, y_shared)
                T.copy(y_shared, y[bx * block_m, 0])

        @T.prim_func
        def _gelu_and_mul_tanh_main(
            x: T.Tensor((M, N), dtype),
            gate: T.Tensor((M, N), dtype),
            y: T.Tensor((M, N), dtype),
        ) -> None:
            gelu_and_mul_tanh_compute(x, gate, y)

        return _gelu_and_mul_tanh_main

    return _gelu_and_mul_tanh_func


def _gelu_and_mul_erf_kernel(M: int, N: int, dtype: str = 'float16') -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gelu_and_mul_erf_func(block_m: int, threads: int) -> Callable:

        @T.macro
        def gelu_and_mul_erf_compute(
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

                T.copy(x[bx * block_m, 0], x_shared)
                T.copy(x_shared, x_local)
                T.copy(gate[bx * block_m, 0], g_shared)
                T.copy(g_shared, g_local)

                # y = gelu_erf(x) * gate
                # gelu_exact(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
                for i, j in T.Parallel(block_m, N):
                    val = x_local[i, j]
                    y_local[i, j] = T.cast(
                        0.5 * val * (1.0 + T.erf(
                            val * T.cast(INV_SQRT_2, accum_dtype)))
                        * g_local[i, j],
                        dtype)

                T.copy(y_local, y_shared)
                T.copy(y_shared, y[bx * block_m, 0])

        @T.prim_func
        def _gelu_and_mul_erf_main(
            x: T.Tensor((M, N), dtype),
            gate: T.Tensor((M, N), dtype),
            y: T.Tensor((M, N), dtype),
        ) -> None:
            gelu_and_mul_erf_compute(x, gate, y)

        return _gelu_and_mul_erf_main

    return _gelu_and_mul_erf_func


@torch.library.custom_op("top::gelu_and_mul_tanh_wrapped_kernel", mutates_args=())
def _gelu_and_mul_tanh_wrapped_kernel(
    M: int, N: int, dtype: str, block_m: int, threads: int,
    x: torch.Tensor, gate: torch.Tensor,
) -> torch.Tensor:
    return _gelu_and_mul_tanh_kernel(M, N, dtype)(block_m, threads)(x, gate)


@_gelu_and_mul_tanh_wrapped_kernel.register_fake
def _(M: int, N: int, dtype: str, block_m: int, threads: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((M, N), dtype=inputs[0].dtype, device=inputs[0].device)


@torch.library.custom_op("top::gelu_and_mul_erf_wrapped_kernel", mutates_args=())
def _gelu_and_mul_erf_wrapped_kernel(
    M: int, N: int, dtype: str, block_m: int, threads: int,
    x: torch.Tensor, gate: torch.Tensor,
) -> torch.Tensor:
    return _gelu_and_mul_erf_kernel(M, N, dtype)(block_m, threads)(x, gate)


@_gelu_and_mul_erf_wrapped_kernel.register_fake
def _(M: int, N: int, dtype: str, block_m: int, threads: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((M, N), dtype=inputs[0].dtype, device=inputs[0].device)


class GeluAndMulKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 M: int,
                 N: int,
                 dtype: torch.dtype,
                 approximate: str = 'tanh',
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.dtype = dtype
        self.approximate = approximate

        if approximate == 'tanh':
            self.kernel = _gelu_and_mul_tanh_kernel(M, N, self.dtype_str)
            self._wrapped = _gelu_and_mul_tanh_wrapped_kernel
        elif approximate == 'none':
            self.kernel = _gelu_and_mul_erf_kernel(M, N, self.dtype_str)
            self._wrapped = _gelu_and_mul_erf_wrapped_kernel
        else:
            raise ValueError(f"Unknown approximate mode: {approximate}")

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

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return self._wrapped(
            self.M, self.N, self.dtype_str,
            self.config["block_m"], self.config["threads"],
            x, gate,
        )
