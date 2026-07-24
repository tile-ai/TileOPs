import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.quantize import _tir_packed_to_unsigned_convert

from tileops.kernels.kernel_base import Kernel

GROUP_SIZE = 128

__all__ = ["GemmW4A16Kernel"]


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    decode_unsigned_int4 = _tir_packed_to_unsigned_convert("uint", 8)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        num_stages: int = 2,
        threads: int = 128,
    ) -> Callable:
        if group_size % block_k != 0:
            raise ValueError(f"group_size={group_size} must be divisible by block_k={block_k}")
        if (block_n * block_k // 2) % (threads * 4) != 0:
            raise ValueError("packed W4 tile must divide into four-byte decode chunks per thread")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 2), "uint8"),  # type: ignore
            weight_scale: T.Tensor((n, k // group_size), "float32"),  # type: ignore
            weight_zero: T.Tensor((n, k // group_size), "uint8"),  # type: ignore
            output: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n),
                T.ceildiv(m, block_m),
                threads=threads,
            ) as (bx, by):
                activation_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                packed_weight_shared = T.alloc_shared((block_n, block_k // 2), "uint8")
                weight_scale_shared = T.alloc_shared((block_n,), "float32")
                weight_zero_shared = T.alloc_shared((block_n,), "uint8")
                output_local = T.alloc_fragment((block_m, block_n), "float")
                packed_local = T.alloc_local((4,), "uint8")
                dequantized_local = T.alloc_local((8,), dtype)
                scale_local = T.alloc_local((1,), dtype)
                zero_local = T.alloc_local((1,), dtype)

                T.annotate_layout(
                    {
                        activation_shared: tilelang.layout.make_swizzled_layout(activation_shared),
                        weight_shared: tilelang.layout.make_swizzled_layout(weight_shared),
                    }
                )

                m_start = by * block_m
                n_start = bx * block_n
                tx = T.get_thread_binding(0)
                T.clear(output_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k

                    for i, j in T.Parallel(block_m, block_k):
                        activation_shared[i, j] = T.if_then_else(
                            (m_start + i < m) & (k_start + j < k),
                            activation[m_start + i, k_start + j],
                            T.cast(0, dtype),
                        )

                    if n % block_n == 0:
                        T.copy(
                            packed_weight[n_start, k_start // 2],
                            packed_weight_shared,
                        )
                    else:
                        for i, j in T.Parallel(block_n, block_k // 2):
                            packed_weight_shared[i, j] = T.if_then_else(
                                n_start + i < n,
                                packed_weight[n_start + i, k_start // 2 + j],
                                T.cast(0, "uint8"),
                            )

                    for i in T.Parallel(block_n):
                        weight_scale_shared[i] = T.if_then_else(
                            n_start + i < n,
                            weight_scale[n_start + i, k_start // group_size],
                            T.cast(0, "float32"),
                        )
                        weight_zero_shared[i] = T.if_then_else(
                            n_start + i < n,
                            weight_zero[n_start + i, k_start // group_size],
                            T.cast(0, "uint8"),
                        )

                    for chunk in T.serial(block_n * block_k // 2 // (threads * 4)):
                        packed_base = chunk * threads * 4 + tx * 4
                        scale_row = packed_base // (block_k // 2)
                        scale_local[0] = T.cast(weight_scale_shared[scale_row], dtype)
                        zero_local[0] = T.cast(weight_zero_shared[scale_row], dtype)

                        for v in T.vectorized(4):
                            packed_index = packed_base + v
                            packed_local[v] = packed_weight_shared[
                                packed_index // (block_k // 2),
                                packed_index % (block_k // 2),
                            ]

                        for v in T.serial(8):
                            dequantized_local[v] = (
                                T.cast(
                                    decode_unsigned_int4(
                                        4,
                                        packed_local[v // 2],
                                        v % 2,
                                        dtype,
                                    ),
                                    dtype,
                                )
                                - zero_local[0]
                            ) * scale_local[0]

                        for v in T.vectorized(8):
                            output_index = chunk * threads * 8 + tx * 8 + v
                            weight_shared[
                                output_index // block_k,
                                output_index % block_k,
                            ] = dequantized_local[v]

                    T.gemm(
                        activation_shared,
                        weight_shared,
                        output_local,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        output[m_start + i, n_start + j] = output_local[i, j]

        return main

    return build


class GemmW4A16Kernel(Kernel):
    """W4A16 GEMM with group128 affine dequantization and A16 Tensor Core GEMM."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        group_size: int = GROUP_SIZE,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self.kernel = _gemm_w4a16_kernel(m, n, k, self.dtype_str, group_size)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 2,
            "threads": 128,
        }

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        compiled = _gemm_w4a16_kernel(self.m, self.n, self.k, self.dtype_str, self.group_size)(
            **self.config
        )
        return compiled(activation, packed_weight, weight_scale, weight_zero)
