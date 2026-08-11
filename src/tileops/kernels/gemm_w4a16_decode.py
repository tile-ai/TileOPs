"""Fused M=1 W4A16 decode kernel derived from the authored HIR."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["GemmW4A16DecodeKernel"]

GROUP_SIZE = 128


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_kernel(m: int, n: int, k: int, dtype: str):
    @tilelang.jit(out_idx=[4])
    def _func(block_n: int, threads: int):
        block_k = GROUP_SIZE

        @T.prim_func
        def main(
            activation: T.Tensor[(m, k), dtype],
            packed_weight: T.Tensor[(n, k // 2), "uint8"],  # noqa: F821
            weight_scale: T.Tensor[(n, k // GROUP_SIZE), "float32"],  # noqa: F821
            weight_zero: T.Tensor[(n, k // GROUP_SIZE), "uint8"],  # noqa: F821
            output: T.Tensor[(m, n), dtype],
        ):
            with T.Kernel(T.ceildiv(n, block_n), m, threads=threads) as (pid_n, pid_m):
                activation_shared = T.alloc_shared((block_k,), dtype)
                packed_values = T.alloc_fragment((block_n, block_k // 2), "int32")
                scales = T.alloc_fragment((block_n,), "float32")
                zeros = T.alloc_fragment((block_n,), "float32")
                products_low = T.alloc_fragment((block_n, block_k // 2), "float32")
                products_high = T.alloc_fragment((block_n, block_k // 2), "float32")
                partial_low = T.alloc_fragment((block_n,), "float32")
                partial_high = T.alloc_fragment((block_n,), "float32")
                accum = T.alloc_fragment((block_n,), "float32")

                T.fill(accum, 0.0)
                for group in T.Serial(k // block_k):
                    T.copy(activation[pid_m, group * block_k], activation_shared)
                    for row in T.Parallel(block_n):
                        scales[row] = T.if_then_else(
                            pid_n * block_n + row < n,
                            weight_scale[pid_n * block_n + row, group],
                            0.0,
                        )
                        zeros[row] = T.if_then_else(
                            pid_n * block_n + row < n,
                            T.cast(weight_zero[pid_n * block_n + row, group], "float32"),
                            0.0,
                        )
                    for row, packed_k in T.Parallel(block_n, block_k // 2):
                        packed_values[row, packed_k] = T.if_then_else(
                            pid_n * block_n + row < n,
                            T.cast(
                                packed_weight[
                                    pid_n * block_n + row,
                                    group * (block_k // 2) + packed_k,
                                ],
                                "int32",
                            ),
                            0,
                        )
                    for row, packed_k in T.Parallel(block_n, block_k // 2):
                        low = packed_values[row, packed_k] % 16
                        dequant_low = T.cast(
                            (T.cast(low, "float32") - zeros[row]) * scales[row],
                            dtype,
                        )
                        products_low[row, packed_k] = T.cast(
                            activation_shared[packed_k * 2], "float32"
                        ) * T.cast(dequant_low, "float32")
                    for row, packed_k in T.Parallel(block_n, block_k // 2):
                        high = packed_values[row, packed_k] // 16
                        dequant_high = T.cast(
                            (T.cast(high, "float32") - zeros[row]) * scales[row],
                            dtype,
                        )
                        products_high[row, packed_k] = T.cast(
                            activation_shared[packed_k * 2 + 1], "float32"
                        ) * T.cast(dequant_high, "float32")

                    T.reduce_sum(products_low, partial_low, dim=1)
                    T.reduce_sum(products_high, partial_high, dim=1)
                    for row in T.Parallel(block_n):
                        accum[row] = accum[row] + partial_low[row] + partial_high[row]

                for row in T.Parallel(block_n):
                    if pid_n * block_n + row < n:
                        output[pid_m, pid_n * block_n + row] = T.cast(accum[row], dtype)

        return main

    return _func


@torch.library.custom_op("top::gemm_w4a16_decode_fwd", mutates_args=())
def _gemm_w4a16_wrapped(
    m: int,
    n: int,
    k: int,
    dtype_str: str,
    block_n: int,
    threads: int,
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero: torch.Tensor,
) -> torch.Tensor:
    return _gemm_w4a16_kernel(m, n, k, dtype_str)(block_n, threads)(
        activation, packed_weight, weight_scale, weight_zero
    )


@_gemm_w4a16_wrapped.register_fake
def _(
    m,
    n,
    k,
    dtype_str,
    block_n,
    threads,
    activation,
    packed_weight,
    weight_scale,
    weight_zero,
):
    del k, dtype_str, block_n, threads, packed_weight, weight_scale, weight_zero
    return torch.empty((m, n), dtype=activation.dtype, device=activation.device)


class GemmW4A16DecodeKernel(Kernel):
    """Fuse nibble unpacking, affine dequantization, and M=1 GEMV."""

    supported_archs = [80, 86, 89, 90]

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
        if m != 1:
            raise ValueError(f"GemmW4A16DecodeKernel requires M=1, got {m}")
        if group_size != GROUP_SIZE:
            raise ValueError(f"only group_size={GROUP_SIZE} is supported")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.kernel = _gemm_w4a16_kernel(m, n, k, self.dtype_str)
        self.init_config(config, tune=tune)

    @property
    def default_config(self) -> dict:
        return {"block_n": 16, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_n": block_n, "threads": threads}
            for block_n in (4, 8, 16)
            for threads in (128, 256)
        ]

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        return _gemm_w4a16_wrapped(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.config["block_n"],
            self.config["threads"],
            activation,
            packed_weight,
            weight_scale,
            weight_zero,
        )
