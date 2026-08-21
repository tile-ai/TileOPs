"""Fused M=1 W4A16 decode kernel: nibble unpack, affine dequant, and GEMV."""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.gemm_call import GemmCall
from tileops.kernels.kernel_base import Kernel

GROUP_SIZE = 128

#: Packed bytes each carried partial sum covers. Four keeps the accumulator
#: fragment small enough to stay in registers at the tile sizes that saturate
#: the weight stream.
BYTES_PER_SLOT = 4

__all__ = ["GemmW4A16DecodeKernel"]


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_decode_kernel(n: int, k: int, dtype: str) -> Callable:
    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_n: int = 32,
        block_k: int = 512,
        threads: int = 128,
        num_stages: int = 4,
    ) -> Callable:
        if block_k % GROUP_SIZE != 0:
            raise ValueError(f"block_k={block_k} must be a multiple of {GROUP_SIZE}")
        packed_k = block_k // 2
        tile_groups = block_k // GROUP_SIZE

        @T.prim_func
        def main(
            activation: T.Tensor((1, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 2), "uint8"),  # type: ignore
            weight_scale: T.Tensor((n, k // GROUP_SIZE), "float32"),  # type: ignore
            weight_zero: T.Tensor((n, k // GROUP_SIZE), "uint8"),  # type: ignore
            output: T.Tensor((1, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), threads=threads) as bx:
                activation_shared = T.alloc_shared((1, block_k), dtype)
                packed_shared = T.alloc_shared((block_n, packed_k), "uint8")
                scale_shared = T.alloc_shared((block_n, tile_groups), "float32")
                zero_shared = T.alloc_shared((block_n, tile_groups), "uint8")
                # Partial sums carried across the whole K loop, so the
                # cross-thread reduction runs once instead of once per tile.
                products = T.alloc_fragment((block_n, packed_k // BYTES_PER_SLOT), "float")
                partial = T.alloc_fragment((block_n,), "float")

                n_start = bx * block_n
                T.clear(products)

                for kk in T.Pipelined(k // block_k, num_stages=num_stages):
                    for j in T.Parallel(block_k):
                        activation_shared[0, j] = activation[0, kk * block_k + j]
                    if n % block_n == 0:
                        T.copy(
                            packed_weight[
                                n_start : n_start + block_n,
                                kk * packed_k : (kk + 1) * packed_k,
                            ],
                            packed_shared,
                        )
                        T.copy(
                            weight_scale[
                                n_start : n_start + block_n,
                                kk * tile_groups : (kk + 1) * tile_groups,
                            ],
                            scale_shared,
                        )
                        T.copy(
                            weight_zero[
                                n_start : n_start + block_n,
                                kk * tile_groups : (kk + 1) * tile_groups,
                            ],
                            zero_shared,
                        )
                    else:
                        for i, j in T.Parallel(block_n, packed_k):
                            packed_shared[i, j] = T.if_then_else(
                                n_start + i < n,
                                packed_weight[n_start + i, kk * packed_k + j],
                                T.cast(0, "uint8"),
                            )
                        for i, g in T.Parallel(block_n, tile_groups):
                            scale_shared[i, g] = T.if_then_else(
                                n_start + i < n,
                                weight_scale[n_start + i, kk * tile_groups + g],
                                T.cast(0, "float32"),
                            )
                            zero_shared[i, g] = T.if_then_else(
                                n_start + i < n,
                                weight_zero[n_start + i, kk * tile_groups + g],
                                T.cast(0, "uint8"),
                            )

                    for i, j in T.Parallel(block_n, packed_k // BYTES_PER_SLOT):
                        for v in T.serial(BYTES_PER_SLOT):
                            byte_k = j * BYTES_PER_SLOT + v
                            byte = T.cast(packed_shared[i, byte_k], "int32")
                            scale = scale_shared[i, byte_k // (GROUP_SIZE // 2)]
                            zero = T.cast(zero_shared[i, byte_k // (GROUP_SIZE // 2)], "float")
                            # Round to the storage dtype before the product: the
                            # contract dequantizes to A16 and only then multiplies.
                            low = T.cast((T.cast(byte % 16, "float") - zero) * scale, dtype)
                            high = T.cast((T.cast(byte // 16, "float") - zero) * scale, dtype)
                            products[i, j] += T.cast(
                                activation_shared[0, byte_k * 2], "float"
                            ) * T.cast(low, "float") + T.cast(
                                activation_shared[0, byte_k * 2 + 1], "float"
                            ) * T.cast(high, "float")

                T.reduce_sum(products, partial, dim=1)
                for i in T.Parallel(block_n):
                    if n_start + i < n:
                        output[0, n_start + i] = T.cast(partial[i], dtype)

        return main

    return build


class GemmW4A16DecodeKernel(Kernel):
    """Fuse nibble unpacking, affine dequantization, and the M=1 GEMV."""

    #: ``packed_weight`` and ``weight_zero`` are uint8 payloads the kernel
    #: unpacks; the K loop is ``k // block_k``, so no value decides how much work
    #: a candidate runs.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        return call.m == 1

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
        self.group_size = group_size
        self.kernel = _gemm_w4a16_decode_kernel(n, k, self.dtype_str)
        self.init_config(config, tune)
        # block_k must tile K exactly; K is a multiple of GROUP_SIZE, so halving
        # a power-of-two block_k always lands on a divisor.
        while k % self.config["block_k"]:
            self.config["block_k"] //= 2

    @property
    def default_config(self) -> dict:
        # Tuned on H200 over the manifest's four M=1 workloads: 32 rows of N per
        # CTA still leaves 224 CTAs at the smallest N, and 128 threads keeps the
        # carried accumulator in registers.
        return {"block_n": 32, "block_k": 512, "threads": 128, "num_stages": 4}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_n": block_n, "block_k": block_k, "threads": threads, "num_stages": num_stages}
            for block_n in (8, 16, 32)
            for block_k in (256, 512)
            for threads in (128, 256)
            for num_stages in (2, 4)
            # Tile K exactly, and keep the carried accumulator off the stack.
            if self.k % block_k == 0
            and threads <= block_n * block_k // (2 * BYTES_PER_SLOT) <= threads * 32
        ]

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        compiled = _gemm_w4a16_decode_kernel(self.n, self.k, self.dtype_str)(**self.config)
        return compiled(activation, packed_weight, weight_scale, weight_zero)
