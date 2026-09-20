"""Fused M=1 W4A16 decode kernel: nibble unpack, affine dequant, and GEMV."""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from .call_spec import GemmCall
from .w4a16 import GROUP_SIZE

# Four packed bytes encode eight weights. Sixteen adjacent lanes therefore
# cover one group128 and can share its sixteen possible dequantized values.
BYTES_PER_SLOT = 4

__all__ = ["GemmW4A16GemvKernel"]


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_gemv_kernel(n: int, k: int, dtype: str) -> Callable:
    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            # Keep lookup entries and carried sums in the same threads;
            # automatic warp specialization separates this thread-local state.
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_n: int = 32,
        block_k: int = 1024,
        threads: int = 128,
        num_stages: int = 2,
        warp_rows: bool = False,
    ) -> Callable:
        if block_k % GROUP_SIZE != 0:
            raise ValueError(f"block_k={block_k} must be a multiple of {GROUP_SIZE}")
        if threads % 32 != 0 or (block_n * block_k // 8) % threads != 0:
            raise ValueError("decode tiles must divide into whole warps of eight-weight slots")
        if warp_rows and (block_k % 256 or block_n % (threads // 32)):
            raise ValueError("warp rows require whole 256-weight chunks and whole rows per warp")
        packed_k = block_k // 2
        tile_groups = block_k // GROUP_SIZE
        row_slots = 32 if warp_rows else block_k // 8
        carried_sums = block_n * row_slots // threads
        words = block_k // 256 if warp_rows else 1

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
                products = T.alloc_fragment((block_n, packed_k // BYTES_PER_SLOT), "float")
                partial = T.alloc_fragment((block_n,), "float")

                lookup_local = T.alloc_local((1,), "float32")
                activation_local = T.alloc_local((8,), dtype)
                packed_local = T.alloc_local((4,), "uint8")
                scale_local = T.alloc_local((1,), "float32")
                zero_local = T.alloc_local((1,), "int32")
                # Carry sums across K and reduce across threads only once.
                products_local = T.alloc_local((carried_sums,), "float32")
                tx = T.get_thread_binding(0)
                if warp_rows:
                    T.annotate_layout(
                        {packed_shared: tilelang.layout.make_swizzled_layout(packed_shared)}
                    )
                T.annotate_layout(
                    {
                        products: T.Fragment(
                            (block_n, block_k // 8),
                            forward_thread_fn=lambda i, j: (i * (block_k // 8) + j) % threads,
                            forward_index_fn=lambda i, j: (i * (block_k // 8) + j) // threads,
                        )
                    }
                )
                T.clear(products_local)
                n_start = bx * block_n

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

                    for chunk in T.serial(carried_sums):
                        index = chunk * threads + tx
                        row = index // row_slots
                        for word in T.serial(words):
                            col = index % row_slots + word * 32
                            for v in T.vectorized(8):
                                activation_local[v] = activation_shared[0, col * 8 + v]
                            for v in T.vectorized(4):
                                packed_local[v] = packed_shared[row, col * 4 + v]
                            scale_local[0] = scale_shared[row, col // 16]
                            zero_local[0] = T.cast(zero_shared[row, col // 16], "int32")
                            # Each half-warp holds the complete lookup for its
                            # group, preserving FP32 affine math and A16 rounding.
                            lookup_local[0] = T.cast(
                                T.cast(
                                    T.cast(tx % 16 - zero_local[0], "float32") * scale_local[0],
                                    dtype,
                                ),
                                "float32",
                            )
                            for v in T.unroll(8):
                                quantized = (
                                    T.cast(packed_local[v // 2], "int32") >> (4 * (v % 2))
                                ) & 15
                                dequantized = T.shfl_sync(lookup_local[0], quantized, width=16)
                                products_local[chunk] += (
                                    T.cast(activation_local[v], "float32") * dequantized
                                )
                if warp_rows:
                    # Each row stays in one warp throughout K. This avoids the
                    # shared-memory exchange needed by a row spanning warps.
                    for chunk in T.serial(carried_sums):
                        for shift in T.unroll(5):
                            products_local[chunk] += T.shfl_xor(products_local[chunk], 1 << shift)
                        row = chunk * (threads // 32) + tx // 32
                        if tx % 32 == 0 and n_start + row < n:
                            output[0, n_start + row] = T.cast(products_local[chunk], dtype)
                else:
                    for chunk in T.serial(carried_sums):
                        index = chunk * threads + tx
                        products[index // (block_k // 8), index % (block_k // 8)] = products_local[
                            chunk
                        ]
                    T.reduce_sum(products, partial, dim=1)
                    for i in T.Parallel(block_n):
                        if n_start + i < n:
                            output[0, n_start + i] = T.cast(partial[i], dtype)

        return main

    return build


class GemmW4A16GemvKernel(Kernel):
    """Fuse nibble unpacking, affine dequantization, and the M=1 GEMV."""

    # ``packed_weight`` and ``weight_zero`` are uint8 payloads; the K loop is
    # ``k // block_k``.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        return call.m == 1

    @classmethod
    def entry_for(cls, call: GemmCall) -> tuple:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.n, call.k, call.dtype, call.group_size, call.tune, index)
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            tune=call.tune,
            group_size=call.group_size,
            device_index=index,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        group_size: int = GROUP_SIZE,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if m != 1:
            raise ValueError(f"GemmW4A16GemvKernel requires M=1, got {m}")
        if group_size != GROUP_SIZE:
            raise ValueError(f"only group_size={GROUP_SIZE} is supported")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self.kernel = _gemm_w4a16_gemv_kernel(n, k, self.dtype_str)
        self.init_config(config, tune)
        # block_k must tile K exactly; K is a multiple of GROUP_SIZE, so halving
        # a power-of-two block_k always lands on a divisor.
        while k % self.config["block_k"]:
            self.config["block_k"] //= 2

    @property
    def default_config(self) -> dict:
        # Limit the new mapping to the measured Hopper W4 case until the other
        # GEMV regions have been evaluated. Sixteen rows give 512 CTAs here.
        warp_rows = (
            self.n == 8192
            and self.k == 8192
            and self.dtype == torch.float16
            and get_sm_version(self.device_index) == 90
        )
        # 32 rows of N per CTA still leaves 224 CTAs at the manifest's smallest
        # N. A wider K tile amortizes staging the lookup metadata.
        return {
            "block_n": 16 if warp_rows else 32,
            "block_k": 1024,
            "threads": 128,
            "num_stages": 2,
            "warp_rows": warp_rows,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {
                "block_n": block_n,
                "block_k": block_k,
                "threads": threads,
                "num_stages": num_stages,
                "warp_rows": self.default_config["warp_rows"],
            }
            for block_n in (8, 16, 32)
            for block_k in (256, 512, 1024)
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
        compiled = _gemm_w4a16_gemv_kernel(self.n, self.k, self.dtype_str)(**self.config)
        return compiled(activation, packed_weight, weight_scale, weight_zero)
