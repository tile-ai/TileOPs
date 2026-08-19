import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.gemm_call import GemmCall
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantize_utils import (
    UINT4_TO_FP16_LOP3_SOURCE,
    _tir_packed_to_unsigned_convert,
)

GROUP_SIZE = 128

__all__ = ["GemmW4A16DecodeKernel", "GemmW4A16Kernel"]


def _plain_packed_activation_index(index):
    """Map LOP3 output lanes back to the public low/high-nibble order."""
    chunk = index // 8 * 8
    within = index % 8
    return chunk + within // 2 + (within % 2) * 4


def _decode_schedule_index(index, size: int, mode: int):
    """Compile-time traversal orders retained from the decode AKO campaign."""
    if mode == 0:
        return index
    if mode == 5:
        bits = (size - 1).bit_length()
        result = index & 1
        for bit in range(1, bits):
            result = (result << 1) | ((index >> bit) & 1)
        return result
    if mode == 7:
        reverse = size - 1 - index
        return reverse ^ (reverse >> 1)
    raise ValueError(f"unsupported decode schedule mode {mode}")


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_decode_direct_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    """M=1 register-dequant GEMV over the public packed-weight layout."""
    if m != 1:
        raise ValueError(f"W4A16 decode requires M=1, got {m}")
    if dtype != "float16":
        raise ValueError(f"W4A16 decode requires float16, got {dtype}")
    if k % 256 != 0:
        raise ValueError(f"W4A16 decode requires K divisible by 256, got {k}")

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def build(
        n_partition: int = 1,
        split_k_warps: int = 1,
        outputs_per_warp: int = 1,
    ) -> Callable:
        reduce_threads = 32
        values_per_thread = 8
        packed_per_thread = values_per_thread // 2
        block_k = reduce_threads * values_per_thread
        split_block_k = block_k * split_k_warps
        total_threads = reduce_threads * n_partition * split_k_warps

        if split_k_warps not in (1, 2, 4, 8):
            raise ValueError("split_k_warps must be 1, 2, 4, or 8")
        if outputs_per_warp not in (1, 2, 4, 8):
            raise ValueError("outputs_per_warp must be 1, 2, 4, or 8")
        if total_threads > 1024:
            raise ValueError(f"decode launch requests {total_threads} threads")
        if k % split_block_k != 0:
            raise ValueError(f"K={k} must be divisible by {split_block_k}")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 2), "uint8"),  # type: ignore
            weight_scale: T.Tensor((n, k // group_size), "float32"),  # type: ignore
            weight_zero: T.Tensor((n, k // group_size), "uint8"),  # type: ignore
            output: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, n_partition * outputs_per_warp),
                threads=(reduce_threads, n_partition * split_k_warps),
            ) as bx:
                activation_local = T.alloc_local((values_per_thread,), dtype)
                packed_local = T.alloc_local((packed_per_thread,), "uint8")
                decoded_local = T.alloc_local((values_per_thread,), dtype)
                accumulator = T.alloc_local((outputs_per_warp,), "float32")
                reduced = T.alloc_local((outputs_per_warp,), "float32")
                scale_local = T.alloc_local((1,), dtype)
                zero_local = T.alloc_local((1,), dtype)
                partials = T.alloc_shared((n_partition, outputs_per_warp, split_k_warps), "float32")

                lane = T.thread_binding(0, reduce_threads, thread="threadIdx.x")
                warp_slot = T.thread_binding(
                    0,
                    n_partition * split_k_warps,
                    thread="threadIdx.y",
                )
                n_slot = warp_slot // split_k_warps
                k_partition = warp_slot % split_k_warps
                output_col_base = bx * n_partition * outputs_per_warp + n_slot * outputs_per_warp

                T.import_source(UINT4_TO_FP16_LOP3_SOURCE)
                T.clear(accumulator)

                for ko in T.serial(k // split_block_k):
                    logical_k = (
                        ko * split_k_warps + k_partition
                    ) * block_k + lane * values_per_thread
                    for v in T.vectorized(values_per_thread):
                        activation_local[v] = activation[0, logical_k + v]

                    for output_slot in T.serial(outputs_per_warp):
                        output_col = output_col_base + output_slot
                        for v in T.vectorized(packed_per_thread):
                            packed_local[v] = T.if_then_else(
                                output_col < n,
                                packed_weight[output_col, logical_k // 2 + v],
                                T.cast(0, "uint8"),
                            )
                        T.call_extern(
                            "decode_i4u_to_f16",
                            T.access_ptr(packed_local[0], "r", packed_per_thread),
                            T.access_ptr(decoded_local[0], "w", values_per_thread),
                            dtype=dtype,
                        )
                        scale_local[0] = T.if_then_else(
                            output_col < n,
                            T.cast(weight_scale[output_col, logical_k // group_size], dtype),
                            T.cast(0, dtype),
                        )
                        zero_local[0] = T.if_then_else(
                            output_col < n,
                            T.cast(weight_zero[output_col, logical_k // group_size], dtype),
                            T.cast(0, dtype),
                        )
                        for v in T.serial(values_per_thread):
                            activation_v = _plain_packed_activation_index(v)
                            accumulator[output_slot] += T.cast(
                                activation_local[activation_v], "float32"
                            ) * T.cast(
                                (decoded_local[v] - zero_local[0]) * scale_local[0],
                                "float32",
                            )

                for output_slot in T.serial(outputs_per_warp):
                    for step in T.serial(5):
                        accumulator[output_slot] += T.shfl_down(
                            accumulator[output_slot], 16 >> step, width=reduce_threads
                        )

                if split_k_warps == 1:
                    if lane == 0:
                        for output_slot in T.serial(outputs_per_warp):
                            output_col = output_col_base + output_slot
                            if output_col < n:
                                output[0, output_col] = accumulator[output_slot]
                else:
                    if lane == 0:
                        for output_slot in T.serial(outputs_per_warp):
                            partials[n_slot, output_slot, k_partition] = accumulator[output_slot]
                    T.sync_threads(barrier_id=1, arrive_count=total_threads)
                    if lane == 0 and k_partition == 0:
                        for output_slot in T.serial(outputs_per_warp):
                            output_col = output_col_base + output_slot
                            reduced[output_slot] = T.cast(0, "float32")
                            for part in T.serial(split_k_warps):
                                reduced[output_slot] += partials[n_slot, output_slot, part]
                            if output_col < n:
                                output[0, output_col] = reduced[output_slot]

        return main

    return build


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_decode_tma_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    """M=1 N64 pipeline: TMA packed W4, register decode, CTA split-K."""
    if m != 1 or dtype != "float16":
        raise ValueError("W4A16 TMA decode requires M=1 and float16")
    if n % 64 != 0 or k % 1024 != 0:
        raise ValueError("W4A16 TMA decode requires N%64==0 and K%1024==0")

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        num_stages: int = 3,
        cache_activation_fp32: bool = False,
        long_k_schedule: bool = False,
        consumer_max_nreg: int = 112,
    ) -> Callable:
        block_n = 64
        block_k = 512
        split_k_warps = 2
        outputs_per_warp = 8
        producer_threads = 128
        consumer_threads = 512
        threads = producer_threads + consumer_threads
        reduce_threads = 32
        consumer_warps = consumer_threads // reduce_threads
        output_warps = consumer_warps // split_k_warps
        values_per_thread = block_k // reduce_threads
        packed_per_thread = values_per_thread // 2
        groups_per_partition = block_k // group_size
        super_block_k = block_k * split_k_warps
        super_tiles = k // super_block_k
        output_mode = 7 if long_k_schedule else 0
        value_mode = 5 if long_k_schedule else 0
        release_after_activation = long_k_schedule

        if num_stages not in (2, 3, 4):
            raise ValueError("num_stages must be 2, 3, or 4")
        if consumer_max_nreg not in (96, 104, 112):
            raise ValueError("consumer_max_nreg must be 96, 104, or 112")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 2), "uint8"),  # type: ignore
            weight_scale: T.Tensor((n, k // group_size), "float32"),  # type: ignore
            weight_zero: T.Tensor((n, k // group_size), "uint8"),  # type: ignore
            output: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(n // block_n, threads=threads) as bx:
                activation_shared = T.alloc_shared((num_stages, split_k_warps, block_k), dtype)
                packed_shared = T.alloc_shared(
                    (num_stages, split_k_warps, block_n, block_k // 2),
                    "uint8",
                )
                scale_shared = T.alloc_shared(
                    (
                        num_stages,
                        block_n,
                        split_k_warps * groups_per_partition,
                    ),
                    "float32",
                )
                zero_shared = T.alloc_shared(
                    (
                        num_stages,
                        block_n,
                        split_k_warps * groups_per_partition,
                    ),
                    "uint8",
                )
                partials = T.alloc_shared(
                    (output_warps, outputs_per_warp, split_k_warps), "float32"
                )

                activation_local = T.alloc_local((values_per_thread,), dtype)
                activation_float = T.alloc_local((values_per_thread,), "float32")
                packed_local = T.alloc_local((outputs_per_warp, packed_per_thread), "uint8")
                decoded_local = T.alloc_local((values_per_thread,), dtype)
                accumulator = T.alloc_local((outputs_per_warp,), "float32")
                raw_partial = T.alloc_local((outputs_per_warp,), "float32")
                activation_sum = T.alloc_local((1,), "float32")
                output_sum = T.alloc_local((1,), "float32")
                scale_local = T.alloc_local((outputs_per_warp,), "float32")
                zero_local = T.alloc_local((outputs_per_warp,), "float32")

                ready = T.alloc_barrier([producer_threads] * num_stages)
                empty = T.alloc_barrier([consumer_threads] * num_stages)
                producer_iteration = T.alloc_var("int32", init=0)
                consumer_iteration = T.alloc_var("int32", init=0)
                tx = T.get_thread_binding()
                n_start = bx * block_n

                T.import_source(UINT4_TO_FP16_LOP3_SOURCE)

                if tx < producer_threads:
                    T.dec_max_nreg(24)
                    for ko in T.serial(super_tiles):
                        stage = producer_iteration % num_stages
                        phase = (producer_iteration // num_stages) % 2
                        k_start = ko * super_block_k
                        for stage_index in range(num_stages):
                            if stage == stage_index:
                                T.barrier_wait(empty[stage_index], phase ^ 1)
                                T.fence_proxy_async()
                                for part, value in T.Parallel(split_k_warps, block_k):
                                    activation_shared[stage_index, part, value] = activation[
                                        0, k_start + part * block_k + value
                                    ]
                                for part in range(split_k_warps):
                                    part_k_start = k_start + part * block_k
                                    T.tma_copy(
                                        packed_weight[
                                            n_start : n_start + block_n,
                                            part_k_start // 2 : (part_k_start + block_k) // 2,
                                        ],
                                        packed_shared[stage_index, part, :, :],
                                        barrier=ready[stage_index],
                                    )
                                T.tma_copy(
                                    weight_scale[
                                        n_start : n_start + block_n,
                                        k_start // group_size : (k_start + super_block_k)
                                        // group_size,
                                    ],
                                    scale_shared[stage_index, :, :],
                                    barrier=ready[stage_index],
                                )
                                T.copy(
                                    weight_zero[
                                        n_start : n_start + block_n,
                                        k_start // group_size : (k_start + super_block_k)
                                        // group_size,
                                    ],
                                    zero_shared[stage_index, :, :],
                                )
                                T.barrier_arrive(ready[stage_index])
                        producer_iteration = producer_iteration + 1
                else:
                    T.inc_max_nreg(consumer_max_nreg)
                    consumer_thread = tx - producer_threads
                    warp_id = consumer_thread // reduce_threads
                    lane = consumer_thread % reduce_threads
                    output_warp = warp_id // split_k_warps
                    k_partition = warp_id % split_k_warps
                    output_col_base = n_start + output_warp * outputs_per_warp
                    T.clear(accumulator)

                    for _ko in T.serial(super_tiles):
                        stage = consumer_iteration % num_stages
                        phase = (consumer_iteration // num_stages) % 2
                        for stage_index in range(num_stages):
                            if stage == stage_index:
                                T.barrier_wait(ready[stage_index], phase)
                                for value in T.vectorized(values_per_thread):
                                    activation_local[value] = activation_shared[
                                        stage_index,
                                        k_partition,
                                        lane * values_per_thread + value,
                                    ]

                                group_in_partition = (lane * values_per_thread) // group_size
                                for output_iter in T.serial(outputs_per_warp):
                                    output_slot = _decode_schedule_index(
                                        output_iter, outputs_per_warp, output_mode
                                    )
                                    output_col_local = output_warp * outputs_per_warp + output_slot
                                    for value in T.vectorized(packed_per_thread):
                                        packed_local[output_slot, value] = packed_shared[
                                            stage_index,
                                            k_partition,
                                            output_col_local,
                                            lane * packed_per_thread + value,
                                        ]
                                    scale_local[output_slot] = scale_shared[
                                        stage_index,
                                        output_col_local,
                                        k_partition * groups_per_partition + group_in_partition,
                                    ]
                                    zero_local[output_slot] = T.cast(
                                        zero_shared[
                                            stage_index,
                                            output_col_local,
                                            k_partition * groups_per_partition + group_in_partition,
                                        ],
                                        "float32",
                                    )

                                if not release_after_activation:
                                    T.barrier_arrive(empty[stage_index])

                                activation_sum[0] = T.cast(0, "float32")
                                for value_iter in T.serial(values_per_thread):
                                    value = _decode_schedule_index(
                                        value_iter, values_per_thread, value_mode
                                    )
                                    if cache_activation_fp32:
                                        activation_float[value] = T.cast(
                                            activation_local[value], "float32"
                                        )
                                        activation_sum[0] += activation_float[value]
                                    else:
                                        activation_sum[0] += T.cast(
                                            activation_local[value], "float32"
                                        )

                                if release_after_activation:
                                    T.barrier_arrive(empty[stage_index])

                                for output_iter in T.serial(outputs_per_warp):
                                    output_slot = _decode_schedule_index(
                                        output_iter, outputs_per_warp, output_mode
                                    )
                                    for chunk in T.serial(values_per_thread // 8):
                                        T.call_extern(
                                            "decode_i4u_to_f16",
                                            T.access_ptr(
                                                packed_local[output_slot, chunk * 4],
                                                "r",
                                                4,
                                            ),
                                            T.access_ptr(decoded_local[chunk * 8], "w", 8),
                                            dtype=dtype,
                                        )

                                    raw_partial[output_slot] = T.cast(0, "float32")
                                    for value_iter in T.serial(values_per_thread):
                                        decoded_index = _decode_schedule_index(
                                            value_iter, values_per_thread, value_mode
                                        )
                                        activation_index = _plain_packed_activation_index(
                                            decoded_index
                                        )
                                        raw_partial[output_slot] += (
                                            activation_float[activation_index]
                                            if cache_activation_fp32
                                            else T.cast(
                                                activation_local[activation_index],
                                                "float32",
                                            )
                                        ) * T.cast(decoded_local[decoded_index], "float32")
                                    accumulator[output_slot] += (
                                        raw_partial[output_slot]
                                        - zero_local[output_slot] * activation_sum[0]
                                    ) * scale_local[output_slot]
                        consumer_iteration = consumer_iteration + 1

                    for output_slot in T.serial(outputs_per_warp):
                        for step in T.serial(5):
                            accumulator[output_slot] += T.shfl_down(
                                accumulator[output_slot],
                                16 >> step,
                                width=reduce_threads,
                            )
                        if lane == 0:
                            partials[output_warp, output_slot, k_partition] = accumulator[
                                output_slot
                            ]

                    T.sync_threads(barrier_id=1, arrive_count=consumer_threads)
                    if lane == 0 and k_partition == 0:
                        for output_slot in T.serial(outputs_per_warp):
                            output_sum[0] = T.cast(0, "float32")
                            for part in T.serial(split_k_warps):
                                output_sum[0] += partials[output_warp, output_slot, part]
                            output[0, output_col_base + output_slot] = output_sum[0]

        return main

    return build


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

    general = True

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


class GemmW4A16DecodeKernel(Kernel):
    """Hopper M=1 W4A16 path with register decode and CTA-local split-K."""

    supported_archs = [90]

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        return call.m == 1 and call.dtype == torch.float16 and call.k % 256 == 0

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
        self.use_tma = n % 64 == 0 and k % 1024 == 0
        self.kernel = (
            _gemm_w4a16_decode_tma_kernel(m, n, k, self.dtype_str, group_size)
            if self.use_tma
            else _gemm_w4a16_decode_direct_kernel(m, n, k, self.dtype_str, group_size)
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.use_tma:
            return {
                "num_stages": 4 if self.k <= 8192 else 3,
                "cache_activation_fp32": self.k == 16384,
                "long_k_schedule": self.k >= 65536,
                "consumer_max_nreg": 104 if self.k <= 32768 else 112,
            }
        split_k_warps = next(split for split in (8, 4, 2, 1) if self.k % (256 * split) == 0)
        outputs_per_warp = 4 if self.k >= 8192 else 1
        return {
            "n_partition": 4 // outputs_per_warp,
            "split_k_warps": split_k_warps,
            "outputs_per_warp": outputs_per_warp,
        }

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        factory = (
            _gemm_w4a16_decode_tma_kernel if self.use_tma else _gemm_w4a16_decode_direct_kernel
        )
        compiled = factory(self.m, self.n, self.k, self.dtype_str, self.group_size)(**self.config)
        return compiled(activation, packed_weight, weight_scale, weight_zero)
