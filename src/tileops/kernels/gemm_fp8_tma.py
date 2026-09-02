"""SM90 FP8 1D2D GEMM with an explicit TMA/WGMMA pipeline.

The kernel matches FlashInfer's 1x128 A-scale and 128x128 B-scale contract,
with K-major physical A scales.  Its Hopper schedule uses:

* one 128-thread consumer warp-group issues WGMMA;
* one producer warp in the second warp-group issues TMA loads;
* two or more shared-memory stages form a barrier-protected ring buffer.

Shape-specific H200 choices are kept in one table below so the measured
pipeline, epilogue, and scheduling decisions remain auditable.
"""

import functools
import os
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_count

__all__ = ["GemmFp8BlockScaled1D2DTMAKMajorScaleKernel"]

_FP8_GEMM_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_fp8_gemm_1d2d_helper.h")
)

_TMA_BFLOAT16 = 9
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_NONE = 0
_TMA_L2_128B = 2
_TMA_OOB_NONE = 0

# FlashInfer-compatible 1D2D specializations measured on H200.  Keeping every
# shape-dependent choice together prevents scheduling, epilogue, and SM-count
# decisions from drifting apart as new workloads are tuned.
_FP8_1D2D_H200_CONFIGS: dict[tuple[int, int, int], dict[str, object]] = {
    (128, 2112, 7168): {
        "kernel": {"block_n": 16, "num_stages": 8, "group_size_m": 16, "mainloop_unroll": 8},
        "shared_epilogue": True,
    },
    (128, 7168, 2048): {
        "kernel": {"block_n": 64, "num_stages": 8, "group_size_m": 16, "mainloop_unroll": 16},
        "shared_epilogue": True,
        "sm_count": 112,
    },
    (4096, 2112, 7168): {
        "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 16, "mainloop_unroll": 7},
        "shared_epilogue": True,
    },
    (4096, 4096, 7168): {
        "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 16, "mainloop_unroll": 8},
        "shared_epilogue": True,
    },
    (4096, 7168, 2048): {
        "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 32, "mainloop_unroll": 16},
        "shared_epilogue": True,
    },
    (4096, 7168, 16384): {
        "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 16, "mainloop_unroll": 4},
        "shared_epilogue": False,
    },
    (4096, 24576, 1536): {
        "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 32, "mainloop_unroll": 12},
        "shared_epilogue": True,
    },
}


@functools.lru_cache(maxsize=32)
def _gemm_fp8_block_scaled_tma_coop_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    *,
    sm_count: int,
    shared_epilogue: bool = False,
) -> Callable:
    """Build the persistent K-major 1D2D cooperative mainloop."""
    block_m = 128
    half_m = 64
    block_k = 128
    accum_dtype = "float"
    scale_k = (k + block_k - 1) // block_k

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[
            "-O3",
            "--use_fast_math",
            "-DENABLE_BF16",
            "-include",
            _FP8_GEMM_HELPER_PATH,
        ],
    )
    def kernel_func(
        block_n: int = 128,
        num_stages: int = 3,
        group_size_m: int = 16,
        mainloop_unroll: int = 0,
    ) -> Callable:
        if block_n not in (16, 32, 56, 64, 128):
            raise ValueError(f"block_n must be one of 16/32/56/64/128, got {block_n}")
        uniform_scale_b = 128 % block_n == 0
        wgmma_helper = f"tl::fp8_gemm_wgmma_64x128_by_128x{block_n}"
        promotion_1d2d_shared_ab = f"tl::fp8_gemm_1d2d_promote_shared_ab_64x{block_n}"
        promotion_1d2d_uniform = f"tl::fp8_gemm_1d2d_promote_shared_ab_uniform_64x{block_n}"
        global_store_helper = f"tl::fp8_gemm_raw_acc_store_global_64x{block_n}_v2"
        smem_store_helper = f"tl::fp8_gemm_raw_acc_stsm_bf16_64x{block_n}"
        fragment_regs = (half_m * block_n) // 128
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        max_waves = -(-total_tiles // sm_count)
        stage_1d2d_b_per_k = max_waves > 1
        effective_unroll = mainloop_unroll or num_stages

        @T.macro
        def decode(flat_id, mt, nt):
            tiles_per_group = T.int32(group_size_m * num_pid_n)
            group_id = flat_id // tiles_per_group
            first_m = group_id * T.int32(group_size_m)
            group_m = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
            mt[0] = first_m + (flat_id % tiles_per_group) % group_m
            nt[0] = (flat_id % tiles_per_group) // group_m

        @T.prim_func
        def main(
            a: T.Tensor((m, k), dtype),
            b: T.Tensor((n, k), dtype),
            scale_a: T.Tensor((scale_k, m), "float32"),
            scale_b: T.Tensor(((n + 127) // 128, scale_k), "float32"),
            c: T.Tensor((m, n), out_dtype),
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_shared = T.alloc_shared((num_stages, block_m, block_k), dtype)
                b_shared = T.alloc_shared((num_stages, block_n, block_k), dtype)
                partial_0 = T.alloc_local((fragment_regs,), accum_dtype)
                partial_1 = T.alloc_local((fragment_regs,), accum_dtype)
                final_0 = T.alloc_local((fragment_regs,), accum_dtype)
                final_1 = T.alloc_local((fragment_regs,), accum_dtype)
                if shared_epilogue:
                    shared_c = T.alloc_shared((block_m, block_n), out_dtype)
                one_scale_a = T.alloc_shared((num_stages, block_m), accum_dtype)
                if stage_1d2d_b_per_k:
                    one_scale_b = T.alloc_shared((num_stages, 2), accum_dtype)
                else:
                    one_scale_b = T.alloc_shared(
                        (1 if uniform_scale_b else 2, scale_k), accum_dtype
                    )
                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )

                full = T.alloc_barrier([1] * num_stages)
                empty = T.alloc_barrier([256] * num_stages)
                producer_index = T.alloc_var("int32", init=0)
                consumer_index_0 = T.alloc_var("int32", init=0)
                consumer_index_1 = T.alloc_var("int32", init=0)
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")
                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for wave in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * wave + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if wave > 0 and not stage_1d2d_b_per_k:
                                T.sync_threads(barrier_id=15, arrive_count=384)
                            if not stage_1d2d_b_per_k and (
                                not uniform_scale_b or scale_k < 16 or max_waves > 1
                            ):
                                for i in T.Parallel((1 if uniform_scale_b else 2) * scale_k):
                                    scale_row = T.min(
                                        n_start // 128 + i // scale_k,
                                        (n + 127) // 128 - 1,
                                    )
                                    one_scale_b[i // scale_k, i % scale_k] = scale_b[
                                        scale_row, i % scale_k
                                    ]
                                T.sync_threads(barrier_id=10, arrive_count=384)
                            producer_steps = (
                                T.if_then_else(tx == 0, scale_k, 0)
                                if scale_k <= 16 and max_waves == 1
                                else scale_k
                            )
                            for kk in T.unroll(producer_steps, unroll_factor=effective_unroll):
                                slot = producer_index % num_stages
                                T.barrier_wait(
                                    empty[slot], ((producer_index // num_stages) & 1) ^ 1
                                )
                                if stage_1d2d_b_per_k and tx == 0:
                                    scale_row = T.min(
                                        n_start // 128,
                                        (n + 127) // 128 - 1,
                                    )
                                    next_scale_row = T.min(
                                        scale_row + 1,
                                        (n + 127) // 128 - 1,
                                    )
                                    one_scale_b[slot, 0] = scale_b[scale_row, kk]
                                    one_scale_b[slot, 1] = scale_b[next_scale_row, kk]
                                T.tma_copy(
                                    a[
                                        m_start : m_start + block_m,
                                        kk * block_k : (kk + 1) * block_k,
                                    ],
                                    a_shared[slot, :, :],
                                    barrier=full[slot],
                                )
                                T.tma_copy(
                                    scale_a[
                                        kk : kk + 1,
                                        m_start : m_start + block_m,
                                    ],
                                    one_scale_a[slot, :],
                                    barrier=full[slot],
                                )
                                if uniform_scale_b and scale_k >= 16 and max_waves == 1 and kk == 0:
                                    scale_row = T.min(
                                        n_start // 128,
                                        (n + 127) // 128 - 1,
                                    )
                                    T.tma_copy(
                                        scale_b[scale_row : scale_row + 1, 0:scale_k],
                                        one_scale_b[:, :],
                                        barrier=full[slot],
                                    )
                                T.tma_copy(
                                    b[
                                        n_start : n_start + block_n,
                                        kk * block_k : (kk + 1) * block_k,
                                    ],
                                    b_shared[slot, :, :],
                                    barrier=full[slot],
                                )
                                if tx == 0:
                                    T.barrier_arrive(full[slot])
                                producer_index = producer_index + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    for wave in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * wave + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if wave > 0 and not stage_1d2d_b_per_k:
                                T.sync_threads(barrier_id=15, arrive_count=384)
                            if not stage_1d2d_b_per_k and (
                                not uniform_scale_b or scale_k < 16 or max_waves > 1
                            ):
                                T.sync_threads(barrier_id=10, arrive_count=384)
                            T.clear(final_0)
                            for kk in T.unroll(scale_k, unroll_factor=effective_unroll):
                                slot = consumer_index_0 % num_stages
                                T.barrier_wait(full[slot], (consumer_index_0 // num_stages) & 1)
                                T.call_extern(
                                    "handle",
                                    wgmma_helper,
                                    partial_0.data,
                                    T.address_of(a_shared[slot, 0, 0]),
                                    T.address_of(b_shared[slot, 0, 0]),
                                )
                                T.wait_wgmma(0)
                                if stage_1d2d_b_per_k:
                                    T.call_extern(
                                        "handle",
                                        promotion_1d2d_shared_ab,
                                        partial_0.data,
                                        final_0.data,
                                        T.address_of(one_scale_a[slot, 0]),
                                        T.address_of(one_scale_b[slot, 0]),
                                        1,
                                        n_start,
                                        0,
                                    )
                                elif uniform_scale_b:
                                    T.call_extern(
                                        "handle",
                                        promotion_1d2d_uniform,
                                        partial_0.data,
                                        final_0.data,
                                        T.address_of(one_scale_a[slot, 0]),
                                        T.address_of(one_scale_b[0, 0]),
                                        kk,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        promotion_1d2d_shared_ab,
                                        partial_0.data,
                                        final_0.data,
                                        T.address_of(one_scale_a[slot, 0]),
                                        T.address_of(one_scale_b[0, 0]),
                                        scale_k,
                                        n_start,
                                        kk,
                                    )
                                T.barrier_arrive(empty[slot])
                                consumer_index_0 = consumer_index_0 + 1
                            if shared_epilogue:
                                T.call_extern(
                                    "handle",
                                    smem_store_helper,
                                    final_0.data,
                                    T.address_of(shared_c[0, 0]),
                                )
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=13, arrive_count=256)
                                if tx == 128:
                                    output_desc = T.create_tma_descriptor(
                                        _TMA_BFLOAT16,
                                        2,
                                        c.data,
                                        n,
                                        m,
                                        1,
                                        n * 2,
                                        block_n,
                                        block_m,
                                        1,
                                        1,
                                        _TMA_INTERLEAVE_NONE,
                                        _TMA_SWIZZLE_NONE,
                                        _TMA_L2_128B,
                                        _TMA_OOB_NONE,
                                    )
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_tma_store_2d_ptx",
                                        output_desc,
                                        T.address_of(shared_c[0, 0]),
                                        n_start,
                                        m_start,
                                    )
                                T.sync_threads(barrier_id=14, arrive_count=256)
                            else:
                                T.call_extern(
                                    "handle",
                                    global_store_helper,
                                    final_0.data,
                                    c.data,
                                    n,
                                    m_start,
                                    n_start,
                                    m,
                                    n,
                                )

                else:
                    T.inc_max_nreg(240)
                    for wave in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * wave + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if wave > 0 and not stage_1d2d_b_per_k:
                                T.sync_threads(barrier_id=15, arrive_count=384)
                            if not stage_1d2d_b_per_k and (
                                not uniform_scale_b or scale_k < 16 or max_waves > 1
                            ):
                                T.sync_threads(barrier_id=10, arrive_count=384)
                            T.clear(final_1)
                            for kk in T.unroll(scale_k, unroll_factor=effective_unroll):
                                slot = consumer_index_1 % num_stages
                                T.barrier_wait(full[slot], (consumer_index_1 // num_stages) & 1)
                                T.call_extern(
                                    "handle",
                                    wgmma_helper,
                                    partial_1.data,
                                    T.address_of(a_shared[slot, half_m, 0]),
                                    T.address_of(b_shared[slot, 0, 0]),
                                )
                                T.wait_wgmma(0)
                                if stage_1d2d_b_per_k:
                                    T.call_extern(
                                        "handle",
                                        promotion_1d2d_shared_ab,
                                        partial_1.data,
                                        final_1.data,
                                        T.address_of(one_scale_a[slot, half_m]),
                                        T.address_of(one_scale_b[slot, 0]),
                                        1,
                                        n_start,
                                        0,
                                    )
                                elif uniform_scale_b:
                                    T.call_extern(
                                        "handle",
                                        promotion_1d2d_uniform,
                                        partial_1.data,
                                        final_1.data,
                                        T.address_of(one_scale_a[slot, half_m]),
                                        T.address_of(one_scale_b[0, 0]),
                                        kk,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        promotion_1d2d_shared_ab,
                                        partial_1.data,
                                        final_1.data,
                                        T.address_of(one_scale_a[slot, half_m]),
                                        T.address_of(one_scale_b[0, 0]),
                                        scale_k,
                                        n_start,
                                        kk,
                                    )
                                T.barrier_arrive(empty[slot])
                                consumer_index_1 = consumer_index_1 + 1
                            if shared_epilogue:
                                T.call_extern(
                                    "handle",
                                    smem_store_helper,
                                    final_1.data,
                                    T.address_of(shared_c[half_m, 0]),
                                )
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=13, arrive_count=256)
                                T.sync_threads(barrier_id=14, arrive_count=256)
                            else:
                                T.call_extern(
                                    "handle",
                                    global_store_helper,
                                    final_1.data,
                                    c.data,
                                    n,
                                    m_start + half_m,
                                    n_start,
                                    m,
                                    n,
                                )

        return main

    return kernel_func


class GemmFp8BlockScaled1D2DTMAKMajorScaleKernel(Kernel):
    """SM90 FP8 GEMM with FlashInfer-compatible 1D2D K-major scales.

    ``scale_a`` is physically contiguous ``[K // 128, M]`` and ``scale_b``
    is contiguous ``[ceildiv(N, 128), K // 128]``.  The kernel uses a
    persistent 384-thread TMA/WGMMA pipeline and returns BF16 output.
    """

    supported_archs = [90]

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        if m < 128:
            raise ValueError(
                "1D2D cooperative TMA GEMM requires M >= 128; "
                "dispatch smaller M to a GEMV/small-M kernel"
            )
        if dtype != torch.float8_e4m3fn:
            raise NotImplementedError("1D2D cooperative TMA GEMM only supports torch.float8_e4m3fn")
        if out_dtype != torch.bfloat16:
            raise NotImplementedError("1D2D cooperative TMA GEMM only supports BF16 output")

        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.sm_count = get_sm_count()
        self.shared_epilogue = False

        tuned = _FP8_1D2D_H200_CONFIGS.get((m, n, k))
        if tuned is not None:
            self.shared_epilogue = bool(tuned["shared_epilogue"])
            if "sm_count" in tuned:
                self.sm_count = int(tuned["sm_count"])

        self.kernel = _gemm_fp8_block_scaled_tma_coop_kernel(
            m,
            n,
            k,
            self.dtype_str,
            self.out_dtype_str,
            sm_count=self.sm_count,
            shared_epilogue=self.shared_epilogue,
        )
        self.init_config(config, tune)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        tuned = _FP8_1D2D_H200_CONFIGS.get((self.m, self.n, self.k))
        if tuned is not None:
            return dict(tuned["kernel"])

        m_tiles = (self.m + 127) // 128
        target_n = self.n * m_tiles / self.sm_count
        if target_n <= 24:
            block_n = 16
        elif target_n <= 48:
            block_n = 32
        elif target_n <= 96:
            block_n = 64
        else:
            block_n = 128
        return {
            "block_n": block_n,
            "num_stages": 3,
            "group_size_m": 16,
            "mainloop_unroll": 3,
        }

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        compiled = _gemm_fp8_block_scaled_tma_coop_kernel(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            sm_count=self.sm_count,
            shared_epilogue=self.shared_epilogue,
        )(**self.config)
        return compiled(a, b, scale_a, scale_b)
