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

__all__ = ["GemmFp81D2DKernel"]

_FP8_1D2D_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_fp8_1d2d_helper.h")
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


def _shape_refusal(m: int, n: int, k: int, *, shared_epilogue: bool) -> Optional[str]:
    """Why this schedule cannot address these shapes, or ``None`` when it can.

    Every operand and the A scale arrive through TMA, whose descriptors address
    the innermost (contiguous) dimension in 16-byte units: ``k`` for the fp8
    ``a`` / ``b``, and ``m`` for the ``[K // 128, M]`` fp32 ``scale_a``. The
    epilogue adds its own unit — two BF16 columns per packed global store, or a
    descriptor row stride of 16 bytes for the shared-memory path.

    Undeclared, an unaligned shape reaches TileLang's descriptor check and dies
    as "Check failed: (result.supported) is false", naming nothing to change.
    """
    if m < 128:
        return "M >= 128 is required; dispatch smaller M to a GEMV/small-M kernel"
    store_unit = 8 if shared_epilogue else 2
    offenders = [
        f"{name}={value} is not a multiple of {unit} ({what})"
        for name, value, unit, what in (
            ("k", k, 16, "a and b are read K-major through TMA, 16 fp8 per 16 bytes"),
            ("m", m, 4, "scale_a is read M-major through TMA, 4 fp32 per 16 bytes"),
            (
                "n",
                n,
                store_unit,
                "the TMA epilogue needs a 16-byte row stride in c"
                if shared_epilogue
                else "the epilogue writes c two BF16 columns at a time",
            ),
        )
        if value % unit
    ]
    if not offenders:
        return None
    return "; ".join(offenders)


@functools.lru_cache(maxsize=32)
def _gemm_fp8_1d2d_kernel(
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
            _FP8_1D2D_HELPER_PATH,
        ],
    )
    def kernel_func(
        block_n: int = 128,
        num_stages: int = 3,
        group_size_m: int = 16,
        mainloop_unroll: int = 0,
    ) -> Callable:
        # Every value divides 128 and is a multiple of 16, and both matter.
        # Dividing 128 keeps a tile inside one 128-wide B-scale block, so the
        # promotion reads a single scale; a value that straddled two would need
        # the mainloop to stage a second scale row for every K step. Being a
        # multiple of 16 is what ``fp8_gemm_raw_acc_stsm_bf16`` needs: its STSM
        # atom writes a 16x16 patch, and a partial patch would leave columns
        # unwritten in shared memory, publishing whatever the last wave left.
        if block_n not in (16, 32, 64, 128):
            raise ValueError(f"block_n must be one of 16/32/64/128, got {block_n}")
        if group_size_m < 1:
            raise ValueError(f"group_size_m must be positive, got {group_size_m}")
        if num_stages < 1:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        wgmma_helper = f"tl::fp8_gemm_wgmma_64x128_by_128x{block_n}"
        promotion_helper = f"tl::fp8_gemm_1d2d_promote_64x{block_n}"
        global_store_helper = f"tl::fp8_gemm_raw_acc_store_global_64x{block_n}_v2"
        smem_store_helper = f"tl::fp8_gemm_raw_acc_stsm_bf16_64x{block_n}"
        fragment_regs = (half_m * block_n) // 128
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        max_waves = -(-total_tiles // sm_count)
        # One wave stages the whole B-scale column in shared memory before the
        # mainloop; several waves cannot, because the tile — and with it the
        # scale row — changes per wave, so they stage that row's scale per K
        # step into the ring instead. The two forms are mutually exclusive,
        # which is why ``not stage_1d2d_b_per_k`` reads as ``max_waves == 1``.
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
                    one_scale_b = T.alloc_shared((num_stages, 1), accum_dtype)
                else:
                    one_scale_b = T.alloc_shared((1, scale_k), accum_dtype)
                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )

                full = T.alloc_barrier([1] * num_stages)
                # One arrival per consumer warp, not per thread: ``wait_wgmma`` is
                # warp-convergent, so once a lane has passed it every lane of that
                # warp is done with the stage, and lane 0 can speak for the warp.
                # Eight arrivals per stage instead of 256.
                empty = T.alloc_barrier([8] * num_stages)
                producer_index = T.alloc_var("int32", init=0)
                consumer_index_0 = T.alloc_var("int32", init=0)
                consumer_index_1 = T.alloc_var("int32", init=0)
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")
                # The three scales a consumer thread needs for one K step, read
                # out of the stage before the WGMMA so the stage can be handed
                # back to the producer before the promotion runs.
                scales = T.alloc_local((3,), accum_dtype)
                tx = T.get_thread_binding()
                # Rows of this thread's WGMMA accumulator within its 64-row half:
                # lanes 4i..4i+3 of warp w hold rows w*16 + i and w*16 + i + 8.
                acc_row0 = ((tx // 32) % 4) * 16 + (tx % 32) // 4
                acc_row1 = acc_row0 + 8

                if tx < 128:
                    T.dec_max_nreg(24)
                    for wave in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * wave + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if not stage_1d2d_b_per_k and scale_k < 16:
                                for i in T.Parallel(scale_k):
                                    scale_row = T.min(n_start // 128, (n + 127) // 128 - 1)
                                    one_scale_b[0, i] = scale_b[scale_row, i]
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
                                    scale_row = T.min(n_start // 128, (n + 127) // 128 - 1)
                                    one_scale_b[slot, 0] = scale_b[scale_row, kk]
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
                                if scale_k >= 16 and max_waves == 1 and kk == 0:
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
                            if not stage_1d2d_b_per_k and scale_k < 16:
                                T.sync_threads(barrier_id=10, arrive_count=384)
                            T.clear(final_0)
                            for kk in T.unroll(scale_k, unroll_factor=effective_unroll):
                                slot = consumer_index_0 % num_stages
                                T.barrier_wait(full[slot], (consumer_index_0 // num_stages) & 1)
                                scales[0] = one_scale_a[slot, acc_row0]
                                scales[1] = one_scale_a[slot, acc_row1]
                                if stage_1d2d_b_per_k:
                                    scales[2] = one_scale_b[slot, 0]
                                else:
                                    scales[2] = one_scale_b[0, kk]
                                T.call_extern(
                                    "handle",
                                    wgmma_helper,
                                    partial_0.data,
                                    T.address_of(a_shared[slot, 0, 0]),
                                    T.address_of(b_shared[slot, 0, 0]),
                                )
                                T.wait_wgmma(0)
                                # The stage's shared memory is no longer read past
                                # this point: release it before the promotion so the
                                # producer's next TMA overlaps the multiply-adds.
                                if tx % 32 == 0:
                                    T.barrier_arrive(empty[slot])
                                T.call_extern(
                                    "handle",
                                    promotion_helper,
                                    partial_0.data,
                                    final_0.data,
                                    scales[0],
                                    scales[1],
                                    scales[2],
                                )
                                consumer_index_0 = consumer_index_0 + 1
                            if shared_epilogue:
                                # The previous tile's TMA store may still be reading
                                # shared_c; its issuing thread waits for that read to
                                # finish, then both consumer warp-groups align before
                                # anyone writes the next tile into it. Deferring the
                                # wait to here lets the store overlap this tile's
                                # mainloop instead of stalling the last one.
                                if tx == 128:
                                    T.tma_store_wait(0)
                                T.sync_threads(barrier_id=13, arrive_count=256)
                                T.call_extern(
                                    "handle",
                                    smem_store_helper,
                                    final_0.data,
                                    T.address_of(shared_c[0, 0]),
                                )
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=14, arrive_count=256)
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
                                        "tl::fp8_tma_store_2d_issue",
                                        output_desc,
                                        T.address_of(shared_c[0, 0]),
                                        n_start,
                                        m_start,
                                    )
                                    T.tma_store_arrive()
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
                    # Drain the last tile's store before this CTA's shared memory
                    # is released with it.
                    if shared_epilogue and tx == 128:
                        T.tma_store_wait(0)

                else:
                    T.inc_max_nreg(240)
                    for wave in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * wave + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if not stage_1d2d_b_per_k and scale_k < 16:
                                T.sync_threads(barrier_id=10, arrive_count=384)
                            T.clear(final_1)
                            for kk in T.unroll(scale_k, unroll_factor=effective_unroll):
                                slot = consumer_index_1 % num_stages
                                T.barrier_wait(full[slot], (consumer_index_1 // num_stages) & 1)
                                scales[0] = one_scale_a[slot, half_m + acc_row0]
                                scales[1] = one_scale_a[slot, half_m + acc_row1]
                                if stage_1d2d_b_per_k:
                                    scales[2] = one_scale_b[slot, 0]
                                else:
                                    scales[2] = one_scale_b[0, kk]
                                T.call_extern(
                                    "handle",
                                    wgmma_helper,
                                    partial_1.data,
                                    T.address_of(a_shared[slot, half_m, 0]),
                                    T.address_of(b_shared[slot, 0, 0]),
                                )
                                T.wait_wgmma(0)
                                # The stage's shared memory is no longer read past
                                # this point: release it before the promotion so the
                                # producer's next TMA overlaps the multiply-adds.
                                if tx % 32 == 0:
                                    T.barrier_arrive(empty[slot])
                                T.call_extern(
                                    "handle",
                                    promotion_helper,
                                    partial_1.data,
                                    final_1.data,
                                    scales[0],
                                    scales[1],
                                    scales[2],
                                )
                                consumer_index_1 = consumer_index_1 + 1
                            if shared_epilogue:
                                T.sync_threads(barrier_id=13, arrive_count=256)
                                T.call_extern(
                                    "handle",
                                    smem_store_helper,
                                    final_1.data,
                                    T.address_of(shared_c[half_m, 0]),
                                )
                                T.fence_proxy_async()
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


class GemmFp81D2DKernel(Kernel):
    """SM90 FP8 GEMM with FlashInfer-compatible 1D2D K-major scales.

    ``scale_a`` is physically contiguous ``[K // 128, M]`` and ``scale_b``
    is contiguous ``[ceildiv(N, 128), K // 128]``.  The kernel uses a
    persistent 384-thread TMA/WGMMA pipeline and returns BF16 output.

    Args:
        m: Rows of ``a``; must be at least 128.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim.
        dtype: Operand dtype; only ``torch.float8_e4m3fn``.
        out_dtype: Output dtype; only ``torch.bfloat16``.
        config: Kernel config override; unset keys take their default.
        tune: Accepted for interface parity. This kernel declares no
            ``autotune_configs``, so its schedule comes from the measured table.
        shared_epilogue: Whether to stage the tile through shared memory and
            store it with TMA. ``None`` takes the measured choice for this
            shape, which is what a caller wants; state it to compare the two
            epilogues on a shape the table does not cover.
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
        shared_epilogue: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if dtype != torch.float8_e4m3fn:
            raise NotImplementedError("1D2D cooperative TMA GEMM only supports torch.float8_e4m3fn")
        if out_dtype != torch.bfloat16:
            raise NotImplementedError("1D2D cooperative TMA GEMM only supports BF16 output")

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
        if shared_epilogue is not None:
            self.shared_epilogue = bool(shared_epilogue)

        refusal = _shape_refusal(m, n, k, shared_epilogue=self.shared_epilogue)
        if refusal is not None:
            raise ValueError(f"1D2D cooperative TMA GEMM cannot serve m={m} n={n} k={k}: {refusal}")

        self.kernel = _gemm_fp8_1d2d_kernel(
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
        compiled = _gemm_fp8_1d2d_kernel(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            sm_count=self.sm_count,
            shared_epilogue=self.shared_epilogue,
        )(**self.config)
        return compiled(a, b, scale_a, scale_b)
