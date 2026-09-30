"""FP8 NT GEMM for 1D2D block scales: ``scale_a`` per 1x128, ``scale_b`` per 128x128.

A persistent grid of one producer and two consumer warp-groups: the producer
fills a shared-memory ring through TMA, each consumer runs WGMMA over 64 of the
tile's 128 rows and folds every K-step's partial in under its two scales.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.gemm.call_spec import GemmCall
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import device_calibration, get_sm_count

__all__ = ["GemmFp81D2DFwdKernel"]

_FP8_1D2D_HELPER_PATH = csrc_path("fp8_1d2d_helper.h")

# K-steps one A-scale staging covers, the most that divides ``ceil(K/128)``: eight fp32
# of a row-major ``scale_a`` row fill one 32-byte sector, four are TMA's 16-byte unit.
_SCALE_A_GROUPS = (16, 8, 4)
# Buffers of the A-scale ring, so one group is written while the other is read.
_SCALE_A_BUFFERS = 2

_TMA_BFLOAT16 = 9
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_NONE = 0
_TMA_L2_128B = 2
_TMA_OOB_NONE = 0

# Schedules measured per (m, n, k), by calibrated board (``tileops.utils.calibration_key``);
# any other board or shape takes the analytic ``block_n`` band of ``default_config``.
_FP8_1D2D_CONFIGS: dict[str, dict[tuple[int, int, int], dict[str, object]]] = {
    "h200": {
        (128, 2112, 7168): {
            "kernel": {"block_n": 16, "num_stages": 8, "group_size_m": 16, "group_unroll": 1},
            "shared_epilogue": True,
        },
        (128, 7168, 2048): {
            "kernel": {"block_n": 64, "num_stages": 8, "group_size_m": 16, "group_unroll": 1},
            "shared_epilogue": True,
            "sm_count": 112,
        },
        (4096, 2112, 7168): {
            "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 16, "group_unroll": 1},
            "shared_epilogue": True,
        },
        (4096, 4096, 7168): {
            "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 16, "group_unroll": 1},
            "shared_epilogue": True,
        },
        (4096, 7168, 2048): {
            "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 32, "group_unroll": 1},
            "shared_epilogue": True,
        },
        (4096, 7168, 16384): {
            "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 16, "group_unroll": 1},
            "shared_epilogue": False,
        },
        (4096, 24576, 1536): {
            "kernel": {"block_n": 128, "num_stages": 4, "group_size_m": 32, "group_unroll": 3},
            "shared_epilogue": True,
        },
    },
}


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
    """Build the persistent 1D2D GEMM; the returned factory takes the tile config.

    The K loop runs in groups of ``scale_a_group`` K-steps, one ``scale_a`` staging
    each; ``group_unroll`` is how many groups one unrolled iteration covers.
    """
    block_m = 128
    half_m = 64
    block_k = 128
    accum_dtype = "float"
    scale_k = (k + block_k - 1) // block_k
    scale_a_group = next(g for g in _SCALE_A_GROUPS if scale_k % g == 0)

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
        group_unroll: int = 1,
    ) -> Callable:
        # Each value divides 128, so a tile reads one B-scale block, and is a multiple
        # of the 16-column STSM atom; any other leaves epilogue columns unwritten.
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
        # One wave stages its tile's whole B-scale row before the mainloop; several
        # waves stage it per K-step in the ring, since the tile changes per wave.
        stage_1d2d_b_per_k = max_waves > 1
        producer_threads = 32 if max_waves == 1 else 128
        num_groups = scale_k // scale_a_group

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
            scale_a: T.Tensor((m, scale_k), "float32"),
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
                scale_a_ring = T.alloc_shared(
                    (_SCALE_A_BUFFERS, block_m, scale_a_group), accum_dtype
                )
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
                scale_a_full = T.alloc_barrier([1] * _SCALE_A_BUFFERS)
                scale_a_empty = T.alloc_barrier([8] * _SCALE_A_BUFFERS)
                # One arrival per consumer warp: ``wait_wgmma`` is warp-convergent, so
                # lane 0 speaks for its warp.
                empty = T.alloc_barrier([8] * num_stages)
                producer_index = T.alloc_var("int32", init=0)
                consumer_index_0 = T.alloc_var("int32", init=0)
                consumer_index_1 = T.alloc_var("int32", init=0)
                scale_a_producer = T.alloc_var("int32", init=0)
                scale_a_consumer_0 = T.alloc_var("int32", init=0)
                scale_a_consumer_1 = T.alloc_var("int32", init=0)
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")
                # A consumer thread's three scales for one K-step, read before the WGMMA
                # so the stage is released before the promotion.
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
                            # One lane issues each TMA; on a single wave the other
                            # producer threads only poll ``empty`` and take issue slots
                            # from the consumers, so one warp (the election is a warp
                            # collective) drives the pipeline there.
                            if tx < producer_threads:
                                for group in T.unroll(num_groups, unroll_factor=group_unroll):
                                    for col in T.unroll(scale_a_group):
                                        kk = group * scale_a_group + col
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
                                        if col == 0:
                                            # The next group's scale_a columns in one TMA box; rows
                                            # past m and columns past scale_k are zero-filled.
                                            buf = scale_a_producer % _SCALE_A_BUFFERS
                                            T.barrier_wait(
                                                scale_a_empty[buf],
                                                ((scale_a_producer // _SCALE_A_BUFFERS) & 1) ^ 1,
                                            )
                                            T.tma_copy(
                                                scale_a[
                                                    m_start : m_start + block_m,
                                                    kk : kk + scale_a_group,
                                                ],
                                                scale_a_ring[buf, :, :],
                                                barrier=scale_a_full[buf],
                                            )
                                            if tx == 0:
                                                T.barrier_arrive(scale_a_full[buf])
                                            scale_a_producer = scale_a_producer + 1
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
                            for group in T.unroll(num_groups, unroll_factor=group_unroll):
                                for col in T.unroll(scale_a_group):
                                    kk = group * scale_a_group + col
                                    slot = consumer_index_0 % num_stages
                                    T.barrier_wait(full[slot], (consumer_index_0 // num_stages) & 1)
                                    buf = scale_a_consumer_0 % _SCALE_A_BUFFERS
                                    if col == 0:
                                        T.barrier_wait(
                                            scale_a_full[buf],
                                            (scale_a_consumer_0 // _SCALE_A_BUFFERS) & 1,
                                        )
                                    scales[0] = scale_a_ring[buf, acc_row0, col]
                                    scales[1] = scale_a_ring[buf, acc_row1, col]
                                    if col == scale_a_group - 1:
                                        if tx % 32 == 0:
                                            T.barrier_arrive(scale_a_empty[buf])
                                        scale_a_consumer_0 = scale_a_consumer_0 + 1
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
                                    # Nothing reads the stage past here; releasing it before
                                    # the promotion overlaps the next TMA with it.
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
                                # The previous tile's TMA store may still read shared_c;
                                # waiting here rather than after issuing it lets the
                                # store overlap this tile's mainloop.
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
                    # Drain the last tile's store before the CTA's shared memory goes.
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
                            for group in T.unroll(num_groups, unroll_factor=group_unroll):
                                for col in T.unroll(scale_a_group):
                                    kk = group * scale_a_group + col
                                    slot = consumer_index_1 % num_stages
                                    T.barrier_wait(full[slot], (consumer_index_1 // num_stages) & 1)
                                    buf = scale_a_consumer_1 % _SCALE_A_BUFFERS
                                    if col == 0:
                                        T.barrier_wait(
                                            scale_a_full[buf],
                                            (scale_a_consumer_1 // _SCALE_A_BUFFERS) & 1,
                                        )
                                    scales[0] = scale_a_ring[buf, half_m + acc_row0, col]
                                    scales[1] = scale_a_ring[buf, half_m + acc_row1, col]
                                    if col == scale_a_group - 1:
                                        if tx % 32 == 0:
                                            T.barrier_arrive(scale_a_empty[buf])
                                        scale_a_consumer_1 = scale_a_consumer_1 + 1
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
                                    # Nothing reads the stage past here; releasing it before
                                    # the promotion overlaps the next TMA with it.
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


class GemmFp81D2DFwdKernel(Kernel):
    """FP8 NT GEMM for 1D2D scales, bfloat16 output, no bias.

    ``scale_a`` is ``[M, ceil(K/128)]`` and ``scale_b`` is
    ``[ceil(N/128), ceil(K/128)]``, both row-major.

    Args:
        m: Rows of ``a``; at least 128.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim.
        dtype: Operand dtype; ``torch.float8_e4m3fn``.
        out_dtype: Output dtype; ``torch.bfloat16``.
        config: Kernel config override; unset keys take their default.
        tune: Accepted for the common kernel interface; no ``autotune_configs``
            are declared, so the schedule comes from ``default_config``.
        device_index: The device the kernel is built for.
        shared_epilogue: Whether to stage the tile through shared memory and store
            it with TMA. ``None`` takes the calibrated choice for this shape.
    """

    @staticmethod
    def _calibrated_epilogue(m: int, n: int, k: int, calibration: Optional[str]) -> bool:
        """Whether the schedule calibrated on *calibration*'s board stores through shared memory.

        False off a calibrated board or shape: the packed global store has the
        smaller unit, so it addresses every shape the other one does.
        """
        tuned = _FP8_1D2D_CONFIGS.get(calibration, {}).get((m, n, k))
        return bool(tuned["shared_epilogue"]) if tuned is not None else False

    @staticmethod
    def _shape_refusal(m: int, n: int, k: int, *, shared_epilogue: bool) -> Optional[str]:
        """Why this schedule cannot address these shapes, or ``None`` when it can.

        ``a``, ``b`` and ``scale_a`` arrive through TMA, whose descriptors address the
        innermost (contiguous) dimension in 16-byte units: ``k`` for the fp8 operands,
        ``ceil(k / 128)`` for the fp32 ``scale_a``. The epilogue adds its own
        unit: two BF16 columns per packed global store, or a descriptor row stride of
        16 bytes for the shared-memory path. Below one 128-row tile the tile is
        mostly padding.
        """
        if m < 128:
            return f"m={m} is below one 128-row tile"
        store_unit = 8 if shared_epilogue else 2
        offenders = [
            f"{name}={value} is not a multiple of {unit} ({what})"
            for name, value, unit, what in (
                ("k", k, 16, "a and b are read K-major through TMA, 16 fp8 per 16 bytes"),
                (
                    "ceil(k / 128)",
                    -(-k // 128),
                    4,
                    "scale_a is read row-major through TMA, 4 fp32 per 16 bytes",
                ),
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

    supported_archs = [90]

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        return (
            call.block_scale_grid == "1d2d"
            and call.dtype == torch.float8_e4m3fn
            and call.out_dtype == torch.bfloat16
            and not call.has_bias
            and cls._shape_refusal(
                call.m,
                call.n,
                call.k,
                shared_epilogue=cls._calibrated_epilogue(call.m, call.n, call.k, call.calibration),
            )
            is None
        )

    @classmethod
    def entry_for(cls, call: GemmCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.n, call.k, call.dtype, call.out_dtype, index)
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            call.out_dtype,
            tune=call.tune,
            device_index=index,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
        shared_epilogue: Optional[bool] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if dtype != torch.float8_e4m3fn:
            raise NotImplementedError(f"{type(self).__name__} takes float8_e4m3fn, got {dtype}")
        if out_dtype != torch.bfloat16:
            raise NotImplementedError(f"{type(self).__name__} writes bfloat16, got {out_dtype}")

        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.sm_count = get_sm_count(self.device_index)
        calibration = device_calibration(self.device_index)
        self._calibrated = _FP8_1D2D_CONFIGS.get(calibration, {}).get((m, n, k))
        if self._calibrated is not None:
            self.sm_count = int(self._calibrated.get("sm_count", self.sm_count))
        self.shared_epilogue = (
            self._calibrated_epilogue(m, n, k, calibration)
            if shared_epilogue is None
            else bool(shared_epilogue)
        )

        refusal = self._shape_refusal(m, n, k, shared_epilogue=self.shared_epilogue)
        if refusal is not None:
            raise ValueError(f"{type(self).__name__} cannot serve m={m} n={n} k={k}: {refusal}")

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
        if self._calibrated is not None:
            return dict(self._calibrated["kernel"])
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
            "group_unroll": 1,
        }

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError(f"{type(self).__name__} has no bias epilogue")
        return self.kernel(**self.config)(a, b, scale_a, scale_b)
