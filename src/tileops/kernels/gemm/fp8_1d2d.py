"""FP8 NT GEMM for block128 scales: ``scale_a`` per 1x128, ``scale_b`` per 128x128 or 1x128.

A persistent grid of one producer and one or two consumer warp-groups: the
producer fills a shared-memory ring through TMA, each consumer runs WGMMA over 64
of the tile's rows and folds every K-step's partial in under its scales. Two
builders share that plan: ``_gemm_fp8_1d2d_kernel`` holds 64- or 128-row tiles
under 128x128 ``scale_b`` blocks, and ``_gemm_fp8_block128_wave_kernel`` holds tiles of
one or two 128-row waves whose K-step products all go through one partial
accumulator, under either ``scale_b`` grid.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import (
    TensorCoreIntrinEmitter as WgmmaEmitter,
)

from tileops._csrc import csrc_include
from tileops.kernels.constants import (
    BLOCK_SHARED_BYTES_OPT_IN,
    SHARED_BUFFER_ALIGN_BYTES,
    TMA_DTYPE_BFLOAT16,
    TMA_DTYPE_UINT8,
    TMA_INTERLEAVE_NONE,
    TMA_L2_PROMOTION_128B,
    TMA_OOB_FILL_NONE,
    TMA_SWIZZLE_NONE,
)
from tileops.kernels.gemm.call_spec import GemmFP8Call, GemmFP8FwdInterface
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import device_calibration, get_sm_count

__all__ = ["GemmFP8Block128WaveFwdKernel", "GemmFP81D2DFwdKernel"]


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
    scale_a_groups: tuple[int, ...],
    scale_a_buffers: int,
) -> Callable:
    """Build the persistent 1D2D GEMM; the returned factory takes the tile config.

    The K loop runs in groups of ``scale_a_group`` K-steps, one ``scale_a`` staging
    each; ``group_unroll`` is how many groups one unrolled iteration covers.
    ``block_m`` is 64 or 128: one consumer warpgroup per 64 rows of the tile.
    """
    half_m = 64
    block_k = 128
    accum_dtype = "float"
    scale_k = (k + block_k - 1) // block_k
    scale_a_group = next(g for g in scale_a_groups if scale_k % g == 0)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[
            "-O3",
            "--use_fast_math",
            "-DENABLE_BF16",
            *csrc_include("fp8_1d2d_helper.h"),
        ],
    )
    def kernel_func(
        block_n: int = 128,
        num_stages: int = 3,
        group_size_m: int = 16,
        group_unroll: int = 1,
        block_m: int = 128,
    ) -> Callable:
        if block_m not in (half_m, 2 * half_m):
            raise ValueError(f"block_m must be {half_m} or {2 * half_m}, got {block_m}")
        if group_size_m < 1:
            raise ValueError(f"group_size_m must be positive, got {group_size_m}")
        if num_stages < 1:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        wgmma_helper = f"tileops::fp8_gemm_wgmma_64x128_by_128x{block_n}"
        promotion_helper = f"tileops::fp8_gemm_1d2d_promote_64x{block_n}"
        global_store_helper = f"tileops::fp8_gemm_raw_acc_store_global_64x{block_n}_v2"
        smem_store_helper = f"tileops::fp8_gemm_raw_acc_stsm_bf16_64x{block_n}"
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
        # The producer warpgroup, then one consumer warpgroup per 64 rows.
        consumer_warpgroups = block_m // half_m
        consumer_threads = 128 * consumer_warpgroups
        threads = 128 + consumer_threads
        # One arrival per consumer warp: ``wait_wgmma`` is warp-convergent, so lane 0
        # speaks for its warp.
        consumer_warps = consumer_threads // 32

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
            with T.Kernel(sm_count, threads=threads) as (pid,):
                a_shared = T.alloc_shared((num_stages, block_m, block_k), dtype)
                b_shared = T.alloc_shared((num_stages, block_n, block_k), dtype)
                partial_0 = T.alloc_local((fragment_regs,), accum_dtype)
                partial_1 = T.alloc_local((fragment_regs,), accum_dtype)
                final_0 = T.alloc_local((fragment_regs,), accum_dtype)
                final_1 = T.alloc_local((fragment_regs,), accum_dtype)
                if shared_epilogue:
                    shared_c = T.alloc_shared((block_m, block_n), out_dtype)
                scale_a_ring = T.alloc_shared(
                    (scale_a_buffers, block_m, scale_a_group), accum_dtype
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
                scale_a_full = T.alloc_barrier([1] * scale_a_buffers)
                scale_a_empty = T.alloc_barrier([consumer_warps] * scale_a_buffers)
                empty = T.alloc_barrier([consumer_warps] * num_stages)
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
                                T.sync_threads(barrier_id=10, arrive_count=threads)
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
                                            buf = scale_a_producer % scale_a_buffers
                                            T.barrier_wait(
                                                scale_a_empty[buf],
                                                ((scale_a_producer // scale_a_buffers) & 1) ^ 1,
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
                                T.sync_threads(barrier_id=10, arrive_count=threads)
                            T.clear(final_0)
                            for group in T.unroll(num_groups, unroll_factor=group_unroll):
                                for col in T.unroll(scale_a_group):
                                    kk = group * scale_a_group + col
                                    slot = consumer_index_0 % num_stages
                                    T.barrier_wait(full[slot], (consumer_index_0 // num_stages) & 1)
                                    buf = scale_a_consumer_0 % scale_a_buffers
                                    if col == 0:
                                        T.barrier_wait(
                                            scale_a_full[buf],
                                            (scale_a_consumer_0 // scale_a_buffers) & 1,
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
                                T.sync_threads(barrier_id=13, arrive_count=consumer_threads)
                                T.call_extern(
                                    "handle",
                                    smem_store_helper,
                                    final_0.data,
                                    T.address_of(shared_c[0, 0]),
                                )
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=14, arrive_count=consumer_threads)
                                if tx == 128:
                                    output_desc = T.create_tma_descriptor(
                                        TMA_DTYPE_BFLOAT16,
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
                                        TMA_INTERLEAVE_NONE,
                                        TMA_SWIZZLE_NONE,
                                        TMA_L2_PROMOTION_128B,
                                        TMA_OOB_FILL_NONE,
                                    )
                                    T.call_extern(
                                        "handle",
                                        "tileops::fp8_tma_store_2d_issue",
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

                elif consumer_warpgroups == 2:
                    T.inc_max_nreg(240)
                    for wave in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * wave + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if not stage_1d2d_b_per_k and scale_k < 16:
                                T.sync_threads(barrier_id=10, arrive_count=threads)
                            T.clear(final_1)
                            for group in T.unroll(num_groups, unroll_factor=group_unroll):
                                for col in T.unroll(scale_a_group):
                                    kk = group * scale_a_group + col
                                    slot = consumer_index_1 % num_stages
                                    T.barrier_wait(full[slot], (consumer_index_1 // num_stages) & 1)
                                    buf = scale_a_consumer_1 % scale_a_buffers
                                    if col == 0:
                                        T.barrier_wait(
                                            scale_a_full[buf],
                                            (scale_a_consumer_1 // scale_a_buffers) & 1,
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
                                T.sync_threads(barrier_id=13, arrive_count=consumer_threads)
                                T.call_extern(
                                    "handle",
                                    smem_store_helper,
                                    final_1.data,
                                    T.address_of(shared_c[half_m, 0]),
                                )
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=14, arrive_count=consumer_threads)
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


@functools.lru_cache(maxsize=32)
def _gemm_fp8_block128_wave_kernel(
    m: int, n: int, k: int, *, sm_count: int, scale_b_per_row: bool = False
) -> Callable:
    """Persistent block-scaled FP8 GEMM whose output tile is ``waves`` x 128 rows by ``block_n``.

    One producer warp issues every TMA: per K block the tile's A waves and its B
    tile into a ``num_stages`` ring, and per group of K blocks the tile's
    ``scale_a`` rows and ``scale_b`` entry into a two-buffer ring. Two consumer
    warpgroups each own 64 rows of every wave. Per K block and wave they form one
    WGMMA product and fold it into that wave's accumulator under the row scale
    ``scale_a[row, kb] * scale_b[n_block, kb]``, so one partial is live beside the
    accumulators whatever the wave count. Each wave lands in bf16 through its own
    shared-memory tile and a TMA store that overlaps the next tile's mainloop.

    With ``scale_b_per_row``, ``scale_b`` is per 1x128 block, ``[n, ceil(k / 128)]``:
    the scale ring stages the tile's ``block_n`` rows of it as it does ``scale_a``'s,
    and the fold multiplies each column by its own entry.

    A 176-wide tile can span three 128x128 ``scale_b`` blocks. Which of its columns
    take which block depends on where the tile starts, so each tile runs one of eight
    copies of the K loop, each folding with that split fixed at compile time.

    Args:
        m: Rows of ``a`` and of the output.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim; ``ceil(k / 128)`` is a multiple of 4.
        sm_count: Persistent grid width.
        scale_b_per_row: Whether ``scale_b`` is per 1x128 block rather than per 128x128.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, num_stages,
        group_size_m, waves, prefetch_b, scales_under_wgmma, column_run, scale_span)``
        returns the compiled ``prim_func``.
    """
    # One M wave: the rows the two consumer warpgroups cover with one 64-row WGMMA each.
    wave_m = 128
    block_k = 128
    # Columns of ``b`` one ``scale_b`` entry covers.
    scale_block_n = 128
    # The two consumer warpgroups, then the producer warpgroup.
    consumer_threads = 256
    producer_threads = 128
    # K steps one staging of the tile's scales covers: the most that divides ``ceil(K/128)``.
    # Eight fp32 of a ``scale_a`` row fill one 32-byte sector; four are TMA's 16-byte unit.
    # A per-row ``scale_b`` stages ``block_n`` rows per group, so it takes the 16-byte unit
    # and leaves the shared memory to the operand ring.
    scale_groups = (4,) if scale_b_per_row else (8, 4)
    # Buffers of the scale ring, so one group is written while the other is read.
    scale_buffers = 2
    # A per-row ``scale_b`` is read in the fold, after the stage that gates its refill is
    # released, so its ring keeps a third buffer: the producer reaches group g + 3 only
    # past stage (g + 3) * scale_group - num_stages, which follows group g's last fold
    # while num_stages <= 2 * scale_group.
    scale_b_buffers = 3 if scale_b_per_row else scale_buffers
    # fp32 per ``scale_b`` staging row. A group reads at most eight; 32 fp32 make each
    # buffer a multiple of 128 bytes, the alignment a TMA shared-memory destination needs.
    scale_b_span = 32
    # Registers a producer and a consumer thread hold after the warpgroup handoff.
    producer_regs = 24
    consumer_regs = 240
    # Named barrier the consumer warpgroups meet at around the output staging tile.
    epilogue_bar = 1

    k_blocks = -(-k // block_k)
    scale_n = n if scale_b_per_row else -(-n // scale_block_n)
    # TMA moves every operand: a global row must be a multiple of 16 bytes, and a scale
    # group must be a whole number of 16-byte scale_a boxes.
    if n % 8 != 0 or k % 16 != 0:
        raise ValueError(f"the wave kernel needs n % 8 == 0 and k % 16 == 0, got n={n}, k={k}")
    if k_blocks % min(scale_groups) != 0:
        raise ValueError(
            f"the wave kernel needs ceil(k / {block_k}) to be a multiple of "
            f"{min(scale_groups)}, got {k_blocks}"
        )
    divisor_group = next(g for g in scale_groups if k_blocks % g == 0)
    # A group spanning all of K stages the tile's ``scale_a`` rows, one contiguous run of
    # fp32, as rows of ``scale_a_span`` fp32: a box of ``block_m`` rows of ``scale_group``
    # fp32 costs TMA one request per 16-byte row, which on a K-512 tile measured longer
    # than the tile's A and B loads.
    scale_a_span = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16", *csrc_include("fp8_1d2d_helper.h")],
    )
    def _gemm_fp8_block128_wave_func(
        block_n: int = 128,
        num_stages: int = 3,
        group_size_m: int = 8,
        waves: int = 2,
        prefetch_b: int = 0,
        scales_under_wgmma: int = 0,
        column_run: int = 1,
        scale_span: int = 0,
    ) -> Callable:
        # ``scale_span`` makes one scale group span all of K, so a tile's ``scale_a`` is one
        # contiguous run staged once a tile; otherwise a group is the largest of
        # ``scale_groups`` that divides ``ceil(K/128)``.
        if scale_span and scale_b_per_row:
            raise ValueError("scale_span stages 128x128 scale_b blocks only")
        scale_group = k_blocks if scale_span else divisor_group
        if scale_group > scale_b_span:
            raise ValueError(
                f"a scale group of {scale_group} K blocks exceeds the {scale_b_span}-wide "
                "scale_b staging row"
            )
        num_scale_groups = k_blocks // scale_group
        # A whole-K ``scale_a`` span takes ``k_blocks / divisor_group`` times the shared
        # memory of a divisor group's, so it keeps one buffer: the producer stages the next
        # tile's span once every consumer warp has folded this tile's last K step
        # (``empty[num_stages]``), while they write the epilogue.
        scale_a_buffers = 1 if scale_span else scale_buffers
        if scale_span and not (
            (wave_m * scale_group) % scale_a_span == 0 and (m * k_blocks) % scale_a_span == 0
        ):
            raise ValueError(
                f"scale_span needs whole {scale_a_span}-fp32 rows of scale_a per tile, got "
                f"m={m} and {k_blocks} K blocks"
            )

        def buf_a(buf):
            """The ``scale_a`` ring buffer of scale group ``buf``'s ring position."""
            return 0 if scale_span else buf

        if group_size_m < 1:
            raise ValueError(f"group_size_m must be positive, got {group_size_m}")
        if num_stages < 1:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        # The widths ``fp8_1d2d_helper.h`` instantiates a wave WGMMA for. A 64- or
        # 128-wide tile reads one B-scale block; a 192-wide tile starts at a multiple
        # of 64 and spans two: columns below ``former`` take the first, the rest the
        # second. A 56-wide tile starts anywhere in a B-scale block and spans two when
        # it crosses a block boundary, with the same ``former`` split. A 176-wide tile
        # starts at a multiple of 16 and spans up to three: below ``former``, the next
        # 128 columns, then the rest.
        if block_n not in (56, 64, 128, 176, 192):
            raise ValueError(f"block_n must be one of 56/64/128/176/192, got {block_n}")
        if block_n == 176 and scale_b_per_row:
            raise ValueError("a 176-wide tile folds 128x128 scale_b blocks only")
        two_sb = block_n == 192 and not scale_b_per_row
        split_sb = block_n < scale_block_n and scale_block_n % block_n != 0 and not scale_b_per_row
        three_sb = block_n == 176
        sb_rows = 3 if three_sb else 2 if two_sb or split_sb else 1
        # Eight copies of a fully unrolled scale group overflow the instruction cache; a
        # 176-wide tile's K loop measured fastest unrolled four K steps deep.
        dispatch_unroll = 4
        # The scale ring has no barriers of its own (see the producer), which holds only
        # while a stage is released before the scale buffer it shares is refilled.
        if num_stages > scale_group + 1:
            raise ValueError(
                f"num_stages must be at most scale_group + 1 = {scale_group + 1}, got {num_stages}"
            )
        if waves not in (1, 2):
            raise ValueError(f"waves must be 1 or 2, got {waves}")
        block_m = wave_m * waves
        # Rows of ``scale_a_span`` fp32 one tile's ``scale_a`` staging takes, or 0 where a
        # group does not span all of K and each staging is a box of ``block_m`` rows.
        scale_a_rows = (
            block_m * scale_group // scale_a_span
            if num_scale_groups == 1
            and (wave_m * scale_group) % scale_a_span == 0
            and (m * k_blocks) % scale_a_span == 0
            else 0
        )

        def consumer_acc_layout(buf):
            """The accumulator layout T.wgmma_gemm infers for the two consumer warpgroups."""
            base = acc_emitter.make_mma_store_layout(buf)
            return T.Fragment(
                [wave_m, block_n],
                forward_thread_fn=lambda i, j: base.map_forward_thread([i, j])[0],
                forward_index_fn=lambda i, j: base.map_forward_index([i, j]),
            )

        def consumer_row_layout(buf):
            """One value per accumulator row, held by the four lanes of that row's quad.

            Row i lives in consumer warp i // 16; its quad is lane (i % 8) * 4..+3 and
            the value is register (i % 16) // 8, as in the accumulator layout.
            """
            return T.Fragment(
                [wave_m],
                forward_fn=lambda i, rep: (
                    (i // 16) * 32 + (i % 8) * 4 + rep,
                    (i % 16) // 8,
                ),
                replicate=4,
            )

        # The accumulator layout of a 128 x block_n WGMMA tile over two consumer
        # warpgroups (eight warps along M), as T.wgmma_gemm infers it.
        acc_emitter = WgmmaEmitter(
            a_dtype=T.float8_e4m3fn,
            b_dtype=T.float8_e4m3fn,
            accum_dtype=T.float32,
            a_transposed=False,
            b_transposed=True,
            block_row_warps=8,
            block_col_warps=1,
            warp_row_tiles=wave_m // 8,
            warp_col_tiles=block_n,
            chunk=block_k,
        )
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        max_tiles = -(-total_tiles // sm_count)
        consumer_warps = consumer_threads // 32
        # ``column_run`` consecutive CTAs take ``column_run`` consecutive column tiles of
        # one row tile, so CTAs launched side by side read the same A rows; a run is one
        # step of the group raster.
        if column_run < 1 or num_pid_n % column_run or sm_count % column_run:
            raise ValueError(
                f"column_run must divide the {num_pid_n} column tiles and sm_count "
                f"{sm_count}, got {column_run}"
            )
        run_cols = num_pid_n // column_run

        @T.macro
        def decode(flat_id, mt, nt):
            if column_run == 1:
                tiles_per_group = T.int32(group_size_m * num_pid_n)
                group_id = flat_id // tiles_per_group
                first_m = group_id * T.int32(group_size_m)
                group_m = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
                mt[0] = first_m + (flat_id % tiles_per_group) % group_m
                nt[0] = (flat_id % tiles_per_group) // group_m
            else:
                run_id = flat_id // column_run
                runs_per_group = T.int32(group_size_m * run_cols)
                first_m = run_id // runs_per_group * T.int32(group_size_m)
                group_m = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
                mt[0] = first_m + (run_id % runs_per_group) % group_m
                nt[0] = (run_id % runs_per_group) // group_m * column_run + flat_id % column_run

        @T.macro
        def wave_wgmma(partial, a_wave, b_smem, slot, consumer_wg):
            """Issue one 128-row wave's WGMMA over a K block into ``partial``."""
            T.call_extern(
                "handle",
                f"tileops::fp8_wave_wgmma_64x{block_n}",
                partial.data,
                T.address_of(a_wave[slot, 0, 0]),
                T.address_of(b_smem[slot, 0, 0]),
                consumer_wg[0],
            )

        @T.macro
        def wave_row_scales(row_scales, former, scale_a_ring, scale_b_ring, buf, col, wave):
            """Form one wave's row scales for a K block from the staged scales.

            ``row_scales`` holds the row scale under each ``scale_b`` block the tile can
            span, then, for a 192-wide tile, the one its columns 64..127 take.
            """
            for i in T.Parallel(wave_m):
                if scale_b_per_row:
                    row_scales[0][i] = scale_a_ring[buf_a(buf), wave * wave_m + i, col]
                else:
                    row_scales[0][i] = (
                        scale_a_ring[buf_a(buf), wave * wave_m + i, col] * scale_b_ring[buf, 0, col]
                    )
                if sb_rows > 1:
                    row_scales[1][i] = (
                        scale_a_ring[buf_a(buf), wave * wave_m + i, col] * scale_b_ring[buf, 1, col]
                    )
                if sb_rows > 2:
                    row_scales[2][i] = (
                        scale_a_ring[buf_a(buf), wave * wave_m + i, col] * scale_b_ring[buf, 2, col]
                    )
                if two_sb:
                    # Columns 64..127 take the first B scale when the tile starts a 128 block.
                    row_scales[2][i] = T.if_then_else(
                        former[0] == scale_block_n, row_scales[0][i], row_scales[1][i]
                    )

        @T.macro
        def wave_step(
            partial,
            row_scales,
            former,
            a_wave,
            b_smem,
            scale_a_ring,
            scale_b_ring,
            slot,
            buf,
            col,
            wave,
            consumer_wg,
        ):
            """Multiply one 128-row wave of a K block into ``partial``; form its row scales.

            With ``scales_under_wgmma`` the WGMMA is issued first and the row scales are
            formed while it runs; otherwise they are formed first.
            """
            if scales_under_wgmma:
                wave_wgmma(partial, a_wave, b_smem, slot, consumer_wg)
                wave_row_scales(row_scales, former, scale_a_ring, scale_b_ring, buf, col, wave)
            else:
                wave_row_scales(row_scales, former, scale_a_ring, scale_b_ring, buf, col, wave)
                wave_wgmma(partial, a_wave, b_smem, slot, consumer_wg)
            T.wait_wgmma(0)

        @T.macro
        def wave_fold(final, partial, row_scales, former, scale_b_ring, buf, col, split):
            """Fold one wave's K-block product into its accumulator under its scales.

            A per-row ``scale_b`` goes through a helper that reads each of the thread's
            column scales once; an element-wise read costs a shared-memory load per
            accumulator element. A tile that can cross a B-scale block goes through a
            helper that picks each 8-column chunk's row scales once, at compile time for
            a 176-wide tile, whose ``split`` is its ``former``.
            """
            if three_sb:
                T.call_extern(
                    "handle",
                    f"tileops::fp8_wave_fold_three_{block_n}_{split}",
                    final.data,
                    partial.data,
                    row_scales[0].data,
                    row_scales[1].data,
                    row_scales[2].data,
                )
            elif split_sb:
                T.call_extern(
                    "handle",
                    f"tileops::fp8_wave_fold_split_{block_n}",
                    final.data,
                    partial.data,
                    row_scales[0].data,
                    row_scales[1].data,
                    former[0],
                )
            elif scale_b_per_row:
                T.call_extern(
                    "handle",
                    f"tileops::fp8_wave_fold_per_column_{block_n}x{scale_group}",
                    final.data,
                    partial.data,
                    row_scales[0].data,
                    T.address_of(scale_b_ring[buf, 0, col]),
                )
            else:
                for i, j in T.Parallel(wave_m, block_n):
                    if two_sb:
                        final[i, j] += partial[i, j] * T.if_then_else(
                            j < scale_block_n // 2,
                            row_scales[0][i],
                            T.if_then_else(j < scale_block_n, row_scales[2][i], row_scales[1][i]),
                        )
                    else:
                        final[i, j] += partial[i, j] * row_scales[0][i]

        @T.macro
        def tile_mainloop(
            final0,
            final1,
            partial,
            row_scales,
            former,
            a_wave0,
            a_wave1,
            b_smem,
            scale_a_ring,
            scale_b_ring,
            full,
            empty,
            consumer_wg,
            counters,
            tx,
            split,
        ):
            """Run one tile's K loop: every wave's WGMMA and fold per K block.

            ``counters`` holds the consumer's ring position and scale-group count; a macro
            cannot rebind a variable it is handed. ``split`` is the tile's ``former`` where
            the fold fixes it at compile time, else 0.
            """
            for _group in T.serial(num_scale_groups):
                for col in T.unroll(scale_group, unroll_factor=dispatch_unroll if split else None):
                    # Every consumer thread holds the same counters; the broadcast lets the
                    # compiler keep what derives from them in uniform registers.
                    slot = T.call_extern(
                        "int32", "tileops::fp8_uniform", T.cast(counters[0] % num_stages, "int32")
                    )
                    buf = T.call_extern(
                        "int32",
                        "tileops::fp8_uniform",
                        T.cast(counters[1] % scale_buffers, "int32"),
                    )
                    buf_b = T.call_extern(
                        "int32",
                        "tileops::fp8_uniform",
                        T.cast(counters[1] % scale_b_buffers, "int32"),
                    )
                    T.barrier_wait(full[slot], (counters[0] // num_stages) & 1)
                    wave_step(
                        partial,
                        row_scales,
                        former,
                        a_wave0,
                        b_smem,
                        scale_a_ring,
                        scale_b_ring,
                        slot,
                        buf,
                        col,
                        0,
                        consumer_wg,
                    )
                    # The stage and, after its last K step, the scale group are released
                    # once the tile's last wave has read them.
                    if waves == 1 and tx % 32 == 0:
                        T.barrier_arrive(empty[slot])
                    wave_fold(final0, partial, row_scales, former, scale_b_ring, buf_b, col, split)
                    if waves == 2:
                        wave_step(
                            partial,
                            row_scales,
                            former,
                            a_wave1,
                            b_smem,
                            scale_a_ring,
                            scale_b_ring,
                            slot,
                            buf,
                            col,
                            1,
                            consumer_wg,
                        )
                        if tx % 32 == 0:
                            T.barrier_arrive(empty[slot])
                        wave_fold(
                            final1, partial, row_scales, former, scale_b_ring, buf_b, col, split
                        )
                    if col == scale_group - 1:
                        counters[1] = counters[1] + 1
                        if scale_span and tx % 32 == 0:
                            T.barrier_arrive(empty[num_stages])
                    counters[0] = counters[0] + 1

        @T.macro
        def tile_mainloop_from(
            split,
            final0,
            final1,
            partial,
            row_scales,
            former,
            a_wave0,
            a_wave1,
            b_smem,
            scale_a_ring,
            scale_b_ring,
            full,
            empty,
            consumer_wg,
            counters,
            tx,
        ):
            """Run the K loop copy whose fold fixes ``former`` to the tile's, from ``split`` up.

            A 176-wide tile starts at a multiple of 16 columns, so ``former`` is one of
            16, 32, ..., 128; the last copy takes 128 unchecked.
            """
            if split == scale_block_n or former[0] == split:
                tile_mainloop(
                    final0,
                    final1,
                    partial,
                    row_scales,
                    former,
                    a_wave0,
                    a_wave1,
                    b_smem,
                    scale_a_ring,
                    scale_b_ring,
                    full,
                    empty,
                    consumer_wg,
                    counters,
                    tx,
                    split,
                )
            else:
                tile_mainloop_from(
                    split + 16,
                    final0,
                    final1,
                    partial,
                    row_scales,
                    former,
                    a_wave0,
                    a_wave1,
                    b_smem,
                    scale_a_ring,
                    scale_b_ring,
                    full,
                    empty,
                    consumer_wg,
                    counters,
                    tx,
                )

        @T.prim_func
        def _gemm_fp8_block128_wave_main(
            a: T.Tensor((m, k), "float8_e4m3fn"),  # type: ignore
            b: T.Tensor((n, k), "float8_e4m3fn"),  # type: ignore
            scale_a: T.Tensor((m, k_blocks), "float32"),  # type: ignore
            scale_b: T.Tensor((scale_n, k_blocks), "float32"),  # type: ignore
            c: T.Tensor((m, n), "bfloat16"),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=consumer_threads + producer_threads) as (pid,):
                if scale_a_rows:
                    scale_a_flat = T.reshape(scale_a, (m * k_blocks // scale_a_span, scale_a_span))
                a_wave0 = T.alloc_shared((num_stages, wave_m, block_k), "float8_e4m3fn")
                a_wave1 = T.alloc_shared((num_stages, wave_m, block_k), "float8_e4m3fn")
                b_smem = T.alloc_shared((num_stages, block_n, block_k), "float8_e4m3fn")
                scale_a_ring = T.alloc_shared((scale_a_buffers, block_m, scale_group), "float32")
                if scale_a_rows:
                    scale_a_ring_flat = T.reshape(
                        scale_a_ring, (scale_a_buffers, scale_a_rows, scale_a_span)
                    )
                if scale_b_per_row:
                    scale_b_ring = T.alloc_shared(
                        (scale_b_buffers, block_n, scale_group), "float32"
                    )
                else:
                    scale_b_ring = T.alloc_shared((scale_buffers, sb_rows, scale_b_span), "float32")
                # One output staging tile per wave: a 128-row tile is what T.copy writes
                # with stmatrix, where a slice of a 256-row tile takes 4-byte stores.
                c_smem0 = T.alloc_shared((wave_m, block_n), "bfloat16")
                c_smem1 = T.alloc_shared((wave_m, block_n), "bfloat16") if waves == 2 else None
                partial = T.alloc_fragment((wave_m, block_n), "float32")
                final0 = T.alloc_fragment((wave_m, block_n), "float32")
                final1 = T.alloc_fragment((wave_m, block_n), "float32")
                c_cast = T.alloc_fragment((wave_m, block_n), "bfloat16")
                row_scale = T.alloc_fragment((wave_m,), "float32")
                row_scale1 = T.alloc_fragment((wave_m,), "float32")
                row_scale2 = T.alloc_fragment((wave_m,), "float32")
                former = T.alloc_local((1,), "int32")
                consumer_wg = T.alloc_local((1,), "int32")
                T.annotate_layout(
                    {
                        a_wave0: tilelang.layout.make_swizzled_layout(a_wave0),
                        a_wave1: tilelang.layout.make_swizzled_layout(a_wave1),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem0: tilelang.layout.make_swizzled_layout(c_smem0),
                        **(
                            {c_smem1: tilelang.layout.make_swizzled_layout(c_smem1)}
                            if waves == 2
                            else {}
                        ),
                        partial: consumer_acc_layout(partial),
                        final0: consumer_acc_layout(final0),
                        final1: consumer_acc_layout(final1),
                        c_cast: consumer_acc_layout(c_cast),
                        row_scale: consumer_row_layout(row_scale),
                        row_scale1: consumer_row_layout(row_scale1),
                        row_scale2: consumer_row_layout(row_scale2),
                    }
                )

                # One arrival per consumer warp releases a slot: ``wait_wgmma`` is
                # warp-convergent, so lane 0 speaks for its warp.
                full = T.alloc_barrier([1] * num_stages)
                # Under ``scale_span`` one more barrier, ``empty[num_stages]``, frees the
                # ``scale_a`` span after each tile's last fold.
                empty = T.alloc_barrier([consumer_warps] * (num_stages + (1 if scale_span else 0)))

                producer_index = T.alloc_var("uint32", init=0)
                scale_producer = T.alloc_var("uint32", init=0)
                # The consumer's ring position and scale-group count.
                counters = T.alloc_local((2,), "uint32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")
                tx = T.get_thread_binding()

                # Consumers are threads 0..255, the producer warpgroup 256..383, so the
                # accumulator layouts' thread numbers are the threads themselves.
                if tx >= consumer_threads:
                    T.dec_max_nreg(producer_regs)
                    if tx < consumer_threads + 32:
                        for w in T.serial(max_tiles):
                            flat_id = T.int32(sm_count) * w + pid
                            if flat_id < total_tiles:
                                decode(flat_id, mt, nt)
                                m_start = mt[0] * block_m
                                n_start = nt[0] * block_n
                                scale_row = n_start // scale_block_n
                                for group in T.serial(num_scale_groups):
                                    for col in T.unroll(scale_group):
                                        kb = group * scale_group + col
                                        ks = kb * block_k
                                        slot = T.cast(producer_index % num_stages, "int32")
                                        T.barrier_wait(
                                            empty[slot], ((producer_index // num_stages) & 1) ^ 1
                                        )
                                        if col == 0:
                                            buf = T.cast(scale_producer % scale_buffers, "int32")
                                            # The group's scales ride the full barrier of
                                            # its first K step. Group g + 2 reuses this
                                            # buffer once stage (kb - num_stages) is
                                            # released, at or after group g's last K step
                                            # since num_stages <= scale_group + 1. Rows
                                            # past m and columns past ceil(k/128) are
                                            # zero-filled.
                                            if scale_span:
                                                pass  # staged after the K step's A and B
                                            elif scale_a_rows:
                                                flat_row = m_start * k_blocks // scale_a_span
                                                T.tma_copy(
                                                    scale_a_flat[
                                                        flat_row : flat_row + scale_a_rows, :
                                                    ],
                                                    scale_a_ring_flat[buf, :, :],
                                                    barrier=full[slot],
                                                )
                                            else:
                                                T.tma_copy(
                                                    scale_a[
                                                        m_start : m_start + block_m,
                                                        kb : kb + scale_group,
                                                    ],
                                                    scale_a_ring[buf, :, :],
                                                    barrier=full[slot],
                                                )
                                            if scale_b_per_row:
                                                buf_b = T.cast(
                                                    scale_producer % scale_b_buffers, "int32"
                                                )
                                                T.tma_copy(
                                                    scale_b[
                                                        n_start : n_start + block_n,
                                                        kb : kb + scale_group,
                                                    ],
                                                    scale_b_ring[buf_b, :, :],
                                                    barrier=full[slot],
                                                )
                                            else:
                                                T.tma_copy(
                                                    scale_b[
                                                        scale_row : scale_row + sb_rows,
                                                        kb : kb + scale_b_span,
                                                    ],
                                                    scale_b_ring[buf, :, :],
                                                    barrier=full[slot],
                                                )
                                            scale_producer = scale_producer + 1
                                        T.tma_copy(
                                            a[m_start : m_start + wave_m, ks : ks + block_k],
                                            a_wave0[slot, :, :],
                                            barrier=full[slot],
                                        )
                                        if waves == 2:
                                            T.tma_copy(
                                                a[
                                                    m_start + wave_m : m_start + 2 * wave_m,
                                                    ks : ks + block_k,
                                                ],
                                                a_wave1[slot, :, :],
                                                barrier=full[slot],
                                            )
                                        T.tma_copy(
                                            b[n_start : n_start + block_n, ks : ks + block_k],
                                            b_smem[slot, :, :],
                                            barrier=full[slot],
                                        )
                                        if prefetch_b:  # noqa: SIM102 -- a trace-time switch
                                            # Pull the next K block of B into L2 now, so its
                                            # TMA load a stage later does not wait on DRAM.
                                            if tx == consumer_threads and ks + block_k < k:
                                                b_prefetch = T.create_tma_descriptor(
                                                    TMA_DTYPE_UINT8,
                                                    2,
                                                    b.data,
                                                    k,
                                                    n,
                                                    1,
                                                    k,
                                                    block_k,
                                                    block_n,
                                                    1,
                                                    1,
                                                    TMA_INTERLEAVE_NONE,
                                                    TMA_SWIZZLE_NONE,
                                                    TMA_L2_PROMOTION_128B,
                                                    TMA_OOB_FILL_NONE,
                                                )
                                                T.call_extern(
                                                    "handle",
                                                    "tileops::fp8_tma_prefetch_2d",
                                                    b_prefetch,
                                                    ks + block_k,
                                                    n_start,
                                                )
                                        if scale_span and col == 0:
                                            # The tile's span waits for the previous tile's
                                            # last fold; its A and B are already in flight.
                                            T.barrier_wait(
                                                empty[num_stages], ((scale_producer - 1) & 1) ^ 1
                                            )
                                            flat_row = m_start * k_blocks // scale_a_span
                                            T.tma_copy(
                                                scale_a_flat[flat_row : flat_row + scale_a_rows, :],
                                                scale_a_ring_flat[0, :, :],
                                                barrier=full[slot],
                                            )
                                        if tx == consumer_threads:
                                            T.barrier_arrive(full[slot])
                                        producer_index = producer_index + 1
                else:
                    T.inc_max_nreg(consumer_regs)
                    counters[0] = 0
                    counters[1] = 0
                    # The consumer warpgroup, broadcast from lane 0 once: the compiler keeps
                    # it, and the A descriptors derived from it, in uniform registers.
                    consumer_wg[0] = T.call_extern("int32", "tileops::fp8_uniform", tx // 128)
                    for w in T.serial(max_tiles):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if two_sb or split_sb or three_sb:
                                former[0] = T.min(
                                    T.int32(block_n), scale_block_n - n_start % scale_block_n
                                )
                            T.clear(final0)
                            if waves == 2:
                                T.clear(final1)
                            row_scales = (row_scale, row_scale1, row_scale2)
                            if three_sb:
                                tile_mainloop_from(
                                    16,
                                    final0,
                                    final1,
                                    partial,
                                    row_scales,
                                    former,
                                    a_wave0,
                                    a_wave1,
                                    b_smem,
                                    scale_a_ring,
                                    scale_b_ring,
                                    full,
                                    empty,
                                    consumer_wg,
                                    counters,
                                    tx,
                                )
                            else:
                                tile_mainloop(
                                    final0,
                                    final1,
                                    partial,
                                    row_scales,
                                    former,
                                    a_wave0,
                                    a_wave1,
                                    b_smem,
                                    scale_a_ring,
                                    scale_b_ring,
                                    full,
                                    empty,
                                    consumer_wg,
                                    counters,
                                    tx,
                                    0,
                                )
                            # The previous tile's TMA stores may still read the staging
                            # tiles; waiting here lets them overlap this tile's mainloop.
                            # The thread that issued them waits; the others meet it at the
                            # 256-thread barrier TileLang places before the first staging
                            # write (checked in test_gemm.py), which orders the writes to
                            # both staging tiles after the wait. A
                            # one-wave tile also meets at the epilogue barrier first: at
                            # the locked clock that measured faster with one wave and
                            # slower with two, where TileLang adds a barrier per wave.
                            T.tma_store_wait(0)
                            if waves == 1:
                                T.sync_threads(
                                    barrier_id=epilogue_bar, arrive_count=consumer_threads
                                )
                            T.copy(final0, c_cast)
                            T.copy(c_cast, c_smem0)
                            if waves == 2:
                                T.copy(final1, c_cast)
                                T.copy(c_cast, c_smem1)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=epilogue_bar, arrive_count=consumer_threads)
                            T.tma_copy(
                                c_smem0, c[m_start : m_start + wave_m, n_start : n_start + block_n]
                            )
                            if waves == 2:
                                T.tma_copy(
                                    c_smem1,
                                    c[
                                        m_start + wave_m : m_start + block_m,
                                        n_start : n_start + block_n,
                                    ],
                                )
                    # Drain the last tile's store before the CTA's shared memory goes.
                    T.tma_store_wait(0)

        return _gemm_fp8_block128_wave_main

    return _gemm_fp8_block128_wave_func


class GemmFP81D2DFwdKernel(Kernel, GemmFP8FwdInterface):
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
        device_index: The device the kernel is built for.
        shared_epilogue: Whether to stage the tile through shared memory and store
            it with TMA. ``None`` takes the calibrated choice for this shape.
    """

    # K-steps one A-scale staging covers, the most that divides ``ceil(K/128)``: eight fp32
    # of a row-major ``scale_a`` row fill one 32-byte sector, four are TMA's 16-byte unit.
    _SCALE_A_GROUPS = (16, 8, 4)
    # Buffers of the A-scale ring, so one group is written while the other is read.
    _SCALE_A_BUFFERS = 2

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

    @classmethod
    def _calibrated_epilogue(cls, m: int, n: int, k: int, calibration: Optional[str]) -> bool:
        """Whether the schedule calibrated on *calibration*'s board stores through shared memory.

        False off a calibrated board or shape: the packed global store has the
        smaller unit, so it addresses every shape the other one does.
        """
        tuned = cls._FP8_1D2D_CONFIGS.get(calibration, {}).get((m, n, k))
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
    def refusal(cls, call: GemmFP8Call) -> "str | None":
        if cls.block_scale_grid(call) != "1d2d":
            return "requires 1d2d block scales"
        if call.dtype != torch.float8_e4m3fn:
            return f"requires float8_e4m3fn operands, got {call.dtype}"
        if call.out_dtype != torch.bfloat16:
            return f"writes bfloat16, not {call.out_dtype}"
        if call.has_bias:
            return "does not serve a bias"

        reason = cls._shape_refusal(
            call.m,
            call.n,
            call.k,
            shared_epilogue=cls._calibrated_epilogue(call.m, call.n, call.k, call.calibration),
        )
        return reason or super().refusal(call)

    @classmethod
    def entry_for(cls, call: GemmFP8Call) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.n, call.k, call.dtype, call.out_dtype, index)
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            call.out_dtype,
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
        self._calibrated = self._FP8_1D2D_CONFIGS.get(calibration, {}).get((m, n, k))
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
            scale_a_groups=self._SCALE_A_GROUPS,
            scale_a_buffers=self._SCALE_A_BUFFERS,
        )
        self.init_config(config)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        if self._calibrated is not None:
            return {"block_m": 128, **self._calibrated["kernel"]}
        # A single 128-row tile splits into two 64-row ones, one consumer warpgroup each:
        # twice the CTAs over the same N, each reading half the ``A`` rows.
        block_m = 64 if self.m <= 128 else 128
        m_tiles = -(-self.m // block_m)
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
            "num_stages": self._deepest_ring(block_m, block_n),
            "group_size_m": 16,
            "group_unroll": 1,
            "block_m": block_m,
        }

    def _deepest_ring(self, block_m: int, block_n: int) -> int:
        """The most K-steps in flight the shared-memory budget holds, at most one tile's K-steps.

        A stage holds a ``block_m``x128 ``A`` tile and a ``block_n``x128 ``B`` tile, both
        fp8. Beside the ring live the two ``scale_a`` buffers, the ``scale_b`` staging (one
        row of K-steps at most) and, on the shared epilogue, the bf16 output tile. TileLang
        starts each of these buffers on a ``SHARED_BUFFER_ALIGN_BYTES`` boundary, and
        places them after the mbarriers, which are static shared memory and fit in one
        such unit. The uncalibrated shapes this serves wait on memory, and a deeper ring
        keeps more of each SM's loads in flight.
        """
        block_k = 128
        align = SHARED_BUFFER_ALIGN_BYTES
        scale_k = -(-self.k // block_k)
        scale_a_group = next(g for g in self._SCALE_A_GROUPS if scale_k % g == 0)
        per_stage = (block_m + block_n) * block_k
        # The mbarriers' unit, the two ``scale_a`` buffers and the ``scale_b`` row, each
        # rounded up to whole units.
        fixed = align
        fixed += -(-self._SCALE_A_BUFFERS * block_m * scale_a_group * 4 // align) * align
        fixed += -(-scale_k * 4 // align) * align
        if self.shared_epilogue:
            fixed += -(-block_m * block_n * 2 // align) * align
        return max(1, min(scale_k, (BLOCK_SHARED_BYTES_OPT_IN[90] - fixed) // per_stage))

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


class GemmFP8Block128WaveFwdKernel(Kernel, GemmFP8FwdInterface):
    """FP8 NT GEMM for block128 scales on calibrated shapes, in tiles of 128-row waves.

    ``scale_a`` is ``[M, ceil(K/128)]``; ``scale_b`` is per 128x128 block,
    ``[ceil(N/128), ceil(K/128)]``, or per 1x128 block, ``[N, ceil(K/128)]``, both
    row-major. It serves only the shapes the device's board holds a schedule for:
    ``_FP8_1D2D_WAVE_CONFIGS`` under 128x128 ``scale_b`` blocks,
    ``_FP8_1D1D_WAVE_CONFIGS`` under 1x128 ones.

    Args:
        m: Rows of ``a``.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim.
        dtype: Operand dtype; ``torch.float8_e4m3fn``.
        out_dtype: Output dtype; ``torch.bfloat16``.
        config: Kernel config override; unset keys take their default.
        device_index: The device the kernel is built for.
        scale_b_per_row: Whether ``scale_b`` is per 1x128 block rather than per 128x128.
    """

    supported_archs = [90]
    preferred_over = frozenset({"gemm_fp8_1d2d", "gemm_fp8_block_scale"})

    # Wave schedules measured per (m, n, k), by calibrated board; this kernel
    # serves exactly these shapes under 128x128 ``scale_b`` blocks.
    _FP8_1D2D_WAVE_CONFIGS: dict[str, dict[tuple[int, int, int], dict[str, object]]] = {
        "h200": {
            (128, 24576, 1536): {
                "kernel": {"block_n": 192, "num_stages": 4, "group_size_m": 8, "waves": 1},
                "sm_count": 128,
            },
            (128, 32768, 512): {
                "kernel": {"block_n": 128, "num_stages": 5, "group_size_m": 8, "waves": 1},
                "sm_count": 132,
            },
            (128, 7168, 16384): {
                "kernel": {
                    "block_n": 56,
                    "num_stages": 8,
                    "group_size_m": 8,
                    "waves": 1,
                    "prefetch_b": 1,
                    "scales_under_wgmma": 1,
                },
                "sm_count": 128,
            },
            (4096, 576, 7168): {
                "kernel": {"block_n": 192, "num_stages": 4, "group_size_m": 8, "waves": 1},
                "sm_count": 96,
            },
            (4096, 2112, 7168): {
                "kernel": {"block_n": 176, "num_stages": 4, "group_size_m": 8, "waves": 1},
                "sm_count": 128,
            },
            (4096, 4096, 7168): {
                "kernel": {
                    "block_n": 128,
                    "num_stages": 3,
                    "group_size_m": 8,
                    "waves": 2,
                    "column_run": 4,
                },
                "sm_count": 128,
            },
            (4096, 7168, 2048): {
                "kernel": {"block_n": 128, "num_stages": 3, "group_size_m": 8, "waves": 2},
                "sm_count": 128,
            },
            (4096, 7168, 16384): {
                "kernel": {
                    "block_n": 128,
                    "num_stages": 3,
                    "group_size_m": 8,
                    "waves": 2,
                    "column_run": 4,
                },
                "sm_count": 128,
            },
            (4096, 24576, 1536): {
                "kernel": {
                    "block_n": 128,
                    "num_stages": 3,
                    "group_size_m": 8,
                    "waves": 2,
                    "prefetch_b": 1,
                    "column_run": 4,
                    "scale_span": 1,
                },
                "sm_count": 128,
            },
            (4096, 32768, 512): {
                "kernel": {
                    "block_n": 128,
                    "num_stages": 3,
                    "group_size_m": 16,
                    "waves": 2,
                    "prefetch_b": 1,
                },
                "sm_count": 132,
            },
        },
    }

    # The same, under 1x128 ``scale_b`` blocks: the scale ring then stages ``block_n`` rows of
    # ``scale_b`` per group, which leaves room for fewer stages than the 128x128 schedule.
    _FP8_1D1D_WAVE_CONFIGS: dict[str, dict[tuple[int, int, int], dict[str, object]]] = {
        "h200": {
            (128, 7168, 16384): {
                "kernel": {"block_n": 64, "num_stages": 5, "group_size_m": 8, "waves": 1},
                "sm_count": 112,
            },
            (4096, 576, 7168): {
                "kernel": {"block_n": 192, "num_stages": 4, "group_size_m": 8, "waves": 1},
                "sm_count": 96,
            },
            (4096, 2112, 7168): {
                "kernel": {"block_n": 192, "num_stages": 4, "group_size_m": 8, "waves": 1},
                "sm_count": 118,
            },
            (4096, 7168, 16384): {
                "kernel": {"block_n": 128, "num_stages": 3, "group_size_m": 8, "waves": 2},
                "sm_count": 128,
            },
        },
    }

    @classmethod
    def refusal(cls, call: GemmFP8Call) -> "str | None":
        if cls.block_scale_grid(call) is None:
            return "requires block128 scales"
        if call.dtype != torch.float8_e4m3fn:
            return f"requires float8_e4m3fn operands, got {call.dtype}"
        if call.out_dtype != torch.bfloat16:
            return f"writes bfloat16, not {call.out_dtype}"
        if call.has_bias:
            return "does not serve a bias"

        configs = cls._configs(cls.block_scale_grid(call) == "1d1d")
        if (call.m, call.n, call.k) not in configs.get(call.calibration, {}):
            return f"has no wave config for {(call.m, call.n, call.k)} on this board"
        return super().refusal(call)

    @classmethod
    def _configs(
        cls, scale_b_per_row: bool
    ) -> dict[str, dict[tuple[int, int, int], dict[str, object]]]:
        """The schedule table for the ``scale_b`` grid."""
        return cls._FP8_1D1D_WAVE_CONFIGS if scale_b_per_row else cls._FP8_1D2D_WAVE_CONFIGS

    @classmethod
    def entry_for(cls, call: GemmFP8Call) -> Entry:
        index = call.device.index if call.device is not None else None
        per_row = cls.block_scale_grid(call) == "1d1d"
        identity = (call.m, call.n, call.k, call.dtype, call.out_dtype, index, per_row)
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            call.out_dtype,
            device_index=index,
            scale_b_per_row=per_row,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
        scale_b_per_row: bool = False,
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
        self.scale_b_per_row = scale_b_per_row
        calibration = device_calibration(self.device_index)
        self._calibrated = self._configs(scale_b_per_row).get(calibration, {}).get((m, n, k))
        if self._calibrated is None:
            raise ValueError(
                f"{type(self).__name__} has no schedule for m={m} n={n} k={k} on {calibration!r}"
            )
        self.sm_count = int(self._calibrated["sm_count"])
        self.kernel = _gemm_fp8_block128_wave_kernel(
            m, n, k, sm_count=self.sm_count, scale_b_per_row=scale_b_per_row
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return dict(self._calibrated["kernel"])

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
