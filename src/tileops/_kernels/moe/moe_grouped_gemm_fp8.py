"""Block-scaled FP8 grouped GEMM on SM90, with TMA loads and WGMMA accumulation.

Supports masked slabs and contiguous expert rows. A scales cover 1x128 blocks;
B scales cover 128x128 blocks. Accumulation is float32 and output is bfloat16.
"""

import itertools
import math
from typing import Callable, ClassVar, Mapping, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import (
    QUANT_SCALE_BLOCK,
    TMA_DTYPE_BFLOAT16,
    TMA_DTYPE_UINT8,
    TMA_INTERLEAVE_NONE,
    TMA_L2_PROMOTION_128B,
    TMA_L2_PROMOTION_256B,
    TMA_OOB_FILL_NONE,
    TMA_SWIZZLE_64B,
    TMA_SWIZZLE_128B,
)
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.moe.call_spec import MGroupedGEMMFP8Call, MGroupedGEMMFP8FwdInterface
from tileops.manifest.primitives import moe_layout_metadata

__all__ = ["MoEGroupedGEMMFP8Kernel"]

_CONFIG_KEYS = ("block_m", "block_n", "num_stages", "num_sms", "stage_scales")


def _b_scale_sub_blocks(block_n: int) -> int:
    """Number of column sub-blocks aligned to the 128-column scale grid."""
    return block_n // math.gcd(QUANT_SCALE_BLOCK, block_n)


def _smem_bytes(block_m: int, block_n: int, num_stages: int, stage_scales: int) -> int:
    """Bytes for the A/B ring, BF16 output tile, scales and barriers."""
    ring = num_stages * (block_m + block_n) * QUANT_SCALE_BLOCK
    scales = _b_scale_sub_blocks(block_n) + (block_m if stage_scales else 0)
    return ring + block_m * block_n * 2 + num_stages * (4 * scales + 16)


@target_cache(maxsize=32)
def _moe_grouped_gemm_fp8_kernel(
    num_groups: int,
    n: int,
    k: int,
    max_m: Optional[int],
    sm_count: int,
    smem_limit: int,
    packing: Optional[str] = None,
    metadata_kind: Optional[str] = None,
    alignment: int = 1,
) -> Callable:
    """Build a persistent FP8 grouped GEMM for the given layout.

    Tight per-row ids must be sorted. Aligned tiles require ``block_m`` to divide
    the alignment; id ``num_groups`` denotes trailing padding. Tight tail stores
    must preserve the next expert's rows. N tails and all output stores are bounded.
    """
    warpgroup_m = 64
    # Group M tiles to share weight tiles within a wave.
    swizzle_m_tiles = 16
    # Cursor fields: expert, preceding M tiles, expert tiles, first row, row count.
    cur_g, cur_m, cur_tiles, cur_row0, cur_rows = 0, 1, 2, 3, 4
    accum_dtype = "float"
    ab_dtype = "float8_e4m3fn"
    cd_dtype = "bfloat16"
    block_k = QUANT_SCALE_BLOCK
    scale_k = k // block_k
    contiguous = packing is not None
    tight = packing == "tight"
    # Aligned per-row ids name each row tile's expert; tight ones need each expert's end.
    rows_walk = metadata_kind == "per_row" and not tight
    recover = metadata_kind == "per_row" and tight

    @tilelang.jit(
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        compile_flags=[
            "-O3",
            "--use_fast_math",
            "-DENABLE_BF16",
            *csrc_include("moe_grouped_gemm_fp8_helper.h"),
        ],
    )
    def _moe_grouped_gemm_fp8_func(
        block_m: int, block_n: int, num_stages: int, num_sms: int, stage_scales: int
    ) -> Callable:
        config = dict(
            zip(_CONFIG_KEYS, (block_m, block_n, num_stages, num_sms, stage_scales), strict=True)
        )
        reason = MoEGroupedGEMMFP8Kernel.config_refusal(
            config,
            sm_count=sm_count,
            smem_limit=smem_limit,
            alignment=alignment if contiguous and not tight else 0,
        )
        if reason is not None:
            raise ValueError(f"cannot build the FP8 grouped GEMM for {config}: {reason}")
        n_tiles = tilelang.cdiv(n, block_n)
        # The last N tile runs past N: its loads read zeros and its stores are clipped.
        ragged_n = n % block_n != 0
        # A tile's columns run in sub-blocks of sub_block_n, each inside one b scale row.
        sub_blocks = _b_scale_sub_blocks(block_n)
        sub_block_n = block_n // sub_blocks
        last_scale_b_row = n // QUANT_SCALE_BLOCK - 1
        swizzle_span = swizzle_m_tiles * n_tiles
        if contiguous:
            m = T.dynamic("m")
            m_tiles = T.ceildiv(m, block_m)
            # A tight expert's last tile may hold fewer than block_m rows; aligned runs tile
            # their padded extent exactly.
            max_tasks = (m_tiles + (num_groups if tight else 0)) * n_tiles
            max_waves = T.ceildiv(max_tasks, num_sms)
            rows_shape = (m,)
        else:
            tiles_per_slab = tilelang.cdiv(max_m, block_m)
            # The sequence is never longer than every tile of every slab.
            max_tasks = num_groups * tiles_per_slab * n_tiles
            max_waves = tilelang.cdiv(max_tasks, num_sms)
            # The last tile of a slab whose rows end inside it reads no scale past max_m.
            ragged_slab = max_m % block_m != 0
            rows_shape = (num_groups, max_m)
        # The previous task's TMA store must have read shared_c before the next is staged.
        multi_wave = contiguous or max_waves > 1
        math_threads = block_m // warpgroup_m * 128
        math_warps = math_threads // 32
        fragment_regs = warpgroup_m * block_n // 128
        # Static rings have compile-time slots; other rings retain position across tasks.
        static_ring = scale_k % num_stages == 0
        rings_per_task = scale_k // num_stages
        k_unroll = num_stages if static_ring else 1
        wgmma_helper = f"tileops::moe_fp8_wgmma_64x128_by_128xN_lo<{block_n}>"
        promote_helper = f"tileops::moe_fp8_promote<{block_n}>"
        stsm_helper = f"tileops::moe_fp8_stsm_bf16_swizzled<{block_m}, {block_n}>"
        store_rows_helper = f"tileops::moe_fp8_store_rows_bf16_swizzled<{block_m}, {block_n}>"
        # The staged tile is boxes of 64 bfloat16 columns under a 128B swizzle, or 32 under
        # a 64B one when the row is no multiple of 128 bytes (160), one TMA store each.
        epilogue_swizzle_bytes = 128 if block_n * 2 % 128 == 0 else 64
        epilogue_swizzle = {64: TMA_SWIZZLE_64B, 128: TMA_SWIZZLE_128B}[epilogue_swizzle_bytes]
        epilogue_block_n = epilogue_swizzle_bytes // 2
        epilogue_store_count = block_n // epilogue_block_n

        def ring_position(kk, ring_index):
            """Ring slot and barrier phase shared by producers and consumers."""
            if static_ring:
                return kk % num_stages, (ring_index * rings_per_task + kk // num_stages) & 1
            return ring_index % num_stages, (ring_index // num_stages) & 1

        def promote(partial, final, scales):
            """Accumulate a partial product under its row and column-block scales."""
            b_scales = [scales[4 + j] for j in range(sub_blocks)]
            return T.call_extern(
                "handle", promote_helper, partial.data, final.data, scales[0], scales[1], *b_scales
            )

        @T.macro
        def walk_to(layout, task_id, cursor):
            """Resolve a masked task to an expert; ``num_groups`` marks exhaustion."""
            while cursor[cur_g] < num_groups:
                cursor[cur_tiles] = (
                    T.max(T.min(layout[cursor[cur_g]], max_m), 0) + block_m - 1
                ) // block_m
                if task_id < (cursor[cur_m] + cursor[cur_tiles]) * n_tiles:
                    break
                cursor[cur_m] = cursor[cur_m] + cursor[cur_tiles]
                cursor[cur_g] = cursor[cur_g] + 1

        @T.macro
        def walk_ends(ends, task_id, cursor):
            """Resolve a task to an expert run, retaining its row bounds and tile count."""
            while cursor[cur_g] < num_groups:
                prev_end = T.if_then_else(cursor[cur_g] > 0, ends[T.max(cursor[cur_g] - 1, 0)], 0)
                if tight:
                    cursor[cur_row0] = prev_end
                else:
                    cursor[cur_row0] = T.ceildiv(prev_end, alignment) * alignment
                cursor[cur_rows] = T.max(ends[cursor[cur_g]] - cursor[cur_row0], 0)
                cursor[cur_tiles] = (cursor[cur_rows] + block_m - 1) // block_m
                if task_id < (cursor[cur_m] + cursor[cur_tiles]) * n_tiles:
                    break
                cursor[cur_m] = cursor[cur_m] + cursor[cur_tiles]
                cursor[cur_g] = cursor[cur_g] + 1

        @T.macro
        def walk_rows(layout, task_id, cursor):
            """Resolve an aligned row tile from its first expert id, with N tiles fastest.

            Id ``num_groups`` denotes padding; each row tile occupies one cursor run.
            """
            cursor[cur_g] = num_groups
            if task_id < m_tiles * n_tiles:
                cursor[cur_m] = task_id // n_tiles
                cursor[cur_tiles] = 1
                cursor[cur_row0] = cursor[cur_m] * block_m
                cursor[cur_rows] = block_m
                cursor[cur_g] = T.min(layout[cursor[cur_row0]], num_groups)

        walk = walk_rows if rows_walk else walk_ends if contiguous else walk_to

        @T.macro
        def tile_of(task_id, cursor, tile):
            """Map a valid expert-local task to its M/N tile.

            Requires ``0 <= local < m_tiles * n_tiles`` and ``m_tiles >= 1``.
            """
            local = task_id - cursor[cur_m] * n_tiles
            m_tiles = cursor[cur_tiles]
            if not contiguous and tiles_per_slab <= swizzle_m_tiles:
                # One group spans every M tile of the expert. A runtime divisor costs a long
                # division, so each of the tiles_per_slab values m_tiles can take gets its own.
                for i in T.unroll(tiles_per_slab):
                    if m_tiles == i + 1:
                        tile[0] = T.truncmod(local, i + 1)
                        tile[1] = T.truncdiv(local, i + 1)
            else:
                first = T.truncdiv(local, swizzle_span) * swizzle_m_tiles
                in_group = T.truncmod(local, swizzle_span)
                group_m_tiles = T.min(swizzle_m_tiles, m_tiles - first)
                tile[0] = first + T.truncmod(in_group, group_m_tiles)
                tile[1] = T.truncdiv(in_group, group_m_tiles)

        @T.prim_func
        def _moe_grouped_gemm_fp8_main(
            A: T.Tensor((*rows_shape, k), ab_dtype),  # type: ignore
            A_scale: T.Tensor((*rows_shape, scale_k), "float32"),  # type: ignore
            B: T.Tensor((num_groups, n, k), ab_dtype),  # type: ignore
            B_scale: T.Tensor((num_groups, last_scale_b_row + 1, scale_k), "float32"),  # type: ignore
            layout: T.Tensor((m,) if metadata_kind == "per_row" else (num_groups,), "int32"),  # type: ignore
            C: T.Tensor((*rows_shape, n), cd_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(num_sms, threads=math_threads + 128) as (pid,):
                a_shared = T.alloc_shared((num_stages, block_m, block_k), ab_dtype)
                b_shared = T.alloc_shared((num_stages, block_n, block_k), ab_dtype)
                # scope="shared": a shared.dyn store here makes TileLang fence it against the TMA
                # writes with a barrier one warp cannot complete.
                b_scale_shared = T.alloc_shared(
                    (num_stages, sub_blocks), accum_dtype, scope="shared"
                )
                if stage_scales:
                    a_scale_shared = T.alloc_shared(
                        (num_stages, block_m), accum_dtype, scope="shared"
                    )
                shared_c = T.alloc_shared((block_m * block_n,), cd_dtype)
                partial = T.alloc_local((fragment_regs,), accum_dtype)
                final = T.alloc_local((fragment_regs,), accum_dtype)
                # This step's A scales (two rows), the next step's, and this step's B scales,
                # one per column sub-block.
                scales = T.alloc_local((4 + sub_blocks,), accum_dtype)
                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )
                # Staged, a stage is full once its TMA bytes land and each producer lane's
                # cp.async copies of its scales do; otherwise once lane 0 has stored the B scales.
                full = T.alloc_barrier([32 if stage_scales else 1] * num_stages)
                empty = T.alloc_barrier([math_warps] * num_stages)
                # Each role's count of the tasks (static ring) or K steps (index ring) it has run.
                # Not the wave index: a skipped task advances no barrier.
                ring_index = T.alloc_var("int32", init=0)
                cursor = T.alloc_local((5 if contiguous else 3,), "int32")
                tile = T.alloc_local((2,), "int32")
                for i in T.unroll(5 if contiguous else 3):
                    cursor[i] = 0
                tx = T.get_thread_binding()
                if recover:
                    # Each expert's end: the first row whose id is past it, ids not decreasing.
                    ends = T.alloc_shared((num_groups,), "int32", scope="shared")
                    for group_step in T.serial(T.ceildiv(num_groups, math_threads + 128)):
                        group = group_step * (math_threads + 128) + tx
                        if group < num_groups:
                            lo = T.alloc_var("int32", init=0)
                            hi = T.alloc_var("int32", init=m)
                            for _ in T.serial(32):
                                if lo < hi:
                                    mid = (lo + hi) >> 1
                                    if layout[mid] <= group:
                                        lo = mid + 1
                                    else:
                                        hi = mid
                            ends[group] = lo
                    T.sync_threads()
                if contiguous:
                    # [m, N], so the store of the last rows is clipped at m.
                    output_desc = T.create_tma_descriptor(
                        TMA_DTYPE_BFLOAT16,
                        2,
                        C.data,
                        n,
                        m,
                        1,
                        n * 2,
                        epilogue_block_n,
                        block_m,
                        1,
                        1,
                        TMA_INTERLEAVE_NONE,
                        epilogue_swizzle,
                        TMA_L2_PROMOTION_128B,
                        TMA_OOB_FILL_NONE,
                    )
                else:
                    # [E, max_m, N], so the store of a slab's last tile is clipped at max_m.
                    output_desc = T.create_tma_descriptor(
                        TMA_DTYPE_BFLOAT16,
                        3,
                        C.data,
                        n,
                        max_m,
                        num_groups,
                        1,
                        n * 2,
                        max_m * n * 2,
                        epilogue_block_n,
                        block_m,
                        1,
                        1,
                        1,
                        1,
                        TMA_INTERLEAVE_NONE,
                        epilogue_swizzle,
                        TMA_L2_PROMOTION_128B,
                        TMA_OOB_FILL_NONE,
                    )

                # Explicit descriptors enable 256-byte L2 promotion.
                if contiguous:
                    a_load_desc = T.create_tma_descriptor(
                        TMA_DTYPE_UINT8,
                        2,
                        A.data,
                        k,
                        m,
                        1,
                        k,
                        block_k,
                        block_m,
                        1,
                        1,
                        TMA_INTERLEAVE_NONE,
                        TMA_SWIZZLE_128B,
                        TMA_L2_PROMOTION_256B,
                        TMA_OOB_FILL_NONE,
                    )
                else:
                    a_load_desc = T.create_tma_descriptor(
                        TMA_DTYPE_UINT8,
                        3,
                        A.data,
                        k,
                        max_m,
                        num_groups,
                        1,
                        k,
                        max_m * k,
                        block_k,
                        block_m,
                        1,
                        1,
                        1,
                        1,
                        TMA_INTERLEAVE_NONE,
                        TMA_SWIZZLE_128B,
                        TMA_L2_PROMOTION_256B,
                        TMA_OOB_FILL_NONE,
                    )
                b_load_desc = T.create_tma_descriptor(
                    TMA_DTYPE_UINT8,
                    3,
                    B.data,
                    k,
                    n,
                    num_groups,
                    1,
                    k,
                    n * k,
                    block_k,
                    block_n,
                    1,
                    1,
                    1,
                    1,
                    TMA_INTERLEAVE_NONE,
                    TMA_SWIZZLE_128B,
                    TMA_L2_PROMOTION_256B,
                    TMA_OOB_FILL_NONE,
                )

                if tx >= math_threads:
                    T.dec_max_nreg(40)
                    producer_tx = tx - math_threads
                    # Shuffled so the value stays warp-uniform for NVCC's uniform datapath.
                    producer_warp = T.tvm_warp_shuffle(
                        T.uint32(0xFFFFFFFF), producer_tx // 32, 0, 32, 32
                    )
                    for wave in T.serial(max_waves):
                        task_id = num_sms * wave + pid
                        # Skipped tasks must advance neither role's barriers.
                        walk(ends if recover else layout, task_id, cursor)
                        if T.And(cursor[cur_g] < num_groups, producer_warp == 0):
                            tile_of(task_id, cursor, tile)
                            # Inline bounds prevent redundant TileLang scale-load guards.
                            g = T.meta_var(T.max(T.min(cursor[cur_g], num_groups - 1), 0))
                            if contiguous:
                                m_start = T.meta_var(cursor[cur_row0] + tile[0] * block_m)
                            else:
                                m_start = T.meta_var(
                                    T.max(T.min(tile[0], tiles_per_slab - 1), 0) * block_m
                                )
                            n_start = T.meta_var(T.max(T.min(tile[1], n_tiles - 1), 0) * block_n)
                            for kk in T.unroll(scale_k, unroll_factor=k_unroll):
                                slot, phase = ring_position(kk, ring_index)
                                T.barrier_wait(empty[slot], phase ^ 1)
                                if T.shuffle_elect(32):
                                    T.mbarrier_expect_tx(full[slot], block_m * block_k)
                                    if contiguous:
                                        T.tma_load(
                                            a_load_desc,
                                            full[slot],
                                            T.address_of(a_shared[slot, 0, 0]),
                                            kk * block_k,
                                            m_start,
                                            0,
                                        )
                                    else:
                                        T.tma_load(
                                            a_load_desc,
                                            full[slot],
                                            T.address_of(a_shared[slot, 0, 0]),
                                            kk * block_k,
                                            m_start,
                                            g,
                                            0,
                                        )
                                    T.mbarrier_expect_tx(full[slot], block_n * block_k)
                                    T.tma_load(
                                        b_load_desc,
                                        full[slot],
                                        T.address_of(b_shared[slot, 0, 0]),
                                        kk * block_k,
                                        n_start,
                                        g,
                                        0,
                                    )
                                if stage_scales:
                                    # The full barrier requires both TMA completion and all 32 lanes' scale copies.
                                    for r in T.unroll(block_m // 32):
                                        row = r * 32 + producer_tx
                                        if contiguous:
                                            a_row = T.meta_var(T.min(m_start + row, m - 1))
                                        elif ragged_slab:
                                            a_row = T.meta_var(T.min(m_start + row, max_m - 1))
                                        else:
                                            a_row = T.meta_var(m_start + row)
                                        T.call_extern(
                                            "handle",
                                            "tileops::moe_fp8_cp_async_f32",
                                            T.address_of(a_scale_shared[slot, row]),
                                            T.address_of(
                                                A_scale[a_row, kk]
                                                if contiguous
                                                else A_scale[g, a_row, kk]
                                            ),
                                        )
                                    if producer_tx < sub_blocks:
                                        b_row = T.meta_var(
                                            T.min(
                                                (n_start + producer_tx * sub_block_n)
                                                // QUANT_SCALE_BLOCK,
                                                last_scale_b_row,
                                            )
                                        )
                                        T.call_extern(
                                            "handle",
                                            "tileops::moe_fp8_cp_async_f32",
                                            T.address_of(b_scale_shared[slot, producer_tx]),
                                            T.address_of(B_scale[g, b_row, kk]),
                                        )
                                    T.call_extern(
                                        "handle",
                                        "tileops::moe_fp8_cp_async_arrive",
                                        T.address_of(full[slot]),
                                    )
                                else:
                                    # The full-barrier release covers the B-scale stores.
                                    if producer_tx == 0:
                                        for j in T.unroll(sub_blocks):
                                            b_scale_shared[slot, j] = B_scale[
                                                g,
                                                T.min(
                                                    (n_start + j * sub_block_n)
                                                    // QUANT_SCALE_BLOCK,
                                                    last_scale_b_row,
                                                ),
                                                kk,
                                            ]
                                        T.barrier_arrive(full[slot])
                                if not static_ring:
                                    ring_index = ring_index + 1
                            if static_ring:
                                ring_index = ring_index + 1
                else:
                    T.inc_max_nreg(232)
                    math_wg_idx = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), tx // 128, 0, 32, 32)
                    math_wg_offset = math_wg_idx * warpgroup_m
                    a_desc_lo = T.call_extern(
                        "uint32",
                        "tileops::moe_fp8_wgmma_desc_lo",
                        T.address_of(a_shared[0, math_wg_offset, 0]),
                    )
                    b_desc_lo = T.call_extern(
                        "uint32",
                        "tileops::moe_fp8_wgmma_desc_lo",
                        T.address_of(b_shared[0, 0, 0]),
                    )
                    # Lanes 4i..4i+3 of warp w hold accumulator rows w*16 + i and w*16 + i + 8.
                    # Read off ``tx``, which TileLang can bound, so the loads carry no guard.
                    acc_row0 = (tx // 32) * 16 + (tx % 32) // 4
                    for wave in T.serial(max_waves):
                        task_id = num_sms * wave + pid
                        walk(ends if recover else layout, task_id, cursor)
                        if cursor[cur_g] < num_groups:
                            tile_of(task_id, cursor, tile)
                            g = T.meta_var(T.max(T.min(cursor[cur_g], num_groups - 1), 0))
                            if contiguous:
                                m_start = T.meta_var(cursor[cur_row0] + tile[0] * block_m)
                                tile_rows = T.meta_var(cursor[cur_rows] - tile[0] * block_m)
                            else:
                                m_start = T.meta_var(
                                    T.max(T.min(tile[0], tiles_per_slab - 1), 0) * block_m
                                )
                            n_start = T.meta_var(T.max(T.min(tile[1], n_tiles - 1), 0) * block_n)
                            short_tile = T.meta_var(tile_rows < block_m) if tight else False
                            if contiguous:
                                scale_row0 = T.meta_var(T.min(m_start + acc_row0, m - 1))
                                scale_row1 = T.meta_var(T.min(m_start + acc_row0 + 8, m - 1))
                            elif ragged_slab:
                                scale_row0 = T.meta_var(T.min(m_start + acc_row0, max_m - 1))
                                scale_row1 = T.meta_var(T.min(m_start + acc_row0 + 8, max_m - 1))
                            else:
                                scale_row0 = T.meta_var(m_start + acc_row0)
                                scale_row1 = T.meta_var(scale_row0 + 8)
                            T.clear(final)
                            if not stage_scales:
                                if contiguous:
                                    scales[0] = A_scale[scale_row0, 0]
                                    scales[1] = A_scale[scale_row1, 0]
                                else:
                                    scales[0] = A_scale[g, scale_row0, 0]
                                    scales[1] = A_scale[g, scale_row1, 0]
                            for kk in T.unroll(scale_k, unroll_factor=k_unroll):
                                slot, phase = ring_position(kk, ring_index)
                                if not stage_scales:
                                    next_kk = T.min(kk + 1, scale_k - 1)
                                    if contiguous:
                                        scales[2] = A_scale[scale_row0, next_kk]
                                        scales[3] = A_scale[scale_row1, next_kk]
                                    else:
                                        scales[2] = A_scale[g, scale_row0, next_kk]
                                        scales[3] = A_scale[g, scale_row1, next_kk]
                                T.barrier_wait(full[slot], phase)
                                if stage_scales:
                                    scales[0] = a_scale_shared[slot, acc_row0]
                                    scales[1] = a_scale_shared[slot, acc_row0 + 8]
                                for j in T.unroll(sub_blocks):
                                    scales[4 + j] = b_scale_shared[slot, j]
                                T.call_extern(
                                    "handle",
                                    wgmma_helper,
                                    partial.data,
                                    a_desc_lo + T.uint32(slot * block_m * (block_k // 16)),
                                    b_desc_lo + T.uint32(slot * block_n * (block_k // 16)),
                                )
                                T.wait_wgmma(0)
                                if tx % 32 == 0:
                                    T.barrier_arrive(empty[slot])
                                promote(partial, final, scales)
                                if not stage_scales:
                                    scales[0] = scales[2]
                                    scales[1] = scales[3]
                                if not static_ring:
                                    ring_index = ring_index + 1
                            if static_ring:
                                ring_index = ring_index + 1
                            # Do not overwrite shared_c until its preceding TMA store has finished reading.
                            if multi_wave:
                                if tx < epilogue_store_count:
                                    T.tma_store_wait(0)
                                T.sync_threads(barrier_id=13, arrive_count=math_threads)
                            T.call_extern(
                                "handle",
                                stsm_helper,
                                final.data,
                                T.address_of(shared_c[0]),
                                math_wg_offset,
                            )
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=14, arrive_count=math_threads)
                            if ragged_n:
                                store_box = T.And(
                                    tx < epilogue_store_count,
                                    n_start + tx * epilogue_block_n < n,
                                )
                            else:
                                store_box = tx < epilogue_store_count
                            if not contiguous:
                                if store_box:
                                    T.call_extern(
                                        "handle",
                                        "tileops::moe_fp8_tma_store_3d_issue",
                                        output_desc,
                                        T.address_of(shared_c[tx * block_m * epilogue_block_n]),
                                        n_start + tx * epilogue_block_n,
                                        m_start,
                                        g,
                                    )
                                    T.tma_store_arrive()
                            elif short_tile:
                                # Tight tails must not overwrite the next expert's rows.
                                T.call_extern(
                                    "handle",
                                    store_rows_helper,
                                    T.address_of(shared_c[0]),
                                    T.address_of(C[m_start, n_start]),
                                    n,
                                    T.min(tile_rows, m - m_start),
                                    n - n_start,
                                    tx,
                                    math_threads,
                                )
                            else:
                                if store_box:
                                    T.call_extern(
                                        "handle",
                                        "tileops::moe_fp8_tma_store_2d_issue",
                                        output_desc,
                                        T.address_of(shared_c[tx * block_m * epilogue_block_n]),
                                        n_start + tx * epilogue_block_n,
                                        m_start,
                                    )
                                    T.tma_store_arrive()
                    if tx < epilogue_store_count:
                        T.tma_store_wait(0)

        return _moe_grouped_gemm_fp8_main

    return _moe_grouped_gemm_fp8_func


class MoEGroupedGEMMFP8Kernel(Kernel, MGroupedGEMMFP8FwdInterface):
    """Block-scaled FP8 grouped GEMM over masked expert slabs or contiguous expert rows, on SM90
    TMA and WGMMA.

    Rows of a slab past its valid count, inside a tile that holds a valid one, are computed and
    stored like the others; rows past ``max_m`` are not written. A count is clamped to
    ``[0, max_m]``. Contiguous rows past an aligned expert's end are its padding and are stored
    like the others; a tight expert's are the next expert's and are not stored. Under aligned
    packing ``block_m`` divides the alignment.

    A config is ``{block_m, block_n, num_stages, num_sms, stage_scales}``; ``stage_scales`` is 1
    where the producer copies each stage's scales into the ring and 0 where the math warp-groups
    read the A scales from global memory. The calibrated table, the default rule, the autotuning
    candidates and a caller's config all pass ``config_refusal``.
    """

    supported_archs: list[int] = [90]

    # Tile heights: one or two 64-row math warp-groups.
    _BLOCK_M: ClassVar[tuple[int, ...]] = (64, 128)
    # Tile widths the helper header instantiates; 160 and 192, which 128 does not divide, span
    # two ``b`` scale rows, so their columns run in 32- and 64-wide sub-blocks.
    _BLOCK_N: ClassVar[tuple[int, ...]] = (64, 128, 160, 192)
    # Ring depths. A depth that divides the K steps runs the constant-slot ring.
    _NUM_STAGES: ClassVar[range] = range(3, 9)
    # Calibrated configs keyed by (E, N, K, max_m); contiguous layouts use max_m=None.
    _CONFIGS: ClassVar[dict[str, dict[tuple[int, int, int, Optional[int]], dict[str, int]]]] = {
        "h200": {
            (8, 4096, 7168, 128): {
                "block_m": 64,
                "block_n": 128,
                "num_stages": 7,
                "num_sms": 132,
                "stage_scales": 1,
            },
            (8, 7168, 2048, 128): {
                "block_m": 128,
                "block_n": 160,
                "num_stages": 4,
                "num_sms": 120,
                "stage_scales": 1,
            },
            (32, 4096, 7168, 64): {
                "block_m": 64,
                "block_n": 192,
                "num_stages": 4,
                "num_sms": 118,
                "stage_scales": 0,
            },
            (32, 7168, 2048, 64): {
                "block_m": 64,
                "block_n": 160,
                "num_stages": 4,
                "num_sms": 131,
                "stage_scales": 1,
            },
            (32, 7168, 2048, 256): {
                "block_m": 64,
                "block_n": 160,
                "num_stages": 4,
                "num_sms": 132,
                "stage_scales": 1,
            },
            (16, 3072, 4096, 64): {
                "block_m": 64,
                "block_n": 192,
                "num_stages": 4,
                "num_sms": 128,
                "stage_scales": 1,
            },
            (16, 3072, 4096, 256): {
                "block_m": 64,
                "block_n": 192,
                "num_stages": 4,
                "num_sms": 128,
                "stage_scales": 1,
            },
            (48, 4096, 7168, 64): {
                "block_m": 64,
                "block_n": 192,
                "num_stages": 4,
                "num_sms": 132,
                "stage_scales": 0,
            },
            (48, 7168, 2048, 64): {
                "block_m": 64,
                "block_n": 160,
                "num_stages": 4,
                "num_sms": 128,
                "stage_scales": 1,
            },
            (48, 4096, 7168, 256): {
                "block_m": 64,
                "block_n": 192,
                "num_stages": 4,
                "num_sms": 132,
                "stage_scales": 0,
            },
            (48, 7168, 2048, 256): {
                "block_m": 64,
                "block_n": 192,
                "num_stages": 4,
                "num_sms": 132,
                "stage_scales": 0,
            },
            (16, 1536, 2048, None): {
                "block_m": 128,
                "block_n": 192,
                "num_stages": 4,
                "num_sms": 132,
                "stage_scales": 0,
            },
            (32, 4096, 7168, None): {
                "block_m": 128,
                "block_n": 128,
                "num_stages": 4,
                "num_sms": 132,
                "stage_scales": 1,
            },
            (32, 7168, 2048, None): {
                "block_m": 128,
                "block_n": 160,
                "num_stages": 4,
                "num_sms": 132,
                "stage_scales": 1,
            },
        },
    }

    @classmethod
    def config_refusal(
        cls, config: Mapping[str, object], *, sm_count: int, smem_limit: int, alignment: int = 0
    ) -> Optional[str]:
        """Why the builder cannot instantiate *config*, or ``None`` when it can.

        Args:
            config: A candidate config.
            sm_count: The device's SM count, which bounds ``num_sms``: the grid is persistent.
            smem_limit: The shared memory one block may take on the device.
            alignment: An aligned layout's alignment, which ``block_m`` divides; 0 for any
                other layout.

        Returns:
            The first violated constraint, or ``None``.
        """
        if set(config) != set(_CONFIG_KEYS):
            return f"needs exactly the keys {list(_CONFIG_KEYS)}, got {sorted(config)}"
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in config.values()):
            return "takes integer values"
        block_m, block_n, stages, num_sms, stage_scales = (config[key] for key in _CONFIG_KEYS)
        if block_m not in cls._BLOCK_M:
            return f"block_m must be one of {cls._BLOCK_M}, got {block_m}"
        if block_n not in cls._BLOCK_N:
            return f"block_n must be one of {cls._BLOCK_N}, got {block_n}"
        if stages not in cls._NUM_STAGES:
            return (
                f"num_stages must lie in [{cls._NUM_STAGES.start}, {cls._NUM_STAGES.stop - 1}], "
                f"got {stages}"
            )
        if not 1 <= num_sms <= sm_count:
            return f"num_sms must lie in [1, {sm_count}], the device's SMs, got {num_sms}"
        if stage_scales not in (0, 1):
            return f"stage_scales must be 0 or 1, got {stage_scales}"
        if alignment % block_m:
            return f"block_m must divide the layout's alignment {alignment}, got {block_m}"
        smem = _smem_bytes(block_m, block_n, stages, stage_scales)
        if smem > smem_limit:
            return f"takes {smem} bytes of shared memory, past the {smem_limit} a block may"
        return None

    @classmethod
    def refusal(cls, call: MGroupedGEMMFP8Call) -> Optional[str]:
        """The layout, dtype, extents and scale grids the kernel instantiates."""
        if call.ab_dtype is not torch.float8_e4m3fn:
            return f"serves float8_e4m3fn operands, got {call.ab_dtype}"
        if call.k % QUANT_SCALE_BLOCK or call.n % QUANT_SCALE_BLOCK:
            return f"needs K and N multiples of {QUANT_SCALE_BLOCK}, got K = {call.k}, N = {call.n}"
        if call.packing == "aligned" and call.alignment % min(cls._BLOCK_M):
            return f"tiles aligned runs in {cls._BLOCK_M}-row tiles, got alignment {call.alignment}"
        scale_k = call.k // QUANT_SCALE_BLOCK
        rows = (call.m,) if call.kind == "contiguous" else (call.num_groups, call.max_m)
        a_scale = (*rows, scale_k)
        b_scale = (call.num_groups, call.n // QUANT_SCALE_BLOCK, scale_k)
        if tuple(call.a_scale_shape) != a_scale or tuple(call.b_scale_shape) != b_scale:
            return (
                f"reads a_scale {a_scale} and b_scale {b_scale}, got "
                f"{tuple(call.a_scale_shape)} and {tuple(call.b_scale_shape)}"
            )
        return super().refusal(call)

    def __init__(self, call: MGroupedGEMMFP8Call) -> None:
        device_index = call.device.index if call.device is not None else None
        super().__init__(device_index=device_index)
        self.call = call
        self.scale_k = call.k // QUANT_SCALE_BLOCK
        self.sm_count = self.build_target.sm_count
        # Kept free below the device's opt-in shared-memory limit for allocator padding
        # ``_smem_bytes`` does not model; no legal config's footprint lies in it.
        smem_headroom = 1024
        self.smem_limit = self.build_target.smem_budget - smem_headroom
        # Tight per-row metadata keeps every expert's recovered end in shared memory.
        if call.packing == "tight" and call.metadata_kind == "per_row":
            self.smem_limit -= 4 * call.num_groups
        self.alignment = call.alignment if call.packing == "aligned" else 0
        calibration = self.build_target.calibration
        calibrated = self._CONFIGS.get(calibration, {}).get(
            (call.num_groups, call.n, call.k, call.max_m)
        )
        # Calibrated configurations must satisfy this layout's resource and alignment limits.
        refused = calibrated is not None and self.config_refusal(
            calibrated, sm_count=self.sm_count, smem_limit=self.smem_limit, alignment=self.alignment
        )
        self._calibrated = None if refused else calibrated
        self.kernel = _moe_grouped_gemm_fp8_kernel(
            call.num_groups,
            call.n,
            call.k,
            call.max_m,
            self.sm_count,
            self.smem_limit,
            call.packing,
            call.metadata_kind,
            call.alignment,
        )
        self.init_config()

    def _check_config(self, config: Mapping[str, object]) -> None:
        """Validate a launch configuration against device and layout limits."""
        reason = self.config_refusal(
            config, sm_count=self.sm_count, smem_limit=self.smem_limit, alignment=self.alignment
        )
        if reason is not None:
            raise ValueError(f"{type(self).__name__} cannot run config {dict(config)}: {reason}")

    def init_config(self, config: Optional[dict] = None) -> None:
        """Set the config as the base class does, then check it.

        Raises:
            ValueError: *config* names a key the kernel does not take, or the resulting
                config is one ``config_refusal`` refuses.
        """
        if config is not None and not set(config) <= set(_CONFIG_KEYS):
            raise ValueError(
                f"{type(self).__name__} takes the config keys {list(_CONFIG_KEYS)}, "
                f"got {sorted(config)}"
            )
        super().init_config(config)
        self._check_config(self.config)

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        super().autotune(warmup=warmup, rep=rep)
        self._check_config(self.config)

    def _stages_for(self, block_m: int, block_n: int, stage_scales: int) -> list[int]:
        """Ring depths that fit beside the tiles, those dividing the K steps when any do."""
        fitting = [
            s
            for s in self._NUM_STAGES
            if _smem_bytes(block_m, block_n, s, stage_scales) <= self.smem_limit
        ]
        dividing = [s for s in fitting if self.scale_k % s == 0]
        return dividing or fitting

    def _balanced_sms(self, block_m: int, block_n: int) -> int:
        """Minimize idle blocks for masked slabs; use all SMs for contiguous layouts."""
        call = self.call
        if call.kind == "contiguous":
            return self.sm_count
        tiles = call.num_groups * -(-call.max_m // block_m) * -(-call.n // block_n)
        if tiles == 0:
            return 1
        return -(-tiles // -(-tiles // self.sm_count))

    @property
    def default_config(self) -> dict:
        if self._calibrated is not None:
            return dict(self._calibrated)
        # A slab of up to 128 rows fits one M tile, so each weight tile is read once. A wider
        # slab takes several M tiles at either height; with the valid counts unknown, 64-row
        # tiles waste less on a sparse slab. Contiguous rows take the taller tile where the
        # alignment allows it.
        if self.call.kind == "contiguous":
            block_m = 64 if self.alignment % 128 else 128
        else:
            block_m = 128 if 64 < self.call.max_m <= 128 else 64
        block_n = 128
        return {
            "block_m": block_m,
            "block_n": block_n,
            "num_stages": max(self._stages_for(block_m, block_n, 0)),
            "num_sms": self._balanced_sms(block_m, block_n),
            "stage_scales": 0,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # A tile wider than N computes only columns it does not store.
        widths = [n for n in self._BLOCK_N if n <= self.call.n]
        heights = [m for m in self._BLOCK_M if self.alignment % m == 0]
        configs = [
            {"block_m": m, "block_n": n, "num_stages": s, "num_sms": sms, "stage_scales": staged}
            for m, n, staged in itertools.product(heights, widths, (0, 1))
            for s in self._stages_for(m, n, staged)
            for sms in sorted({self._balanced_sms(m, n), self.sm_count})
        ]
        for config in configs:
            self._check_config(config)
        return configs

    @property
    def autotune_supply_prog(self) -> Callable:
        """Supply masked full/half slabs or evenly distributed contiguous rows for tuning."""
        from tilelang.utils.device import get_current_device

        call, scale_k = self.call, self.scale_k

        def supply_prog(params: list) -> list:
            if len(params) != 6:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects 6 parameters "
                    f"(a, a_scale, b, b_scale, layout, c), got {len(params)}"
                )
            device = get_current_device()
            groups = call.num_groups
            rows = (call.m,) if call.kind == "contiguous" else (groups, call.max_m)
            metadata = moe_layout_metadata(call, math.prod(rows), groups)
            fp8 = torch.float8_e4m3fn
            return [
                torch.randn(*rows, call.k, device=device).to(fp8),
                torch.rand(*rows, scale_k, device=device) + 0.5,
                torch.randn(groups, call.n, call.k, device=device).to(fp8),
                torch.rand(groups, call.n // QUANT_SCALE_BLOCK, scale_k, device=device) + 0.5,
                torch.tensor(metadata, dtype=torch.int32, device=device),
                torch.empty(*rows, call.n, dtype=torch.bfloat16, device=device),
            ]

        return supply_prog

    def forward(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run each expert's block-scaled product over its rows, into *out* if given."""
        if out is None:
            out = torch.empty((*a.shape[:-1], self.call.n), dtype=torch.bfloat16, device=a.device)
        if out.numel() == 0:
            return out
        self.program(self._compile)(a, a_scale, b, b_scale, layout_metadata, out)
        return out

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self.kernel(**cfg)
