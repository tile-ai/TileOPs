"""MXFP4 grouped GEMM on SM90: packed e2m1 weights times FP8 rows on an fp8 WGMMA.

Supports masked slabs and contiguous expert rows. A scales cover 1x128 blocks of ``a``, B
scales 1x32 blocks of ``b``, both e8m0 or both float32. The weights are the WGMMA's M operand,
decoded to e4m3 in registers; the rows (tokens) are its N operand, read as TMA loads them.
Each 128-wide K step is promoted into a float32 accumulator; output is bfloat16.
"""

import math
from collections.abc import Callable, Mapping
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import (
    QUANT_SCALE_BLOCK,
    SETMAXNREG_GRANULARITY,
    SHARED_BUFFER_ALIGN_BYTES,
    SM90_REGISTERS,
    TMA_DTYPE_BFLOAT16,
    TMA_DTYPE_UINT8,
    TMA_L2_PROMOTION_128B,
    TMA_L2_PROMOTION_256B,
    TMA_SWIZZLE_64B,
    TMA_SWIZZLE_128B,
    TMA_SWIZZLE_NONE,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.moe.call_spec import (
    MGroupedGEMMMXFP4Call,
    MGroupedGEMMMXFP4FwdInterface,
)
from tileops._kernels.moe.grouped_tasks import CURSOR, GroupedGEMMConfigMixin, task_walk
from tileops.manifest.primitives import moe_layout_metadata

__all__ = ["MoEGroupedGEMMMXFP4Kernel", "MoEGroupedGEMMMXFP4SharedDecodeKernel"]

# K values one ``b`` scale covers.
_B_SCALE_BLOCK = 32
# Sub-blocks of one 128-wide K step, each under its own ``b`` scale.
_SUB_BLOCKS = QUANT_SCALE_BLOCK // _B_SCALE_BLOCK


def _math_warps(block_n: int) -> int:
    """Math warps of a tile: a 64-row warp-group of four warps per 64 weight rows."""
    return block_n // WGMMA_ROWS * 4


def _table_box(scale_k: int) -> int:
    """Bytes of each code-table row one TMA box carries: 128 or 64, whichever divides the row of
    ``4 * scale_k`` bytes; 0 when neither does and the table comes by one bulk copy."""
    row_bytes = 4 * scale_k
    return 128 if row_bytes % 128 == 0 else 64 if row_bytes % 64 == 0 else 0


def _smem_bytes(
    block_m: int, block_n: int, num_stages: int, scale_bytes: int, scale_k: int, tables: bool
) -> int:
    """Bytes of the fp8 row and packed-weight ring; with ``tables``, two tasks' e8m0 code tables
    (1024 bytes of alignment) and, when they come by TMA, token code tables, else the weight
    scales a stage; two steps' token scales, the bfloat16 output tile, and the ring's and code
    tables' barriers."""
    ring = num_stages * (block_m * QUANT_SCALE_BLOCK + block_n * QUANT_SCALE_BLOCK // 2)
    if tables:
        token_codes = 2 * block_m * scale_k if _table_box(scale_k) else 0
        tables_bytes = 2 * block_n * scale_k * 4 + SHARED_BUFFER_ALIGN_BYTES + token_codes
    else:
        tables_bytes = num_stages * block_n * _SUB_BLOCKS * scale_bytes
    # Each math warp's token scales of two steps, as floats.
    token_slices = _math_warps(block_n) * 2 * block_m * 4
    barriers = (2 * num_stages + 4) * 8
    return ring + tables_bytes + token_slices + block_m * block_n * 2 + barriers


def _uses_tables(
    block_m: int, block_n: int, num_stages: int, scale_bytes: int, scale_k: int, smem_limit: int
) -> bool:
    """Whether a config reads its e8m0 scales from per-task tables: they grow with K, so past
    the shared memory a block may take beside the ring, the scales come a stage at a time. A
    table that comes by one bulk copy, without token code tables, is taken under the bound of a
    table that comes by TMA: only where the two token code tables would fit beside it too."""
    bulk_room = 0 if _table_box(scale_k) else 2 * block_m * scale_k
    return (
        scale_bytes == 1
        and _smem_bytes(block_m, block_n, num_stages, 1, scale_k, True) + bulk_room <= smem_limit
    )


def _config_smem(
    block_m: int, block_n: int, num_stages: int, scale_bytes: int, scale_k: int, smem_limit: int
) -> int:
    """Shared memory of a config, with or without the tables as ``_uses_tables`` picks."""
    tables = _uses_tables(block_m, block_n, num_stages, scale_bytes, scale_k, smem_limit)
    return _smem_bytes(block_m, block_n, num_stages, scale_bytes, scale_k, tables)


def _e8m0_value(code):
    """The float a ``uint32`` e8m0 code stands for: 2 ** (code - 127), 2 ** -127 at code 0 and
    NaN at 255, as torch's float8_e8m0fnu reads them."""
    bits = T.if_then_else(
        code == 0,
        T.uint32(0x00400000),
        T.if_then_else(code == 255, T.uint32(0x7FC00000), code << 23),
    )
    return T.reinterpret(bits, "float32")


@target_cache(maxsize=32)
def _moe_grouped_gemm_mxfp4_kernel(
    num_groups: int,
    n: int,
    k: int,
    max_m: int | None,
    e8m0: bool,
    sm_count: int,
    smem_limit: int,
    packing: str | None = None,
    metadata_kind: str | None = None,
    alignment: int = 1,
) -> Callable:
    """Build a persistent MXFP4 grouped GEMM for the given layout and scale dtype.

    Tight per-row ids must be sorted. Aligned tiles require ``block_m`` to divide the
    alignment; id ``num_groups`` denotes trailing padding. Tight tail stores must preserve the
    next expert's rows. N tails and all output stores are bounded.
    """
    cur_g = CURSOR.g
    accum_dtype = "float"
    a_dtype = "float8_e4m3fn"
    cd_dtype = "bfloat16"
    scale_dtype = "uint8" if e8m0 else "float32"
    # Columns of one output staging box: 128 bytes of bfloat16, the TMA store's 128B swizzle.
    store_box_cols = 64
    scale_bytes = 1 if e8m0 else 4
    block_k = QUANT_SCALE_BLOCK
    packed_k = block_k // 2
    scale_k = k // block_k
    contiguous = packing is not None
    tight = packing == "tight"

    @tilelang.jit(
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        compile_flags=[
            "-O3",
            "--use_fast_math",
            "-DENABLE_BF16",
            *csrc_include("moe_grouped_gemm_mxfp4_helper.h"),
        ],
    )
    def _moe_grouped_gemm_mxfp4_func(
        block_m: int, block_n: int, num_stages: int, num_sms: int
    ) -> Callable:
        config = dict(
            zip(
                MoEGroupedGEMMMXFP4Kernel._CONFIG_KEYS,
                (block_m, block_n, num_stages, num_sms),
                strict=True,
            )
        )
        reason = MoEGroupedGEMMMXFP4Kernel.config_refusal(
            config,
            sm_count=sm_count,
            smem_limit=smem_limit,
            alignment=alignment if contiguous and not tight else 0,
            scale_bytes=scale_bytes,
            scale_k=scale_k,
        )
        if reason is not None:
            raise ValueError(f"cannot build the MXFP4 grouped GEMM for {config}: {reason}")
        tasks = task_walk(
            num_groups=num_groups,
            block_m=block_m,
            block_n=block_n,
            n=n,
            num_sms=num_sms,
            max_m=max_m,
            packing=packing,
            metadata_kind=metadata_kind,
            alignment=alignment,
        )
        m, rows_shape, max_waves = tasks.m, tasks.rows_shape, tasks.max_waves
        ragged_slab = tasks.ragged_slab
        row_tiles_descriptor = tasks.row_tiles_descriptor
        walk, tile_of, recover = tasks.walk, tasks.tile_of, tasks.recover
        # One math warp-group per 64 weight rows, each over every row (token) of the tile.
        math_threads = block_n // WGMMA_ROWS * WARPGROUP_THREADS
        math_warps = _math_warps(block_n)
        fragment_regs = WGMMA_ROWS * block_m // WARPGROUP_THREADS
        # Named barriers of the math threads: shared_c free to restage, the vote on a task's
        # scale check, and shared_c staged.
        reuse_barrier, check_barrier, staged_barrier = 12, 13, 14
        # A step contracted sub-block by sub-block (float32 scales, or e8m0 ones that do not
        # fold) holds up to four sub-blocks at once in independent accumulator sets of at
        # most 64 registers in all; a folded step takes the first set alone.
        acc_sets = max(1, min(_SUB_BLOCKS, 64 // fragment_regs))
        # Contraction passes of a step contracted sub-block by sub-block: 1, 2 or 4.
        passes = _SUB_BLOCKS // acc_sets
        # The producer keeps 40 registers a thread (its walk and copies fit them) and the
        # math threads share what it frees:
        # setmaxnreg.inc waits for registers another warp-group released, so the math count
        # is the launch count (the register file over the block's threads) plus that share.
        launch_regs = (
            SM90_REGISTERS // (math_threads + WARPGROUP_THREADS) // SETMAXNREG_GRANULARITY
        ) * SETMAXNREG_GRANULARITY
        producer_regs = 40
        math_regs = (
            launch_regs
            + (WARPGROUP_THREADS * (launch_regs - producer_regs) // math_threads)
            // SETMAXNREG_GRANULARITY
            * SETMAXNREG_GRANULARITY
        )
        # Token scales a lane holds: tokens lane + 32 j of the tile.
        token_regs = -(-block_m // 32)
        # 16-byte units of one stage's fp8 row tile.
        stage_units = block_m * block_k // 16
        # Producer lanes each stage block_n / 32 weight rows of B scales.
        scale_rows_per_lane = block_n // 32
        # Static rings have compile-time slots; other rings retain position across tasks.
        static_ring = scale_k % num_stages == 0
        rings_per_task = scale_k // num_stages
        # Even unrolling keeps each step's register fragment buffer a compile-time choice.
        k_unroll = math.lcm(num_stages if static_ring else 1, 2)
        store_count = block_n // store_box_cols
        # A task's e8m0 code table: rows of 4 * scale_k bytes. Rows of whole 64-byte units
        # come by one swizzled TMA in boxes of up to 128 bytes of each row; others by one
        # contiguous bulk copy, rows back to back.
        tables = _uses_tables(block_m, block_n, num_stages, scale_bytes, scale_k, smem_limit)
        table_row_bytes = scale_k * 4
        table_box = _table_box(scale_k) if tables else 0

        def ring_position(kk, ring_index):
            """Ring slot and barrier phase of K step ``kk`` of a task, shared by producers and
            consumers; ``ring_index`` counts the role's earlier tasks (static ring) or K steps
            (index ring)."""
            if static_ring:
                return kk % num_stages, (ring_index * rings_per_task + kk // num_stages) & 1
            return (ring_index + kk) % num_stages, ((ring_index + kk) // num_stages) & 1

        @T.macro
        def load_token_scales(A_scale, token_scales, g, m_start, kk, tx):
            """The A scales at K step ``kk`` of tokens ``lane + 32 j`` of the tile, as floats;
            a token past the tile's rows, or a row past ``a``'s, reads the last one. Under the
            code table ``A_scale`` is the task's token codes, ``A_scale(token, kk)``."""
            for j in T.unroll(token_regs):
                row = m_start + T.min(tx % 32 + 32 * j, block_m - 1)
                if callable(A_scale):
                    value = A_scale(T.min(tx % 32 + 32 * j, block_m - 1), kk)
                elif contiguous:
                    value = A_scale[T.min(row, m - 1), kk]
                elif ragged_slab:
                    value = A_scale[g, T.min(row, max_m - 1), kk]
                else:
                    value = A_scale[g, row, kk]
                if e8m0:
                    token_scales[j] = _e8m0_value(T.cast(value, "uint32"))
                else:
                    token_scales[j] = value

        @T.macro
        def wgmma_slices(step, frags, a_desc_lo, slot, first, count, sets):
            """K32 slices ``first .. first + count`` of a warp-group's 64 x block_m x 128 fp8
            product: the weights from the register fragments, the tokens from ring ``slot``;
            slice i into accumulator set i % sets, each set overwritten at its first slice."""
            # C++ (tl::wgmma_rs): T.ptx_wgmma_rs gets a fence.proxy.async before every issue
            # from TileLang's InjectFenceProxy, since the promotion reads shared memory between
            # issues.
            T.call_extern(
                "handle",
                f"tileops::mxfp4_wgmma_rs<{block_m}, {first}, {count}, {sets}>",
                step.data,
                frags.data,
                a_desc_lo + T.uint32(slot * stage_units),
            )

        @T.macro
        def promote(step, final, scales, token_slices, buf, first, sets, fold, safe, tx):
            """final += step * row scale * token scale over the warp-group's 64 x block_m
            accumulator. The thread's rows are g and g + 8 (scales ``[0]`` and ``[1]`` when
            ``fold``, else ``[sub]`` and ``[4 + sub]`` for sub-block ``first + set``); its
            tokens 8 i + 2 (lane % 4) and the next, from the warp's shared slice. With every
            row scale in [2 ** -63, 2 ** 63], (step * row) * token cannot overflow before the
            result and loses only results under 2 ** -38 to a flushed subnormal token scale;
            ``safe`` steps go through the exact split."""
            pair = tx % 4 * 2
            for i in T.unroll(block_m // 8):
                for e in T.unroll(4):
                    token = token_slices[tx // 32, buf, i * 8 + pair + e % 2]
                    for s in T.unroll(sets):
                        # An if block: TileLang's macro parser takes no conditional expression.
                        if fold:  # noqa: SIM108
                            row = scales[e // 2]
                        else:
                            row = scales[e // 2 * _SUB_BLOCKS + first + s]
                        value = step[s * fragment_regs + i * 4 + e]
                        if safe:
                            # Splits each scale into mantissa and exponent by bit operations
                            # that --use_fast_math does not flush (an e8m0 code 0 is 2 ** -127).
                            final[i * 4 + e] = final[i * 4 + e] + T.call_extern(
                                "float", "tileops::mxfp4_scaled_exact", value, row, token
                            )
                        else:
                            final[i * 4 + e] = final[i * 4 + e] + value * row * token

        @T.macro
        def copy_check(
            table_desc,
            token_desc,
            token_codes,
            walk_source,
            B_scale,
            ahead,
            ahead_tile,
            task_id,
            check_codes,
            check_full,
            check_empty,
            check_index,
        ):
            """Copy the tile codes of the first task from ``task_id`` on, a wave apart, that
            runs (``ahead`` walked on to it; a skipped task takes no table) into table
            ``check_index % 2`` once the math threads have released it: a word of four codes a
            K step, rows up to N's last, 16-byte aligned (N a multiple of 8, tiles of 64
            rows)."""
            next_task = T.alloc_var("int32", init=task_id)
            walk(walk_source, next_task, ahead)
            while T.And(ahead[cur_g] >= num_groups, next_task + num_sms < max_waves * num_sms):
                next_task = next_task + num_sms
                walk(walk_source, next_task, ahead)
            if ahead[cur_g] < num_groups:
                tile_of(next_task, ahead, ahead_tile)
                ahead_g = tasks.expert(ahead)
                ahead_n = tasks.first_col(ahead_tile)
                buf = check_index % 2
                T.barrier_wait(check_empty[buf], (check_index // 2 & 1) ^ 1)
                if T.shuffle_elect(32):
                    if table_box:
                        # Whole boxes land, rows past N (or past the tokens) zero-filled.
                        T.mbarrier_arrive_expect_tx(
                            check_full[buf], block_n * table_row_bytes + block_m * scale_k
                        )
                        if contiguous:
                            T.tma_load(
                                token_desc,
                                check_full[buf],
                                T.address_of(token_codes[buf, 0]),
                                0,
                                tasks.first_row(ahead, ahead_tile),
                                0,
                            )
                        else:
                            T.tma_load(
                                token_desc,
                                check_full[buf],
                                T.address_of(token_codes[buf, 0]),
                                0,
                                tasks.first_row(ahead, ahead_tile),
                                ahead_g,
                                0,
                            )
                        for box in T.unroll(table_row_bytes // table_box):
                            T.tma_load(
                                table_desc,
                                check_full[buf],
                                T.address_of(check_codes[buf, box * block_n * table_box // 4]),
                                box * table_box,
                                ahead_n,
                                ahead_g,
                                0,
                            )
                    else:
                        # C++: arrive.expect_tx and the 1-D bulk copy on it.
                        T.call_extern(
                            "handle",
                            "tileops::mxfp4_bulk_load",
                            T.address_of(check_codes[buf, 0]),
                            T.address_of(B_scale[ahead_g, ahead_n, 0]),
                            T.address_of(check_full[buf]),
                            T.min(block_n, n - ahead_n) * table_row_bytes,
                        )

        @T.macro
        def decode_step(
            A_scale,
            b_packed,
            scale_at,
            fold,
            token_slices,
            token_scales,
            frags,
            scales,
            slot,
            kk,
            wg_row0,
            g,
            m_start,
            tx,
        ):
            """K step ``kk``'s weight fragments from ring ``slot`` and scales from
            ``scale_at(slot, kk)``, and its token
            scales (``token_scales``, loaded a step ahead) into the warp's slice ``kk % 2``;
            loads step ``kk + 1``'s."""
            # C++: the e2m1 -> e4m3 decode is byte permutes (prmt, its sign-fill mode) into the
            # WGMMA register-fragment layout, which no TileLang primitive expresses.
            # The code table's reader takes its layout and the step; a ring stage's does not.
            mode = ("Fold" if fold else "SplitE8M0") if e8m0 else "SplitF32"
            layout_args = f", {scale_k}, {table_box}, {block_n}" if tables else ""
            T.call_extern(
                "handle",
                f"tileops::mxfp4_weight_step<tileops::kMxfp4{mode}{layout_args}>",
                T.address_of(b_packed[slot, 0, 0]),
                scale_at(slot, kk),
                wg_row0,
                frags.data,
                scales.data,
                *((kk,) if tables else ()),
            )
            # C++: a TileLang store to shared memory here gets block-wide bar.sync from the
            # storage-sync pass; the slice is warp-private.
            T.call_extern(
                "handle",
                f"tileops::mxfp4_stage_tokens<{block_m}>",
                T.address_of(token_slices[tx // 32, kk % 2, 0]),
                token_scales.data,
            )
            load_token_scales(A_scale, token_scales, g, m_start, T.min(kk + 1, scale_k - 1), tx)

        @T.macro
        def run_step(
            A_scale,
            b_packed,
            scale_at,
            fold,
            safe,
            token_slices,
            token_scales,
            final,
            step,
            frags,
            scales,
            frags_next,
            scales_next,
            full,
            empty,
            kk,
            ring_index,
            a_desc,
            wg_row0,
            g,
            m_start,
            tx,
        ):
            """K step ``kk``: its WGMMAs on the tokens of its ring slot while the next step's
            weights decode, then its promotion."""
            slot, _ = ring_position(kk, ring_index)
            # A folded step's four slices accumulate in the tensor core; otherwise each
            # sub-block goes to its own set, acc_sets at a time.
            if fold:
                wgmma_slices(step, frags, a_desc, slot, 0, _SUB_BLOCKS, 1)
            else:
                wgmma_slices(step, frags, a_desc, slot, 0, acc_sets, acc_sets)
            next_slot, next_phase = ring_position(kk + 1, ring_index)
            if kk + 1 < scale_k:
                T.barrier_wait(full[next_slot], next_phase)
                decode_step(
                    A_scale,
                    b_packed,
                    scale_at,
                    fold,
                    token_slices,
                    token_scales,
                    frags_next,
                    scales_next,
                    next_slot,
                    kk + 1,
                    wg_row0,
                    g,
                    m_start,
                    tx,
                )
            T.wait_wgmma(0)
            if fold:
                promote(step, final, scales, token_slices, kk % 2, 0, 1, True, safe, tx)
            else:
                promote(step, final, scales, token_slices, kk % 2, 0, acc_sets, False, safe, tx)
                # The remaining sub-blocks, acc_sets at a time (passes of a compile-time count).
                if passes > 1:
                    wgmma_slices(step, frags, a_desc, slot, acc_sets, acc_sets, acc_sets)
                    T.wait_wgmma(0)
                    promote(
                        step,
                        final,
                        scales,
                        token_slices,
                        kk % 2,
                        acc_sets,
                        acc_sets,
                        False,
                        safe,
                        tx,
                    )
                if passes > 2:
                    wgmma_slices(step, frags, a_desc, slot, 2, 1, 1)
                    T.wait_wgmma(0)
                    promote(step, final, scales, token_slices, kk % 2, 2, 1, False, safe, tx)
                    wgmma_slices(step, frags, a_desc, slot, 3, 1, 1)
                    T.wait_wgmma(0)
                    promote(step, final, scales, token_slices, kk % 2, 3, 1, False, safe, tx)
            # Every math warp has read this step's tokens.
            if tx % 32 == 0:
                T.barrier_arrive(empty[slot])

        @T.macro
        def task_steps(
            A_scale,
            b_packed,
            scale_at,
            token_slices,
            token_scales,
            final,
            step,
            frags0,
            scales0,
            frags1,
            scales1,
            full,
            empty,
            ring_index,
            a_desc,
            wg_row0,
            g,
            m_start,
            tx,
            fold_steps,
            safe_steps,
        ):
            """A task's K steps from its first ring slot, each folded into one product when
            ``fold_steps``, else contracted sub-block by sub-block, promoted exactly when
            ``safe_steps``."""
            first_slot, first_phase = ring_position(0, ring_index)
            T.barrier_wait(full[first_slot], first_phase)
            decode_step(
                A_scale,
                b_packed,
                scale_at,
                fold_steps,
                token_slices,
                token_scales,
                frags0,
                scales0,
                first_slot,
                0,
                wg_row0,
                g,
                m_start,
                tx,
            )
            for kk in T.unroll(scale_k, unroll_factor=k_unroll):
                if kk % 2 == 0:
                    run_step(
                        A_scale,
                        b_packed,
                        scale_at,
                        fold_steps,
                        safe_steps,
                        token_slices,
                        token_scales,
                        final,
                        step,
                        frags0,
                        scales0,
                        frags1,
                        scales1,
                        full,
                        empty,
                        kk,
                        ring_index,
                        a_desc,
                        wg_row0,
                        g,
                        m_start,
                        tx,
                    )
                else:
                    run_step(
                        A_scale,
                        b_packed,
                        scale_at,
                        fold_steps,
                        safe_steps,
                        token_slices,
                        token_scales,
                        final,
                        step,
                        frags1,
                        scales1,
                        frags0,
                        scales0,
                        full,
                        empty,
                        kk,
                        ring_index,
                        a_desc,
                        wg_row0,
                        g,
                        m_start,
                        tx,
                    )

        @T.prim_func
        def _moe_grouped_gemm_mxfp4_main(
            A: T.Tensor((*rows_shape, k), a_dtype),  # type: ignore
            A_scale: T.Tensor((*rows_shape, scale_k), scale_dtype),  # type: ignore
            B: T.Tensor((num_groups, n, k // 2), "uint8"),  # type: ignore
            B_scale: T.Tensor((num_groups, n, k // _B_SCALE_BLOCK), scale_dtype),  # type: ignore
            layout: T.Tensor((m,) if metadata_kind == "per_row" else (num_groups,), "int32"),  # type: ignore
            C: T.Tensor((*rows_shape, n), cd_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(num_sms, threads=math_threads + WARPGROUP_THREADS) as (pid,):
                a_shared = T.alloc_shared((num_stages, block_m, block_k), a_dtype)
                b_packed = T.alloc_shared((num_stages, block_n, packed_k), "uint8")
                # B scales a stage, unless the code tables hold them (then one unused row).
                # scope="shared": a shared.dyn store here makes TileLang fence it against the TMA
                # writes with a barrier one warp cannot complete.
                b_scale_shared = T.alloc_shared(
                    (1 if tables else num_stages, block_n * _SUB_BLOCKS),
                    scale_dtype,
                    scope="shared",
                )
                shared_c = T.alloc_shared((block_m * block_n,), cd_dtype)
                final = T.alloc_local((fragment_regs,), accum_dtype)
                step = T.alloc_local((acc_sets * fragment_regs,), accum_dtype)
                # Two steps' weight fragments and row scales: one a WGMMA reads while the next
                # step's decode.
                frags0 = T.alloc_local((_SUB_BLOCKS * 4,), "uint32")
                frags1 = T.alloc_local((_SUB_BLOCKS * 4,), "uint32")
                scales0 = T.alloc_local((2 * _SUB_BLOCKS,), accum_dtype)
                scales1 = T.alloc_local((2 * _SUB_BLOCKS,), accum_dtype)
                # Each math warp's token scales of two steps: the promotion of one reads them
                # while the next step's are staged. Lane l loads tokens l + 32 j a step ahead.
                token_slices = T.alloc_shared((math_warps, 2, block_m), accum_dtype)
                token_scales = T.alloc_local((token_regs,), accum_dtype)
                T.annotate_layout({a_shared: tilelang.layout.make_swizzled_layout(a_shared)})
                # A stage is full once its TMA bytes land and, under float32 scales, each
                # producer lane's cp.async copies of its B scales do.
                full = T.alloc_barrier([1 if tables else 32] * num_stages)
                empty = T.alloc_barrier([math_warps] * num_stages)
                # With the tables, two tables of a tile's weight-scale codes, which a task's
                # check scans and its decode reads its steps' scales from: the producer copies
                # a task's when it starts the one before; the math threads release a table
                # when the task ends. Words: a byte buffer of 2 * 32768 overflows TVM's 16-bit
                # vector lanes.
                check_codes = T.alloc_shared((2, (block_n if tables else 1) * scale_k), "uint32")
                # With the code table by TMA, the tile's token codes, rows of scale_k bytes.
                token_codes = T.alloc_shared((2, block_m * scale_k if table_box else 16), "uint8")
                check_full = T.alloc_barrier([1, 1])
                check_empty = T.alloc_barrier([1, 1])
                # The producer's walk a wave ahead, to the task whose codes it copies next.
                ahead = T.alloc_local((tasks.cursor_len,), "int32")
                ahead_tile = T.alloc_local((2,), "int32")
                # Each role's count of the tasks it has run, the check table's phase.
                check_index = T.alloc_var("int32", init=0)
                # Each role's count of the tasks (static ring) or K steps (index ring) it has run.
                # Not the wave index: a skipped task advances no barrier.
                ring_index = T.alloc_var("int32", init=0)

                cursor = T.alloc_local((tasks.cursor_len,), "int32")
                tile = T.alloc_local((2,), "int32")
                for i in T.unroll(tasks.cursor_len):
                    cursor[i] = 0
                    ahead[i] = 0
                tx = T.get_thread_binding()
                if recover:
                    ends = T.alloc_shared((num_groups,), "int32", scope="shared")
                    tasks.recover_ends(layout, ends, tx, math_threads + WARPGROUP_THREADS)
                T.sync_threads()
                # [m, N] or [E, max_m, N], so the store of a run's or slab's last rows is
                # clipped at its end.
                output_desc = row_tiles_descriptor(
                    TMA_DTYPE_BFLOAT16,
                    C,
                    n,
                    n * 2,
                    store_box_cols,
                    TMA_SWIZZLE_128B,
                    TMA_L2_PROMOTION_128B,
                )
                a_load_desc = row_tiles_descriptor(
                    TMA_DTYPE_UINT8, A, k, k, block_k, TMA_SWIZZLE_128B, TMA_L2_PROMOTION_256B
                )
                # Rows of 64 packed bytes, 64B-swizzled so a warp's reads of eight rows
                # spread over the banks.
                b_load_desc = tasks.expert_rows_descriptor(
                    TMA_DTYPE_UINT8,
                    B,
                    k // 2,
                    k // 2,
                    packed_k,
                    TMA_SWIZZLE_64B,
                    TMA_L2_PROMOTION_256B,
                )

                if table_box:
                    token_desc = row_tiles_descriptor(
                        TMA_DTYPE_UINT8,
                        A_scale,
                        scale_k,
                        scale_k,
                        scale_k,
                        TMA_SWIZZLE_NONE,
                        TMA_L2_PROMOTION_128B,
                    )
                    table_desc = tasks.expert_rows_descriptor(
                        TMA_DTYPE_UINT8,
                        B_scale,
                        table_row_bytes,
                        table_row_bytes,
                        table_box,
                        TMA_SWIZZLE_128B if table_box == 128 else TMA_SWIZZLE_64B,
                        TMA_L2_PROMOTION_128B,
                    )

                if tx >= math_threads:
                    T.dec_max_nreg(producer_regs)
                    producer_tx = tx - math_threads
                    # Shuffled so the value stays warp-uniform for NVCC's uniform datapath.
                    producer_warp = T.tvm_warp_shuffle(
                        T.uint32(0xFFFFFFFF), producer_tx // 32, 0, 32, 32
                    )
                    if T.And(tables, producer_warp == 0):
                        copy_check(
                            table_desc if table_box else None,
                            token_desc if table_box else None,
                            token_codes,
                            ends if recover else layout,
                            B_scale,
                            ahead,
                            ahead_tile,
                            pid,
                            check_codes,
                            check_full,
                            check_empty,
                            0,
                        )
                    for wave in T.serial(max_waves):
                        task_id = num_sms * wave + pid
                        # Skipped tasks must advance neither role's barriers.
                        walk(ends if recover else layout, task_id, cursor)
                        if T.And(cursor[cur_g] < num_groups, producer_warp == 0):
                            tile_of(task_id, cursor, tile)
                            g = T.meta_var(tasks.expert(cursor))
                            m_start = T.meta_var(tasks.first_row(cursor, tile))
                            n_start = T.meta_var(tasks.first_col(tile))
                            if tables:
                                # The next task's codes, a task ahead of its check.
                                copy_check(
                                    table_desc if table_box else None,
                                    token_desc if table_box else None,
                                    token_codes,
                                    ends if recover else layout,
                                    B_scale,
                                    ahead,
                                    ahead_tile,
                                    task_id + num_sms,
                                    check_codes,
                                    check_full,
                                    check_empty,
                                    check_index + 1,
                                )
                                check_index = check_index + 1
                            for kk in T.unroll(scale_k, unroll_factor=k_unroll):
                                slot, phase = ring_position(kk, ring_index)
                                T.barrier_wait(empty[slot], phase ^ 1)
                                if T.shuffle_elect(32):
                                    if tables:
                                        T.mbarrier_arrive_expect_tx(
                                            full[slot], block_m * block_k + block_n * packed_k
                                        )
                                    else:
                                        T.mbarrier_expect_tx(
                                            full[slot], block_m * block_k + block_n * packed_k
                                        )
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
                                    T.tma_load(
                                        b_load_desc,
                                        full[slot],
                                        T.address_of(b_packed[slot, 0, 0]),
                                        kk * packed_k,
                                        n_start,
                                        g,
                                        0,
                                    )
                                if not tables:
                                    # Each lane copies the four B scales of its rows; rows past
                                    # N read the last row, which no stored column uses.
                                    for r in T.unroll(scale_rows_per_lane):
                                        row = r * 32 + producer_tx
                                        b_row = T.meta_var(T.min(n_start + row, n - 1))
                                        T.ptx_cp_async(
                                            T.access_ptr(
                                                b_scale_shared[slot, row * _SUB_BLOCKS], "w", 4
                                            ),
                                            T.access_ptr(
                                                B_scale[g, b_row, kk * _SUB_BLOCKS], "r", 4
                                            ),
                                            _SUB_BLOCKS,
                                        )
                                    # Arrives once each lane's copies land.
                                    T.cp_async_barrier_noinc(full[slot])
                            ring_index = ring_index + (1 if static_ring else scale_k)
                else:
                    T.inc_max_nreg(math_regs)
                    math_wg_idx = T.tvm_warp_shuffle(
                        T.uint32(0xFFFFFFFF), tx // WARPGROUP_THREADS, 0, 32, 32
                    )
                    # The warp-group's 64 weight rows of the tile.
                    wg_row0 = math_wg_idx * WGMMA_ROWS
                    # The rows' ring as a 128B-swizzled K-major WGMMA operand; a slot or K32
                    # slice is an offset in 16-byte units of the descriptor's low word, kept
                    # warp-uniform (the low word the C++ WGMMA issue above takes).
                    a_desc = T.call_extern(
                        "uint32", "tileops::moe_fp8_wgmma_desc_lo", T.address_of(a_shared[0, 0, 0])
                    )

                    for wave in T.serial(max_waves):
                        task_id = num_sms * wave + pid
                        walk(ends if recover else layout, task_id, cursor)
                        if cursor[cur_g] < num_groups:
                            tile_of(task_id, cursor, tile)
                            g = T.meta_var(tasks.expert(cursor))
                            m_start = T.meta_var(tasks.first_row(cursor, tile))
                            n_start = T.meta_var(tasks.first_col(tile))
                            T.clear(final)
                            # Whether every weight scale the task reads lies in [2 ** -63,
                            # 2 ** 63] and, under e8m0 scales, every step of every weight row
                            # folds exactly: one barrier a task makes the choice uniform. A
                            # task that does not contracts sub-block by sub-block, exactly. A
                            # variable: an expression would be inlined into its use, past later
                            # writes of what it reads.
                            fast = T.alloc_var("int32")
                            # With the tables a step's weight scales are the task's code
                            # table's (and, with it by TMA, its token scales the token table's);
                            # otherwise the ring slot's. The table is the task's
                            # for the whole task (bound here, check_index moves at its end).
                            table = T.meta_var(check_index % 2)
                            if table_box:
                                token_at = T.meta_var(
                                    lambda token, kk, t=table: token_codes[t, token * scale_k + kk]
                                )
                            else:
                                token_at = A_scale
                            if tables:
                                scale_at = T.meta_var(
                                    lambda slot, kk, t=table: T.address_of(check_codes[t, 0])
                                )
                            else:
                                scale_at = T.meta_var(
                                    lambda slot, kk: T.address_of(b_scale_shared[slot, 0])
                                )
                            if tables:
                                T.barrier_wait(check_full[check_index % 2], check_index // 2 & 1)
                                # C++: barrier.red.and over the math threads, of the scan of the
                                # codes the producer copied.
                                fast = T.call_extern(
                                    "int32",
                                    "tileops::mxfp4_all_threads",
                                    T.call_extern(
                                        "bool",
                                        f"tileops::mxfp4_rows_fast_shared<{scale_k}, "
                                        f"{table_box}, {block_n}>",
                                        T.address_of(check_codes[check_index % 2, 0]),
                                        wg_row0,
                                        n - 1 - n_start,
                                    ),
                                    check_barrier,
                                    math_threads,
                                )
                            else:
                                # C++: barrier.red.and over the math threads, and the scale
                                # scan with all its loads issued before any is used.
                                fast = T.call_extern(
                                    "int32",
                                    "tileops::mxfp4_all_threads",
                                    T.call_extern(
                                        "bool",
                                        f"tileops::mxfp4_rows_fast<{scale_k}, {str(e8m0).lower()}>",
                                        B_scale.data,
                                        g * n + n_start + wg_row0,
                                        g * n + n - 1,
                                    ),
                                    check_barrier,
                                    math_threads,
                                )
                            load_token_scales(token_at, token_scales, g, m_start, 0, tx)
                            if fast != 0:
                                task_steps(
                                    token_at,
                                    b_packed,
                                    scale_at,
                                    token_slices,
                                    token_scales,
                                    final,
                                    step,
                                    frags0,
                                    scales0,
                                    frags1,
                                    scales1,
                                    full,
                                    empty,
                                    ring_index,
                                    a_desc,
                                    wg_row0,
                                    g,
                                    m_start,
                                    tx,
                                    e8m0,
                                    False,
                                )
                            else:
                                task_steps(
                                    token_at,
                                    b_packed,
                                    scale_at,
                                    token_slices,
                                    token_scales,
                                    final,
                                    step,
                                    frags0,
                                    scales0,
                                    frags1,
                                    scales1,
                                    full,
                                    empty,
                                    ring_index,
                                    a_desc,
                                    wg_row0,
                                    g,
                                    m_start,
                                    tx,
                                    False,
                                    True,
                                )
                            ring_index = ring_index + (1 if static_ring else scale_k)
                            # The previous task's output stores, issued by some math threads
                            # and read by all on a tight tail, must be done with shared_c
                            # before any math thread rewrites it.
                            if tx < store_count:
                                T.tma_store_wait(0)
                            T.sync_threads(barrier_id=reuse_barrier, arrive_count=math_threads)
                            if tables:
                                # Every math thread has decoded the task's last step (the
                                # barrier above): its table goes back to the producer.
                                if tx == 0:
                                    T.barrier_arrive(check_empty[check_index % 2])
                                check_index = check_index + 1
                            # C++: transposed bf16 stores into the swizzled staging tile; a
                            # TileLang shared store gets the storage-sync pass's bar.sync.
                            T.call_extern(
                                "handle",
                                f"tileops::mxfp4_stage_transposed<{block_m}>",
                                final.data,
                                T.address_of(shared_c[0]),
                                wg_row0,
                            )
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=staged_barrier, arrive_count=math_threads)
                            tasks.store_tile(
                                output_desc,
                                shared_c,
                                C,
                                cursor,
                                tile,
                                store_box_cols,
                                store_count,
                                tx,
                                math_threads,
                            )
                    if tx < store_count:
                        T.tma_store_wait(0)

        return _moe_grouped_gemm_mxfp4_main

    return _moe_grouped_gemm_mxfp4_func


# Rows of a shared-decode tile: two 64-row math warp-groups of tokens, the WGMMA M operand.
_SHARED_DECODE_ROWS = 2 * WGMMA_ROWS
# Weight rows of a shared-decode tile, the WGMMA N extent: a math thread holds a step's and the
# task's accumulators, 64 floats each.
_SHARED_DECODE_COLUMNS = 128
# Decoded fp8 stages: the decode warps fill one while the math warp-groups contract the other.
_DECODED_STAGES = 2
# Stages of packed weights: the decode warps release a stage once they have read it, a step
# ahead of the math warp-groups, so the ring of fp8 token rows sets the loads' lead.
_PACKED_STAGES = 3


def _shared_decode_smem_bytes(block_n: int, num_stages: int, scale_k: int) -> int:
    """Bytes of a shared-decode config: the ring of fp8 token rows, the packed-weight and
    decoded fp8 stages, two tasks' weight-code and token-code tables (1024 bytes of alignment),
    two tasks' row references, column scales and flags, the bfloat16 output tile and the
    barriers."""
    block_m = _SHARED_DECODE_ROWS
    ring = num_stages * block_m * QUANT_SCALE_BLOCK
    packed = _PACKED_STAGES * block_n * QUANT_SCALE_BLOCK // 2
    decoded = _DECODED_STAGES * block_n * QUANT_SCALE_BLOCK
    tables = 2 * (block_n * scale_k * 4 + block_m * scale_k) + SHARED_BUFFER_ALIGN_BYTES
    task_info = 2 * (2 * block_n * 4 + 4)
    barriers = (2 * num_stages + 2 * _PACKED_STAGES + 2 * _DECODED_STAGES + 4) * 8
    return ring + packed + decoded + tables + task_info + block_m * block_n * 2 + barriers


@target_cache(maxsize=32)
def _moe_grouped_gemm_mxfp4_shared_decode_kernel(
    num_groups: int,
    n: int,
    k: int,
    sm_count: int,
    smem_limit: int,
    metadata_kind: str,
    alignment: int,
) -> Callable:
    """Build a persistent MXFP4 grouped GEMM over aligned contiguous rows under e8m0 scales that
    decodes each weight tile once into shared memory.

    The tokens are the WGMMA M operand (two 64-row math warp-groups), the decoded weights its N
    operand, both read from shared memory. The first warp of two producer warp-groups issues
    the TMA loads; their other seven decode each step's packed weights into an fp8 stage. A
    task whose rows fold (``mxfp4_scan_task``) contracts each step at once and promotes it under
    its token scales alone, its columns scaled once at the end; another contracts each
    sub-block alone and promotes it by ``mxfp4_scaled_exact``.
    """
    cur_g = CURSOR.g
    block_k = QUANT_SCALE_BLOCK
    packed_k = block_k // 2
    scale_k = k // block_k
    table_row_bytes = scale_k * 4
    table_box = _table_box(scale_k)
    # Columns of one output staging box: 128 bytes of bfloat16, the TMA store's 128B swizzle.
    store_box_cols = 64

    @tilelang.jit(
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        compile_flags=[
            "-O3",
            "--use_fast_math",
            "-DENABLE_BF16",
            *csrc_include("moe_grouped_gemm_mxfp4_helper.h"),
        ],
    )
    def _moe_grouped_gemm_mxfp4_shared_decode_func(
        block_m: int, block_n: int, num_stages: int, num_sms: int
    ) -> Callable:
        config = dict(
            zip(
                MoEGroupedGEMMMXFP4Kernel._CONFIG_KEYS,
                (block_m, block_n, num_stages, num_sms),
                strict=True,
            )
        )
        reason = MoEGroupedGEMMMXFP4SharedDecodeKernel.config_refusal(
            config,
            sm_count=sm_count,
            smem_limit=smem_limit,
            alignment=alignment,
            scale_k=scale_k,
        )
        if reason is not None:
            raise ValueError(f"cannot build the MXFP4 grouped GEMM for {config}: {reason}")
        tasks = task_walk(
            num_groups=num_groups,
            block_m=block_m,
            block_n=block_n,
            n=n,
            num_sms=num_sms,
            max_m=None,
            packing="aligned",
            metadata_kind=metadata_kind,
            alignment=alignment,
        )
        m, max_waves = tasks.m, tasks.max_waves
        row_tiles_descriptor = tasks.row_tiles_descriptor
        walk, tile_of = tasks.walk, tasks.tile_of
        math_threads = block_m // WGMMA_ROWS * WARPGROUP_THREADS
        math_warps = math_threads // 32
        fragment_regs = WGMMA_ROWS * block_n // WARPGROUP_THREADS
        store_count = block_n // store_box_cols
        # Named barriers: the math threads' shared_c free to restage and shared_c staged, and
        # the decode threads' vote on a task's fold.
        reuse_barrier, staged_barrier, vote_barrier = 12, 13, 14
        # Producer warp-groups: their first warp issues the TMA loads, the other seven decode,
        # two to each scheduler but one, so one decode warp's dependent chain issues while the
        # other's waits.
        producer_groups = 2
        decode_warps = producer_groups * 4 - 1
        # Registers of a math thread: a step's and the task's 64 accumulators. The producer
        # threads take what the register file leaves.
        math_regs = 168
        layout_args = f"{scale_k}, {table_box}, {block_n}"
        decode_threads = decode_warps * 32
        # Decode units (a sub-block of a weight row) a decode thread takes each step.
        unit_slots = -(-_SUB_BLOCKS * block_n // decode_threads)
        producer_threads = producer_groups * WARPGROUP_THREADS
        producer_regs = (
            (SM90_REGISTERS - math_threads * math_regs)
            // producer_threads
            // SETMAXNREG_GRANULARITY
            * SETMAXNREG_GRANULARITY
        )

        def position(step, depth):
            """Stage and barrier phase of a role's ``step``-th K step in a ring of ``depth``."""
            return step % depth, (step // depth) & 1

        @T.macro
        def copy_tables(
            table_desc,
            token_desc,
            layout,
            ahead,
            ahead_tile,
            task_id,
            tab_full,
            tab_empty,
            codes,
            token_codes,
            index,
        ):
            """Copy the weight and token codes of the first task from ``task_id`` on, a wave
            apart, that runs (``ahead`` walked on to it) into tables ``index % 2`` once both
            consuming roles have released them."""
            next_task = T.alloc_var("int32", init=task_id)
            walk(layout, next_task, ahead)
            while T.And(ahead[cur_g] >= num_groups, next_task + num_sms < max_waves * num_sms):
                next_task = next_task + num_sms
                walk(layout, next_task, ahead)
            if ahead[cur_g] < num_groups:
                tile_of(next_task, ahead, ahead_tile)
                buf = index % 2
                T.barrier_wait(tab_empty[buf], (index // 2 & 1) ^ 1)
                if T.shuffle_elect(32):
                    # Whole boxes land, rows past N zero-filled.
                    T.mbarrier_arrive_expect_tx(
                        tab_full[buf], block_n * table_row_bytes + block_m * scale_k
                    )
                    T.tma_load(
                        token_desc,
                        tab_full[buf],
                        T.address_of(token_codes[buf, 0]),
                        0,
                        tasks.first_row(ahead, ahead_tile),
                        0,
                    )
                    for box in T.unroll(table_row_bytes // table_box):
                        T.tma_load(
                            table_desc,
                            tab_full[buf],
                            T.address_of(codes[buf, box * block_n * table_box // 4]),
                            box * table_box,
                            tasks.first_col(ahead_tile),
                            tasks.expert(ahead),
                            0,
                        )

        @T.macro
        def math_step(
            fold,
            buf,
            kk,
            step_index,
            a_desc_lo,
            b_desc_lo,
            token_codes,
            codes,
            tokens,
            step,
            final,
            full,
            empty,
            decoded_full,
            decoded_empty,
            acc_row0,
            tx,
        ):
            """K step ``kk`` of the task: its WGMMAs, the release of its stages, its promotion;
            all at once under the task's ``fold`` flag, sub-block by sub-block otherwise."""
            slot, phase = position(step_index, num_stages)
            stage, stage_phase = position(step_index, _DECODED_STAGES)
            # The token scales of the thread's two accumulator rows.
            tokens[0] = _e8m0_value(T.cast(token_codes[buf, acc_row0 * scale_k + kk], "uint32"))
            tokens[1] = _e8m0_value(
                T.cast(token_codes[buf, (acc_row0 + 8) * scale_k + kk], "uint32")
            )
            T.barrier_wait(full[slot], phase)
            T.barrier_wait(decoded_full[stage], stage_phase)
            a_lo = T.meta_var(a_desc_lo + T.uint32(slot * block_m * (block_k // 16)))
            b_lo = T.meta_var(b_desc_lo + T.uint32(stage * block_n * (block_k // 16)))
            if fold != 0:
                T.call_extern(
                    "handle",
                    f"tileops::moe_fp8_wgmma_64x128_by_128xN_lo<{block_n}>",
                    step.data,
                    a_lo,
                    b_lo,
                )
                T.wait_wgmma(0)
                if tx % 32 == 0:
                    T.barrier_arrive(empty[slot])
                    T.barrier_arrive(decoded_empty[stage])
                T.call_extern(
                    "handle",
                    f"tileops::moe_fp8_promote<{block_n}>",
                    step.data,
                    final.data,
                    tokens[0],
                    tokens[1],
                    T.float32(1.0),
                )
            else:
                for sub_block in T.unroll(_SUB_BLOCKS):
                    T.call_extern(
                        "handle",
                        f"tileops::mxfp4_wgmma_ss_slice<{block_n}>",
                        step.data,
                        a_lo,
                        b_lo,
                        sub_block,
                    )
                    T.wait_wgmma(0)
                    T.call_extern(
                        "handle",
                        f"tileops::mxfp4_promote_exact<{layout_args}>",
                        step.data,
                        final.data,
                        T.address_of(codes[buf, 0]),
                        kk,
                        sub_block,
                        tokens[0],
                        tokens[1],
                    )
                if tx % 32 == 0:
                    T.barrier_arrive(empty[slot])
                    T.barrier_arrive(decoded_empty[stage])

        @T.prim_func
        def _moe_grouped_gemm_mxfp4_shared_decode_main(
            A: T.Tensor((m, k), "float8_e4m3fn"),  # type: ignore
            A_scale: T.Tensor((m, scale_k), "uint8"),  # type: ignore
            B: T.Tensor((num_groups, n, k // 2), "uint8"),  # type: ignore
            B_scale: T.Tensor((num_groups, n, k // _B_SCALE_BLOCK), "uint8"),  # type: ignore
            layout: T.Tensor((m,) if metadata_kind == "per_row" else (num_groups,), "int32"),  # type: ignore
            C: T.Tensor((m, n), "bfloat16"),  # type: ignore
        ) -> None:
            with T.Kernel(num_sms, threads=math_threads + producer_threads) as (pid,):
                a_shared = T.alloc_shared((num_stages, block_m, block_k), "float8_e4m3fn")
                b_packed = T.alloc_shared((_PACKED_STAGES, block_n, packed_k), "uint8")
                b_decoded = T.alloc_shared((_DECODED_STAGES, block_n, block_k), "float8_e4m3fn")
                shared_c = T.alloc_shared((block_m * block_n,), "bfloat16")
                # Words: a byte buffer of 2 * 32768 overflows TVM's 16-bit vector lanes.
                codes = T.alloc_shared((2, block_n * scale_k), "uint32")
                token_codes = T.alloc_shared((2, block_m * scale_k), "uint8")
                # A task's row references, column scales and fold flag, which the decode threads
                # write. scope="shared": a shared.dyn store here makes TileLang fence it against
                # the TMA writes with a barrier one warp cannot complete.
                refs = T.alloc_shared((2, block_n), "int32", scope="shared")
                column_scales = T.alloc_shared((2, block_n), "float32", scope="shared")
                flags = T.alloc_shared((2,), "int32", scope="shared")
                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_decoded: tilelang.layout.make_swizzled_layout(b_decoded),
                    }
                )
                step = T.alloc_local((fragment_regs,), "float32")
                final = T.alloc_local((fragment_regs,), "float32")
                tokens = T.alloc_local((2,), "float32")
                unit_refs = T.alloc_local((unit_slots,), "int32")
                # A token-row stage is full once its TMA bytes land; the math warps release it.
                full = T.alloc_barrier([1] * num_stages)
                empty = T.alloc_barrier([math_warps] * num_stages)
                packed_full = T.alloc_barrier([1] * _PACKED_STAGES)
                packed_empty = T.alloc_barrier([decode_warps] * _PACKED_STAGES)
                decoded_full = T.alloc_barrier([decode_warps] * _DECODED_STAGES)
                decoded_empty = T.alloc_barrier([math_warps] * _DECODED_STAGES)
                tab_full = T.alloc_barrier([1, 1])
                tab_empty = T.alloc_barrier([decode_warps + math_warps] * 2)
                ahead = T.alloc_local((tasks.cursor_len,), "int32")
                ahead_tile = T.alloc_local((2,), "int32")
                cursor = T.alloc_local((tasks.cursor_len,), "int32")
                tile = T.alloc_local((2,), "int32")
                # Each role's count of the K steps and of the tasks it has run. Not the wave
                # index: a skipped task advances no barrier.
                step_index = T.alloc_var("int32", init=0)
                task_index = T.alloc_var("int32", init=0)
                for i in T.unroll(tasks.cursor_len):
                    cursor[i] = 0
                    ahead[i] = 0
                tx = T.get_thread_binding()
                T.sync_threads()
                output_desc = row_tiles_descriptor(
                    TMA_DTYPE_BFLOAT16,
                    C,
                    n,
                    n * 2,
                    store_box_cols,
                    TMA_SWIZZLE_128B,
                    TMA_L2_PROMOTION_128B,
                )
                a_load_desc = row_tiles_descriptor(
                    TMA_DTYPE_UINT8, A, k, k, block_k, TMA_SWIZZLE_128B, TMA_L2_PROMOTION_256B
                )
                # Rows of 64 packed bytes, unswizzled: a decode warp reads 512 consecutive bytes.
                b_load_desc = tasks.expert_rows_descriptor(
                    TMA_DTYPE_UINT8,
                    B,
                    k // 2,
                    k // 2,
                    packed_k,
                    TMA_SWIZZLE_NONE,
                    TMA_L2_PROMOTION_256B,
                )
                token_desc = row_tiles_descriptor(
                    TMA_DTYPE_UINT8,
                    A_scale,
                    scale_k,
                    scale_k,
                    scale_k,
                    TMA_SWIZZLE_NONE,
                    TMA_L2_PROMOTION_128B,
                )
                table_desc = tasks.expert_rows_descriptor(
                    TMA_DTYPE_UINT8,
                    B_scale,
                    table_row_bytes,
                    table_row_bytes,
                    table_box,
                    TMA_SWIZZLE_128B if table_box == 128 else TMA_SWIZZLE_64B,
                    TMA_L2_PROMOTION_128B,
                )

                if tx >= math_threads:
                    T.dec_max_nreg(producer_regs)
                    producer_tx = tx - math_threads
                    # Shuffled so the value stays warp-uniform for NVCC's uniform datapath.
                    producer_warp = T.tvm_warp_shuffle(
                        T.uint32(0xFFFFFFFF), producer_tx // 32, 0, 32, 32
                    )
                    if producer_warp == 0:
                        copy_tables(
                            table_desc,
                            token_desc,
                            layout,
                            ahead,
                            ahead_tile,
                            pid,
                            tab_full,
                            tab_empty,
                            codes,
                            token_codes,
                            0,
                        )
                        for wave in T.serial(max_waves):
                            task_id = num_sms * wave + pid
                            walk(layout, task_id, cursor)
                            if cursor[cur_g] < num_groups:
                                tile_of(task_id, cursor, tile)
                                g = T.meta_var(tasks.expert(cursor))
                                m_start = T.meta_var(tasks.first_row(cursor, tile))
                                n_start = T.meta_var(tasks.first_col(tile))
                                for kk in T.serial(scale_k):
                                    packed_slot, packed_phase = position(step_index, _PACKED_STAGES)
                                    T.barrier_wait(packed_empty[packed_slot], packed_phase ^ 1)
                                    if T.shuffle_elect(32):
                                        T.mbarrier_arrive_expect_tx(
                                            packed_full[packed_slot], block_n * packed_k
                                        )
                                        T.tma_load(
                                            b_load_desc,
                                            packed_full[packed_slot],
                                            T.address_of(b_packed[packed_slot, 0, 0]),
                                            kk * packed_k,
                                            n_start,
                                            g,
                                            0,
                                        )
                                    slot, phase = position(step_index, num_stages)
                                    T.barrier_wait(empty[slot], phase ^ 1)
                                    if T.shuffle_elect(32):
                                        T.mbarrier_arrive_expect_tx(full[slot], block_m * block_k)
                                        T.tma_load(
                                            a_load_desc,
                                            full[slot],
                                            T.address_of(a_shared[slot, 0, 0]),
                                            kk * block_k,
                                            m_start,
                                            0,
                                        )
                                    # The next task's codes once this task's step num_stages - 1
                                    # has its slot: by then both consumers are past the task
                                    # before this one.
                                    if kk == num_stages - 1:
                                        copy_tables(
                                            table_desc,
                                            token_desc,
                                            layout,
                                            ahead,
                                            ahead_tile,
                                            task_id + num_sms,
                                            tab_full,
                                            tab_empty,
                                            codes,
                                            token_codes,
                                            task_index + 1,
                                        )
                                    step_index = step_index + 1
                                task_index = task_index + 1
                    else:
                        decode_tx = producer_tx - 32
                        for wave in T.serial(max_waves):
                            task_id = num_sms * wave + pid
                            walk(layout, task_id, cursor)
                            if cursor[cur_g] < num_groups:
                                tile_of(task_id, cursor, tile)
                                n_start = T.meta_var(tasks.first_col(tile))
                                buf = task_index % 2
                                T.barrier_wait(tab_full[buf], task_index // 2 & 1)
                                # C++: barrier.red.and over the decode threads, of the scan.
                                fold = T.call_extern(
                                    "int32",
                                    "tileops::mxfp4_all_threads",
                                    T.call_extern(
                                        "bool",
                                        f"tileops::mxfp4_scan_task<{layout_args}, {block_m}, "
                                        f"{decode_threads}>",
                                        T.address_of(codes[buf, 0]),
                                        T.address_of(token_codes[buf, 0]),
                                        T.address_of(refs[buf, 0]),
                                        T.address_of(column_scales[buf, 0]),
                                        n - 1 - n_start,
                                        decode_tx,
                                    ),
                                    vote_barrier,
                                    decode_threads,
                                )
                                T.call_extern(
                                    "handle",
                                    f"tileops::mxfp4_unit_refs<{block_n}, {decode_threads}>",
                                    T.address_of(refs[buf, 0]),
                                    unit_refs.data,
                                    decode_tx,
                                )
                                if decode_tx == 0:
                                    T.call_extern(
                                        "handle",
                                        "tileops::mxfp4_store_flag",
                                        T.address_of(flags[buf]),
                                        fold,
                                    )
                                for kk in T.serial(scale_k):
                                    packed_slot, packed_phase = position(step_index, _PACKED_STAGES)
                                    stage, stage_phase = position(step_index, _DECODED_STAGES)
                                    T.barrier_wait(packed_full[packed_slot], packed_phase)
                                    T.barrier_wait(decoded_empty[stage], stage_phase ^ 1)
                                    T.call_extern(
                                        "handle",
                                        f"tileops::mxfp4_decode_shared<{layout_args}, "
                                        f"{decode_threads}>",
                                        T.address_of(b_packed[packed_slot, 0, 0]),
                                        T.address_of(codes[buf, 0]),
                                        T.address_of(b_decoded[stage, 0, 0]),
                                        T.address_of(packed_empty[packed_slot]),
                                        unit_refs.data,
                                        fold != 0,
                                        kk,
                                        decode_tx,
                                    )
                                    # C++: the WGMMA reads the stage through the async proxy, so
                                    # each thread fences its stores before its warp arrives.
                                    T.call_extern(
                                        "handle",
                                        "tileops::mxfp4_publish_stage",
                                        T.address_of(decoded_full[stage]),
                                    )
                                    step_index = step_index + 1
                                T.call_extern(
                                    "handle",
                                    "tileops::mxfp4_warp_arrive",
                                    T.address_of(tab_empty[buf]),
                                )
                                task_index = task_index + 1
                else:
                    T.inc_max_nreg(math_regs)
                    math_wg_idx = T.tvm_warp_shuffle(
                        T.uint32(0xFFFFFFFF), tx // WARPGROUP_THREADS, 0, 32, 32
                    )
                    math_wg_offset = math_wg_idx * WGMMA_ROWS
                    a_desc_lo = T.call_extern(
                        "uint32",
                        "tileops::moe_fp8_wgmma_desc_lo",
                        T.address_of(a_shared[0, math_wg_offset, 0]),
                    )
                    b_desc_lo = T.call_extern(
                        "uint32",
                        "tileops::moe_fp8_wgmma_desc_lo",
                        T.address_of(b_decoded[0, 0, 0]),
                    )
                    # Lanes 4i..4i+3 of warp w hold accumulator rows w*16 + i and w*16 + i + 8.
                    acc_row0 = (tx // 32) * 16 + (tx % 32) // 4
                    for wave in T.serial(max_waves):
                        task_id = num_sms * wave + pid
                        walk(layout, task_id, cursor)
                        if cursor[cur_g] < num_groups:
                            tile_of(task_id, cursor, tile)
                            buf = task_index % 2
                            T.clear(final)
                            T.barrier_wait(tab_full[buf], task_index // 2 & 1)
                            # The decode threads set the task's flag before its first stage.
                            first_stage, first_phase = position(step_index, _DECODED_STAGES)
                            T.barrier_wait(decoded_full[first_stage], first_phase)
                            fold = T.alloc_var("int32")
                            fold = flags[buf]
                            for kk in T.serial(scale_k):
                                math_step(
                                    fold,
                                    buf,
                                    kk,
                                    step_index,
                                    a_desc_lo,
                                    b_desc_lo,
                                    token_codes,
                                    codes,
                                    tokens,
                                    step,
                                    final,
                                    full,
                                    empty,
                                    decoded_full,
                                    decoded_empty,
                                    acc_row0,
                                    tx,
                                )
                                step_index = step_index + 1
                            if fold != 0:
                                T.call_extern(
                                    "handle",
                                    f"tileops::mxfp4_scale_columns<{block_n}>",
                                    final.data,
                                    T.address_of(column_scales[buf, 0]),
                                )
                            # After every lane's last read of the task's codes and column
                            # scales (C++: __syncwarp, then one arrival a warp): the release lets
                            # the next tables land and the decode warps rescan.
                            T.call_extern(
                                "handle",
                                "tileops::mxfp4_warp_arrive",
                                T.address_of(tab_empty[buf]),
                            )
                            # The previous task's stores must be done with shared_c.
                            if tx < store_count:
                                T.tma_store_wait(0)
                            T.sync_threads(barrier_id=reuse_barrier, arrive_count=math_threads)
                            T.call_extern(
                                "handle",
                                f"tileops::moe_fp8_stsm_bf16_swizzled<{block_m}, {block_n}>",
                                final.data,
                                T.address_of(shared_c[0]),
                                math_wg_offset,
                            )
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=staged_barrier, arrive_count=math_threads)
                            tasks.store_tile(
                                output_desc,
                                shared_c,
                                C,
                                cursor,
                                tile,
                                store_box_cols,
                                store_count,
                                tx,
                                math_threads,
                            )
                            task_index = task_index + 1
                    if tx < store_count:
                        T.tma_store_wait(0)

        return _moe_grouped_gemm_mxfp4_shared_decode_main

    return _moe_grouped_gemm_mxfp4_shared_decode_func


class MoEGroupedGEMMMXFP4Kernel(GroupedGEMMConfigMixin, Kernel, MGroupedGEMMMXFP4FwdInterface):
    """MXFP4 grouped GEMM over masked expert slabs or contiguous expert rows, on SM90 TMA and
    WGMMA.

    Each task first checks that its weight scales lie in ``[2 ** -63, 2 ** 63]`` and, under
    e8m0, that each step's spans at most five binades. A passing task folds each e8m0 step's
    four sub-blocks into one contraction (float32 steps go sub-block by sub-block); a failing
    one contracts sub-block by sub-block, its scale products split into mantissa and exponent.
    Rows of a slab past its valid count, inside a tile that holds a valid one, are stored like
    the others with unspecified values; rows past ``max_m`` are not written. A count is
    clamped to ``[0, max_m]``.
    Contiguous rows past an aligned expert's end are its padding and are stored like the
    others; a tight expert's are the next expert's and are not stored. Under aligned packing
    ``block_m`` divides the alignment.

    A config is ``{block_m, block_n, num_stages, num_sms}``: ``block_m`` rows of ``a`` (the
    WGMMA N extent) by ``block_n`` weight rows (a math warp-group per 64). The default rule,
    the autotuning candidates and a caller's config all pass ``config_refusal``.
    """

    supported_archs: list[int] = [90]

    _CONFIG_KEYS: ClassVar[tuple[str, ...]] = ("block_m", "block_n", "num_stages", "num_sms")
    # Row (token) tiles: the WGMMA N extent, a multiple of 8 rows of TMA box.
    _BLOCK_M: ClassVar[tuple[int, ...]] = (8, 16, 32, 64, 128)
    # Weight tiles: one 64-row math warp-group each 64 rows, whole 64-column store boxes; two
    # to four warp-groups, whose register rebalance holds their fragments.
    _BLOCK_N: ClassVar[tuple[int, ...]] = (128, 192, 256)
    # Ring depths. A depth that divides the K steps runs the constant-slot ring.
    _NUM_STAGES: ClassVar[range] = range(2, 9)
    # Kept free below the device's opt-in shared-memory limit for allocator padding
    # ``_smem_bytes`` does not model; no legal config's footprint lies in it.
    _SMEM_HEADROOM: ClassVar[int] = 2048
    # The default (block_m, block_n) of a contiguous call without row alignment; aligned
    # contiguous calls keep its block_n.
    _CONTIGUOUS_TILE: ClassVar[tuple[int, int]] = (64, 128)

    @classmethod
    def config_refusal(
        cls,
        config: Mapping[str, object],
        *,
        sm_count: int,
        smem_limit: int,
        scale_bytes: int,
        scale_k: int,
        alignment: int = 0,
    ) -> str | None:
        """Why the builder cannot instantiate *config*, or ``None`` when it can.

        Args:
            config: A candidate config.
            sm_count: The device's SM count, which bounds ``num_sms``: the grid is persistent.
            smem_limit: The shared memory one block may take on the device.
            scale_bytes: Bytes of one scale, 1 (e8m0) or 4 (float32).
            scale_k: The 128-wide K steps, which size the e8m0 check table.
            alignment: An aligned layout's alignment, which ``block_m`` divides; 0 for any
                other layout.

        Returns:
            The first violated constraint, or ``None``.
        """
        reason = cls._common_refusal(config, sm_count=sm_count, alignment=alignment)
        if reason is not None:
            return reason
        block_m, block_n, stages, _ = (config[key] for key in cls._CONFIG_KEYS)
        smem = _config_smem(block_m, block_n, stages, scale_bytes, scale_k, smem_limit)
        if smem > smem_limit:
            return f"takes {smem} bytes of shared memory, past the {smem_limit} a block may"
        return None

    @classmethod
    def refusal(cls, call: MGroupedGEMMMXFP4Call) -> str | None:
        """The layout, dtypes, extents and scale grids the kernel instantiates."""
        if call.scale_dtype not in (torch.float8_e8m0fnu, torch.float32):
            return f"reads float8_e8m0fnu or float32 scales, got {call.scale_dtype}"
        if call.k <= 0 or call.k % QUANT_SCALE_BLOCK:
            return f"needs K a positive multiple of {QUANT_SCALE_BLOCK}, got K = {call.k}"
        # The output's TMA store moves rows of whole 16-byte units.
        if call.n % 8:
            return f"needs N a multiple of 8, got N = {call.n}"
        if call.packing == "aligned" and call.alignment % min(cls._BLOCK_M):
            return f"tiles aligned runs in {cls._BLOCK_M}-row tiles, got alignment {call.alignment}"
        # Tight per-row ids keep every expert's row end in shared memory, beside at least the
        # default tile of an unaligned contiguous call on the shallowest ring.
        if call.packing == "tight" and call.metadata_kind == "per_row":
            scale_bytes = 1 if call.scale_dtype == torch.float8_e8m0fnu else 4
            least = _smem_bytes(
                *cls._CONTIGUOUS_TILE,
                min(cls._NUM_STAGES),
                scale_bytes,
                call.k // QUANT_SCALE_BLOCK,
                False,
            )
            room = call.smem_budget - cls._SMEM_HEADROOM - least
            if 4 * call.num_groups > room:
                return (
                    f"keeps each expert's row end in shared memory: {call.num_groups} experts "
                    f"take {4 * call.num_groups} bytes, past the {room} beside the default tile"
                )
        return super().refusal(call)

    def __init__(self, call: MGroupedGEMMMXFP4Call) -> None:
        device_index = call.device.index if call.device is not None else None
        super().__init__(device_index=device_index)
        self.call = call
        self.scale_bytes = 1 if call.scale_dtype == torch.float8_e8m0fnu else 4
        self.scale_k = call.k // QUANT_SCALE_BLOCK
        self.sm_count = self.build_target.sm_count
        self.smem_limit = self.build_target.smem_budget - self._SMEM_HEADROOM
        # Tight per-row metadata keeps every expert's recovered end in shared memory.
        if call.packing == "tight" and call.metadata_kind == "per_row":
            self.smem_limit -= 4 * call.num_groups
        self.alignment = call.alignment if call.packing == "aligned" else 0
        self.kernel = _moe_grouped_gemm_mxfp4_kernel(
            call.num_groups,
            call.n,
            call.k,
            call.max_m,
            self.scale_bytes == 1,
            self.sm_count,
            self.smem_limit,
            call.packing,
            call.metadata_kind,
            call.alignment,
        )
        self.init_config()

    def _refusal_inputs(self) -> dict:
        return {
            "sm_count": self.sm_count,
            "smem_limit": self.smem_limit,
            "scale_bytes": self.scale_bytes,
            "scale_k": self.scale_k,
            "alignment": self.alignment,
        }

    def _stages_for(self, block_m: int, block_n: int) -> list[int]:
        """Ring depths that fit beside the tiles, those dividing the K steps when any do."""
        fitting = [
            s
            for s in self._NUM_STAGES
            if _config_smem(block_m, block_n, s, self.scale_bytes, self.scale_k, self.smem_limit)
            <= self.smem_limit
        ]
        dividing = [s for s in fitting if self.scale_k % s == 0]
        return dividing or fitting

    @property
    def default_config(self) -> dict:
        # Every row tile converts its weights once, so aligned runs take the tallest tile
        # their alignment allows. A masked slab's tile is the narrowest holding twice an
        # expert's mean rows; float32 scales' sub-block path spills in four warp-groups.
        # Four warp-groups take a 4-deep ring.
        call = self.call
        if call.kind == "contiguous":
            unaligned_m, block_n = self._CONTIGUOUS_TILE
            block_m = (
                max(m for m in self._BLOCK_M if self.alignment % m == 0)
                if self.alignment
                else unaligned_m
            )
        else:
            rows = 2 * call.max_m / call.num_groups
            block_m = next((m for m in self._BLOCK_M if m >= rows), max(self._BLOCK_M))
            groups, sms = call.num_groups, self.sm_count
            occupied = groups * (1 - (1 - 1 / groups) ** call.max_m)
            widths = (192, 256) if self.scale_bytes == 1 else (192,)
            block_n = min(
                widths, key=lambda w: math.ceil(occupied * -(-call.n // w) / sms - 1e-9) * w
            )
        stages = self._stages_for(block_m, block_n)
        shallow = [s for s in stages if s <= 4]
        return {
            "block_m": block_m,
            "block_n": block_n,
            "num_stages": max(shallow) if block_n == 256 and shallow else max(stages),
            "num_sms": self._balanced_sms(block_m, block_n),
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # A tile wider than N computes only columns it does not store.
        widths = [n for n in self._BLOCK_N if n <= max(self.call.n, min(self._BLOCK_N))]
        heights = [m for m in self._BLOCK_M if self.alignment % m == 0]
        configs = [
            {"block_m": m, "block_n": n, "num_stages": s, "num_sms": sms}
            for m in heights
            for n in widths
            for s in self._stages_for(m, n)
            for sms in sorted({self._balanced_sms(m, n), self.sm_count})
        ]
        return [
            config
            for config in configs
            if self.config_refusal(config, **self._refusal_inputs()) is None
        ]

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

            def scales(*shape):
                # The program reads e8m0 codes as bytes, as forward hands them.
                if call.scale_dtype == torch.float8_e8m0fnu:
                    return torch.randint(123, 129, shape, dtype=torch.uint8, device=device)
                return torch.rand(*shape, device=device) + 0.5

            return [
                torch.randn(*rows, call.k, device=device).to(torch.float8_e4m3fn),
                scales(*rows, scale_k),
                torch.randint(
                    0, 256, (groups, call.n, call.k // 2), dtype=torch.uint8, device=device
                ),
                scales(groups, call.n, call.k // _B_SCALE_BLOCK),
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
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run each expert's block-scaled product over its rows, into *out* if given."""
        if out is None:
            out = torch.empty((*a.shape[:-1], self.call.n), dtype=torch.bfloat16, device=a.device)
        if out.numel() == 0:
            return out
        if a_scale.dtype == torch.float8_e8m0fnu:
            # The program reads e8m0 codes as bytes.
            a_scale, b_scale = a_scale.view(torch.uint8), b_scale.view(torch.uint8)
        self.program(self._compile)(a, a_scale, b, b_scale, layout_metadata, out)
        return out

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self.kernel(**cfg)


class MoEGroupedGEMMMXFP4SharedDecodeKernel(MoEGroupedGEMMMXFP4Kernel):
    """MXFP4 grouped GEMM over aligned contiguous rows under e8m0 scales that decodes each
    weight tile once into shared memory (``_moe_grouped_gemm_mxfp4_shared_decode_kernel``).

    128 rows of ``a`` are the WGMMA M operand, 128 decoded weight rows its N operand. A task
    folds when each of its weight rows' codes lie in [64, 190] and span at most eleven binades
    over K, and each of its token codes, its padding rows' among them, is 0 or lies in
    [64, 190]: its sub-blocks then carry
    their scales relative to their row's largest code less six, each step is added under its
    token scales alone, and each column is scaled by its row's reference after the last step.
    Another task contracts sub-block by sub-block, its products split into mantissa and
    exponent. Rows past an aligned expert's end are its padding and are stored like the others.

    A config is ``{block_m, block_n, num_stages, num_sms}`` with ``block_m`` and ``block_n``
    128; ``num_stages`` is the depth of the ring of ``a``'s rows.
    """

    preferred_over: ClassVar[frozenset[str]] = frozenset({"grouped_gemm_mxfp4"})
    _BLOCK_M: ClassVar[tuple[int, ...]] = (_SHARED_DECODE_ROWS,)
    _BLOCK_N: ClassVar[tuple[int, ...]] = (_SHARED_DECODE_COLUMNS,)

    @classmethod
    def config_refusal(
        cls,
        config: Mapping[str, object],
        *,
        sm_count: int,
        smem_limit: int,
        scale_k: int,
        alignment: int,
    ) -> str | None:
        """Why the builder cannot instantiate *config*, or ``None`` when it can.

        Args:
            config: A candidate config.
            sm_count: The device's SM count, which bounds ``num_sms``: the grid is persistent.
            smem_limit: The shared memory one block may take on the device.
            scale_k: The 128-wide K steps, which size the code tables.
            alignment: The layout's alignment, which ``block_m`` divides.

        Returns:
            The first violated constraint, or ``None``.
        """
        reason = cls._common_refusal(config, sm_count=sm_count, alignment=alignment)
        if reason is not None:
            return reason
        smem = _shared_decode_smem_bytes(config["block_n"], config["num_stages"], scale_k)
        if smem > smem_limit:
            return f"takes {smem} bytes of shared memory, past the {smem_limit} a block may"
        return None

    @classmethod
    def refusal(cls, call: MGroupedGEMMMXFP4Call) -> str | None:
        """Aligned contiguous runs of whole 128-row tiles under e8m0 scales, K whose token code
        rows (K / 128 bytes) are whole 16-byte TMA units, and the shallowest ring fitting."""
        if call.packing != "aligned" or call.alignment % _SHARED_DECODE_ROWS:
            return f"decodes aligned runs of whole {_SHARED_DECODE_ROWS}-row tiles"
        if call.scale_dtype != torch.float8_e8m0fnu:
            return f"folds e8m0 scales only, got {call.scale_dtype}"
        scale_k = call.k // QUANT_SCALE_BLOCK
        if call.k % QUANT_SCALE_BLOCK or not _table_box(scale_k):
            return f"loads code tables by TMA, which needs K a multiple of 2048, got K = {call.k}"
        least = _shared_decode_smem_bytes(_SHARED_DECODE_COLUMNS, min(cls._NUM_STAGES), scale_k)
        room = call.smem_budget - cls._SMEM_HEADROOM
        if least > room:
            return f"takes {least} bytes of shared memory on the shallowest ring, past {room}"
        return super().refusal(call)

    def __init__(self, call: MGroupedGEMMMXFP4Call) -> None:
        Kernel.__init__(self, device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.scale_k = call.k // QUANT_SCALE_BLOCK
        self.sm_count = self.build_target.sm_count
        self.smem_limit = self.build_target.smem_budget - self._SMEM_HEADROOM
        self.alignment = call.alignment
        self.kernel = _moe_grouped_gemm_mxfp4_shared_decode_kernel(
            call.num_groups,
            call.n,
            call.k,
            self.sm_count,
            self.smem_limit,
            call.metadata_kind,
            call.alignment,
        )
        self.init_config()

    def _refusal_inputs(self) -> dict:
        return {
            "sm_count": self.sm_count,
            "smem_limit": self.smem_limit,
            "scale_k": self.scale_k,
            "alignment": self.alignment,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # Every ring depth that fits beside the tiles; the ring keeps its position across tasks.
        return [
            {
                "block_m": _SHARED_DECODE_ROWS,
                "block_n": _SHARED_DECODE_COLUMNS,
                "num_stages": stages,
                "num_sms": self.sm_count,
            }
            for stages in self._NUM_STAGES
            if _shared_decode_smem_bytes(_SHARED_DECODE_COLUMNS, stages, self.scale_k)
            <= self.smem_limit
        ]

    @property
    def default_config(self) -> dict:
        # The deepest ring that fits: it sets how far the loads of a's rows run ahead.
        return max(self.autotune_configs, key=lambda config: config["num_stages"])
