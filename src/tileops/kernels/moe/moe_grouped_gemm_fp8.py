"""Block-scaled FP8 M-grouped GEMM over masked expert slabs, a Tensor Memory Accelerator (TMA)
producer feeding warp-group MMA (WGMMA) consumers.

``a`` carries one float32 scale per row and 128 K columns, ``b`` one per 128x128 block. A
persistent grid walks the ``block_m x block_n`` output tiles of every expert's slab: a producer
warp fills a shared-memory ring of A and B tiles through TMA across tiles, each math warp-group
contracts its 64 rows of every 128-wide K step into a partial and folds it into the float32
result under that step's two scales, and the tile leaves through shared memory and a TMA
store. A tile wider than 128 columns spans two ``b`` scale rows, and the last N tile may run
past N: its loads read zeros there and its stores are clipped.
"""

import functools
import itertools
from typing import Callable, ClassVar, Mapping, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

from tileops._csrc import csrc_include
from tileops.kernels.constants import (
    TMA_DTYPE_BFLOAT16,
    TMA_DTYPE_UINT8,
    TMA_INTERLEAVE_NONE,
    TMA_L2_PROMOTION_128B,
    TMA_L2_PROMOTION_256B,
    TMA_OOB_FILL_NONE,
    TMA_SWIZZLE_64B,
    TMA_SWIZZLE_128B,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import MGroupedGemmFP8Call, MGroupedGemmFP8FwdInterface
from tileops.manifest.primitives import moe_layout_metadata
from tileops.utils import device_calibration, get_shared_memory_optin, get_sm_count

__all__ = ["MoEGroupedGemmFP8Kernel"]

# The K extent one scale covers and the N extent one ``b`` scale row covers, fixed by the
# operands' scale grids; also the K step, so each step reads one scale of each operand.
_SCALE_BLOCK = 128
# The rows one math warp-group's WGMMA covers.
_WARPGROUP_M = 64
# An expert's tiles run in groups of this many M tiles, N tile slowest inside a group, so the
# blocks of one wave that share a weight tile run at once. A schedule constant, not tuned.
_SWIZZLE_M_TILES = 16
# Slots of a role's task-sequence cursor.
_CUR_G, _CUR_M, _CUR_TILES = 0, 1, 2
# The keys of a config, each a parameter of the builder.
_CONFIG_KEYS = ("block_m", "block_n", "num_stages", "num_sms")
# Kept free below the device's opt-in shared-memory limit, as the on-chip RMSNorm kernel does,
# for allocator padding ``_smem_bytes`` does not model; no legal config's footprint lies in it.
_SMEM_HEADROOM = 1024


def _scale_b_rows(block_n: int) -> int:
    """The ``b`` scale rows one tile reads: two when 128 does not divide its width."""
    return 1 if _SCALE_BLOCK % block_n == 0 else 2


def _smem_bytes(block_m: int, block_n: int, num_stages: int) -> int:
    """Shared memory one block takes: the A/B ring (one byte per element), the bfloat16 tile
    the TMA store reads, and per stage its B scales and a ``full`` and an ``empty`` barrier."""
    ring = num_stages * (block_m + block_n) * _SCALE_BLOCK
    return ring + block_m * block_n * 2 + num_stages * (4 * _scale_b_rows(block_n) + 16)


@functools.lru_cache(maxsize=32)
def _moe_grouped_gemm_fp8_kernel(
    num_groups: int, n: int, k: int, max_m: int, sm_count: int, smem_limit: int
) -> Callable:
    """A persistent grid over the ``block_m x block_n`` output tiles of every expert's slab.

    Block ``pid`` runs tasks ``pid, pid + num_sms, ...`` of a sequence holding only the tiles
    that contain a valid row: expert ``g`` contributes ``cdiv(clamp(count, 0, max_m), block_m)``
    M tiles times ``n_tiles``, the experts in order. Inside an expert, tiles run in groups of
    ``_SWIZZLE_M_TILES`` M tiles with the M tile fastest, so the blocks that read one weight tile
    run side by side. Each role walks the sequence itself, reading the counts and carrying its
    expert cursor across its tasks; a task past the sequence's end is skipped by both roles and
    touches no barrier. A ``block_n`` dividing 128 reads one ``b_scale`` row; a wider one two,
    the second clamped to the last row, and its first ``(128 - n0 % 128) / 8`` eight-column
    fragments take the first. N need not be a multiple of ``block_n``.

    ``sm_count`` and ``smem_limit`` are the device's, which bound ``num_sms`` and the block's
    shared memory; every config is checked by ``MoEGroupedGemmFP8Kernel.config_refusal``
    before anything is traced.
    """
    accum_dtype = "float"
    ab_dtype = "float8_e4m3fn"
    cd_dtype = "bfloat16"
    block_k = _SCALE_BLOCK
    scale_k = k // block_k

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
        block_m: int, block_n: int, num_stages: int, num_sms: int
    ) -> Callable:
        config = dict(zip(_CONFIG_KEYS, (block_m, block_n, num_stages, num_sms), strict=True))
        reason = MoEGroupedGemmFP8Kernel.config_refusal(
            config, sm_count=sm_count, smem_limit=smem_limit
        )
        if reason is not None:
            raise ValueError(f"cannot build the FP8 masked grouped GEMM for {config}: {reason}")
        tiles_per_slab = tilelang.cdiv(max_m, block_m)
        n_tiles = tilelang.cdiv(n, block_n)
        # The last N tile runs past N: its loads read zeros and its stores are clipped.
        ragged_n = n % block_n != 0
        scale_b_rows = _scale_b_rows(block_n)
        last_scale_b_row = n // _SCALE_BLOCK - 1
        # The sequence is never longer than every tile of every slab.
        max_tasks = num_groups * tiles_per_slab * n_tiles
        max_waves = tilelang.cdiv(max_tasks, num_sms)
        swizzle_span = _SWIZZLE_M_TILES * n_tiles
        # The last tile of a slab whose rows end inside it reads no scale past max_m.
        ragged_slab = max_m % block_m != 0
        math_threads = block_m // _WARPGROUP_M * 128
        math_warps = math_threads // 32
        fragment_regs = _WARPGROUP_M * block_n // 128
        # A whole number of rings per task starts every task on slot 0, so unrolled by its
        # length every slot is a constant and the phase follows from the tasks run so far.
        # Otherwise each role carries its ring position across tasks.
        static_ring = scale_k % num_stages == 0
        rings_per_task = scale_k // num_stages
        k_unroll = num_stages if static_ring else 1
        wgmma_helper = f"tileops::moe_fp8_wgmma_64x128_by_128x{block_n}_lo"
        if scale_b_rows == 1:
            promote_helper = f"tileops::moe_fp8_promote_64x{block_n}"
        else:
            promote_helper = f"tileops::moe_fp8_promote_two_b_scales_64x{block_n}"
        stsm_helper = f"tileops::moe_fp8_stsm_bf16_swizzled_bm{block_m}_64x{block_n}"
        # The staged tile is boxes of 64 bfloat16 columns under a 128B swizzle, or 32 under
        # a 64B one when the row is no multiple of 128 bytes (160), one TMA store each.
        epilogue_swizzle_bytes = 128 if block_n * 2 % 128 == 0 else 64
        epilogue_swizzle = {64: TMA_SWIZZLE_64B, 128: TMA_SWIZZLE_128B}[epilogue_swizzle_bytes]
        epilogue_block_n = epilogue_swizzle_bytes // 2
        epilogue_store_count = block_n // epilogue_block_n

        def ring_position(kk, ring_index):
            """The ring slot and phase of K step ``kk``, the one definition both roles read, so
            the producer fills each slot in the order the consumers drain it."""
            if static_ring:
                return kk % num_stages, (ring_index * rings_per_task + kk // num_stages) & 1
            return ring_index % num_stages, (ring_index // num_stages) & 1

        @T.macro
        def walk_to(layout, task_id, cursor):
            """Advance ``cursor`` to the expert holding ``task_id``, or to ``num_groups`` past the
            sequence's end. A macro's arguments bind immutably, so the cursor is a register
            array: the expert, the M tiles of the experts before it, and the expert's own."""
            while cursor[_CUR_G] < num_groups:
                cursor[_CUR_TILES] = (
                    T.max(T.min(layout[cursor[_CUR_G]], max_m), 0) + block_m - 1
                ) // block_m
                if task_id < (cursor[_CUR_M] + cursor[_CUR_TILES]) * n_tiles:
                    break
                cursor[_CUR_M] = cursor[_CUR_M] + cursor[_CUR_TILES]
                cursor[_CUR_G] = cursor[_CUR_G] + 1

        @T.macro
        def tile_of(task_id, cursor, tile):
            """The M and N tile of task ``task_id``, inside the cursor's expert.

            Called only where ``walk_to`` stopped inside the sequence, so ``0 <= local <
            m_tiles * n_tiles`` and ``1 <= m_tiles <= tiles_per_slab``: every operand is
            non-negative, truncating division is exact, and below exactly one branch of the
            constant-divisor form matches.
            """
            local = task_id - cursor[_CUR_M] * n_tiles
            m_tiles = cursor[_CUR_TILES]
            if tiles_per_slab <= _SWIZZLE_M_TILES:
                # One group spans every M tile of the expert. A runtime divisor costs a long
                # division, so each of the tiles_per_slab values m_tiles can take gets its own.
                for i in T.unroll(tiles_per_slab):
                    if m_tiles == i + 1:
                        tile[0] = T.truncmod(local, i + 1)
                        tile[1] = T.truncdiv(local, i + 1)
            else:
                first = T.truncdiv(local, swizzle_span) * _SWIZZLE_M_TILES
                in_group = T.truncmod(local, swizzle_span)
                group_m_tiles = T.min(_SWIZZLE_M_TILES, m_tiles - first)
                tile[0] = first + T.truncmod(in_group, group_m_tiles)
                tile[1] = T.truncdiv(in_group, group_m_tiles)

        @T.prim_func
        def _moe_grouped_gemm_fp8_main(
            A: T.Tensor((num_groups, max_m, k), ab_dtype),  # type: ignore
            A_scale: T.Tensor((num_groups, max_m, scale_k), "float32"),  # type: ignore
            B: T.Tensor((num_groups, n, k), ab_dtype),  # type: ignore
            B_scale: T.Tensor((num_groups, last_scale_b_row + 1, scale_k), "float32"),  # type: ignore
            layout: T.Tensor((num_groups,), "int32"),  # type: ignore
            C: T.Tensor((num_groups, max_m, n), cd_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(num_sms, threads=math_threads + 128) as (pid,):
                a_shared = T.alloc_shared((num_stages, block_m, block_k), ab_dtype)
                b_shared = T.alloc_shared((num_stages, block_n, block_k), ab_dtype)
                # scope="shared": a shared.dyn store here makes TileLang fence it against the TMA
                # writes with a barrier one warp cannot complete.
                b_scale_shared = T.alloc_shared(
                    (num_stages, scale_b_rows), accum_dtype, scope="shared"
                )
                shared_c = T.alloc_shared((block_m * block_n,), cd_dtype)
                partial = T.alloc_local((fragment_regs,), accum_dtype)
                final = T.alloc_local((fragment_regs,), accum_dtype)
                # This step's A scales (two rows), the next step's, and this step's B scales.
                scales = T.alloc_local((4 + scale_b_rows,), accum_dtype)
                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )
                full = T.alloc_barrier([1] * num_stages)
                empty = T.alloc_barrier([math_warps] * num_stages)
                # Each role's count of the tasks (static ring) or K steps (index ring) it has run.
                # Not the wave index: a skipped task advances no barrier.
                ring_index = T.alloc_var("int32", init=0)
                # Each role's place in the task sequence (see walk_to), and its task's M and N
                # tile.
                cursor = T.alloc_local((3,), "int32")
                tile = T.alloc_local((2,), "int32")
                for i in T.unroll(3):
                    cursor[i] = 0
                tx = T.get_thread_binding()
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

                # The operand loads ask L2 for 256-byte promotion; ``T.tma_copy`` asks for 128 and
                # its descriptors cannot say otherwise.
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
                        walk_to(layout, task_id, cursor)
                        # Both roles walk one sequence from one read-only ``layout``, so they skip
                        # the same tasks, and a skipped task touches no barrier.
                        if T.And(cursor[_CUR_G] < num_groups, producer_warp == 0):
                            tile_of(task_id, cursor, tile)
                            # The clamps change no value; inlined rather than let-bound, they
                            # bound the indices for TileLang, which otherwise guards every scale
                            # load against them.
                            g = T.meta_var(T.max(T.min(cursor[_CUR_G], num_groups - 1), 0))
                            m_start = T.meta_var(
                                T.max(T.min(tile[0], tiles_per_slab - 1), 0) * block_m
                            )
                            n_start = T.meta_var(T.max(T.min(tile[1], n_tiles - 1), 0) * block_n)
                            for kk in T.unroll(scale_k, unroll_factor=k_unroll):
                                slot, phase = ring_position(kk, ring_index)
                                T.barrier_wait(empty[slot], phase ^ 1)
                                if T.shuffle_elect(32):
                                    T.mbarrier_expect_tx(full[slot], block_m * block_k)
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
                                # After the TMAs are in flight, lane 0 stores the stage's B scales
                                # and then arrives on ``full``, so its release covers them.
                                if producer_tx == 0:
                                    for row in T.unroll(scale_b_rows):
                                        b_scale_shared[slot, row] = B_scale[
                                            g,
                                            T.min(n_start // _SCALE_BLOCK + row, last_scale_b_row),
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
                    math_wg_offset = math_wg_idx * _WARPGROUP_M
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
                        walk_to(layout, task_id, cursor)
                        if cursor[_CUR_G] < num_groups:
                            tile_of(task_id, cursor, tile)
                            g = T.meta_var(T.max(T.min(cursor[_CUR_G], num_groups - 1), 0))
                            m_start = T.meta_var(
                                T.max(T.min(tile[0], tiles_per_slab - 1), 0) * block_m
                            )
                            n_start = T.meta_var(T.max(T.min(tile[1], n_tiles - 1), 0) * block_n)
                            # Inlined, as m_start is, so the scale loads stay unguarded.
                            if ragged_slab:
                                scale_row0 = T.meta_var(T.min(m_start + acc_row0, max_m - 1))
                                scale_row1 = T.meta_var(T.min(m_start + acc_row0 + 8, max_m - 1))
                            else:
                                scale_row0 = T.meta_var(m_start + acc_row0)
                                scale_row1 = T.meta_var(scale_row0 + 8)
                            if scale_b_rows == 2:
                                # The fragments the first B scale row covers.
                                first_scale_iters = (_SCALE_BLOCK - n_start % _SCALE_BLOCK) // 8
                            T.clear(final)
                            scales[0] = A_scale[g, scale_row0, 0]
                            scales[1] = A_scale[g, scale_row1, 0]
                            for kk in T.unroll(scale_k, unroll_factor=k_unroll):
                                slot, phase = ring_position(kk, ring_index)
                                # The next step's, a whole step ahead of its use.
                                next_kk = T.min(kk + 1, scale_k - 1)
                                scales[2] = A_scale[g, scale_row0, next_kk]
                                scales[3] = A_scale[g, scale_row1, next_kk]
                                T.barrier_wait(full[slot], phase)
                                for row in T.unroll(scale_b_rows):
                                    scales[4 + row] = b_scale_shared[slot, row]
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
                                if scale_b_rows == 1:
                                    T.call_extern(
                                        "handle",
                                        promote_helper,
                                        partial.data,
                                        final.data,
                                        scales[0],
                                        scales[1],
                                        scales[4],
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        promote_helper,
                                        partial.data,
                                        final.data,
                                        scales[0],
                                        scales[1],
                                        scales[4],
                                        scales[5],
                                        first_scale_iters,
                                    )
                                scales[0] = scales[2]
                                scales[1] = scales[3]
                                if not static_ring:
                                    ring_index = ring_index + 1
                            if static_ring:
                                ring_index = ring_index + 1
                            # The previous task's TMA store has read shared_c.
                            if max_waves > 1:
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
                                # A box of the last N tile wholly past N is not stored.
                                store_box = T.And(
                                    tx < epilogue_store_count,
                                    n_start + tx * epilogue_block_n < n,
                                )
                            else:
                                store_box = tx < epilogue_store_count
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
                    if tx < epilogue_store_count:
                        T.tma_store_wait(0)

        return _moe_grouped_gemm_fp8_main

    return _moe_grouped_gemm_fp8_func


class MoEGroupedGemmFP8Kernel(Kernel, MGroupedGemmFP8FwdInterface):
    """Block-scaled FP8 grouped GEMM over masked expert slabs, on SM90 TMA and WGMMA.

    Rows of a slab past its valid count, inside a tile that holds a valid one, are computed and
    stored like the others; rows past ``max_m`` are not written. A count is clamped to
    ``[0, max_m]``.

    A config is ``{block_m, block_n, num_stages, num_sms}``. The calibrated table, the default
    rule, the autotuning candidates and a caller's config all pass ``config_refusal``.
    """

    supported_archs: list[int] = [90]

    # Tile heights: one or two 64-row math warp-groups.
    _BLOCK_M: ClassVar[tuple[int, ...]] = (64, 128)
    # Tile widths the helper header instantiates; 160 and 192, which 128 does not divide, span
    # two ``b`` scale rows.
    _BLOCK_N: ClassVar[tuple[int, ...]] = (64, 128, 160, 192)
    # Ring depths. A depth that divides the K steps runs the constant-slot ring.
    _NUM_STAGES: ClassVar[range] = range(3, 9)
    # Calibrated configs by board (``tileops.utils.device_calibration``), keyed by
    # (E, N, K, max_m): the fastest point of the block_m x block_n x num_stages x num_sms sweep.
    # A shape the default rule already serves best is not listed.
    _CONFIGS: ClassVar[dict[str, dict[tuple[int, int, int, int], dict[str, int]]]] = {
        "h200": {
            (8, 4096, 7168, 128): {"block_m": 128, "block_n": 128, "num_stages": 4, "num_sms": 128},
            (8, 7168, 2048, 128): {"block_m": 128, "block_n": 160, "num_stages": 4, "num_sms": 120},
            (32, 4096, 7168, 64): {"block_m": 64, "block_n": 192, "num_stages": 4, "num_sms": 118},
            (32, 7168, 2048, 64): {"block_m": 64, "block_n": 160, "num_stages": 4, "num_sms": 131},
            (32, 7168, 2048, 256): {"block_m": 64, "block_n": 160, "num_stages": 4, "num_sms": 132},
            (16, 3072, 4096, 64): {"block_m": 64, "block_n": 192, "num_stages": 4, "num_sms": 128},
            (16, 3072, 4096, 256): {"block_m": 64, "block_n": 192, "num_stages": 4, "num_sms": 128},
            (48, 4096, 7168, 64): {"block_m": 64, "block_n": 192, "num_stages": 4, "num_sms": 132},
            (48, 7168, 2048, 64): {"block_m": 64, "block_n": 160, "num_stages": 4, "num_sms": 128},
            (48, 4096, 7168, 256): {"block_m": 64, "block_n": 192, "num_stages": 4, "num_sms": 132},
            (48, 7168, 2048, 256): {"block_m": 64, "block_n": 192, "num_stages": 4, "num_sms": 132},
        },
    }

    @classmethod
    def config_refusal(
        cls, config: Mapping[str, object], *, sm_count: int, smem_limit: int
    ) -> Optional[str]:
        """Why the builder cannot instantiate *config*, or ``None`` when it can.

        Args:
            config: A candidate config.
            sm_count: The device's SM count, which bounds ``num_sms``: the grid is persistent.
            smem_limit: The shared memory one block may take on the device.

        Returns:
            The first violated constraint, or ``None``.
        """
        if set(config) != set(_CONFIG_KEYS):
            return f"needs exactly the keys {list(_CONFIG_KEYS)}, got {sorted(config)}"
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in config.values()):
            return "takes integer values"
        block_m, block_n, stages, num_sms = (config[key] for key in _CONFIG_KEYS)
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
        smem = _smem_bytes(block_m, block_n, stages)
        if smem > smem_limit:
            return f"takes {smem} bytes of shared memory, past the {smem_limit} a block may"
        return None

    @classmethod
    def applies(cls, call: MGroupedGemmFP8Call) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: MGroupedGemmFP8Call) -> Optional[str]:
        """The layout, dtype, extents and scale grids the kernel instantiates."""
        if call.kind != "masked":
            return f"serves only the masked layout, got {call.kind}"
        if call.ab_dtype is not torch.float8_e4m3fn:
            return f"serves float8_e4m3fn operands, got {call.ab_dtype}"
        if call.k % _SCALE_BLOCK or call.n % _SCALE_BLOCK:
            return f"needs K and N multiples of {_SCALE_BLOCK}, got K = {call.k}, N = {call.n}"
        scale_k = call.k // _SCALE_BLOCK
        a_scale = (call.num_groups, call.max_m, scale_k)
        b_scale = (call.num_groups, call.n // _SCALE_BLOCK, scale_k)
        if tuple(call.a_scale_shape) != a_scale or tuple(call.b_scale_shape) != b_scale:
            return (
                f"reads a_scale {a_scale} and b_scale {b_scale}, got "
                f"{tuple(call.a_scale_shape)} and {tuple(call.b_scale_shape)}"
            )
        return None

    def __init__(self, call: MGroupedGemmFP8Call) -> None:
        device_index = call.device.index if call.device is not None else None
        super().__init__(device_index=device_index)
        self.call = call
        self.scale_k = call.k // _SCALE_BLOCK
        self.sm_count = get_sm_count(self.device_index)
        self.smem_limit = get_shared_memory_optin(self.device_index) - _SMEM_HEADROOM
        calibration = device_calibration(self.device_index)
        self._calibrated = self._CONFIGS.get(calibration, {}).get(
            (call.num_groups, call.n, call.k, call.max_m)
        )
        self.kernel = _moe_grouped_gemm_fp8_kernel(
            call.num_groups, call.n, call.k, call.max_m, self.sm_count, self.smem_limit
        )
        self.init_config()

    def _check_config(self, config: Mapping[str, object]) -> None:
        """Raise ``ValueError`` on a config ``config_refusal`` refuses on this device."""
        reason = self.config_refusal(config, sm_count=self.sm_count, smem_limit=self.smem_limit)
        if reason is not None:
            raise ValueError(f"{type(self).__name__} cannot run config {dict(config)}: {reason}")

    def init_config(self, config: Optional[dict] = None, tune: bool = False) -> None:
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
        super().init_config(config, tune)
        self._check_config(self.config)

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        super().autotune(warmup=warmup, rep=rep)
        self._check_config(self.config)

    def _stages_for(self, block_m: int, block_n: int) -> list[int]:
        """Ring depths that fit beside the tiles, those dividing the K steps when any do."""
        fitting = [
            s for s in self._NUM_STAGES if _smem_bytes(block_m, block_n, s) <= self.smem_limit
        ]
        dividing = [s for s in fitting if self.scale_k % s == 0]
        return dividing or fitting

    def _balanced_sms(self, block_m: int, block_n: int) -> int:
        """The fewest blocks that run the tiles in as many waves as all SMs do, so every block
        runs the same number of tiles; one for a call with no tile."""
        call = self.call
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
        # tiles waste less on a sparse slab.
        block_m = 128 if 64 < self.call.max_m <= 128 else 64
        block_n = 128
        return {
            "block_m": block_m,
            "block_n": block_n,
            "num_stages": max(self._stages_for(block_m, block_n)),
            "num_sms": self._balanced_sms(block_m, block_n),
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # A tile wider than N computes only columns it does not store.
        widths = [n for n in self._BLOCK_N if n <= self.call.n]
        configs = [
            {"block_m": m, "block_n": n, "num_stages": s, "num_sms": sms}
            for m, n in itertools.product(self._BLOCK_M, widths)
            for s in self._stages_for(m, n)
            for sms in sorted({self._balanced_sms(m, n), self.sm_count})
        ]
        for config in configs:
            self._check_config(config)
        return configs

    @property
    def autotune_supply_prog(self) -> Callable:
        """Supply autotuning operands of the call's shapes and the manifest generator's masked
        counts, ``max_m`` and ``max_m // 2`` alternating.

        The call carries no counts, since reading them would synchronise, so the candidates
        are timed on that fill whatever the counts of the calls that follow.
        """
        from tilelang.utils.device import get_current_device

        call, scale_k = self.call, self.scale_k

        def supply_prog(params: list) -> list:
            if len(params) != 6:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects 6 parameters "
                    f"(a, a_scale, b, b_scale, layout, c), got {len(params)}"
                )
            device = get_current_device()
            groups, rows = call.num_groups, call.max_m
            metadata = moe_layout_metadata(call, groups * rows, groups)
            fp8 = torch.float8_e4m3fn
            return [
                torch.randn(groups, rows, call.k, device=device).to(fp8),
                torch.rand(groups, rows, scale_k, device=device) + 0.5,
                torch.randn(groups, call.n, call.k, device=device).to(fp8),
                torch.rand(groups, call.n // _SCALE_BLOCK, scale_k, device=device) + 0.5,
                torch.tensor(metadata, dtype=torch.int32, device=device),
                torch.empty(groups, rows, call.n, dtype=torch.bfloat16, device=device),
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
        """Run each expert's block-scaled product over its valid rows, into *out* if given."""
        call = self.call
        if out is None:
            out = torch.empty(
                (call.num_groups, call.max_m, call.n), dtype=torch.bfloat16, device=a.device
            )
        if out.numel() == 0:
            return out
        self.kernel(**self.config)(a, a_scale, b, b_scale, layout_metadata, out)
        return out
