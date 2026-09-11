"""Persistent grouped GEMM template for grouped, batched, and MoE layouts.

Inputs and output shapes are documented by ``GemmTemplate``.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.elementwise._erf import erf
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_count

from .heuristics import (
    ACTIVATIONS,
    PER_GROUP_TYPES,
    PER_ROW_TYPES,
    GemmDesc,
    GemmType,
    GroupedGemmSpec,
    Major,
    get_best_config,
    spec_from_config,
)

__all__ = [
    "GemmDesc",
    "GemmType",
    "Major",
    "GemmTemplate",
    "GroupedGemmTemplate",
    "GroupedGemmSpec",
]


def _num_1d_blocks_per_group(block_m: int, block_n: int, num_sms: int) -> int:
    """Return the number of M tiles in one L2 swizzle group."""
    best, best_usage = 0, None
    for candidate in (8, 16):
        usage = candidate * block_m + -(-num_sms // candidate) * block_n
        if best_usage is None or usage < best_usage:
            best, best_usage = candidate, usage
    return best


def _make_prim_func(
    gemm_type: str,
    a_k_major: bool,
    b_k_major: bool,
    ab_dtype: str,
    cd_dtype: str,
    num_groups: int,
    shape_m: int,
    shape_n: int,
    shape_k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    num_math_wgs: int,
    num_sms: int,
    activation: str,
    epilogue_stage_n: int,
    swizzle_group_m: int,
):
    """Build the ``@T.prim_func`` for one spec; parameters are the spec's scalars."""
    dtype = ab_dtype
    accum_dtype = "float"
    gtype = GemmType(gemm_type)
    tma_threads = 128
    math_threads = 128 * num_math_wgs
    threads = tma_threads + math_threads
    wg_rows = block_m // num_math_wgs
    math_warps = math_threads // 32
    blocks_per_group = (
        swizzle_group_m
        if gtype is GemmType.DENSE and swizzle_group_m
        else _num_1d_blocks_per_group(block_m, block_n, num_sms)
    )
    dense_coop2 = gtype is GemmType.DENSE and num_math_wgs == 2
    tma_regs = 24 if dense_coop2 else 48
    math_regs = 240 if dense_coop2 else (248 if num_math_wgs == 1 else 224)
    swizzle_atom = 64
    epilogue_barrier_base = 8
    cast_output = cd_dtype != "float32"
    fused = activation != "none"
    c_tile_n = block_n // 2 if fused else block_n  # output columns per tile
    staged_n = epilogue_stage_n or c_tile_n
    epilogue_chunks = c_tile_n // staged_n

    # Scheduler family and what each operand looks like.
    masked = gtype is GemmType.M_GROUPED_MASKED
    aligned_psum = gtype is GemmType.M_GROUPED_ALIGNED_PSUM
    tight_per_row = gtype is GemmType.M_GROUPED_TIGHT_PER_ROW
    tight = gtype in (GemmType.M_GROUPED_TIGHT_PSUM, GemmType.M_GROUPED_TIGHT_PER_ROW)
    dense = gtype is GemmType.DENSE
    batched = gtype is GemmType.BATCHED
    k_grouped = gtype is GemmType.K_GROUPED_CONTIGUOUS
    per_group = gtype in PER_GROUP_TYPES
    a_has_group = masked or batched
    b_has_group = not (dense or k_grouped)
    c_has_group = a_has_group or k_grouped
    # A K-major operand's TMA box must start 16 bytes aligned along K, so a
    # K-grouped GEMM with one rounds each group's start down to 8 elements and
    # masks the head in shared memory like the tail.
    k_head_align = 8 if k_grouped and (a_k_major or b_k_major) else 0
    search_steps = max(1, (num_groups - 1).bit_length())

    m = T.dynamic("m") if shape_m == 0 else shape_m
    n = T.dynamic("n") if shape_n == 0 else shape_n
    k = T.dynamic("k") if shape_k == 0 else shape_k

    a_2d = (m, k) if a_k_major else (k, m)
    a_shape = (num_groups,) + a_2d if a_has_group else a_2d
    b_2d = (n, k) if b_k_major else (k, n)
    b_shape = (num_groups,) + b_2d if b_has_group else b_2d
    c_cols = n // 2 if fused else n  # act(gate) * up halves the width
    c_shape = (num_groups, m, c_cols) if c_has_group else (m, c_cols)
    if gtype in PER_ROW_TYPES:
        layout_shape = (m,)
    elif per_group or k_grouped:
        layout_shape = (num_groups,)
    else:
        layout_shape = (1,)
    a_tile = (wg_rows, block_k) if a_k_major else (block_k, wg_rows)
    b_tile = (block_n, block_k) if b_k_major else (block_k, block_n)
    half_n = block_n // 2

    def a_region(A, group, row0, k0):
        if a_has_group:
            if a_k_major:
                return A[group, row0 : row0 + wg_rows, k0 : k0 + block_k]
            return A[group, k0 : k0 + block_k, row0 : row0 + wg_rows]
        if a_k_major:
            return A[row0 : row0 + wg_rows, k0 : k0 + block_k]
        return A[k0 : k0 + block_k, row0 : row0 + wg_rows]

    def b_region(B, group, n0, k0):
        if b_has_group:
            if b_k_major:
                return B[group, n0 : n0 + block_n, k0 : k0 + block_k]
            return B[group, k0 : k0 + block_k, n0 : n0 + block_n]
        if b_k_major:
            return B[n0 : n0 + block_n, k0 : k0 + block_k]
        return B[k0 : k0 + block_k, n0 : n0 + block_n]

    k_atoms = block_k // swizzle_atom  # 128-byte swizzle atoms along K

    def b_half_region(B, group, n0, k0, atom):
        """``block_n / 2`` columns and one K atom of a K-major B (the fused path takes no other)."""
        kk = k0 + atom * swizzle_atom
        if dense:
            return B[n0 : n0 + half_n, kk : kk + swizzle_atom]
        return B[group, n0 : n0 + half_n, kk : kk + swizzle_atom]

    def align_up(x):
        return ((x + T.int32(block_m - 1)) // T.int32(block_m)) * T.int32(block_m)

    @T.macro
    def swizzle_block(block_idx, num_m_blocks, num_n_blocks, out_m, out_n):
        """Map a tile id to its M and N blocks.

        Groups ``blocks_per_group`` tiles along M so the CTAs of one wave share
        the same ``B`` tile in L2.
        """
        num_blocks_per_group = num_n_blocks * T.int32(blocks_per_group)
        group_idx = block_idx // num_blocks_per_group
        first = group_idx * T.int32(blocks_per_group)
        in_group = block_idx % num_blocks_per_group
        in_group_blocks = T.min(T.int32(blocks_per_group), num_m_blocks - first)
        out_m[0] = first + in_group % in_group_blocks
        out_n[0] = in_group // in_group_blocks

    @T.macro
    def recover_ends(grouped_layout, s_ends, tx):
        """Recover each group's end row from non-decreasing per-row group ids."""
        for g in T.serial(T.ceildiv(num_groups, threads)):
            group = g * T.int32(threads) + tx
            if group < num_groups:
                lo = T.alloc_var("int32", init=0)
                hi = T.alloc_var("int32", init=m)
                for _ in T.serial(32):
                    if lo < hi:
                        mid = (lo + hi) >> T.int32(1)
                        if grouped_layout[mid] <= group:
                            lo = mid + T.int32(1)
                        else:
                            hi = mid
                s_ends[group] = lo

    @T.macro
    def tile_cumsum(ends, s_cum, s_total, num_n_blocks):
        """Per-group tile-count prefix sum; ``s_total`` is the tile count of the call.

        Masked groups are ``masked_m[g]`` rows from row 0 of their slab; psum
        groups run from the previous end (aligned up for the aligned variant)
        to ``ends[g]``.
        """
        s_cum[0] = T.int32(0)
        prev_end = T.alloc_var("int32", init=0)
        for g in T.serial(num_groups):
            if masked:
                rows = T.max(T.min(ends[g], m), T.int32(0))
            elif aligned_psum:
                rows = T.max(ends[g] - align_up(prev_end), T.int32(0))
            else:
                rows = T.max(ends[g] - prev_end, T.int32(0))
            s_cum[g + 1] = s_cum[g] + (rows + T.int32(block_m - 1)) // T.int32(block_m)
            prev_end = ends[g]
        s_total[0] = s_cum[num_groups] * num_n_blocks

    @T.macro
    def k_cumsum(sizes, s_cum):
        """K-grouped: the K prefix sum, group ``g`` contracting ``[s_cum[g], s_cum[g + 1])``."""
        s_cum[0] = T.int32(0)
        for g in T.serial(num_groups):
            s_cum[g + 1] = s_cum[g] + T.max(sizes[g], T.int32(0))

    @T.macro
    def resolve_tile(
        block_idx,
        grouped_layout,
        ends,
        s_cum,
        num_m_blocks,
        num_n_blocks,
        t_group,
        t_row0,
        t_col0,
        t_rows,
        t_k0,
        t_klen,
        t_khead,
    ):
        """Tile id -> group, first row, first column, valid rows, K range.

        ``t_row0`` indexes the A/C slab the tile belongs to. ``t_rows`` is
        ``block_m`` except on a tight group's last tile. ``ends`` contains
        per-group ends for psum, masked and recovered per-row layouts. The K
        range ``[t_k0, t_k0 + t_klen)`` is the whole ``K`` except for the
        K-grouped type, where it is the group's slice of the prefix sum; its
        first ``t_khead`` columns are alignment padding to mask.
        """
        m_blk = T.alloc_local((1,), "int32")
        n_blk = T.alloc_local((1,), "int32")
        t_k0[0] = T.int32(0)
        t_klen[0] = k
        t_khead[0] = T.int32(0)
        if dense:
            swizzle_block(block_idx, num_m_blocks, num_n_blocks, m_blk, n_blk)
            t_group[0] = T.int32(0)
            t_row0[0] = m_blk[0] * T.int32(block_m)
            t_rows[0] = T.int32(block_m)
        elif batched or k_grouped:
            per_batch = num_m_blocks * num_n_blocks
            rem = block_idx % per_batch
            t_group[0] = block_idx // per_batch
            if k_grouped:
                swizzle_block(rem, num_m_blocks, num_n_blocks, m_blk, n_blk)
                k_start = s_cum[t_group[0]]
                k_len = s_cum[t_group[0] + 1] - k_start
                if k_head_align:
                    t_khead[0] = k_start % T.int32(k_head_align)
                t_k0[0] = k_start - t_khead[0]
                t_klen[0] = k_len + t_khead[0]
            else:
                m_blk[0] = rem % num_m_blocks
                n_blk[0] = rem // num_m_blocks
            t_row0[0] = m_blk[0] * T.int32(block_m)
            t_rows[0] = T.int32(block_m)
        elif per_group:
            lo = T.alloc_var("int32", init=0)
            hi = T.alloc_var("int32", init=num_groups - 1)
            for _ in T.serial(search_steps):
                mid = (lo + hi) >> T.int32(1)
                if s_cum[mid + 1] * num_n_blocks <= block_idx:
                    lo = mid + T.int32(1)
                else:
                    hi = mid
            t_group[0] = lo
            rel = block_idx - s_cum[lo] * num_n_blocks
            swizzle_block(rel, s_cum[lo + 1] - s_cum[lo], num_n_blocks, m_blk, n_blk)
            if masked:
                t_row0[0] = m_blk[0] * T.int32(block_m)
                t_rows[0] = T.int32(block_m)
            else:
                prev_end = T.if_then_else(lo > 0, ends[T.max(lo - 1, 0)], T.int32(0))
                if aligned_psum:
                    t_row0[0] = align_up(prev_end) + m_blk[0] * T.int32(block_m)
                    t_rows[0] = T.int32(block_m)
                else:
                    t_row0[0] = prev_end + m_blk[0] * T.int32(block_m)
                    t_rows[0] = T.min(T.int32(block_m), ends[lo] - t_row0[0])
        else:  # aligned per-row: the flat grid, a tile's group read off its first row
            swizzle_block(block_idx, num_m_blocks, num_n_blocks, m_blk, n_blk)
            t_row0[0] = m_blk[0] * T.int32(block_m)
            t_rows[0] = T.int32(block_m)
            # Rows past the last group carry `num_groups`; clamp so a tile of
            # padding reads a real B and its rows are ignored.
            t_group[0] = T.min(
                T.max(grouped_layout[t_row0[0]], T.int32(0)), T.int32(num_groups - 1)
            )
        t_col0[0] = n_blk[0] * T.int32(c_tile_n)

    def tile_k_iters(t_klen):
        """K blocks of a tile. A K-grouped group with no K still runs one, fully
        masked, block so the accumulator remains initialized by WGMMA."""
        iters = T.ceildiv(t_klen[0], T.int32(block_k))
        return T.max(iters, T.int32(1)) if k_grouped else iters

    @T.macro
    def mask_k(A_s, B_s, slot, lo, hi):
        """K-grouped: zero a stage's columns outside ``[lo, hi)`` along K.

        TMA clips against the tensor, not the group, so a group's last K block
        holds the next group's first rows past ``hi``, and its first block holds
        the previous group's last rows below ``lo`` when the start was rounded
        down for a K-major operand. Both operands are zeroed so a stray
        non-finite value cannot survive as ``0 * inf``; the caller fences the
        generic-proxy writes before the WGMMA reads them.
        """
        if a_k_major:
            for i, kk in T.Parallel(wg_rows, block_k):
                if kk < lo or kk >= hi:
                    A_s[slot, i, kk] = T.cast(0, A_s.dtype)
        else:
            for kk, i in T.Parallel(block_k, wg_rows):
                if kk < lo or kk >= hi:
                    A_s[slot, kk, i] = T.cast(0, A_s.dtype)
        if b_k_major:
            for i, kk in T.Parallel(block_n, block_k):
                if kk < lo or kk >= hi:
                    B_s[slot, i, kk] = T.cast(0, B_s.dtype)
        else:
            for kk, i in T.Parallel(block_k, block_n):
                if kk < lo or kk >= hi:
                    B_s[slot, kk, i] = T.cast(0, B_s.dtype)

    @T.macro
    def load_a(A, dst, full, slot, group, row0, k0):
        T.tma_copy(a_region(A, group, row0, k0), dst[slot, :, :], barrier=full[slot])

    @T.macro
    def load_b(B, dst, full, slot, group, n0, k0):
        if fused:
            # Gate columns fill the tile's first half, the matching up columns its
            # second, one box per 128-byte K atom so each lands in one swizzle span.
            for atom in range(k_atoms):
                lo, hi = atom * swizzle_atom, (atom + 1) * swizzle_atom
                T.tma_copy(
                    b_half_region(B, group, n0, k0, atom),
                    dst[slot, 0:half_n, lo:hi],
                    barrier=full[slot],
                )
                T.tma_copy(
                    b_half_region(B, group, c_cols + n0, k0, atom),
                    dst[slot, half_n:block_n, lo:hi],
                    barrier=full[slot],
                )
        else:
            T.tma_copy(b_region(B, group, n0, k0), dst[slot, :, :], barrier=full[slot])

    @T.macro
    def gate_multiply(C_l, C_up, C_s):
        """The fused epilogue: ``C_s[i, j] = act(C_l[i, j]) * C_l[i, j + block_n / 2]``.

        Both accumulator columns of a pair live in the calling thread, so this is
        register arithmetic followed by the same shared staging a plain tile does.
        The up half goes through ``C_up`` because one parallel loop may read a
        fragment under a single index pattern.
        """
        T.copy(C_l[:, half_n:block_n], C_up)
        for i, j in T.Parallel(wg_rows, half_n):
            g = C_l[i, j]
            u = C_up[i, j]
            if activation == "silu_and_mul":
                C_s[i, j] = T.cast(g * T.sigmoid(g) * u, cd_dtype)
            else:  # gelu_and_mul, exact erf form
                half = T.cast(0.5, accum_dtype)
                one = T.cast(1.0, accum_dtype)
                inv_sqrt2 = T.cast(0.7071067811865476, accum_dtype)
                C_s[i, j] = T.cast(half * g * (one + erf(g * inv_sqrt2, cd_dtype)) * u, cd_dtype)

    @T.macro
    def store_tile(C, C_src, C_up, C_s, group, row0, col0, rows, wg):
        """Stage and store one warp-group's rows of a tile."""
        for chunk in range(epilogue_chunks):
            chunk_col = chunk * staged_n
            T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=128)
            if fused:
                gate_multiply(C_src, C_up, C_s)
            else:
                T.copy(C_src[:, chunk_col : chunk_col + staged_n], C_s)
            if tight and rows < T.int32(wg_rows):
                T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=128)
                if rows > 0:
                    for i, j in T.Parallel(wg_rows, staged_n):
                        if i < rows and col0 + chunk_col + j < c_cols:
                            C[row0 + i, col0 + chunk_col + j] = C_s[i, j]
            else:
                T.fence_proxy_async()
                T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=128)
                if c_has_group:
                    T.copy(C_s, C[group, row0, col0 + chunk_col])
                else:
                    T.copy(C_s, C[row0, col0 + chunk_col])

    @T.macro
    def release_stage(empty, slot, lane):
        """Lane 0 of each math warp arrives on the stage's empty barrier."""
        if lane == 0:
            T.mbarrier_arrive(empty[slot])

    @T.macro
    def math_warpgroup(
        A_s,
        B_s,
        C,
        full,
        empty,
        grouped_layout,
        ends,
        s_cum,
        C_l,
        C_cast,
        C_up,
        C_s,
        wg,
        pid,
        num_m_blocks,
        num_n_blocks,
        num_blocks,
        num_waves,
    ):
        """One math warp-group: drain the ring over K for each tile, then store."""
        T.inc_max_nreg(math_regs)
        tx = T.get_thread_binding()
        lane = tx % 32
        gi = T.alloc_var("int32", init=0)
        prev_slot = T.alloc_local((1,), "int32")
        t_group = T.alloc_local((1,), "int32")
        t_row0 = T.alloc_local((1,), "int32")
        t_col0 = T.alloc_local((1,), "int32")
        t_rows = T.alloc_local((1,), "int32")
        t_k0 = T.alloc_local((1,), "int32")
        t_klen = T.alloc_local((1,), "int32")
        t_khead = T.alloc_local((1,), "int32")

        for w in T.serial(num_waves):
            block_idx = w * T.int32(num_sms) + pid
            if block_idx < num_blocks:
                resolve_tile(
                    block_idx,
                    grouped_layout,
                    ends,
                    s_cum,
                    num_m_blocks,
                    num_n_blocks,
                    t_group,
                    t_row0,
                    t_col0,
                    t_rows,
                    t_k0,
                    t_klen,
                    t_khead,
                )
                k_iters = tile_k_iters(t_klen)
                for ki in T.serial(k_iters):
                    slot = gi % num_stages
                    phase = (gi // num_stages) & 1
                    T.barrier_wait(full[slot], phase)
                    if k_grouped:
                        lo = T.if_then_else(ki == 0, t_khead[0], T.int32(0))
                        hi = t_klen[0] - ki * T.int32(block_k)
                        if lo > 0 or hi < T.int32(block_k):
                            mask_k(A_s, B_s, slot, lo, hi)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=128)
                    T.wgmma_gemm(
                        A_s[slot, :, :],
                        B_s[slot, :, :],
                        C_l,
                        transpose_A=not a_k_major,
                        transpose_B=b_k_major,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=(ki == 0),
                    )
                    # Drain the previous WGMMA before releasing its pipeline stage.
                    if ki > 0:
                        T.wait_wgmma(1)
                        release_stage(empty, prev_slot[0], lane)
                    prev_slot[0] = slot
                    gi = gi + 1
                T.wait_wgmma(0)
                release_stage(empty, prev_slot[0], lane)
                T.warpgroup_fence_operand(C_l, num_regs=(wg_rows * block_n) // 128)

                row0 = t_row0[0] + T.int32(wg * wg_rows)
                rows = t_rows[0] - T.int32(wg * wg_rows)
                if fused or not cast_output:
                    # The fused epilogue casts as it multiplies; fp32 out needs no cast.
                    store_tile(C, C_l, C_up, C_s, t_group[0], row0, t_col0[0], rows, wg)
                else:
                    T.copy(C_l, C_cast)
                    store_tile(C, C_cast, C_up, C_s, t_group[0], row0, t_col0[0], rows, wg)

    @T.prim_func
    def grouped_gemm(
        A: T.Tensor(a_shape, dtype),  # type: ignore
        B: T.Tensor(b_shape, dtype),  # type: ignore
        C: T.Tensor(c_shape, cd_dtype),  # type: ignore
        grouped_layout: T.Tensor(layout_shape, "int32"),  # type: ignore
    ):
        with T.Kernel(num_sms, threads=threads) as pid:
            A_s0 = T.alloc_shared((num_stages,) + a_tile, dtype)
            if num_math_wgs > 1:
                A_s1 = T.alloc_shared((num_stages,) + a_tile, dtype)
            B_s = T.alloc_shared((num_stages,) + b_tile, dtype)
            # The cast fragment serves a plain 16-bit output; the fused epilogue casts
            # as it multiplies and takes the up half through its own fragment.
            C_l0 = T.alloc_fragment((wg_rows, block_n), accum_dtype)
            C_cast0 = (
                T.alloc_fragment((wg_rows, block_n), cd_dtype)
                if cast_output and not fused
                else C_l0
            )
            C_up0 = T.alloc_fragment((wg_rows, half_n), accum_dtype) if fused else C_l0
            C_s0 = T.alloc_shared((wg_rows, staged_n), cd_dtype)
            if num_math_wgs > 1:
                C_l1 = T.alloc_fragment((wg_rows, block_n), accum_dtype)
                C_cast1 = (
                    T.alloc_fragment((wg_rows, block_n), cd_dtype)
                    if cast_output and not fused
                    else C_l1
                )
                C_up1 = T.alloc_fragment((wg_rows, half_n), accum_dtype) if fused else C_l1
                C_s1 = T.alloc_shared((wg_rows, staged_n), cd_dtype)
            if dense:
                s_cum = T.alloc_local((1,), "int32")
                s_total = T.alloc_local((1,), "int32")
            else:
                s_cum = T.alloc_shared((num_groups + 1,), "int32")
                s_total = T.alloc_shared((1,), "int32")
            if tight_per_row:
                s_ends = T.alloc_shared((num_groups,), "int32")

            swizzled = {
                A_s0: tilelang.layout.make_swizzled_layout(A_s0),
                B_s: tilelang.layout.make_swizzled_layout(B_s),
                C_s0: tilelang.layout.make_swizzled_layout(C_s0),
            }
            if num_math_wgs > 1:
                swizzled[A_s1] = tilelang.layout.make_swizzled_layout(A_s1)
                swizzled[C_s1] = tilelang.layout.make_swizzled_layout(C_s1)
            T.annotate_layout(swizzled)

            # full tracks producer arrivals; empty tracks consumer releases.
            full = T.alloc_barrier([tma_threads] * num_stages)
            empty = T.alloc_barrier([math_warps] * num_stages)

            num_m_blocks = T.ceildiv(m, block_m)
            num_n_blocks = T.ceildiv(c_cols, c_tile_n)

            tx = T.get_thread_binding()

            if tight_per_row:
                recover_ends(grouped_layout, s_ends, tx)
                T.sync_threads()
                if tx == 0:
                    tile_cumsum(s_ends, s_cum, s_total, num_n_blocks)
            elif per_group:
                if tx == 0:
                    tile_cumsum(grouped_layout, s_cum, s_total, num_n_blocks)
            elif k_grouped:
                if tx == 0:
                    k_cumsum(grouped_layout, s_cum)

            if not dense:
                T.sync_threads()

            if per_group:
                num_blocks = s_total[0]
            elif batched or k_grouped:
                num_blocks = num_m_blocks * num_n_blocks * T.int32(num_groups)
            else:
                num_blocks = num_m_blocks * num_n_blocks
            num_waves = T.ceildiv(num_blocks, num_sms)

            if tx < tma_threads:
                # ── TMA warp-group ──
                T.dec_max_nreg(tma_regs)
                gi = T.alloc_var("int32", init=0)
                t_group = T.alloc_local((1,), "int32")
                t_row0 = T.alloc_local((1,), "int32")
                t_col0 = T.alloc_local((1,), "int32")
                t_rows = T.alloc_local((1,), "int32")
                t_k0 = T.alloc_local((1,), "int32")
                t_klen = T.alloc_local((1,), "int32")
                t_khead = T.alloc_local((1,), "int32")

                for w in T.serial(num_waves):
                    block_idx = w * T.int32(num_sms) + pid
                    if block_idx < num_blocks:
                        resolve_tile(
                            block_idx,
                            grouped_layout,
                            s_ends if tight_per_row else grouped_layout,
                            s_cum,
                            num_m_blocks,
                            num_n_blocks,
                            t_group,
                            t_row0,
                            t_col0,
                            t_rows,
                            t_k0,
                            t_klen,
                            t_khead,
                        )
                        k_iters = tile_k_iters(t_klen)
                        for ki in T.serial(k_iters):
                            slot = gi % num_stages
                            phase = (gi // num_stages) & 1
                            k0 = t_k0[0] + ki * T.int32(block_k)
                            T.barrier_wait(empty[slot], phase ^ 1)
                            load_a(A, A_s0, full, slot, t_group[0], t_row0[0], k0)
                            if num_math_wgs > 1:
                                load_a(
                                    A,
                                    A_s1,
                                    full,
                                    slot,
                                    t_group[0],
                                    t_row0[0] + T.int32(wg_rows),
                                    k0,
                                )
                            load_b(B, B_s, full, slot, t_group[0], t_col0[0], k0)
                            T.barrier_arrive(full[slot])
                            gi = gi + 1
            elif tx < tma_threads + 128:
                math_warpgroup(
                    A_s0,
                    B_s,
                    C,
                    full,
                    empty,
                    grouped_layout,
                    s_ends if tight_per_row else grouped_layout,
                    s_cum,
                    C_l0,
                    C_cast0,
                    C_up0,
                    C_s0,
                    0,
                    pid,
                    num_m_blocks,
                    num_n_blocks,
                    num_blocks,
                    num_waves,
                )
            else:
                if num_math_wgs > 1:
                    math_warpgroup(
                        A_s1,
                        B_s,
                        C,
                        full,
                        empty,
                        grouped_layout,
                        s_ends if tight_per_row else grouped_layout,
                        s_cum,
                        C_l1,
                        C_cast1,
                        C_up1,
                        C_s1,
                        1,
                        pid,
                        num_m_blocks,
                        num_n_blocks,
                        num_blocks,
                        num_waves,
                    )

    return grouped_gemm


# maxsize=None: an entry is one compiled kernel, so evicting it would only
# force a recompile. The key space is the specs the selector emits, a few
# hundred layouts per (gemm_type, majors, dtypes, static n and k), plus any
# explicit config a caller pins.
@functools.lru_cache(maxsize=None)
def _grouped_gemm_kernel(spec: GroupedGemmSpec):
    """JIT factory for one ``spec``; ``_grouped_gemm_kernel(spec)()`` compiles it.

    The builder closes over the spec's scalars only, so TileLang's cache key
    is exactly the template parameter pack.
    """
    gemm_type = spec.gemm_type.value
    a_k_major = spec.major_a is Major.K
    b_k_major = spec.major_b is Major.K
    ab_dtype = spec.ab_dtype
    cd_dtype = spec.cd_dtype
    num_groups = spec.num_groups
    shape_m, shape_n, shape_k = spec.shape_m, spec.shape_n, spec.shape_k
    block_m, block_n, block_k = spec.block_m, spec.block_n, spec.block_k
    num_stages = spec.num_stages
    num_math_wgs = spec.num_math_warpgroups
    num_sms = spec.num_sms
    activation = spec.activation
    epilogue_stage_n = spec.epilogue_stage_n
    swizzle_group_m = spec.swizzle_group_m

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            "tl.disable_warp_specialized": True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func():
        return _make_prim_func(
            gemm_type,
            a_k_major,
            b_k_major,
            ab_dtype,
            cd_dtype,
            num_groups,
            shape_m,
            shape_n,
            shape_k,
            block_m,
            block_n,
            block_k,
            num_stages,
            num_math_wgs,
            num_sms,
            activation,
            epilogue_stage_n,
            swizzle_group_m,
        )

    return _func


def _major_of(t: torch.Tensor, k_dim: int) -> Major:
    """K-major when the K dim is the contiguous one, MN-major when the other is."""
    if t.stride(k_dim) == 1:
        return Major.K
    other = t.ndim - 1 if k_dim == t.ndim - 2 else t.ndim - 2
    if t.stride(other) == 1:
        return Major.MN
    raise ValueError(f"neither of the last two dims of a {tuple(t.shape)} tensor is contiguous")


def _torch_dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


class GemmTemplate(Kernel):
    """16-bit grouped GEMM template for Hopper.

    ``C = A @ B^T`` per group on bf16 or fp16 operands with fp32 accumulation and
    a ``C`` in the operand dtype or fp32. Majorness is read off the operands'
    strides; batched operands may use either layout and m-grouped types need a K-major ``A``. Operand shapes per
    ``gemm_type`` (logical; an MN-major
    operand is a transposed view):

    | ``gemm_type``               | ``a``            | ``b``          | ``c``            | ``grouped_layout``           |
    | --------------------------- | ---------------- | -------------- | ---------------- | ---------------------------- |
    | ``DENSE``                   | ``[M, K]``       | ``[N, K]``     | ``[M, N]``       | none                         |
    | ``M_GROUPED_ALIGNED_PER_ROW`` | ``[M, K]``     | ``[G, N, K]``  | ``[M, N]``       | ``[M]`` group of each row    |
    | ``M_GROUPED_ALIGNED_PSUM``  | ``[M, K]``       | ``[G, N, K]``  | ``[M, N]``       | ``[G]`` psum row ends        |
    | ``M_GROUPED_TIGHT_PSUM``    | ``[M, K]``       | ``[G, N, K]``  | ``[M, N]``       | ``[G]`` psum row ends        |
    | ``M_GROUPED_TIGHT_PER_ROW`` | ``[M, K]``       | ``[G, N, K]``  | ``[M, N]``       | ``[M]`` group of each row    |
    | ``M_GROUPED_MASKED``        | ``[G, max_m, K]``| ``[G, N, K]``  | ``[G, max_m, N]``| ``[G]`` valid rows per group |
    | ``BATCHED``                 | ``[G, M, K]``    | ``[G, N, K]``  | ``[G, M, N]``    | none                         |
    | ``K_GROUPED_CONTIGUOUS``    | ``[M, sum_k]``   | ``[N, sum_k]`` | ``[G, M, N]``    | ``[G]`` K per group          |

    With ``activation`` set, ``b`` stacks gate and up along ``N`` and ``c`` has
    half as many columns.

    A selector chooses a legal tile layout per call; ``config`` may pin one.
    Dimensions absent from ``static_dims`` remain dynamic.

    Example:
        ```python linenums="1"
        kernel = GemmTemplate(GemmType.M_GROUPED_TIGHT_PSUM, num_groups=E)
        c = kernel(a, b, grouped_layout=ends)  # a: [M, K], b: [E, N, K], ends: [E] int32
        kernel = GemmTemplate(GemmType.BATCHED, num_groups=G)
        c = kernel(a, b)  # a: [G, M, K], b: [G, N, K]
        ```
    """

    supported_archs: list[int] = [90]

    def __init__(
        self,
        gemm_type: GemmType,
        *,
        num_groups: int = 1,
        cd_dtype: Optional[torch.dtype] = None,
        activation: str = "none",
        static_dims: str = "nk",
        m_alignment: int = 128,
        expected_m: int = 0,
        sm_count: Optional[int] = None,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        """Fix the template's operand-side parameters; the schedule is chosen per call.

        Args:
            gemm_type: Which rows of ``a`` and which ``b`` a tile reads.
            num_groups: Groups, experts or batches.
            cd_dtype: Output dtype: the operand dtype (default) or ``torch.float32``.
            activation: ``"none"``, or a gated activation (``"silu_and_mul"``,
                ``"gelu_and_mul"``) fused into the epilogue over a gate||up ``b``.
            static_dims: Dims compiled into the kernel, a subset of ``"mnk"``.
            m_alignment: Segment alignment of the aligned grouped layouts; it fixes ``block_m``.
            expected_m: Rows per group the cost model should plan for in the masked and
                psum layouts; the slab height, or the mean over groups, when 0. A caller
                whose routing is skewed passes the large experts' row count here.
            sm_count: Persistent grid size; the device's SM count when ``None``.
            config: Pins ``block_m``, ``block_n`` and optionally ``num_stages`` instead of
                running the selector.
            tune: Kernel-protocol flag; this kernel has no autotune space, the selector
                stands in for it, so ``True`` only warns.
            device_index: Device the kernel is built for; the current one when ``None``.
        """
        super().__init__(device_index=device_index)
        if num_groups < 1:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if GemmType(gemm_type) is GemmType.DENSE and num_groups != 1:
            raise ValueError(f"dense requires num_groups=1, got {num_groups}")
        if m_alignment < 1:
            raise ValueError(f"m_alignment must be positive, got {m_alignment}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
        if activation != "none" and GemmType(gemm_type) is GemmType.K_GROUPED_CONTIGUOUS:
            raise ValueError("a fused gated activation needs a per-group B; K-grouped has one B")
        self.gemm_type = GemmType(gemm_type)
        self.activation = activation
        self.num_groups = num_groups
        self.cd_dtype = cd_dtype
        self.static_dims = static_dims
        self.m_alignment = m_alignment
        self.expected_m = expected_m
        self.sm_count = get_sm_count(device_index) if sm_count is None else sm_count
        self.explicit_config = config
        self._spec_cache: dict[GemmDesc, GroupedGemmSpec] = {}
        self._empty_layout = None
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # The selector picks per call; nothing to author here.
        return {}

    @property
    def _a_has_group(self) -> bool:
        return self.gemm_type in (GemmType.M_GROUPED_MASKED, GemmType.BATCHED)

    @property
    def _c_has_group(self) -> bool:
        return self._a_has_group or self.gemm_type is GemmType.K_GROUPED_CONTIGUOUS

    def describe(self, a: torch.Tensor, b: torch.Tensor) -> GemmDesc:
        """The selector's view of a call; see the class docstring for the shapes."""
        if self.gemm_type is GemmType.K_GROUPED_CONTIGUOUS:
            if a.ndim != 2 or b.ndim != 2:
                raise ValueError("k_grouped_contiguous takes A as [M, sum_k] and B as [N, sum_k]")
        elif self.gemm_type is GemmType.DENSE:
            if a.ndim != 2 or b.ndim != 2:
                raise ValueError("dense takes A as [M, K] and B as [N, K]")
        else:
            a_ndim = 3 if self._a_has_group else 2
            if a.ndim != a_ndim:
                raise ValueError(f"{self.gemm_type.value} takes a {a_ndim}-D A, got {a.ndim}-D")
            if b.ndim != 3 or b.shape[0] != self.num_groups:
                raise ValueError(f"{self.gemm_type.value} takes B as [{self.num_groups}, N, K]")
            if self._a_has_group and a.shape[0] != self.num_groups:
                raise ValueError(f"{self.gemm_type.value} takes A as [{self.num_groups}, M, K]")
        m, k = a.shape[-2], a.shape[-1]
        n, k_b = b.shape[-2], b.shape[-1]
        if k != k_b:
            raise ValueError(f"A and B disagree on K: {k} vs {k_b}")
        if self.activation != "none" and n % 16:
            raise ValueError(
                f"a fused gated activation splits N={n} into gate and up halves whose width "
                "must be a multiple of 8 (the output row pitch): N must be a multiple of 16"
            )
        return GemmDesc(
            gemm_type=self.gemm_type,
            m=m,
            n=n,
            k=k,
            num_groups=self.num_groups,
            major_a=_major_of(a, a.ndim - 1),
            major_b=_major_of(b, b.ndim - 1),
            ab_dtype=_torch_dtype_str(a.dtype),
            cd_dtype=_torch_dtype_str(self.output_dtype(a)),
            num_sms=self.sm_count,
            static_dims=self.static_dims,
            m_alignment=self.m_alignment,
            expected_m=self.expected_m,
            activation=self.activation,
        )

    def spec_for(self, a: torch.Tensor, b: torch.Tensor) -> GroupedGemmSpec:
        """The template instantiation this call runs."""
        return self._spec_of(self.describe(a, b))

    def _spec_of(self, desc: GemmDesc) -> GroupedGemmSpec:
        spec = self._spec_cache.get(desc)
        if spec is None:
            spec = (
                spec_from_config(desc, self.explicit_config)
                if self.explicit_config
                else get_best_config(desc)
            )
            self._spec_cache[desc] = spec
        return spec

    def output_dtype(self, a: torch.Tensor) -> torch.dtype:
        """The dtype ``C`` is written in: ``cd_dtype`` when set, else the operand dtype."""
        return a.dtype if self.cd_dtype is None else self.cd_dtype

    @staticmethod
    def refusal_for(a: torch.Tensor, b: torch.Tensor) -> Optional[str]:
        """Why TMA cannot address these operands or the output, or ``None`` when it can.

        Every TMA-addressed row pitch must be a multiple of 16 bytes: the contiguous
        extent of each operand, and ``N`` for the output row. ``K`` must be positive.
        """
        if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
            return f"bf16 or fp16 operands of one dtype only, got {a.dtype} and {b.dtype}"
        if a.ndim < 2 or b.ndim < 2:
            return f"operands need at least two dims, got {a.ndim}-D and {b.ndim}-D"
        if a.shape[-1] == 0 and b.ndim != 2:  # a 2-D B is K-grouped, whose sum_k may be 0
            return "K must be positive"
        for name, t in (("a", a), ("b", b)):
            inner = t.shape[-1] if t.stride(-1) == 1 else t.shape[-2]
            if inner % 8:
                return f"{name}'s contiguous extent {inner} is not a multiple of 8 elements"
        if b.shape[-2] % 8:
            return f"N={b.shape[-2]} is not a multiple of 8 elements (the output row pitch)"
        return None

    def _check_layout(self, desc: GemmDesc, grouped_layout: Optional[torch.Tensor]) -> None:
        gtype = self.gemm_type
        if gtype in (GemmType.DENSE, GemmType.BATCHED):
            if grouped_layout is not None:
                raise ValueError(f"{gtype.value} takes no grouped_layout")
            return
        if grouped_layout is None:
            raise ValueError(f"{gtype.value} needs grouped_layout")
        length = desc.m if gtype in PER_ROW_TYPES else self.num_groups
        if grouped_layout.dtype != torch.int32 or grouped_layout.shape != (length,):
            raise ValueError(f"grouped_layout must be [{length}] int32")
        if not grouped_layout.is_contiguous():
            raise ValueError("grouped_layout must be contiguous")
        if gtype is GemmType.M_GROUPED_ALIGNED_PER_ROW and desc.m % self.m_alignment:
            raise ValueError(
                f"aligned_per_row rows ({desc.m}) must be a multiple of m_alignment "
                f"({self.m_alignment})"
            )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        grouped_layout: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run dense ``C = A @ B^T`` or its grouped/batched forms.

        Args:
            a: ``[M, K]``, ``[G, M, K]`` for masked/batched, or ``[M, sum_k]``
                for K-grouped.
            b: ``[N, K]`` for dense, ``[G, N, K]`` for grouped/batched, or
                ``[N, sum_k]`` for K-grouped.
            grouped_layout: int32 metadata per the class table; none for dense/batched.
            out: Optional preallocated output in ``cd_dtype``.
        """
        self._require_cuda(a=a, b=b, grouped_layout=grouped_layout, out=out)
        for name, t in (("b", b), ("grouped_layout", grouped_layout), ("out", out)):
            if t is not None and t.device != a.device:
                raise ValueError(f"{name} must be on {a.device}, got {t.device}")
        why = self.refusal_for(a, b)
        if why is not None:
            raise ValueError(f"{type(self).__name__}: {why}")
        desc = self.describe(a, b)
        self._check_layout(desc, grouped_layout)
        # The kernel takes the physical storage: an MN-major operand's contiguous
        # tensor is the transpose of its logical view. A dynamic dim hides a stride
        # from TileLang's ABI check, so require contiguity here.
        a_phys = a if desc.major_a is Major.K else a.transpose(-2, -1)
        b_phys = b if desc.major_b is Major.K else b.transpose(-2, -1)
        if not a_phys.is_contiguous() or not b_phys.is_contiguous():
            raise ValueError("a and b must be contiguous in their physical (K- or MN-major) layout")
        if grouped_layout is None:
            if self._empty_layout is None or self._empty_layout.device != a.device:
                self._empty_layout = torch.zeros(1, dtype=torch.int32, device=a.device)
            grouped_layout = self._empty_layout
        c_cols = desc.c_cols
        c_shape = (self.num_groups, desc.m, c_cols) if self._c_has_group else (desc.m, c_cols)
        cd_dtype = self.output_dtype(a)
        if out is None:
            out = torch.empty(c_shape, dtype=cd_dtype, device=a.device)
        elif tuple(out.shape) != c_shape or out.dtype != cd_dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {list(c_shape)} {cd_dtype}")
        if out.numel() == 0:  # no rows or no columns: nothing to launch
            return out
        if desc.k == 0:  # K-grouped with no tokens in any group: every product is zero
            return out.zero_()
        if out.data_ptr() in (
            a.data_ptr(),
            b.data_ptr(),
        ):  # same start; offset overlap is not caught
            raise ValueError("out must not alias an operand: tiles store while others still load")
        spec = self._spec_of(desc)
        fn = _grouped_gemm_kernel(spec)()
        fn(a_phys, b_phys, out, grouped_layout)
        return out


GroupedGemmTemplate = GemmTemplate
