"""Persistent GEMM template for dense, grouped, batched, and MoE layouts.

Inputs and output shapes are documented by ``GEMMTemplate``.
"""

import dataclasses
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import WARPGROUP_THREADS
from tileops._kernels.elementwise._erf import erf
from tileops._kernels.gemm.dense import wgmma_acc_layout
from tileops._kernels.gemm.persistent.heuristics import (
    ACTIVATIONS,
    INTERLEAVED_ACTIVATIONS,
    PER_GROUP_TYPES,
    PER_ROW_TYPES,
    ROW_STORE_VECTOR,
    SWIGLUOAI_ALPHA,
    SWIGLUOAI_LIMIT,
    GEMMDesc,
    GEMMType,
    GroupedGEMMSpec,
    Major,
    get_best_config,
    spec_from_config,
)
from tileops._kernels.kernel_base import Kernel
from tileops._utils import WARP_LANES

__all__ = ["GEMMTemplate"]


def _make_prim_func(
    gemm_type: str,
    a_k_major: bool,
    b_k_major: bool,
    dtype: str,
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
    epilogue_buffers: int,
    swizzle_group_m: int,
    has_bias: bool,
    pingpong: bool,
    a_box_rows: int,
    group_starts: bool,
    native_wgmma: bool,
    vector_row_store: bool,
    a_strides: tuple,
    c_strides: tuple,
):
    """Build the ``@T.prim_func`` for one spec; parameters are the spec's fields, scalars and
    the stride tuples of a strided batched A or C."""
    gtype = GEMMType(gemm_type)
    accum_dtype = "float"
    tma_threads = WARPGROUP_THREADS
    math_threads = WARPGROUP_THREADS * num_math_wgs
    threads = tma_threads + math_threads
    # Ping-pong: each math warp-group runs whole tiles of its own, the CTA's tiles taken in
    # turn, so one group's epilogue runs under the other's mainloop. Otherwise the groups
    # split one tile's rows.
    wg_rows = block_m if pingpong else block_m // num_math_wgs
    # The warps that release a ring stage: those of the one group whose tile it holds.
    release_warps = (WARPGROUP_THREADS if pingpong else math_threads) // WARP_LANES
    # M tiles of one L2 swizzle group: the pinned dense count, else the group whose resident
    # footprint along M and N is smallest.
    blocks_per_group = (
        swizzle_group_m
        if gtype is GEMMType.DENSE and swizzle_group_m
        else min((8, 16), key=lambda g: g * block_m + -(-num_sms // g) * block_n)
    )
    dense_coop2 = gtype is GEMMType.DENSE and num_math_wgs == 2
    tma_regs = 24 if dense_coop2 else 48
    math_regs = 240 if dense_coop2 else (248 if num_math_wgs == 1 else 224)
    swizzle_atom = 64
    epilogue_barrier_base = 8
    cast_output = cd_dtype != "float32"
    fused = activation != "none"
    # An interleaved B is taken as its [.., N / 2, 2, K] view.
    interleaved = activation in INTERLEAVED_ACTIVATIONS
    c_tile_n = block_n // 2 if fused else block_n  # output columns per tile
    staged_n = epilogue_stage_n or c_tile_n
    epilogue_chunks = c_tile_n // staged_n

    # Scheduler family and what each operand looks like.
    masked = gtype is GEMMType.M_GROUPED_MASKED
    aligned_psum = gtype is GEMMType.M_GROUPED_ALIGNED_PSUM
    tight_per_row = gtype is GEMMType.M_GROUPED_TIGHT_PER_ROW
    tight = gtype in (GEMMType.M_GROUPED_TIGHT_PSUM, GEMMType.M_GROUPED_TIGHT_PER_ROW)
    dense = gtype is GEMMType.DENSE
    batched = gtype is GEMMType.BATCHED
    k_grouped = gtype is GEMMType.K_GROUPED_CONTIGUOUS
    per_group = gtype in PER_GROUP_TYPES
    a_has_group = masked or batched
    b_has_group = not (dense or k_grouped)
    c_has_group = a_has_group or k_grouped
    # Whole output tiles across more than one wave: each tile's TMA store overlaps the next
    # tile. A partial tile or a single wave takes the synchronous copy.
    batched_async_store = (
        batched
        and shape_m > 0
        and shape_n > 0
        and shape_m % block_m == 0
        and shape_n % c_tile_n == 0
        and num_groups * (shape_m // block_m) * (shape_n // c_tile_n) > num_sms
    )
    # A dense GEMM over more than one wave stores each tile without waiting, so the store
    # overlaps the next tile's mainloop; TMA clips a ragged last tile at the output's edge.
    static_cols = shape_n // 2 if fused else shape_n
    dense_async_store = (
        dense
        and shape_m > 0
        and shape_n > 0
        and -(-shape_m // block_m) * -(-static_cols // c_tile_n) > num_sms
    )
    # Batched tiles hold their cast output and store it under the next tile's first WGMMA.
    # A K shorter than the ring cannot cover the store; ping-pong already hides it.
    deferred_store = (
        batched_async_store
        and cast_output
        and not fused
        and epilogue_buffers == 1
        and shape_k >= num_stages * block_k
        and not pingpong
    )
    # A K-major operand's TMA box must start 16-byte aligned along K, so a K-grouped GEMM
    # rounds each group's start down to 8 elements and masks the head like the tail.
    k_head_align = 8 if k_grouped and (a_k_major or b_k_major) else 0
    search_steps = max(1, (num_groups - 1).bit_length())

    m = T.dynamic("m") if shape_m == 0 else shape_m
    n = T.dynamic("n") if shape_n == 0 else shape_n
    k = T.dynamic("k") if shape_k == 0 else shape_k

    a_2d = (m, k) if a_k_major else (k, m)
    a_shape = (num_groups,) + a_2d if a_has_group else a_2d
    b_2d = (n, k) if b_k_major else (k, n)
    c_cols = n // 2 if fused else n  # act(gate, up) halves the width
    if interleaved:
        # A dynamic output width is its own symbol: n // 2 of a symbol does not give back
        # the view's shape.
        c_cols = shape_n // 2 if shape_n else T.dynamic("c_cols")
        b_2d = (c_cols, 2, k)
    b_shape = (num_groups,) + b_2d if b_has_group else b_2d
    # One bias row per group in B's row order; an interleaved one is viewed as
    # [G, N / 2, 2] like B.
    bias_shape = (num_groups, c_cols, 2) if interleaved else (num_groups, n)
    c_shape = (num_groups, m, c_cols) if c_has_group else (m, c_cols)
    a_param = T.StridedTensor(a_shape, a_strides, dtype) if a_strides else T.Tensor(a_shape, dtype)
    c_param = (
        T.StridedTensor(c_shape, c_strides, cd_dtype) if c_strides else T.Tensor(c_shape, cd_dtype)
    )
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

    def b_half_region(B, group, n0, k0, atom, part):
        """``block_n / 2`` gate (*part* 0) or up (*part* 1) rows and one K atom of a K-major B.

        An interleaved B is read through its ``[G, N / 2, 2, K]`` view, so a ragged last tile
        reads zeros past ``N / 2`` rather than the next rows.
        """
        kk = k0 + atom * swizzle_atom
        if interleaved:
            return B[group, n0 : n0 + half_n, part, kk : kk + swizzle_atom]
        row0 = n0 if part == 0 else c_cols + n0
        if dense:
            return B[row0 : row0 + half_n, kk : kk + swizzle_atom]
        return B[group, row0 : row0 + half_n, kk : kk + swizzle_atom]

    def bias_at(Bias, group, col0, j):
        """The bias of accumulator column *j* of the tile at output column *col0*; a column
        past the output's width, whose result is not stored, reads the last column's."""
        if fused:
            col = T.min(col0 + j % half_n, c_cols - 1)
            index = (col, j // half_n) if interleaved else (col + (j // half_n) * c_cols,)
        else:
            index = (T.min(col0 + j, c_cols - 1),)
        return Bias[(group,) + index]

    def align_up(x):
        return ((x + T.int32(block_m - 1)) // T.int32(block_m)) * T.int32(block_m)

    def group_start(ends, g):
        """First row of a psum group *g*: the previous group's end, or its own entry where
        the layout holds starts."""
        if group_starts:
            return ends[g]
        return T.if_then_else(g > 0, ends[T.max(g - 1, 0)], T.int32(0))

    def group_end(ends, g):
        """End row of a psum group *g*: its own entry, or the next group's start where the
        layout holds starts, the last group ending at ``m``."""
        if group_starts:
            return T.if_then_else(g + 1 < num_groups, ends[T.min(g + 1, num_groups - 1)], m)
        return ends[g]

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
        to ``ends[g]``, or from ``ends[g]`` to the next start where the layout holds
        group starts (``group_start`` and ``group_end``).
        """
        s_cum[0] = T.int32(0)
        prev_end = T.alloc_var("int32", init=0)
        for g in T.serial(num_groups):
            if masked:
                rows = T.max(T.min(ends[g], m), T.int32(0))
            elif aligned_psum:
                rows = T.max(ends[g] - align_up(prev_end), T.int32(0))
            elif group_starts:
                rows = T.max(group_end(ends, g) - group_start(ends, g), T.int32(0))
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
        per-group ends for psum, masked and recovered per-row layouts, or per-group
        starts for a tight psum layout with ``group_starts``. The K
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
            elif pingpong:
                # Row-major, so the CTAs a wave launches side by side read one A row block.
                m_blk[0] = rem // num_n_blocks
                n_blk[0] = rem % num_n_blocks
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
                prev_end = group_start(ends, lo)
                if aligned_psum:
                    t_row0[0] = align_up(prev_end) + m_blk[0] * T.int32(block_m)
                    t_rows[0] = T.int32(block_m)
                else:
                    t_row0[0] = prev_end + m_blk[0] * T.int32(block_m)
                    t_rows[0] = T.min(T.int32(block_m), group_end(ends, lo) - t_row0[0])
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
        """K-grouped: zero a stage's columns outside ``[lo, hi)`` along K, which TMA loaded
        from the neighbouring groups.

        Both operands are zeroed so a non-finite neighbour cannot survive as ``0 * inf``;
        the caller fences these generic-proxy writes before the WGMMA reads them.
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
    def load_a(A, dst, full, slot, group, row0, k0, rows):
        if a_box_rows:
            for box in range(wg_rows // a_box_rows):
                lo = box * a_box_rows
                if T.int32(lo) < rows:
                    # One box per 128-byte K atom, each landing in one swizzle span.
                    for atom in range(k_atoms):
                        k_lo, k_hi = atom * swizzle_atom, (atom + 1) * swizzle_atom
                        T.tma_copy(
                            A[row0 + lo : row0 + lo + a_box_rows, k0 + k_lo : k0 + k_hi],
                            dst[slot, lo : lo + a_box_rows, k_lo:k_hi],
                            barrier=full[slot],
                        )
        else:
            T.tma_copy(a_region(A, group, row0, k0), dst[slot, :, :], barrier=full[slot])

    @T.macro
    def load_b(B, dst, full, slot, group, n0, k0):
        if fused:
            # Gate columns fill the tile's first half, the matching up columns its
            # second, one box per 128-byte K atom so each lands in one swizzle span.
            for atom in range(k_atoms):
                lo, hi = atom * swizzle_atom, (atom + 1) * swizzle_atom
                T.tma_copy(
                    b_half_region(B, group, n0, k0, atom, 0),
                    dst[slot, 0:half_n, lo:hi],
                    barrier=full[slot],
                )
                T.tma_copy(
                    b_half_region(B, group, n0, k0, atom, 1),
                    dst[slot, half_n:block_n, lo:hi],
                    barrier=full[slot],
                )
        else:
            T.tma_copy(b_region(B, group, n0, k0), dst[slot, :, :], barrier=full[slot])

    @T.macro
    def stage_bias(Bias, bias_s, group, col0, tx):
        """Copy the tile's ``block_n`` bias values into the warp-group's shared row; the
        previous tile's readers of the row are past the barrier that opens its epilogue
        (``store_tile`` or ``store_chunk``)."""
        for step in range(-(-block_n // WARPGROUP_THREADS)):
            j = step * WARPGROUP_THREADS + tx % WARPGROUP_THREADS
            if j < block_n:
                bias_s[j] = bias_at(Bias, group, col0, j)

    @T.macro
    def add_bias(C_l, bias_s, wg):
        """Add the bias of each accumulator column, in fp32, before the activation and cast."""
        T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=WARPGROUP_THREADS)
        for i, j in T.Parallel(wg_rows, block_n):
            C_l[i, j] = C_l[i, j] + T.cast(bias_s[j], accum_dtype)

    @T.macro
    def gate_multiply(C_l, C_up, C_s):
        """The fused epilogue: ``C_s[i, j] = act(C_l[i, j], C_l[i, j + block_n / 2])``.

        The up half goes through ``C_up`` because one parallel loop may read a fragment
        under a single index pattern.
        """
        T.copy(C_l[:, half_n:block_n], C_up)
        for i, j in T.Parallel(wg_rows, half_n):
            # The loop takes its threads from the first fragment it reads: C_up's layout is
            # inferred in this warp-group's thread range, which an annotated C_l lacks.
            u = C_up[i, j]
            g = C_l[i, j]
            if activation == "silu_and_mul" and dense:
                # The tanh form on one MUFU.TANH, where tanhf is a software sequence; TileLang
                # has no tanh.approx primitive.
                half = T.cast(0.5, accum_dtype)
                th = T.call_extern(accum_dtype, "tileops::approx_tanh", half * g)
                C_s[i, j] = T.cast(g * (half + half * th) * u, cd_dtype)
            elif activation == "silu_and_mul":
                # sigmoid(g) as 0.5 + 0.5 * tanh(g / 2), cheaper on SM90 than expf + division.
                half = T.cast(0.5, accum_dtype)
                C_s[i, j] = T.cast(g * (half + half * T.tanh(half * g)) * u, cd_dtype)
            elif activation == "swigluoai":
                # swigluoai as MoEGroupedGEMMFwdOp defines it, the sigmoid in the tanh form above.
                half = T.cast(0.5, accum_dtype)
                limit = T.cast(SWIGLUOAI_LIMIT, accum_dtype)
                gate = T.min(g, limit)
                up = T.max(T.min(u, limit), -limit)
                sig = half + half * T.tanh(T.cast(SWIGLUOAI_ALPHA / 2, accum_dtype) * gate)
                C_s[i, j] = T.cast(gate * sig * (up + T.cast(1.0, accum_dtype)), cd_dtype)
            else:  # gelu_and_mul, exact erf form
                half = T.cast(0.5, accum_dtype)
                one = T.cast(1.0, accum_dtype)
                inv_sqrt2 = T.cast(0.7071067811865476, accum_dtype)
                C_s[i, j] = T.cast(half * g * (one + erf(g * inv_sqrt2, cd_dtype)) * u, cd_dtype)

    # A row-masked store over a width that is not a power of two divides each element's index
    # by the width; ptxas hoists those loop-invariant quotients into registers that spill. Such
    # a width stores in runs of its widest power-of-two divisor, whose index splits by shifts.
    store_run = staged_n & -staged_n

    @T.macro
    def store_tile(C, C_src, C_up, C_s, group, row0, col0, rows, wg):
        """Stage and store one warp-group's rows of a tile."""
        for chunk in range(epilogue_chunks):
            chunk_col = chunk * staged_n
            if (
                batched_async_store or dense_async_store
            ) and T.get_thread_binding() % WARPGROUP_THREADS == 0:
                T.tma_store_wait(0)
            T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=WARPGROUP_THREADS)
            if fused:
                gate_multiply(C_src, C_up, C_s)
            else:
                T.copy(C_src[:, chunk_col : chunk_col + staged_n], C_s)
            if tight and rows < T.int32(wg_rows):
                T.sync_threads(
                    barrier_id=epilogue_barrier_base + wg, arrive_count=WARPGROUP_THREADS
                )
                if rows > 0:
                    if vector_row_store:
                        # The output row pitch is a multiple of 8 elements
                        # (``GEMMTemplate.refusal_for``), so a vector at a multiple of
                        # ``ROW_STORE_VECTOR`` lies wholly inside or wholly past the last column.
                        for run in range(staged_n // store_run):
                            run_col = col0 + chunk_col + run * store_run
                            for i, v in T.Parallel(wg_rows, store_run // ROW_STORE_VECTOR):
                                if i < rows and run_col + v * ROW_STORE_VECTOR < c_cols:
                                    T.call_extern(
                                        "handle",
                                        "tileops::copy16",
                                        T.address_of(C[row0 + i, run_col + v * ROW_STORE_VECTOR]),
                                        T.address_of(
                                            C_s[i, run * store_run + v * ROW_STORE_VECTOR]
                                        ),
                                    )
                    else:
                        for run in range(staged_n // store_run):
                            run_col = col0 + chunk_col + run * store_run
                            for i, j in T.Parallel(wg_rows, store_run):
                                if i < rows and run_col + j < c_cols:
                                    C[row0 + i, run_col + j] = C_s[i, run * store_run + j]
            else:
                T.fence_proxy_async()
                T.sync_threads(
                    barrier_id=epilogue_barrier_base + wg, arrive_count=WARPGROUP_THREADS
                )
                if c_has_group:
                    if batched_async_store:
                        T.tma_copy(
                            C_s,
                            C[group, row0, col0 + chunk_col],
                            leader_scope_threads=WARPGROUP_THREADS,
                        )
                    else:
                        T.copy(C_s, C[group, row0, col0 + chunk_col])
                elif dense_async_store:
                    T.tma_copy(
                        C_s, C[row0, col0 + chunk_col], leader_scope_threads=WARPGROUP_THREADS
                    )
                else:
                    T.copy(C_s, C[row0, col0 + chunk_col])

    @T.macro
    def store_chunk(C, C_l, buf, group, row0, col0, chunk_col, rows, wg):
        """Cast one chunk of the accumulator into ``buf`` and store it without waiting.

        A full tile waits for all but the newest TMA store, the one reading the other
        buffer. A ragged tile issues no TMA store for its chunks, so it waits for all.
        """
        if T.get_thread_binding() % WARPGROUP_THREADS == 0:
            if tight and rows < T.int32(wg_rows):
                T.tma_store_wait(0)
            else:
                T.tma_store_wait(1)
        T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=WARPGROUP_THREADS)
        for i, j in T.Parallel(wg_rows, staged_n):
            buf[i, j] = T.cast(C_l[i, chunk_col + j], cd_dtype)
        if tight and rows < T.int32(wg_rows):
            T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=WARPGROUP_THREADS)
            if rows > 0:
                for i, j in T.Parallel(wg_rows, staged_n):
                    if i < rows and col0 + chunk_col + j < c_cols:
                        C[row0 + i, col0 + chunk_col + j] = buf[i, j]
        else:
            T.fence_proxy_async()
            T.sync_threads(barrier_id=epilogue_barrier_base + wg, arrive_count=WARPGROUP_THREADS)
            if c_has_group:
                T.tma_copy(
                    buf, C[group, row0, col0 + chunk_col], leader_scope_threads=WARPGROUP_THREADS
                )
            else:
                T.tma_copy(buf, C[row0, col0 + chunk_col], leader_scope_threads=WARPGROUP_THREADS)

    def store_tile_buffered(C, C_l, bufs, group, row0, col0, rows, wg):
        """``store_tile`` over two staging buffers in turn; a plain loop, so it unrolls."""
        for chunk in range(epilogue_chunks):
            store_chunk(C, C_l, bufs[chunk % 2], group, row0, col0, chunk * staged_n, rows, wg)

    def native_acc_layout(buf, width):
        """The layout ``T.wgmma_gemm`` gives a math warp-group's ``64 x width`` accumulator,
        which the native-width WGMMA writes register for register."""
        return wgmma_acc_layout(buf, wg_rows, width, block_k, dtype, "float32")

    @T.macro
    def mma(A_s, B_s, C_l, slot, ki):
        """One K block of the warp-group's tile into ``C_l``."""
        if native_wgmma:
            T.call_extern(
                "handle",
                f"tileops::dense_wgmma_{'bf16' if dtype == 'bfloat16' else 'fp16'}"
                f"_64x{block_n}_k64",
                C_l.access_ptr("rw"),
                T.address_of(A_s[slot, 0, 0]),
                T.address_of(B_s[slot, 0, 0]),
                T.if_then_else(ki > 0, 1, 0),
            )
        else:
            T.wgmma_gemm(
                A_s[slot, :, :],
                B_s[slot, :, :],
                C_l,
                transpose_A=not a_k_major,
                transpose_B=b_k_major,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=(ki == 0),
            )

    @T.macro
    def release_stage(empty, slot, lane):
        """Lane 0 of each math warp arrives on the stage's empty barrier."""
        if lane == 0:
            T.mbarrier_arrive(empty[slot])

    def tile_owned(block_idx, num_blocks, w, wg):
        """Whether math warp-group *wg* runs the CTA's tile *w*: every live tile, or under
        ping-pong the live tiles of its parity. A helper so non-ping-pong kernels bind no
        extra variable."""
        live = block_idx < num_blocks
        return T.And(live, w % 2 == wg) if pingpong else live

    @T.macro
    def math_warpgroup(
        A_s,
        B_s,
        C,
        Bias,
        full,
        empty,
        turn,
        epi,
        grouped_layout,
        ends,
        s_cum,
        C_l,
        C_cast,
        C_up,
        C_s,
        bias_s,
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
        if deferred_store:
            # The tile held in C_cast and not yet stored: group, first row, first column.
            p_tile = T.alloc_local((3,), "int32")
            pending = T.alloc_var("int32", init=0)

        for w in T.serial(num_waves):
            block_idx = w * T.int32(num_sms) + pid
            if tile_owned(block_idx, num_blocks, w, wg):
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
                if has_bias:
                    stage_bias(Bias, bias_s, t_group[0], t_col0[0], tx)
                k_iters = tile_k_iters(t_klen)
                if pingpong:
                    # The producer fills the ring with every tile of the CTA in turn, so tile
                    # w starts at stage w * k_iters.
                    gi = w * k_iters
                    # A stage barrier tells only two consecutive phases apart: wait for the
                    # other group's mainloop before waiting on any stage.
                    if w > 0:
                        T.barrier_wait(turn[1 - wg], ((w - 1) // 2) & 1)
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
                            T.sync_threads(
                                barrier_id=epilogue_barrier_base + wg,
                                arrive_count=WARPGROUP_THREADS,
                            )
                    mma(A_s, B_s, C_l, slot, ki)
                    if deferred_store and ki == 0 and pending == 1:
                        store_tile(
                            C, C_cast, C_up, C_s, p_tile[0], p_tile[1], p_tile[2], wg_rows, wg
                        )
                        pending = 0
                    # Drain the previous WGMMA before releasing its pipeline stage.
                    if ki > 0:
                        T.wait_wgmma(1)
                        release_stage(empty, prev_slot[0], lane)
                    prev_slot[0] = slot
                    gi = gi + 1
                if pingpong and lane == 0:
                    T.mbarrier_arrive(turn[wg])
                T.wait_wgmma(0)
                release_stage(empty, prev_slot[0], lane)
                T.warpgroup_fence_operand(C_l, num_regs=(wg_rows * block_n) // WARPGROUP_THREADS)
                if has_bias:
                    add_bias(C_l, bias_s, wg)

                row_offset = 0 if pingpong else wg * wg_rows
                row0 = t_row0[0] + T.int32(row_offset)
                rows = t_rows[0] - T.int32(row_offset)
                # Ping-pong: wait until the other group's previous tile has left the shared buffers.
                if pingpong and w > 0:
                    T.barrier_wait(epi[1 - wg], ((w - 1) // 2) & 1)
                if deferred_store:
                    T.copy(C_l, C_cast)
                    p_tile[0] = t_group[0]
                    p_tile[1] = row0
                    p_tile[2] = t_col0[0]
                    pending = 1
                elif epilogue_buffers > 1:
                    store_tile_buffered(C, C_l, C_s, t_group[0], row0, t_col0[0], rows, wg)
                elif fused or not cast_output or batched:
                    # The fused epilogue casts as it multiplies; fp32 out needs no cast.
                    # Batched GEMM casts each staged chunk directly into shared memory.
                    store_tile(C, C_l, C_up, C_s, t_group[0], row0, t_col0[0], rows, wg)
                else:
                    T.copy(C_l, C_cast)
                    store_tile(C, C_cast, C_up, C_s, t_group[0], row0, t_col0[0], rows, wg)
                if pingpong:
                    # Drain this group's reads and bulk stores before the other group reuses
                    # the buffers.
                    if (batched_async_store or epilogue_buffers > 1) and (
                        tx % WARPGROUP_THREADS == 0
                    ):
                        T.tma_store_wait(0)
                    T.sync_threads(
                        barrier_id=epilogue_barrier_base + wg, arrive_count=WARPGROUP_THREADS
                    )
                    if lane == 0:
                        T.mbarrier_arrive(epi[wg])
        if deferred_store and pending == 1:
            store_tile(C, C_cast, C_up, C_s, p_tile[0], p_tile[1], p_tile[2], wg_rows, wg)
        if (
            batched_async_store or dense_async_store or epilogue_buffers > 1
        ) and tx % WARPGROUP_THREADS == 0:
            T.tma_store_wait(0)

    def add_swizzled(layouts, buffers):
        """Swizzle one staging buffer or each of a pair; a plain loop, so it unrolls."""
        for buf in buffers if isinstance(buffers, list) else [buffers]:
            layouts[buf] = tilelang.layout.make_swizzled_layout(buf)

    @T.macro
    def persistent_body(A, B, C, grouped_layout, Bias, pid):
        """The persistent CTA: one TMA producer warp-group and the math warp-groups."""
        A_s0 = T.alloc_shared((num_stages,) + a_tile, dtype)
        split_rows = num_math_wgs > 1 and not pingpong
        if split_rows:
            A_s1 = T.alloc_shared((num_stages,) + a_tile, dtype)
        B_s = T.alloc_shared((num_stages,) + b_tile, dtype)
        # The cast fragment serves a plain 16-bit output; the fused epilogue casts
        # as it multiplies and takes the up half through its own fragment.
        C_l0 = T.alloc_fragment((wg_rows, block_n), accum_dtype)
        # A staged epilogue over two buffers casts each chunk as it stages it.
        separate_cast = (
            cast_output and not fused and epilogue_buffers == 1 and (deferred_store or not batched)
        )
        C_cast0 = T.alloc_fragment((wg_rows, block_n), cd_dtype) if separate_cast else C_l0
        C_up0 = T.alloc_fragment((wg_rows, half_n), accum_dtype) if fused else C_l0
        C_s0 = T.alloc_shared((wg_rows, staged_n), cd_dtype)
        if epilogue_buffers > 1:
            C_s0 = [C_s0, T.alloc_shared((wg_rows, staged_n), cd_dtype)]
        bias_s0 = T.alloc_shared((block_n,), dtype) if has_bias else None
        if num_math_wgs > 1:
            C_l1 = T.alloc_fragment((wg_rows, block_n), accum_dtype)
            C_cast1 = T.alloc_fragment((wg_rows, block_n), cd_dtype) if separate_cast else C_l1
            C_up1 = T.alloc_fragment((wg_rows, half_n), accum_dtype) if fused else C_l1
            # Ping-pong's two groups share the staging buffers.
            C_s1 = C_s0 if pingpong else T.alloc_shared((wg_rows, staged_n), cd_dtype)
            if epilogue_buffers > 1 and not pingpong:
                C_s1 = [C_s1, T.alloc_shared((wg_rows, staged_n), cd_dtype)]
            bias_s1 = T.alloc_shared((block_n,), dtype) if has_bias else None
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
        }
        add_swizzled(swizzled, C_s0)
        if num_math_wgs > 1:
            if split_rows:
                swizzled[A_s1] = tilelang.layout.make_swizzled_layout(A_s1)
            if not pingpong:
                add_swizzled(swizzled, C_s1)
        if native_wgmma:
            # The native-width WGMMA has no T.wgmma_gemm to infer the accumulators' layout.
            swizzled[C_l0] = native_acc_layout(C_l0, block_n)
            if num_math_wgs > 1:
                swizzled[C_l1] = native_acc_layout(C_l1, block_n)
        T.annotate_layout(swizzled)

        # full tracks the producer warp's arrivals; empty tracks consumer releases.
        full = T.alloc_barrier([WARP_LANES] * num_stages)
        empty = T.alloc_barrier([release_warps] * num_stages)
        # Ping-pong: each math warp-group's warps arrive once its tile's mainloop has waited
        # on every stage.
        turn = T.alloc_barrier([release_warps] * 2) if pingpong else None
        # Ping-pong: each math warp-group's warps arrive once its tile has left the shared
        # staging buffer.
        epi = T.alloc_barrier([release_warps] * 2) if pingpong else None

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
            # The producer warp-group TMA-loads each tile's A and B stages into the ring.
            T.dec_max_nreg(tma_regs)
            # One elected thread issues every copy, so one warp runs the loop; the
            # warp-group's other three warps only hand their registers back.
            if tx < WARP_LANES:
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
                            load_a(A, A_s0, full, slot, t_group[0], t_row0[0], k0, t_rows[0])
                            if split_rows:
                                load_a(
                                    A,
                                    A_s1,
                                    full,
                                    slot,
                                    t_group[0],
                                    t_row0[0] + T.int32(wg_rows),
                                    k0,
                                    t_rows[0] - T.int32(wg_rows),
                                )
                            load_b(B, B_s, full, slot, t_group[0], t_col0[0], k0)
                            T.barrier_arrive(full[slot])
                            gi = gi + 1
        elif tx < tma_threads + WARPGROUP_THREADS:
            math_warpgroup(
                A_s0,
                B_s,
                C,
                Bias,
                full,
                empty,
                turn,
                epi,
                grouped_layout,
                s_ends if tight_per_row else grouped_layout,
                s_cum,
                C_l0,
                C_cast0,
                C_up0,
                C_s0,
                bias_s0,
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
                    A_s1 if split_rows else A_s0,
                    B_s,
                    C,
                    Bias,
                    full,
                    empty,
                    turn,
                    epi,
                    grouped_layout,
                    s_ends if tight_per_row else grouped_layout,
                    s_cum,
                    C_l1,
                    C_cast1,
                    C_up1,
                    C_s1,
                    bias_s1,
                    1,
                    pid,
                    num_m_blocks,
                    num_n_blocks,
                    num_blocks,
                    num_waves,
                )

    # The bias is a kernel argument only where the call passes one.
    if has_bias:

        @T.prim_func
        def grouped_gemm(
            A: a_param,  # type: ignore
            B: T.Tensor(b_shape, dtype),  # type: ignore
            C: c_param,  # type: ignore
            grouped_layout: T.Tensor(layout_shape, "int32"),  # type: ignore
            Bias: T.Tensor(bias_shape, dtype),  # type: ignore
        ):
            with T.Kernel(num_sms, threads=threads) as pid:
                persistent_body(A, B, C, grouped_layout, Bias, pid)

    else:

        @T.prim_func
        def grouped_gemm(
            A: a_param,  # type: ignore
            B: T.Tensor(b_shape, dtype),  # type: ignore
            C: c_param,  # type: ignore
            grouped_layout: T.Tensor(layout_shape, "int32"),  # type: ignore
        ):
            with T.Kernel(num_sms, threads=threads) as pid:
                persistent_body(A, B, C, grouped_layout, None, pid)

    return grouped_gemm


@target_cache(maxsize=None)
def _grouped_gemm_kernel(spec: GroupedGEMMSpec):
    """JIT factory for one ``spec``; ``_grouped_gemm_kernel(spec)()`` compiles it."""
    # The builder closes over the spec's fields only, so TileLang's cache key is exactly the
    # template parameter pack.
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
    epilogue_buffers = spec.epilogue_buffers
    swizzle_group_m = spec.swizzle_group_m
    has_bias = spec.has_bias
    pingpong = spec.pingpong
    a_box_rows = spec.a_box_rows
    group_starts = spec.group_starts
    native_wgmma = spec.native_wgmma
    vector_row_store = spec.vector_row_store
    a_strides, c_strides = spec.a_strides, spec.c_strides
    headers = []
    if spec.native_wgmma:
        headers.append("dense_wgmma_helper.h")
    if spec.gemm_type is GEMMType.DENSE and spec.activation == "silu_and_mul":
        headers.append("approx_math.h")
    if spec.vector_row_store:
        headers.append("streaming_load.h")  # copy16
    compile_flags = [*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"]
    if headers:
        compile_flags += csrc_include(*headers)

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            "tl.disable_warp_specialized": True,
        },
        compile_flags=compile_flags,
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
            epilogue_buffers,
            swizzle_group_m,
            has_bias,
            pingpong,
            a_box_rows,
            group_starts,
            native_wgmma,
            vector_row_store,
            a_strides,
            c_strides,
        )

    return _func


class GEMMTemplate(Kernel):
    """16-bit grouped GEMM template for SM90.

    ``C = A @ B^T`` per group on bf16 or fp16 operands with fp32 accumulation and
    a ``C`` in the operand dtype or fp32. Majorness is read off the operands'
    strides; batched operands may use either layout and m-grouped types need a K-major ``A``.
    A batched ``A`` with one of its last two dims unit-stride, and a batched ``C``, may also
    lie strided: storage whose last stride is 1 and whose others are whole 16-byte steps, as
    TMA addresses it, is read and written in place, other storage through a contiguous copy;
    an ``A`` with neither of its last two dims unit-stride is refused. Operand shapes per
    ``gemm_type`` (logical; an MN-major
    operand is a transposed view):

    | ``gemm_type``               | ``a``            | ``b``          | ``c``            | ``grouped_layout``           |
    | --------------------------- | ---------------- | -------------- | ---------------- | ---------------------------- |
    | ``DENSE``                   | ``[M, K]``       | ``[N, K]``     | ``[M, N]``       | none                         |
    | ``M_GROUPED_ALIGNED_PER_ROW`` | ``[M, K]``     | ``[G, N, K]``  | ``[M, N]``       | ``[M]`` group of each row    |
    | ``M_GROUPED_ALIGNED_PSUM``  | ``[M, K]``       | ``[G, N, K]``  | ``[M, N]``       | ``[G]`` psum row ends        |
    | ``M_GROUPED_TIGHT_PSUM``    | ``[M, K]``       | ``[G, N, K]``  | ``[M, N]``       | ``[G]`` row ends (or starts) |
    | ``M_GROUPED_TIGHT_PER_ROW`` | ``[M, K]``       | ``[G, N, K]``  | ``[M, N]``       | ``[M]`` group of each row    |
    | ``M_GROUPED_MASKED``        | ``[G, max_m, K]``| ``[G, N, K]``  | ``[G, max_m, N]``| ``[G]`` valid rows per group |
    | ``BATCHED``                 | ``[G, M, K]``    | ``[G, N, K]``  | ``[G, M, N]``    | none                         |
    | ``K_GROUPED_CONTIGUOUS``    | ``[M, sum_k]``   | ``[N, sum_k]`` | ``[G, M, N]``    | ``[G]`` K per group          |

    With ``activation`` set, ``b`` holds gate and up along ``N`` and ``c`` has
    half as many columns: stacked (the ``N / 2`` gate rows, then the up rows) for
    ``silu_and_mul`` and ``gelu_and_mul``, interleaved (row ``2j`` gate ``j``, row
    ``2j + 1`` up ``j``) for ``swigluoai``.

    An optional ``[G, N]`` ``bias``, in ``b``'s row order and the operand dtype, is added
    to the fp32 accumulator before the activation and the cast. Only the M-grouped GEMMs
    take one.

    A selector chooses a legal tile layout per call; ``config`` may pin one.
    Dimensions absent from ``static_dims`` remain dynamic.

    Example:
        ```python linenums="1"
        kernel = GEMMTemplate(GEMMType.M_GROUPED_TIGHT_PSUM, num_groups=E)
        c = kernel(a, b, grouped_layout=ends)  # a: [M, K], b: [E, N, K], ends: [E] int32
        kernel = GEMMTemplate(GEMMType.BATCHED, num_groups=G)
        c = kernel(a, b)  # a: [G, M, K], b: [G, N, K]
        ```
    """

    supported_archs: list[int] = [90]

    @staticmethod
    def _major_of(t: torch.Tensor, k_dim: int) -> Major:
        """K-major when the K dim is the contiguous one, MN-major when the other is."""
        if t.stride(k_dim) == 1:
            return Major.K
        other = t.ndim - 1 if k_dim == t.ndim - 2 else t.ndim - 2
        if t.stride(other) == 1:
            return Major.MN
        raise ValueError(f"neither of the last two dims of a {tuple(t.shape)} tensor is contiguous")

    def __init__(
        self,
        gemm_type: GEMMType,
        *,
        num_groups: int = 1,
        cd_dtype: torch.dtype | None = None,
        activation: str = "none",
        static_dims: str = "nk",
        m_alignment: int = 128,
        expected_m: int = 0,
        sm_count: int | None = None,
        config: dict | None = None,
        group_starts: bool = False,
        device_index: int | None = None,
    ) -> None:
        """Fix the template's operand-side parameters; the schedule is chosen per call.

        Args:
            gemm_type: Which rows of ``a`` and which ``b`` a tile reads.
            num_groups: Groups, experts or batches.
            cd_dtype: Output dtype: the operand dtype (default) or ``torch.float32``.
            activation: ``"none"``, or a gated activation (``"silu_and_mul"``,
                ``"gelu_and_mul"`` over a stacked gate||up ``b``, ``"swigluoai"`` over an
                interleaved one) fused into the epilogue.
            static_dims: Dims compiled into the kernel, a subset of ``"mnk"``.
            m_alignment: Segment alignment of the aligned grouped layouts; it fixes ``block_m``.
            expected_m: Rows per group the cost model should plan for in the masked and
                psum layouts; the slab height, or the mean over groups, when 0. A caller
                whose routing is skewed passes the large experts' row count here.
            sm_count: Persistent grid size; the device's SM count when ``None``.
            config: Pins ``block_m``, ``block_n`` and optionally the other keys
                ``spec_from_config`` takes, instead of running the selector.
            group_starts: Whether a tight psum ``grouped_layout`` holds group starts rather
                than ends; the last group then ends at ``M``.
            device_index: Device the kernel is built for; the current one when ``None``.
        """
        super().__init__(device_index=device_index)
        if num_groups < 1:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if GEMMType(gemm_type) is GEMMType.DENSE and num_groups != 1:
            raise ValueError(f"dense requires num_groups=1, got {num_groups}")
        if m_alignment < 1:
            raise ValueError(f"m_alignment must be positive, got {m_alignment}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
        if activation != "none" and GEMMType(gemm_type) is GEMMType.K_GROUPED_CONTIGUOUS:
            raise ValueError("a fused gated activation needs a per-group B; K-grouped has one B")
        self.gemm_type = GEMMType(gemm_type)
        self.group_starts = group_starts
        self.activation = activation
        self.num_groups = num_groups
        self.cd_dtype = cd_dtype
        self.static_dims = static_dims
        self.m_alignment = m_alignment
        self.expected_m = expected_m
        self.sm_count = self.build_target.sm_count if sm_count is None else sm_count
        self._empty_layout = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # The selector picks per call; nothing to author here.
        return {}

    def init_config(self, config: dict | None = None) -> None:
        """Pin every key *config* states; an empty config leaves the choice to the selector."""
        self.config = dict(config or {})

    @property
    def _a_has_group(self) -> bool:
        return self.gemm_type in (GEMMType.M_GROUPED_MASKED, GEMMType.BATCHED)

    @property
    def _c_has_group(self) -> bool:
        return self._a_has_group or self.gemm_type is GEMMType.K_GROUPED_CONTIGUOUS

    def describe(self, a: torch.Tensor, b: torch.Tensor, has_bias: bool = False) -> GEMMDesc:
        """The selector's view of a call; see the class docstring for the shapes."""
        if self.gemm_type in (GEMMType.K_GROUPED_CONTIGUOUS, GEMMType.DENSE):
            if a.ndim != 2 or b.ndim != 2:
                k = "sum_k" if self.gemm_type is GEMMType.K_GROUPED_CONTIGUOUS else "K"
                raise ValueError(f"{self.gemm_type.value} takes A as [M, {k}] and B as [N, {k}]")
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
        return GEMMDesc(
            gemm_type=self.gemm_type,
            m=m,
            n=n,
            k=k,
            num_groups=self.num_groups,
            major_a=GEMMTemplate._major_of(a, a.ndim - 1),
            major_b=GEMMTemplate._major_of(b, b.ndim - 1),
            ab_dtype=self.dtype_to_str(a.dtype),
            cd_dtype=self.dtype_to_str(self.output_dtype(a)),
            num_sms=self.sm_count,
            static_dims=self.static_dims,
            m_alignment=self.m_alignment,
            expected_m=self.expected_m,
            activation=self.activation,
            calibration=self.build_target.calibration,
            has_bias=has_bias,
            group_starts=self.group_starts,
        )

    def spec_for(self, a: torch.Tensor, b: torch.Tensor) -> GroupedGEMMSpec:
        """The template instantiation this call runs."""
        return self._spec_of(self.config, self.describe(a, b))

    @staticmethod
    def _spec_of(cfg: dict, desc: GEMMDesc) -> GroupedGEMMSpec:
        return spec_from_config(desc, cfg) if cfg else get_best_config(desc)

    def output_dtype(self, a: torch.Tensor) -> torch.dtype:
        """The dtype ``C`` is written in: ``cd_dtype`` when set, else the operand dtype."""
        return a.dtype if self.cd_dtype is None else self.cd_dtype

    @staticmethod
    def refusal_for(a: torch.Tensor, b: torch.Tensor) -> str | None:
        """Why TMA cannot address these operands or the output, or ``None`` when it can.

        Every TMA-addressed row pitch must be a multiple of 16 bytes: the contiguous
        extent of each operand, and ``N`` for the output row. ``K`` must be positive
        unless ``b`` is 2-D.
        """
        if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
            return f"bf16 or fp16 operands of one dtype only, got {a.dtype} and {b.dtype}"
        if a.ndim < 2 or b.ndim < 2:
            return f"operands need at least two dims, got {a.ndim}-D and {b.ndim}-D"
        if a.shape[-1] == 0 and b.ndim != 2:  # a 2-D B (dense or K-grouped) may have K = 0
            return "K must be positive"
        for name, t in (("a", a), ("b", b)):
            inner = t.shape[-1] if t.stride(-1) == 1 else t.shape[-2]
            if inner % 8:
                return f"{name}'s contiguous extent {inner} is not a multiple of 8 elements"
        if b.shape[-2] % 8:
            return f"N={b.shape[-2]} is not a multiple of 8 elements (the output row pitch)"
        return None

    def _check_layout(self, desc: GEMMDesc, grouped_layout: torch.Tensor | None) -> None:
        gtype = self.gemm_type
        if gtype in (GEMMType.DENSE, GEMMType.BATCHED):
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
        if gtype is GEMMType.M_GROUPED_ALIGNED_PER_ROW and desc.m % self.m_alignment:
            raise ValueError(
                f"aligned_per_row rows ({desc.m}) must be a multiple of m_alignment "
                f"({self.m_alignment})"
            )

    def _check_bias(self, b: torch.Tensor, bias: torch.Tensor) -> None:
        n = b.shape[-2] if GEMMTemplate._major_of(b, b.ndim - 1) is Major.K else b.shape[-1]
        shape = (self.num_groups, n)
        if tuple(bias.shape) != shape or bias.dtype != b.dtype or not bias.is_contiguous():
            raise ValueError(f"bias must be a contiguous {list(shape)} {b.dtype}")

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        grouped_layout: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run dense ``C = A @ B^T`` or its grouped/batched forms.

        Each operand's physical storage (an MN-major operand's is the transpose of its
        logical view) must be contiguous, except a batched A's and C's, which may be strided
        as the class docstring states.

        Args:
            a: ``[M, K]``, ``[G, M, K]`` for masked/batched, or ``[M, sum_k]``
                for K-grouped.
            b: ``[N, K]`` for dense, ``[G, N, K]`` for grouped/batched, or
                ``[N, sum_k]`` for K-grouped.
            grouped_layout: int32 metadata per the class table; none for dense/batched.
            out: Optional preallocated output in ``cd_dtype``, sharing no byte with a tensor
                the kernel reads; contiguous, or strided as the class docstring states.
            bias: Optional ``[G, N]`` bias in ``b``'s row order and dtype; M-grouped only.
        """
        self._require_devices(a=a, b=b, grouped_layout=grouped_layout, out=out, bias=bias)
        for name, t in (("b", b), ("grouped_layout", grouped_layout), ("out", out), ("bias", bias)):
            if t is not None and t.device != a.device:
                raise ValueError(f"{name} must be on {a.device}, got {t.device}")
        why = self.refusal_for(a, b)
        if why is not None:
            raise ValueError(f"{type(self).__name__}: {why}")
        if bias is not None:
            self._check_bias(b, bias)
        desc = self.describe(a, b, bias is not None)
        self._check_layout(desc, grouped_layout)
        a_phys = a if desc.major_a is Major.K else a.transpose(-2, -1)
        b_phys = b if desc.major_b is Major.K else b.transpose(-2, -1)
        a_strides = self._declared_strides(a_phys)
        if a_strides is None and self.gemm_type is GEMMType.BATCHED:
            a_phys, a_strides = a_phys.contiguous(), ()
        if a_strides is None or not b_phys.is_contiguous():
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
            c_strides = ()
        else:
            c_strides = self._declared_strides(out)
            if (
                tuple(out.shape) != c_shape
                or out.dtype != cd_dtype
                or (c_strides is None and self.gemm_type is not GEMMType.BATCHED)
            ):
                raise ValueError(
                    f"out must be a {list(c_shape)} {cd_dtype} tensor, contiguous unless the "
                    "GEMM is batched"
                )
        if out.numel() == 0:  # no rows or no columns: nothing to launch
            return out
        if desc.k == 0:  # K-grouped with no tokens in any group: every product is zero
            return out.zero_()
        target = out
        if c_strides is None:
            target, c_strides = torch.empty(c_shape, dtype=cd_dtype, device=a.device), ()
        # The selector's tile layout follows the call's description, so it selects the program.
        fn = self.program(self._compile, desc, a_strides, c_strides)
        if self.activation in INTERLEAVED_ACTIVATIONS:
            # The kernel takes an interleaved B as [.., N / 2, 2, K] and its bias as
            # [.., N / 2, 2].
            b_phys = b_phys.unflatten(-2, (-1, 2))
            bias = None if bias is None else bias.unflatten(-1, (-1, 2))
        if bias is None:
            fn(a_phys, b_phys, target, grouped_layout)
        else:
            fn(a_phys, b_phys, target, grouped_layout, bias)
        return out if target is out else out.copy_(target)

    def _declared_strides(self, t: torch.Tensor) -> "tuple[int, ...] | None":
        """The strides the program declares for the storage *t*: none where it is contiguous,
        its own where a batched GEMM's TMA addresses them, a unit last stride and the others
        positive whole 16-byte steps, and ``None`` where neither holds."""
        if t.is_contiguous():
            return ()
        step = 16 // t.element_size()
        strides = t.stride()
        if (
            self.gemm_type is GEMMType.BATCHED
            and strides[-1] == 1
            and all(s > 0 and s % step == 0 for s in strides[:-1])
        ):
            return strides
        return None

    def _compile(self, cfg: dict, desc: GEMMDesc, a_strides: tuple, c_strides: tuple) -> Callable:
        """The program of the template instantiation *cfg* pins for *desc*, or the selector
        picks when *cfg* is empty, over A and C storage of *a_strides* and *c_strides*."""
        spec = self._spec_of(cfg, desc)
        if a_strides or c_strides:
            spec = dataclasses.replace(spec, a_strides=a_strides, c_strides=c_strides)
        return _grouped_gemm_kernel(spec)()
