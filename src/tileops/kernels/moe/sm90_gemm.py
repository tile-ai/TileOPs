"""SM90 GEMM template, a TileLang port of DeepGEMM's ``sm90_bf16_gemm_impl``.

One kernel body, specialised by ``SM90GemmSpec`` the way the CUDA template
is specialised by its parameter pack: the GEMM type, the majorness of ``A``
and ``B``, the output dtype, which dims are compiled in, the tile, the pipeline
depth, one or two math warp-groups, and TMA multicast across a 2-CTA cluster.
Python-level ``if`` on spec fields plays the role of ``if constexpr``; every
branch not taken is absent from the compiled kernel.

Structure, as in DeepGEMM:

* a persistent grid of ``num_sms`` CTAs, each walking tile ids
  ``iter * num_sms + blockIdx.x`` through a scheduler that swizzles the tile
  order in groups of 8 or 16 along one dim for L2 reuse;
* one TMA warp-group filling a ``num_stages``-deep ring of ``A``/``B`` tiles,
  and one (``block_m <= 64``) or two (``block_m >= 128``) math warp-groups
  draining it, the two splitting the tile's rows;
* one WGMMA group stays in flight: a k-step drains the previous one with
  ``wgmma.wait_group 1`` and releases that stage, arriving at both CTAs of the
  cluster when multicasting;
* the epilogue casts to the output dtype, stages through shared memory and
  TMA-stores the tile, so ragged edges are clipped by the descriptor.

The GEMM types fall into three scheduler families:

* ``flat``: ``NORMAL`` and ``M_GROUPED_ALIGNED_PER_ROW`` enumerate the
  ``M x N`` tile grid once; the grouped one reads a tile's group off its first
  row.
* ``batched``: ``BATCHED`` repeats the tile grid per batch.
* ``per_group``: ``M_GROUPED_MASKED``, ``M_GROUPED_ALIGNED_PSUM`` and
  ``M_GROUPED_TIGHT_PSUM`` enumerate tiles group by group from a per-group
  row count, through a tile-count prefix sum built in shared memory at
  kernel start. Only the tight variant can end a group mid-tile, so only it
  masks the store of a group's last tile.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_count

from .sm90_gemm_heuristics import (
    PER_GROUP_TYPES,
    GemmDesc,
    GemmType,
    Major,
    SM90GemmSpec,
    get_best_config,
    spec_from_config,
)

__all__ = [
    "GemmDesc",
    "GemmType",
    "Major",
    "SM90GemmFwdKernel",
    "SM90GemmSpec",
]

# Register budgets after `setmaxnreg`, DeepGEMM's numbers.
_TMA_REGS = 48
_MATH_REGS_ONE_WG = 248
_MATH_REGS_TWO_WG = 224
# Named barriers private to each math warp-group's epilogue.
_EPILOGUE_BARRIER_BASE = 8
# Both CTAs of the cluster; TMA multicast is only ever 2-wide on SM90.
_CLUSTER_MASK = 0b11


def _num_1d_blocks_per_group(block_m: int, block_n: int, num_sms: int, multicast_on_a: bool) -> int:
    """How many tiles along the primary dim share one L2 group; DeepGEMM's rule."""
    best, best_usage = 0, None
    for candidate in (8, 16):
        if multicast_on_a:
            usage = candidate * block_n + -(-num_sms // candidate) * block_m
        else:
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
    num_multicast: int,
    multicast_on_a: bool,
    num_sms: int,
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
    blocks_per_group = _num_1d_blocks_per_group(block_m, block_n, num_sms, multicast_on_a)
    math_regs = _MATH_REGS_ONE_WG if num_math_wgs == 1 else _MATH_REGS_TWO_WG
    cast_output = cd_dtype != "float32"

    # Scheduler family and what each operand looks like.
    contiguous = gtype is GemmType.M_GROUPED_ALIGNED_PER_ROW
    masked = gtype is GemmType.M_GROUPED_MASKED
    aligned_psum = gtype is GemmType.M_GROUPED_ALIGNED_PSUM
    tight_psum = gtype is GemmType.M_GROUPED_TIGHT_PSUM
    batched = gtype is GemmType.BATCHED
    per_group = gtype in PER_GROUP_TYPES
    b_has_group = gtype is not GemmType.NORMAL
    a_has_group = masked or batched  # A and C carry a leading group dim
    search_steps = max(1, (num_groups - 1).bit_length())

    m = T.dynamic("m") if shape_m == 0 else shape_m
    n = T.dynamic("n") if shape_n == 0 else shape_n
    k = T.dynamic("k") if shape_k == 0 else shape_k

    a_2d = (m, k) if a_k_major else (k, m)
    a_shape = (num_groups,) + a_2d if a_has_group else a_2d
    b_2d = (n, k) if b_k_major else (k, n)
    b_shape = (num_groups,) + b_2d if b_has_group else b_2d
    c_shape = (num_groups, m, n) if a_has_group else (m, n)
    if contiguous:
        layout_shape = (m,)
    elif per_group:
        layout_shape = (num_groups,)
    else:
        layout_shape = (1,)
    a_tile = (wg_rows, block_k) if a_k_major else (block_k, wg_rows)
    b_tile = (block_n, block_k) if b_k_major else (block_k, block_n)

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

    def align_up(x):
        return ((x + T.int32(block_m - 1)) // T.int32(block_m)) * T.int32(block_m)

    @T.macro
    def swizzle_block(block_idx, num_m_blocks, num_n_blocks, out_m, out_n, out_in_group):
        """Tile id -> (m block, n block, tiles in this L2 group), DeepGEMM's swizzle.

        Groups ``blocks_per_group`` tiles along the primary dim (N when
        multicasting A, M otherwise) so the CTAs of one wave share the other
        operand in L2. With a cluster, an odd-sized group is split so the
        peer of every multicast pair is in the same group.
        """
        primary = num_n_blocks if multicast_on_a else num_m_blocks
        secondary = num_m_blocks if multicast_on_a else num_n_blocks
        num_blocks_per_group = secondary * T.int32(blocks_per_group)
        group_idx = block_idx // num_blocks_per_group
        first = T.alloc_var("int32", init=group_idx * T.int32(blocks_per_group))
        in_group = T.alloc_var("int32", init=block_idx % num_blocks_per_group)
        in_group_blocks = T.alloc_var(
            "int32", init=T.min(T.int32(blocks_per_group), primary - first)
        )
        if num_multicast > 1:  # noqa: SIM102 -- compile-time guard around a runtime test
            if in_group_blocks % 2 != 0:
                if in_group < (in_group_blocks ^ 1) * secondary:
                    in_group_blocks = in_group_blocks ^ 1
                else:
                    in_group = in_group - (in_group_blocks ^ 1) * secondary
                    first = first + (in_group_blocks ^ 1)
                    in_group_blocks = T.int32(1)
        if multicast_on_a:
            out_m[0] = in_group // in_group_blocks
            out_n[0] = first + in_group % in_group_blocks
        else:
            out_m[0] = first + in_group % in_group_blocks
            out_n[0] = in_group // in_group_blocks
        out_in_group[0] = in_group_blocks

    @T.macro
    def tile_cumsum(grouped_layout, s_cum, s_total, num_n_blocks):
        """Per-group tile-count prefix sum; ``s_total`` is the tile count of the call.

        Masked groups are ``masked_m[g]`` rows from row 0 of their slab; psum
        groups run from the previous end (aligned up for the aligned variant)
        to ``psum[g]``.
        """
        s_cum[0] = T.int32(0)
        prev_end = T.alloc_var("int32", init=0)
        for g in T.serial(num_groups):
            if masked:
                rows = T.max(T.min(grouped_layout[g], m), T.int32(0))
            elif aligned_psum:
                rows = T.max(grouped_layout[g] - align_up(prev_end), T.int32(0))
            else:
                rows = T.max(grouped_layout[g] - prev_end, T.int32(0))
            s_cum[g + 1] = s_cum[g] + (rows + T.int32(block_m - 1)) // T.int32(block_m)
            prev_end = grouped_layout[g]
        s_total[0] = s_cum[num_groups] * num_n_blocks

    @T.macro
    def resolve_tile(
        block_idx,
        grouped_layout,
        s_cum,
        num_m_blocks,
        num_n_blocks,
        t_group,
        t_row0,
        t_col0,
        t_rows,
        t_in_group,
    ):
        """Tile id -> group, first row, first column, valid rows, L2-group size.

        ``t_row0`` indexes the A/C slab the tile belongs to: the whole tensor
        for the flat and psum families, the group's slab for masked and
        batched. ``t_rows`` is ``block_m`` except on a tight group's last tile.
        """
        m_blk = T.alloc_local((1,), "int32")
        n_blk = T.alloc_local((1,), "int32")
        if batched:
            per_batch = num_m_blocks * num_n_blocks
            rem = block_idx % per_batch
            t_group[0] = block_idx // per_batch
            if multicast_on_a:
                m_blk[0] = rem // num_n_blocks
                n_blk[0] = rem % num_n_blocks
            else:
                m_blk[0] = rem % num_m_blocks
                n_blk[0] = rem // num_m_blocks
            t_in_group[0] = T.int32(1)
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
            swizzle_block(rel, s_cum[lo + 1] - s_cum[lo], num_n_blocks, m_blk, n_blk, t_in_group)
            if masked:
                t_row0[0] = m_blk[0] * T.int32(block_m)
                t_rows[0] = T.int32(block_m)
            else:
                prev_end = T.if_then_else(lo > 0, grouped_layout[T.max(lo - 1, 0)], T.int32(0))
                if aligned_psum:
                    t_row0[0] = align_up(prev_end) + m_blk[0] * T.int32(block_m)
                    t_rows[0] = T.int32(block_m)
                else:
                    t_row0[0] = prev_end + m_blk[0] * T.int32(block_m)
                    t_rows[0] = T.min(T.int32(block_m), grouped_layout[lo] - t_row0[0])
        else:
            swizzle_block(block_idx, num_m_blocks, num_n_blocks, m_blk, n_blk, t_in_group)
            t_row0[0] = m_blk[0] * T.int32(block_m)
            t_rows[0] = T.int32(block_m)
            if contiguous:
                # Rows past the last group carry `num_groups`; clamp so a tile of
                # padding reads a real B and its rows are ignored.
                t_group[0] = T.min(
                    T.max(grouped_layout[t_row0[0]], T.int32(0)), T.int32(num_groups - 1)
                )
            else:
                t_group[0] = T.int32(0)
        t_col0[0] = n_blk[0] * T.int32(block_n)

    @T.macro
    def load_a(A, dst, full, slot, group, row0, k0, use_mc):
        if multicast_on_a:
            if use_mc != 0:
                T.tma_copy(
                    a_region(A, group, row0, k0),
                    dst[slot, :, :],
                    barrier=full[slot],
                    annotations={"cluster_mask": _CLUSTER_MASK},
                )
            else:
                T.tma_copy(a_region(A, group, row0, k0), dst[slot, :, :], barrier=full[slot])
        else:
            T.tma_copy(a_region(A, group, row0, k0), dst[slot, :, :], barrier=full[slot])

    @T.macro
    def load_b(B, dst, full, slot, group, n0, k0, use_mc):
        if num_multicast > 1 and not multicast_on_a:
            if use_mc != 0:
                T.tma_copy(
                    b_region(B, group, n0, k0),
                    dst[slot, :, :],
                    barrier=full[slot],
                    annotations={"cluster_mask": _CLUSTER_MASK},
                )
            else:
                T.tma_copy(b_region(B, group, n0, k0), dst[slot, :, :], barrier=full[slot])
        else:
            T.tma_copy(b_region(B, group, n0, k0), dst[slot, :, :], barrier=full[slot])

    @T.macro
    def store_tile(C, C_src, C_s, group, row0, col0, rows, wg):
        """Store one warp-group's rows of a tile from its shared staging buffer.

        Every tile is staged fragment -> shared first; the previous tile's TMA store
        may still be reading ``C_s``, hence the barrier before. A full tile then
        goes out through TMA. A tight group's ragged last tile cannot: TMA clips
        against the tensor, not the group, and would overwrite the next group's
        rows, so its valid rows are written back with row-predicated vector stores.
        (Writing the accumulator fragment straight from registers scattered 4-byte
        stores across the tile and cost 12% to 24% on short-K grouped GEMMs.)
        """
        T.sync_threads(barrier_id=_EPILOGUE_BARRIER_BASE + wg, arrive_count=128)
        T.copy(C_src, C_s)
        if tight_psum:
            if rows < T.int32(wg_rows):
                T.sync_threads(barrier_id=_EPILOGUE_BARRIER_BASE + wg, arrive_count=128)
                if rows > 0:
                    for i, j in T.Parallel(wg_rows, block_n):
                        if i < rows:
                            C[row0 + i, col0 + j] = C_s[i, j]
            else:
                T.fence_proxy_async()
                T.sync_threads(barrier_id=_EPILOGUE_BARRIER_BASE + wg, arrive_count=128)
                T.copy(C_s, C[row0, col0])
        else:
            T.fence_proxy_async()
            T.sync_threads(barrier_id=_EPILOGUE_BARRIER_BASE + wg, arrive_count=128)
            if a_has_group:
                T.copy(C_s, C[group, row0, col0])
            else:
                T.copy(C_s, C[row0, col0])

    @T.macro
    def release_stage(empty, slot, lane, rank, peer_alive):
        """Lane 0 of each math warp arrives on the stage's empty barrier, at both CTAs of a cluster."""
        if num_multicast > 1:
            if lane < num_multicast:
                T.mbarrier_arrive(empty[slot], T.if_then_else(peer_alive, lane, rank))
        else:
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
        s_cum,
        C_l,
        C_cast,
        C_s,
        wg,
        pid,
        num_m_blocks,
        num_n_blocks,
        num_blocks,
        num_waves,
        k_iters,
    ):
        """One math warp-group: drain the ring over K for each tile, then store."""
        T.inc_max_nreg(math_regs)
        tx = T.get_thread_binding()
        lane = tx % 32
        rank = T.block_rank_in_cluster() if num_multicast > 1 else T.int32(0)
        gi = T.alloc_var("int32", init=0)
        prev_slot = T.alloc_local((1,), "int32")
        t_group = T.alloc_local((1,), "int32")
        t_row0 = T.alloc_local((1,), "int32")
        t_col0 = T.alloc_local((1,), "int32")
        t_rows = T.alloc_local((1,), "int32")
        t_in_group = T.alloc_local((1,), "int32")

        for w in T.serial(num_waves):
            block_idx = w * T.int32(num_sms) + pid
            if block_idx < num_blocks:
                resolve_tile(
                    block_idx,
                    grouped_layout,
                    s_cum,
                    num_m_blocks,
                    num_n_blocks,
                    t_group,
                    t_row0,
                    t_col0,
                    t_rows,
                    t_in_group,
                )
                peer_alive = (
                    (num_n_blocks % T.int32(num_multicast) == 0)
                    | (num_m_blocks % T.int32(num_multicast) == 0)
                    | ((block_idx ^ 1) < num_blocks)
                )
                for ki in T.serial(k_iters):
                    slot = gi % num_stages
                    phase = (gi // num_stages) & 1
                    T.barrier_wait(full[slot], phase)
                    T.wgmma_gemm(
                        A_s[slot, :, :],
                        B_s[slot, :, :],
                        C_l,
                        transpose_A=not a_k_major,
                        transpose_B=b_k_major,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=(ki == 0),
                    )
                    # One WGMMA group stays in flight: drain the previous k-step
                    # and release its stage while this one runs. DeepGEMM drains
                    # every step (`warpgroup_wait<0>`) and merges stages instead,
                    # which it only does for a dense single-warp-group NT GEMM;
                    # the 1-deep pipeline is what TileOPs' grouped kernels use
                    # and is worth 4% to 7% on Mixtral-shaped grouped GEMMs here.
                    if ki > 0:
                        T.wait_wgmma(1)
                        release_stage(empty, prev_slot[0], lane, rank, peer_alive)
                    prev_slot[0] = slot
                    gi = gi + 1
                T.wait_wgmma(0)
                release_stage(empty, prev_slot[0], lane, rank, peer_alive)
                T.warpgroup_fence_operand(C_l, num_regs=(wg_rows * block_n) // 128)

                row0 = t_row0[0] + T.int32(wg * wg_rows)
                rows = t_rows[0] - T.int32(wg * wg_rows)
                if cast_output:
                    T.copy(C_l, C_cast)
                    store_tile(C, C_cast, C_s, t_group[0], row0, t_col0[0], rows, wg)
                else:
                    store_tile(C, C_l, C_s, t_group[0], row0, t_col0[0], rows, wg)

    @T.prim_func
    def sm90_gemm(
        A: T.Tensor(a_shape, dtype),  # type: ignore
        B: T.Tensor(b_shape, dtype),  # type: ignore
        C: T.Tensor(c_shape, cd_dtype),  # type: ignore
        grouped_layout: T.Tensor(layout_shape, "int32"),  # type: ignore
    ):
        with T.ClusterKernel(num_sms, threads=threads, cluster_dims=(num_multicast, 1, 1)) as (
            pid,
        ):
            A_s0 = T.alloc_shared((num_stages,) + a_tile, dtype)
            if num_math_wgs > 1:
                A_s1 = T.alloc_shared((num_stages,) + a_tile, dtype)
            B_s = T.alloc_shared((num_stages,) + b_tile, dtype)
            C_l0 = T.alloc_fragment((wg_rows, block_n), accum_dtype)
            C_cast0 = T.alloc_fragment((wg_rows, block_n), cd_dtype)
            C_s0 = T.alloc_shared((wg_rows, block_n), cd_dtype)
            if num_math_wgs > 1:
                C_l1 = T.alloc_fragment((wg_rows, block_n), accum_dtype)
                C_cast1 = T.alloc_fragment((wg_rows, block_n), cd_dtype)
                C_s1 = T.alloc_shared((wg_rows, block_n), cd_dtype)
            # Per-group tile prefix sum and the call's tile count (per_group family).
            s_cum = T.alloc_shared((num_groups + 1,), "int32")
            s_total = T.alloc_shared((1,), "int32")

            if num_math_wgs > 1:
                T.annotate_layout(
                    {
                        A_s0: tilelang.layout.make_swizzled_layout(A_s0),
                        A_s1: tilelang.layout.make_swizzled_layout(A_s1),
                        B_s: tilelang.layout.make_swizzled_layout(B_s),
                        C_s0: tilelang.layout.make_swizzled_layout(C_s0),
                        C_s1: tilelang.layout.make_swizzled_layout(C_s1),
                    }
                )
            else:
                T.annotate_layout(
                    {
                        A_s0: tilelang.layout.make_swizzled_layout(A_s0),
                        B_s: tilelang.layout.make_swizzled_layout(B_s),
                        C_s0: tilelang.layout.make_swizzled_layout(C_s0),
                    }
                )

            # full: the TMA warp-group arrives once per thread after issuing a
            # stage. empty: lane 0 of every math warp arrives, at both CTAs of
            # the cluster when multicasting (DeepGEMM's arrive counts).
            full = T.alloc_barrier([tma_threads] * num_stages)
            if num_multicast > 1:
                empty = T.alloc_cluster_barrier([math_warps * num_multicast] * num_stages)
            else:
                empty = T.alloc_barrier([math_warps] * num_stages)

            num_m_blocks = T.ceildiv(m, block_m)
            num_n_blocks = T.ceildiv(n, block_n)
            k_iters = T.ceildiv(k, block_k)

            tx = T.get_thread_binding()

            if per_group:  # noqa: SIM102 -- compile-time guard around a runtime test
                if tx == 0:
                    tile_cumsum(grouped_layout, s_cum, s_total, num_n_blocks)

            if num_multicast > 1:
                # Barrier init must be visible cluster-wide before any remote arrive.
                T.cluster_sync()
            else:
                T.sync_threads()

            if per_group:
                num_blocks = s_total[0]
            elif batched:
                num_blocks = num_m_blocks * num_n_blocks * T.int32(num_groups)
            else:
                num_blocks = num_m_blocks * num_n_blocks
            num_waves = T.ceildiv(num_blocks, num_sms)

            if tx < tma_threads:
                # ── TMA warp-group ──
                T.dec_max_nreg(_TMA_REGS)
                gi = T.alloc_var("int32", init=0)
                t_group = T.alloc_local((1,), "int32")
                t_row0 = T.alloc_local((1,), "int32")
                t_col0 = T.alloc_local((1,), "int32")
                t_rows = T.alloc_local((1,), "int32")
                t_in_group = T.alloc_local((1,), "int32")
                use_mc = T.alloc_var("int32", init=1)

                for w in T.serial(num_waves):
                    block_idx = w * T.int32(num_sms) + pid
                    if block_idx < num_blocks:
                        resolve_tile(
                            block_idx,
                            grouped_layout,
                            s_cum,
                            num_m_blocks,
                            num_n_blocks,
                            t_group,
                            t_row0,
                            t_col0,
                            t_rows,
                            t_in_group,
                        )
                        if num_multicast > 1:
                            if contiguous and not multicast_on_a:
                                # The peer's tile must read the same B.
                                peer_row = (t_row0[0] // T.int32(block_m) ^ 1) * T.int32(block_m)
                                use_mc = T.if_then_else(
                                    (t_in_group[0] != 1)
                                    & (grouped_layout[t_row0[0]] == grouped_layout[peer_row]),
                                    T.int32(1),
                                    T.int32(0),
                                )
                            else:
                                use_mc = T.if_then_else(t_in_group[0] != 1, T.int32(1), T.int32(0))
                        for ki in T.serial(k_iters):
                            slot = gi % num_stages
                            phase = (gi // num_stages) & 1
                            k0 = ki * T.int32(block_k)
                            T.barrier_wait(empty[slot], phase ^ 1)
                            load_a(A, A_s0, full, slot, t_group[0], t_row0[0], k0, use_mc)
                            if num_math_wgs > 1:
                                load_a(
                                    A,
                                    A_s1,
                                    full,
                                    slot,
                                    t_group[0],
                                    t_row0[0] + T.int32(wg_rows),
                                    k0,
                                    use_mc,
                                )
                            load_b(B, B_s, full, slot, t_group[0], t_col0[0], k0, use_mc)
                            T.barrier_arrive(full[slot])
                            gi = gi + 1
                if num_multicast > 1:
                    # Let every remote empty-arrive land before this CTA exits.
                    for i in T.serial(num_stages):
                        slot = (gi + i) % num_stages
                        phase = ((gi + i) // num_stages) & 1
                        T.barrier_wait(empty[slot], phase ^ 1)
            elif tx < tma_threads + 128:
                math_warpgroup(
                    A_s0,
                    B_s,
                    C,
                    full,
                    empty,
                    grouped_layout,
                    s_cum,
                    C_l0,
                    C_cast0,
                    C_s0,
                    0,
                    pid,
                    num_m_blocks,
                    num_n_blocks,
                    num_blocks,
                    num_waves,
                    k_iters,
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
                        s_cum,
                        C_l1,
                        C_cast1,
                        C_s1,
                        1,
                        pid,
                        num_m_blocks,
                        num_n_blocks,
                        num_blocks,
                        num_waves,
                        k_iters,
                    )

    return sm90_gemm


# maxsize=None: an entry is one compiled kernel, so evicting it would only
# force a recompile. The key space is the specs the selector emits, a few
# hundred layouts per (gemm_type, majors, dtypes, static n and k), plus any
# explicit config a caller pins.
@functools.lru_cache(maxsize=None)
def _sm90_gemm_kernel(spec: SM90GemmSpec):
    """JIT factory for one ``spec``; ``_sm90_gemm_kernel(spec)()`` compiles it.

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
    num_multicast = spec.num_tma_multicast
    multicast_on_a = spec.is_tma_multicast_on_a
    num_sms = spec.num_sms

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
            num_multicast,
            multicast_on_a,
            num_sms,
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


class SM90GemmFwdKernel(Kernel):
    """DeepGEMM-style 16-bit GEMM on Hopper: dense, batched, or m-grouped.

    ``C = A @ B^T`` on bf16 or fp16 operands with fp32 accumulation and a ``C``
    in the operand dtype or fp32. Any of the four layouts is accepted and read off the operands'
    strides. Operand shapes per ``gemm_type`` (logical; an MN-major operand is
    a transposed view):

    | ``gemm_type``               | ``a``            | ``b``          | ``c``            | ``grouped_layout``           |
    | --------------------------- | ---------------- | -------------- | ---------------- | ---------------------------- |
    | ``NORMAL``                  | ``[M, K]``       | ``[N, K]``     | ``[M, N]``       | none                         |
    | ``M_GROUPED_ALIGNED_PER_ROW`` | ``[M, K]``     | ``[G, N, K]``  | ``[M, N]``       | ``[M]`` group of each row    |
    | ``M_GROUPED_ALIGNED_PSUM``  | ``[M, K]``       | ``[G, N, K]``  | ``[M, N]``       | ``[G]`` psum row ends        |
    | ``M_GROUPED_TIGHT_PSUM``    | ``[M, K]``       | ``[G, N, K]``  | ``[M, N]``       | ``[G]`` psum row ends        |
    | ``M_GROUPED_MASKED``        | ``[G, max_m, K]``| ``[G, N, K]``  | ``[G, max_m, N]``| ``[G]`` valid rows per group |
    | ``BATCHED``                 | ``[G, M, K]``    | ``[G, N, K]``  | ``[G, M, N]``    | none                         |

    The kernel is selected per call by DeepGEMM's cost model over the legal
    tile/cluster layouts, so a config never has to be authored; ``config``
    pins one for tuning. Dims not named in ``static_dims`` stay dynamic, and a
    call with a new value of a dynamic dim reuses the compiled kernel.

    Example:
        ```python linenums="1"
        kernel = SM90GemmFwdKernel(GemmType.NORMAL)
        c = kernel(a, b)                       # a: [M, K], b: [N, K], both bf16
        kernel = SM90GemmFwdKernel(GemmType.M_GROUPED_TIGHT_PSUM, num_groups=E)
        c = kernel(a, b, grouped_layout=ends)  # b: [E, N, K], ends: [E] int32
        ```
    """

    supported_archs: list[int] = [90]

    def __init__(
        self,
        gemm_type: GemmType = GemmType.NORMAL,
        *,
        num_groups: int = 1,
        cd_dtype: Optional[torch.dtype] = None,
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
            num_groups: Groups, experts or batches; 1 for ``NORMAL``.
            cd_dtype: Output dtype: the operand dtype (default) or ``torch.float32``.
            static_dims: Dims compiled into the kernel, a subset of ``"mnk"``.
            m_alignment: Segment alignment of the aligned grouped layouts; it fixes ``block_m``.
            expected_m: Rows per group the cost model should plan for in the masked and
                psum layouts; the slab height, or the mean over groups, when 0. A caller
                whose routing is skewed passes the large experts' row count here.
            sm_count: Persistent grid size; the device's SM count when ``None``.
            config: Pins ``block_m``, ``block_n`` and optionally ``block_k``, ``num_stages``,
                ``cluster_m``, ``cluster_n`` instead of running the selector.
            tune: Kernel-protocol flag; this kernel has no autotune space, the selector
                stands in for it, so ``True`` only warns.
            device_index: Device the kernel is built for; the current one when ``None``.
        """
        super().__init__(device_index=device_index)
        self.gemm_type = GemmType(gemm_type)
        self.num_groups = num_groups
        self.cd_dtype = cd_dtype
        self.static_dims = static_dims
        self.m_alignment = m_alignment
        self.expected_m = expected_m
        self.sm_count = get_sm_count(device_index) if sm_count is None else sm_count
        self.explicit_config = config
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # The selector picks per call; nothing to author here.
        return {}

    @property
    def _a_has_group(self) -> bool:
        return self.gemm_type in (GemmType.M_GROUPED_MASKED, GemmType.BATCHED)

    def describe(self, a: torch.Tensor, b: torch.Tensor) -> GemmDesc:
        """The selector's view of a call; see the class docstring for the shapes."""
        a_ndim = 3 if self._a_has_group else 2
        if a.ndim != a_ndim:
            raise ValueError(f"{self.gemm_type.value} takes a {a_ndim}-D A, got {a.ndim}-D")
        if self.gemm_type is GemmType.NORMAL:
            if b.ndim != 2:
                raise ValueError("a normal GEMM takes a 2-D B")
        elif b.ndim != 3 or b.shape[0] != self.num_groups:
            raise ValueError(f"{self.gemm_type.value} takes B as [{self.num_groups}, N, K]")
        if self._a_has_group and a.shape[0] != self.num_groups:
            raise ValueError(f"{self.gemm_type.value} takes A as [{self.num_groups}, M, K]")
        m, k = a.shape[-2], a.shape[-1]
        n, k_b = b.shape[-2], b.shape[-1]
        if k != k_b:
            raise ValueError(f"A and B disagree on K: {k} vs {k_b}")
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
        )

    def spec_for(self, a: torch.Tensor, b: torch.Tensor) -> SM90GemmSpec:
        """The template instantiation this call runs."""
        return self._spec_of(self.describe(a, b))

    def _spec_of(self, desc: GemmDesc) -> SM90GemmSpec:
        if self.explicit_config:
            return spec_from_config(desc, self.explicit_config)
        return get_best_config(desc)

    def output_dtype(self, a: torch.Tensor) -> torch.dtype:
        """The dtype ``C`` is written in: ``cd_dtype`` when set, else the operand dtype."""
        return a.dtype if self.cd_dtype is None else self.cd_dtype

    @staticmethod
    def refusal_for(a: torch.Tensor, b: torch.Tensor) -> Optional[str]:
        """Why TMA cannot address these operands, or ``None`` when it can."""
        if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
            return f"bf16 or fp16 operands of one dtype only, got {a.dtype} and {b.dtype}"
        for name, t in (("a", a), ("b", b)):
            inner = t.shape[-1] if t.stride(-1) == 1 else t.shape[-2]
            if inner % 8:
                return f"{name}'s contiguous extent {inner} is not a multiple of 8 elements"
        return None

    def _check_layout(self, desc: GemmDesc, grouped_layout: Optional[torch.Tensor]) -> None:
        gtype = self.gemm_type
        if gtype in (GemmType.NORMAL, GemmType.BATCHED):
            if grouped_layout is not None:
                raise ValueError(f"{gtype.value} takes no grouped_layout")
            return
        if grouped_layout is None:
            raise ValueError(f"{gtype.value} needs grouped_layout")
        length = desc.m if gtype is GemmType.M_GROUPED_ALIGNED_PER_ROW else self.num_groups
        if grouped_layout.dtype != torch.int32 or grouped_layout.shape != (length,):
            raise ValueError(f"grouped_layout must be [{length}] int32")
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
        """Run ``C = A @ B^T`` (per group or batch when grouped) and return ``C``.

        Args:
            a: Logical ``[M, K]`` bf16 or fp16, ``[G, M, K]`` for masked and batched; a
                transposed view of contiguous storage is MN-major.
            b: Logical ``[N, K]`` in ``a``'s dtype, with a leading ``G`` dim for every type but
                ``NORMAL``; a transposed view of ``[K, N]`` storage is MN-major.
            grouped_layout: int32 layout metadata per the class table; ``None`` for
                ``NORMAL`` and ``BATCHED``.
            out: Optional preallocated output in ``cd_dtype``.
        """
        self._require_cuda(a=a, b=b, grouped_layout=grouped_layout, out=out)
        why = self.refusal_for(a, b)
        if why is not None:
            raise ValueError(f"{type(self).__name__}: {why}")
        desc = self.describe(a, b)
        self._check_layout(desc, grouped_layout)
        if grouped_layout is None:
            grouped_layout = torch.zeros(1, dtype=torch.int32, device=a.device)
        c_shape = (self.num_groups, desc.m, desc.n) if self._a_has_group else (desc.m, desc.n)
        cd_dtype = self.output_dtype(a)
        if out is None:
            out = torch.empty(c_shape, dtype=cd_dtype, device=a.device)
        elif tuple(out.shape) != c_shape or out.dtype != cd_dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {list(c_shape)} {cd_dtype}")
        spec = self._spec_of(desc)
        # The kernel takes the physical storage: an MN-major operand's
        # contiguous tensor is the transpose of its logical view.
        a_phys = a if desc.major_a is Major.K else a.transpose(-2, -1)
        b_phys = b if desc.major_b is Major.K else b.transpose(-2, -1)
        fn = _sm90_gemm_kernel(spec)()
        fn(a_phys, b_phys, out, grouped_layout)
        return out
