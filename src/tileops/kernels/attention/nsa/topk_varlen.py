import functools
import math
from typing import Callable, Optional

import tilelang
import torch
from tilelang import language as T

from tileops.kernels.attention.call_spec import NSACall, NSATopKFwdInterface
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel

# One warpgroup's WGMMA rows: a CTA's (token, query head) rows are a multiple of this.
_TILE_ROWS = 64
_WARP = 32


def _sort_widths(chunks: int) -> list[int]:
    """Slots per lane of each sort width, 32 lanes times a power of two, up to ``chunks``."""
    widths = [1]
    while _WARP * widths[-1] < chunks:
        widths.append(widths[-1] * 2)
    return widths


@functools.lru_cache(maxsize=32)
def _nsa_topk_varlen_kernel(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    chunk_num: int,
    group: int,
    scale: float,
    selected_block_num: int,
    block_n: int,
    bs: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    """Build the selection kernel for one call shape.

    A CTA takes ``block_t`` tokens of one compressed block of one request and one KV head:
    every token of a block has the same candidates, the chunks of its request up to that
    block. It stages their ``group`` query heads, scaled in the input dtype as FLA does, as
    one K-major tile and streams the request's compressed keys in ``block_n``-chunk tiles
    twice: pass 1 keeps each row's online log-sum-exp over the closed chunks, pass 2 sums
    the open chunks' probabilities over the group in registers into one importance row per
    token. Each warp then sorts whole rows best first, ties to the lower chunk id, and
    writes the first ``selected_block_num`` ids, -1 past the candidates.
    """
    head_kv = heads // group
    # The fewest tokens whose rows fill whole WGMMA row tiles.
    block_t = _TILE_ROWS // math.gcd(group, _TILE_ROWS)
    rows = block_t * group
    split = (bs + block_t - 1) // block_t
    threads = 128
    warps = threads // _WARP
    # A warp takes whole rows; with fewer rows than warps the spare warps take none.
    rows_per_warp = max(1, block_t // warps)
    # The deepest candidate window: every chunk of the longest request.
    n_max = (chunk_num + block_n - 1) // block_n * block_n
    search = max(1, seq_num.bit_length())

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            # Each selected slot has one writer: the sort gives every candidate one rank.
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        },
    )
    def _nsa_topk_varlen_func():
        widths = _sort_widths(n_max)
        slots = widths[-1]
        # (exclusive floor, slots per lane) of the sort width each candidate count selects.
        bands = [(0 if i == 0 else _WARP * widths[i - 1], w) for i, w in enumerate(widths)]
        neg_inf = -T.infinity(accum_dtype)

        def bitonic(slots_per_lane, vals, keys, lane, tv, tk, flag):
            """Sort 32 * slots_per_lane (importance, id) pairs best first in one warp.

            Element i lives in lane i % 32, slot i // 32. Stage k merges runs of length k with
            strides k / 2 ... 1: a stride under 32 pairs lanes through a shuffle, a wider one
            pairs slots inside the lane. Every operand is staged in a scratch local before a
            slot is overwritten. Plain Python, so the per-stride slot pattern unrolls while the
            stage loops stay loops.
            """
            levels = (_WARP * slots_per_lane).bit_length() - 1
            with T.serial(levels) as level:
                k = 2 << level
                with T.serial(level + 1) as down:
                    stride_log = level - down
                    with T.If(stride_log < 5):
                        with T.Then():
                            j = 1 << stride_log
                            for e in range(slots_per_lane):
                                descending = ((e * _WARP + lane) & k) == 0
                                T.buffer_store(tv, T.shfl_xor(vals[e], j), [0])
                                T.buffer_store(tk, T.shfl_xor(keys[e], j), [0])
                                other_better = T.Or(
                                    tv[0] > vals[e], T.And(tv[0] == vals[e], tk[0] < keys[e])
                                )
                                lower = (lane & j) == 0
                                T.buffer_store(
                                    flag,
                                    T.Cast("int32", other_better == (lower == descending)),
                                    [0],
                                )
                                T.buffer_store(
                                    vals, T.if_then_else(flag[0] != 0, tv[0], vals[e]), [e]
                                )
                                T.buffer_store(
                                    keys, T.if_then_else(flag[0] != 0, tk[0], keys[e]), [e]
                                )
                        with T.Else():
                            for wide in range(5, levels):
                                slot_stride = 1 << (wide - 5)
                                with T.If(stride_log == wide), T.Then():
                                    for e in range(slots_per_lane):
                                        if e & slot_stride:
                                            continue
                                        f = e | slot_stride
                                        descending = ((e * _WARP + lane) & k) == 0
                                        T.buffer_store(tv, vals[e], [0])
                                        T.buffer_store(tk, keys[e], [0])
                                        T.buffer_store(tv, vals[f], [1])
                                        T.buffer_store(tk, keys[f], [1])
                                        second_better = T.Or(
                                            tv[1] > tv[0], T.And(tv[1] == tv[0], tk[1] < tk[0])
                                        )
                                        T.buffer_store(
                                            flag, T.Cast("int32", second_better == descending), [0]
                                        )
                                        T.buffer_store(
                                            vals, T.if_then_else(flag[0] != 0, tv[1], tv[0]), [e]
                                        )
                                        T.buffer_store(
                                            keys, T.if_then_else(flag[0] != 0, tk[1], tk[0]), [e]
                                        )
                                        T.buffer_store(
                                            vals, T.if_then_else(flag[0] != 0, tv[0], tv[1]), [f]
                                        )
                                        T.buffer_store(
                                            keys, T.if_then_else(flag[0] != 0, tk[0], tk[1]), [f]
                                        )

        def select_row(t, n_cand, tok0, h, imp_s, vals, keys, lane, tv, tk, flag, block_indices):
            """Sort row t at the narrowest width holding n_cand and write its first ids."""
            for floor, width in bands:
                with T.If(T.And(n_cand > floor, n_cand <= _WARP * width)), T.Then():
                    for e in range(width):
                        i = e * _WARP + lane
                        T.buffer_store(vals, T.if_then_else(i < n_cand, imp_s[t, i], neg_inf), [e])
                        T.buffer_store(keys, i, [e])
                    bitonic(width, vals, keys, lane, tv, tk, flag)
                    for e in range(width):
                        if e * _WARP < selected_block_num:
                            p = e * _WARP + lane
                            with T.If(p < selected_block_num), T.Then():
                                T.buffer_store(
                                    block_indices,
                                    T.if_then_else(p < n_cand, keys[e], -1),
                                    [tok0 + t, h, p],
                                )

        @T.prim_func
        def _nsa_topk_varlen_main(
            q: T.Tensor((c_seq_len, heads, dim), dtype),
            k_cmp: T.Tensor((chunk_num, head_kv, dim), dtype),
            offsets: T.Tensor((seq_num + 1,), T.int32),
            chunk_offsets: T.Tensor((seq_num + 1,), T.int32),
            token_indices: T.Tensor((c_seq_len, 2), T.int32),
            block_indices: T.Tensor((c_seq_len, head_kv, selected_block_num), T.int32),
        ):
            with T.Kernel(chunk_num * split, head_kv, threads=threads) as (bx, by):
                q_s = T.alloc_shared([rows, dim], dtype)
                k_s = T.alloc_shared([block_n, dim], dtype)
                imp_s = T.alloc_shared([block_t, n_max], accum_dtype)
                s_f = T.alloc_fragment([rows, block_n], accum_dtype)
                p3 = T.alloc_fragment([block_t, group, block_n], accum_dtype)
                imp_f = T.alloc_fragment([block_t, block_n], accum_dtype)
                m_f = T.alloc_fragment([rows], accum_dtype)
                m_prev = T.alloc_fragment([rows], accum_dtype)
                l_f = T.alloc_fragment([rows], accum_dtype)
                t_sum = T.alloc_fragment([rows], accum_dtype)
                vals = T.alloc_local([slots], accum_dtype)
                keys = T.alloc_local([slots], T.int32)
                tv = T.alloc_local([2], accum_dtype)
                tk = T.alloc_local([2], T.int32)
                flag = T.alloc_local([1], T.int32)
                # Both WGMMA operands are K-major tiles with the 128-byte swizzle.
                T.annotate_layout(
                    {
                        q_s: tilelang.layout.make_swizzled_layout(q_s),
                        k_s: tilelang.layout.make_swizzled_layout(k_s),
                    }
                )
                tx = T.get_thread_binding()
                lane = tx % _WARP
                warp = tx // _WARP

                chunk = bx // split
                sub = bx % split
                h = by

                # The request holding this chunk: the last one starting at or before it.
                lo = T.alloc_var(T.int32, init=0)
                hi = T.alloc_var(T.int32, init=seq_num - 1)
                for _ in T.serial(search):
                    mid = (lo + hi + 1) // 2
                    if chunk_offsets[mid] <= chunk:
                        lo = mid
                    else:
                        hi = mid - 1
                bos = offsets[lo]
                eos = offsets[lo + 1]
                boc = chunk_offsets[lo]
                cur = chunk - boc
                tok0 = bos + cur * bs + sub * block_t
                ntok = T.min(T.min(eos - tok0, block_t), bs - sub * block_t)
                n_cand = cur + 1
                n_tiles = T.ceildiv(n_cand, block_n)

                if ntok > 0:
                    for r, d in T.Parallel(rows, dim):
                        src = T.min(tok0 + r // group, c_seq_len - 1)
                        q_s[r, d] = T.Cast(
                            dtype, T.Cast(accum_dtype, q[src, h * group + r % group, d]) * scale
                        )

                    # Pass 1: online log-sum-exp over the closed chunks of each (token, head) row.
                    T.fill(m_f, neg_inf)
                    T.fill(l_f, 0.0)
                    for it in T.serial(n_tiles):
                        n0 = it * block_n
                        for n, d in T.Parallel(block_n, dim):
                            k_s[n, d] = T.if_then_else(
                                n0 + n < n_cand,
                                k_cmp[T.min(boc + n0 + n, chunk_num - 1), h, d],
                                T.Cast(dtype, 0),
                            )
                        T.clear(s_f)
                        T.gemm(q_s, k_s, s_f, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        for r, n in T.Parallel(rows, block_n):
                            closed = (cur * bs + sub * block_t + r // group + 1) // bs
                            s_f[r, n] = T.if_then_else(n0 + n < closed, s_f[r, n], neg_inf)
                        T.copy(m_f, m_prev)
                        T.reduce_max(s_f, m_f, dim=1, clear=False)
                        for r, n in T.Parallel(rows, block_n):
                            s_f[r, n] = T.if_then_else(
                                s_f[r, n] > neg_inf, T.exp2((s_f[r, n] - m_f[r]) * LOG2E), 0.0
                            )
                        T.reduce_sum(s_f, t_sum, dim=1)
                        for r in T.Parallel(rows):
                            l_f[r] = (
                                T.if_then_else(
                                    m_prev[r] > neg_inf,
                                    l_f[r] * T.exp2((m_prev[r] - m_f[r]) * LOG2E),
                                    0.0,
                                )
                                + t_sum[r]
                            )
                    for r in T.Parallel(rows):
                        m_f[r] = T.if_then_else(l_f[r] > 0, m_f[r] + T.log2(l_f[r]) / LOG2E, 0.0)

                    # Pass 2: probabilities of the open chunks, summed over the group in registers.
                    for it in T.serial(n_tiles):
                        n0 = it * block_n
                        for n, d in T.Parallel(block_n, dim):
                            k_s[n, d] = T.if_then_else(
                                n0 + n < n_cand,
                                k_cmp[T.min(boc + n0 + n, chunk_num - 1), h, d],
                                T.Cast(dtype, 0),
                            )
                        T.clear(s_f)
                        T.gemm(q_s, k_s, s_f, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        for t, g, n in T.Parallel(block_t, group, block_n):
                            p3[t, g, n] = T.if_then_else(
                                n0 + n < cur,
                                T.exp2((s_f[t * group + g, n] - m_f[t * group + g]) * LOG2E),
                                0.0,
                            )
                        T.reduce_sum(p3, imp_f, dim=1)
                        for t, n in T.Parallel(block_t, block_n):
                            idx = n0 + n
                            # The first, previous and current blocks score one per query head.
                            priority = (idx == 0) or (idx == cur - 1) or (idx == cur)
                            imp_s[t, idx] = T.if_then_else(
                                idx <= cur,
                                T.if_then_else(priority, T.Cast(accum_dtype, group), imp_f[t, n]),
                                neg_inf,
                            )

                    for rr in T.serial(rows_per_warp):
                        t = warp * rows_per_warp + rr
                        if t < ntok:
                            select_row(
                                t,
                                n_cand,
                                tok0,
                                h,
                                imp_s,
                                vals,
                                keys,
                                lane,
                                tv,
                                tk,
                                flag,
                                block_indices,
                            )

        return _nsa_topk_varlen_main

    return _nsa_topk_varlen_func()


@functools.lru_cache(maxsize=32)
def _nsa_topk_varlen_pool_kernel(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    chunk_num: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bc: int,
    bs: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    """The per-token kernel: one CTA per (token, KV head), merging each chunk tile into a pool.

    It serves the groups the tiled kernel does not, those that neither divide nor are a
    multiple of one WGMMA row tile.
    """
    head_kv = heads // group
    # The shared-memory tiles span the whole head dimension: a narrower tile
    # would silently truncate the QK contraction.
    bk = dim

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _nsa_topk_varlen_func(threads: int):
        @T.macro
        def odd_even_sort(indices, values, size):
            for _ in T.serial(size):
                for i in T.Parallel(size // 2):
                    v1 = values[i * 2]
                    v2 = values[i * 2 + 1]
                    idx1 = indices[i * 2]
                    idx2 = indices[i * 2 + 1]
                    swap = v1 < v2 or ((v1 == v2) and idx1 < idx2)
                    values[i * 2] = T.if_then_else(swap, v2, v1)
                    values[i * 2 + 1] = T.if_then_else(swap, v1, v2)
                    indices[i * 2] = T.if_then_else(swap, idx2, idx1)
                    indices[i * 2 + 1] = T.if_then_else(swap, idx1, idx2)
                T.sync_threads()
                # Full lanes avoid divergent barriers for bs=64; the last pair
                # compares an element with itself.
                for i in T.Parallel(size // 2):
                    right = T.min(i * 2 + 2, size - 1)
                    v1 = values[i * 2 + 1]
                    v2 = values[right]
                    idx1 = indices[i * 2 + 1]
                    idx2 = indices[right]
                    swap = v1 < v2 or ((v1 == v2) and idx1 < idx2)
                    values[i * 2 + 1] = T.if_then_else(swap, v2, v1)
                    values[right] = T.if_then_else(swap, v1, v2)
                    indices[i * 2 + 1] = T.if_then_else(swap, idx2, idx1)
                    indices[right] = T.if_then_else(swap, idx1, idx2)
                T.sync_threads()

        @T.prim_func
        def _parallel_nsa_topk_varlen_main(
            q: T.Tensor((c_seq_len, heads, dim), dtype),
            k_cmp: T.Tensor((chunk_num, head_kv, dim), dtype),
            offsets: T.Tensor((seq_num + 1,), T.int32),
            chunk_offsets: T.Tensor((seq_num + 1,), T.int32),
            token_indices: T.Tensor((c_seq_len, 2), T.int32),
            block_indices: T.Tensor((c_seq_len, head_kv, selected_block_num), T.int32),
        ):
            with T.Kernel(c_seq_len, head_kv, threads=threads) as (bx, by):
                q_shared = T.alloc_shared([group, bk], dtype)
                k_shared = T.alloc_shared([bc, bk], dtype)

                pool_scores_s = T.alloc_shared([bc * 2], accum_dtype)
                pool_indices_s = T.alloc_shared([bc * 2], T.int32)

                i_c, i_h = bx, by
                i_n, i_t = token_indices[i_c, 0], token_indices[i_c, 1]

                bos = offsets[i_n]
                boc = chunk_offsets[i_n]
                nc = (i_t + 1) // bs

                T.copy(q[bos + i_t, i_h * group : (i_h + 1) * group, :bk], q_shared)
                # FLA scales Q in the input dtype before the FP32 dot product.
                for g, d in T.Parallel(group, bk):
                    q_shared[g, d] = q_shared[g, d] * scale

                b_lse = T.alloc_fragment([group], accum_dtype)
                acc_s = T.alloc_fragment([group, bc], accum_dtype)
                scores_max = T.alloc_fragment([group], accum_dtype)
                scores_max_prev = T.alloc_fragment([group], accum_dtype)
                scores_scale = T.alloc_fragment([group], accum_dtype)
                scores_sum = T.alloc_fragment([group], accum_dtype)
                logsum = T.alloc_fragment([group], accum_dtype)

                T.fill(scores_max, -T.infinity(accum_dtype))
                T.fill(logsum, 0.0)

                for p in T.Parallel(bc * 2):
                    pool_scores_s[p] = -T.infinity(accum_dtype)
                    pool_indices_s[p] = 0

                # step1: LSE calculation
                for i_loop in T.Pipelined(T.ceildiv(nc, bc), num_stages=3):
                    curr_bc = T.min(bc, nc - i_loop * bc)
                    T.copy(
                        k_cmp[boc + i_loop * bc : boc + i_loop * bc + curr_bc, i_h, :bk],
                        k_shared[:curr_bc, :bk],
                    )

                    for g_m, c_m in T.Parallel(group, bc):
                        acc_s[g_m, c_m] = T.if_then_else(
                            c_m < curr_bc, 0.0, -T.infinity(accum_dtype)
                        )

                    T.gemm(
                        q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for i in T.Parallel(group):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        scores_scale[i] = T.if_then_else(
                            scores_max[i] > -T.infinity(accum_dtype),
                            T.exp2((scores_max_prev[i] - scores_max[i]) * LOG2E),
                            0.0,
                        )

                    for i, j in T.Parallel(group, bc):
                        acc_s[i, j] = T.if_then_else(
                            acc_s[i, j] > -T.infinity(accum_dtype),
                            T.exp2((acc_s[i, j] - scores_max[i]) * LOG2E),
                            0.0,
                        )

                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(group):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i in T.Parallel(group):
                    if nc == 0 or logsum[i] <= 0:
                        b_lse[i] = 0.0
                    else:
                        b_lse[i] = scores_max[i] + T.log2(logsum[i]) / LOG2E

                # step2: Importance Scores alignment and streaming Top-K
                T.sync_threads()
                nc_topk = i_t // bs + 1
                k_shared_topk = T.alloc_shared([bc, bk], dtype)

                for i_tk in T.Pipelined(T.ceildiv(nc_topk, bc), num_stages=3):
                    curr_bc_tk = T.min(bc, nc_topk - i_tk * bc)
                    T.copy(
                        k_cmp[boc + i_tk * bc : boc + i_tk * bc + curr_bc_tk, i_h, :bk],
                        k_shared_topk[:curr_bc_tk, :bk],
                    )

                    for g_m2, c_m2 in T.Parallel(group, bc):
                        acc_s[g_m2, c_m2] = T.if_then_else(
                            c_m2 < curr_bc_tk, 0.0, -T.infinity(accum_dtype)
                        )

                    T.gemm(
                        q_shared,
                        k_shared_topk,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for g_idx, c_idx in T.Parallel(group, bc):
                        curr_blk = i_tk * bc + c_idx
                        is_priority = (
                            (curr_blk == 0)
                            or (curr_blk == i_t // bs - 1)
                            or (curr_blk == i_t // bs)
                        )
                        is_hist = curr_blk < i_t // bs
                        imp = T.if_then_else(
                            is_priority,
                            1.0,
                            T.if_then_else(
                                is_hist,
                                T.exp2((acc_s[g_idx, c_idx] - b_lse[g_idx]) * LOG2E),
                                0.0,
                            ),
                        )
                        acc_s[g_idx, c_idx] = imp

                    b_i_current = T.alloc_fragment([bc], accum_dtype)
                    T.reduce_sum(acc_s, b_i_current, dim=0)

                    # Rank raw scores; equal scores prefer the larger block id.
                    for c_in in T.Parallel(bc):
                        pool_scores_s[bc + c_in] = T.if_then_else(
                            c_in < curr_bc_tk,
                            b_i_current[c_in],
                            -T.infinity(accum_dtype),
                        )
                        pool_indices_s[bc + c_in] = T.if_then_else(
                            c_in < curr_bc_tk, i_tk * bc + c_in + 1, 0
                        )

                    T.sync_threads()
                    odd_even_sort(pool_indices_s, pool_scores_s, bc * 2)

                for s_out in T.Parallel(selected_block_num):
                    idx_final = pool_indices_s[s_out]
                    block_indices[i_c, i_h, s_out] = idx_final - 1

        return _parallel_nsa_topk_varlen_main

    return _nsa_topk_varlen_func


class NSATopKVarlenKernel(Kernel, NSATopKFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]
    # Chunks one key tile holds: the WGMMA n of each score tile.
    _BLOCK_N = 64

    @classmethod
    def entry_for(cls, call: NSACall) -> Entry:
        return call, lambda: cls(
            seq_num=call.batch,
            c_seq_len=call.c_seq_len,
            heads=call.heads,
            dim=call.dim,
            chunk_num=call.chunk_num,
            group=call.heads // call.heads_kv,
            scale=call.scale,
            selected_block_num=call.selected_blocks,
            block_n=cls._BLOCK_N,
            bs=call.block_size,
            dtype=call.dtype,
            accum_dtype=torch.float32,
        )

    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim: int,
        chunk_num: int,
        group: int,
        scale: float,
        selected_block_num: int,
        block_n: int,
        bs: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim = dim
        self.chunk_num = chunk_num
        self.group = group
        self.scale = scale
        self.selected_block_num = selected_block_num
        self.block_n = block_n
        self.bs = bs
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.accum_dtype_str = self.dtype_to_str(self.accum_dtype)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # The per-token fallback runs one warp per CTA; the tiled kernel fixes its own.
        return {"threads": 32}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": 32}]

    @property
    def _tiled(self) -> bool:
        """Whether whole tokens fill the tiled kernel's WGMMA row tiles."""
        return _TILE_ROWS % self.group == 0 or self.group % _TILE_ROWS == 0

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (
            q.to(self.dtype),
            k_cmp.to(self.dtype),
            offsets.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )
        if not self._tiled:
            return _nsa_topk_varlen_pool_kernel(
                self.seq_num,
                self.c_seq_len,
                self.heads,
                self.dim,
                self.chunk_num,
                self.group,
                self.scale,
                self.selected_block_num,
                self.bs,
                self.bs,
                self.dtype_str,
                self.accum_dtype_str,
            )(self.config["threads"])(*tensors)
        return _nsa_topk_varlen_kernel(
            self.seq_num,
            self.c_seq_len,
            self.heads,
            self.dim,
            self.chunk_num,
            self.group,
            self.scale,
            self.selected_block_num,
            self.block_n,
            self.bs,
            self.dtype_str,
            self.accum_dtype_str,
        )(*tensors)
