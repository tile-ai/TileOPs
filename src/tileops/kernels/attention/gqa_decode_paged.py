"""Paged decode attention for any head grouping, MHA included.

One CTA owns ``block_M`` query rows of one KV head. Row ``r`` of a request is
query position ``r // group`` of head ``r % group`` within the heads sharing
that KV head. MHA is the ``group == 1`` case; a one-token GQA decode is the
``seqlen_q == 1`` case.
"""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import (
    AttentionCall,
    paged_decode_refusal,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel

from .online_softmax import (
    make_apply_softcap,
    make_online_softmax,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

__all__ = ["GQADecodePagedKernel"]

# Below this, no split of the row saw a key. A threshold, not an equality with -inf:
# fast math folds comparisons with infinity away.
_NO_KEY_LSE = -1.0e30


def gqa_decode_paged_block_ns(page_size: int) -> tuple[int, ...]:
    """Return the key tile heights that keep one tile inside one page, widest first.

    A page shorter than 64 rows is also one tile.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    block_ns = tuple(n for n in (128, 64, 32, 16) if page_size % n == 0)
    if page_size < 64 and page_size not in block_ns:
        block_ns = (page_size, *block_ns)
    if block_ns:
        return block_ns
    raise ValueError(f"page_size={page_size} matches no supported block_N")


def gqa_decode_paged_block_n(page_size: int) -> int:
    """Return the widest key tile height that keeps one tile inside one page."""
    return gqa_decode_paged_block_ns(page_size)[0]


def _softmax_scale(dim, sm_scale, softcap):
    """The score scale, the exp2-domain factor the softmax applies, and whether scores are 0.

    A zero scale makes every score zero. The kernel then zeroes the scores before the
    mask and applies a unit factor, so a masked key stays at -inf instead of -inf * 0.
    """
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    if softcap > 0.0:
        return score_scale, LOG2E, False
    if score_scale == 0.0:
        return score_scale, 1.0, True
    return score_scale, score_scale * LOG2E, False


def _tail_block_n(block_N: int) -> int:
    """The key tile height of the pass over the tile crossing the cache end."""
    return 16 if block_N % 16 == 0 else block_N


def _make_tile_steps(
    block_M, block_N, dim, dtype, page_size, group, seqlen_q, is_causal, sm_scale, softcap
):
    """The macros both variants share: the Q gather and the key tile updates.

    Full tiles run pipelined at ``block_N`` rows. The tile crossing the cache end runs
    first, unpipelined, in ``_tail_block_n`` rows through buffers of its own, and loads
    zeros past the cache: a non-finite value there would survive a zero weight.
    """
    accum_dtype = "float"
    score_scale, softmax_scale, zero_scores = _softmax_scale(dim, sm_scale, softcap)
    # A causal mask can hide a whole tile from a query row.
    make_softmax = make_online_softmax_with_mask_guard if is_causal else make_online_softmax
    rescale = make_rescale(block_M, dim)
    rows = seqlen_q * group

    @T.macro
    def load_q(Q, Q_shared, bid, kv_head, row0):
        for i, d in T.Parallel(block_M, dim):
            r = T.min(row0 + i, rows - 1)
            Q_shared[i, d] = T.if_then_else(
                row0 + i < rows, Q[bid, r // group, kv_head * group + r % group, d], 0
            )

    def make_tile_step(tile_n, crosses_cache_end):
        online_softmax = make_softmax(softmax_scale, accum_dtype, block_M, tile_n)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_M, tile_n)
            if softcap > 0.0
            else None
        )

        @T.macro
        def tile_step(
            K,
            V,
            block_table,
            Q_shared,
            K_shared,
            V_shared,
            acc_s,
            acc_s_cast,
            acc_o,
            scores_max,
            scores_max_prev,
            scores_scale,
            scores_sum,
            logsum,
            bid,
            kv_head,
            row0,
            k,
            kv_len,
        ):
            key0 = k * tile_n
            base = block_table[bid, key0 // page_size] * page_size + key0 % page_size
            T.copy(K[base : base + tile_n, kv_head, :], K_shared)
            # Issue both K/V loads before QK so copies can overlap the GEMMs.
            if crosses_cache_end:
                for j, d in T.Parallel(tile_n, dim):
                    V_shared[j, d] = T.if_then_else(
                        key0 + j < kv_len, V[base + j, kv_head, d], T.cast(0, dtype)
                    )
            else:
                T.copy(V[base : base + tile_n, kv_head, :], V_shared)
            T.clear(acc_s)
            # The GEMM runs even for zero scores: it fixes the layout the row statistics share.
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            if zero_scores:
                T.clear(acc_s)
            if softcap > 0.0:
                apply_softcap(acc_s)
            if is_causal:
                # Causal queries sit at the end of the cache.
                for i, j in T.Parallel(block_M, tile_n):
                    key = key0 + j
                    visible = (key < kv_len) & (key <= (row0 + i) // group + kv_len - seqlen_q)
                    acc_s[i, j] = T.if_then_else(visible, acc_s[i, j], -T.infinity(accum_dtype))
            else:
                for i, j in T.Parallel(block_M, tile_n):
                    acc_s[i, j] = T.if_then_else(
                        key0 + j < kv_len, acc_s[i, j], -T.infinity(accum_dtype)
                    )
            online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
            T.copy(acc_s, acc_s_cast)
            rescale(acc_o, scores_scale)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        return tile_step

    def visible_end(row0, kv_len):
        """One past the last key a row of this block sees."""
        if not is_causal:
            return kv_len
        last_pos = T.min(seqlen_q - 1, (row0 + block_M - 1) // group)
        return T.max(0, T.min(kv_len, kv_len - seqlen_q + last_pos + 1))

    full_tile = make_tile_step(block_N, False)
    tail_tile = make_tile_step(_tail_block_n(block_N), True)
    return load_q, full_tile, tail_tile, visible_end


@functools.lru_cache(maxsize=32)
def _gqa_decode_no_split_paged_kernel(
    batch,
    heads,
    heads_kv,
    seqlen_q,
    seqlen_kv,
    dim,
    page_size,
    max_pages_per_req,
    is_causal,
    sm_scale,
    softcap,
    dtype,
):
    accum_dtype = "float"
    group = heads // heads_kv
    rows = seqlen_q * group

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_M, block_N, num_stages, threads):
        shape_q = [batch, seqlen_q, heads, dim]
        shape_kv = [seqlen_kv, heads_kv, dim]
        tail_n = _tail_block_n(block_N)
        load_q, full_tile, tail_tile, visible_end = _make_tile_steps(
            block_M, block_N, dim, dtype, page_size, group, seqlen_q, is_causal, sm_scale, softcap
        )

        @T.prim_func
        def gqa_decode_no_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, max_pages_per_req], T.int32),
            Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(T.ceildiv(rows, block_M), heads_kv, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                K_tail = T.alloc_shared([tail_n, dim], dtype)
                V_tail = T.alloc_shared([tail_n, dim], dtype)
                acc_tail = T.alloc_fragment([block_M, tail_n], accum_dtype)
                acc_tail_cast = T.alloc_fragment([block_M, tail_n], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                row0 = bx * block_M
                kv_len = real_seqlen_kv[bz]

                load_q(Q, Q_shared, bz, by, row0)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(visible_end(row0, kv_len), block_N)
                # The tile crossing the cache end, if this block reaches it, runs first.
                full_range = T.min(loop_range, kv_len // block_N)
                tail_start = full_range * block_N
                tail_end = T.min(visible_end(row0, kv_len), loop_range * block_N)
                for t in T.serial(T.max(0, T.ceildiv(tail_end - tail_start, tail_n))):
                    tail_tile(
                        K,
                        V,
                        block_table,
                        Q_shared,
                        K_tail,
                        V_tail,
                        acc_tail,
                        acc_tail_cast,
                        acc_o,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                        bz,
                        by,
                        row0,
                        tail_start // tail_n + t,
                        kv_len,
                    )
                for k in T.Pipelined(full_range, num_stages=num_stages):
                    full_tile(
                        K,
                        V,
                        block_table,
                        Q_shared,
                        K_shared,
                        V_shared,
                        acc_s,
                        acc_s_cast,
                        acc_o,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                        bz,
                        by,
                        row0,
                        k,
                        kv_len,
                    )
                for i, d in T.Parallel(block_M, dim):
                    if row0 + i < rows:
                        Output[bz, (row0 + i) // group, by * group + (row0 + i) % group, d] = (
                            T.if_then_else(logsum[i] == 0, 0, acc_o[i, d] / logsum[i])
                        )

        return gqa_decode_no_split

    return _func


@functools.lru_cache(maxsize=32)
def _gqa_decode_split_paged_kernel(
    batch,
    heads,
    heads_kv,
    seqlen_q,
    seqlen_kv,
    dim,
    page_size,
    max_pages_per_req,
    is_causal,
    sm_scale,
    softcap,
    dtype,
):
    accum_dtype = "float"
    group = heads // heads_kv
    rows = seqlen_q * group
    _, softmax_scale, _ = _softmax_scale(dim, sm_scale, softcap)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_M, block_N, num_split, num_stages, threads):
        shape_q = [batch, seqlen_q, heads, dim]
        shape_kv = [seqlen_kv, heads_kv, dim]
        shape_lse = [batch, heads_kv, num_split, rows]
        part_shape = [batch, heads_kv, num_split, rows, dim]
        tail_n = _tail_block_n(block_N)
        load_q, full_tile, tail_tile, visible_end = _make_tile_steps(
            block_M, block_N, dim, dtype, page_size, group, seqlen_q, is_causal, sm_scale, softcap
        )

        @T.macro
        def _gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, max_pages_per_req], T.int32),
            glse: T.Tensor(shape_lse, accum_dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            split_length: T.Tensor([batch, num_split], "int32"),
        ):
            with T.Kernel(
                T.ceildiv(rows, block_M), heads_kv * batch, num_split, threads=threads
            ) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                K_tail = T.alloc_shared([tail_n, dim], dtype)
                V_tail = T.alloc_shared([tail_n, dim], dtype)
                acc_tail = T.alloc_fragment([block_M, tail_n], accum_dtype)
                acc_tail_cast = T.alloc_fragment([block_M, tail_n], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                split_length_shared = T.alloc_shared([num_split], "int32")

                row0 = bx * block_M
                kv_head = by % heads_kv
                bid = by // heads_kv
                sid = bz
                T.copy(split_length[bid, :], split_length_shared, disable_tma=True)
                kv_len = real_seqlen_kv[bid]

                load_q(Q, Q_shared, bid, kv_head, row0)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # Each split runs its tiles up to the last key a row of this block sees,
                # so a split past a short cache runs none.
                offset = T.if_then_else(sid > 0, split_length_shared[sid - 1] // block_N, 0)
                blocks_in_split = T.if_then_else(
                    sid > 0,
                    T.ceildiv(split_length_shared[sid] - split_length_shared[sid - 1], block_N),
                    T.ceildiv(split_length_shared[0], block_N),
                )
                loop_range = T.max(
                    0,
                    T.min(blocks_in_split, T.ceildiv(visible_end(row0, kv_len), block_N) - offset),
                )
                # The tile crossing the cache end, if this block reaches it, runs first.
                full_range = T.max(0, T.min(loop_range, kv_len // block_N - offset))
                tail_start = (offset + full_range) * block_N
                tail_end = T.min(visible_end(row0, kv_len), (offset + loop_range) * block_N)
                for t in T.serial(T.max(0, T.ceildiv(tail_end - tail_start, tail_n))):
                    tail_tile(
                        K,
                        V,
                        block_table,
                        Q_shared,
                        K_tail,
                        V_tail,
                        acc_tail,
                        acc_tail_cast,
                        acc_o,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                        bid,
                        kv_head,
                        row0,
                        tail_start // tail_n + t,
                        kv_len,
                    )
                for k in T.Pipelined(full_range, num_stages=num_stages):
                    full_tile(
                        K,
                        V,
                        block_table,
                        Q_shared,
                        K_shared,
                        V_shared,
                        acc_s,
                        acc_s_cast,
                        acc_o,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                        bid,
                        kv_head,
                        row0,
                        k + offset,
                        kv_len,
                    )
                for i, d in T.Parallel(block_M, dim):
                    acc_o[i, d] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, d] / logsum[i])
                for i in T.Parallel(block_M):
                    # A row this split saw no key for weighs nothing in combine.
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.if_then_else(
                        logsum[i] == 0,
                        -T.infinity(accum_dtype),
                        T.log2(logsum_safe) + scores_max[i] * softmax_scale,
                    )
                T.copy(logsum, glse[bid, kv_head, sid, row0 : row0 + block_M])
                T.copy(acc_o, Output_partial[bid, kv_head, sid, row0 : row0 + block_M, :])

        @T.macro
        def combine(
            glse: T.Tensor(shape_lse, accum_dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(rows, heads_kv, batch, threads=128) as (r, kv_head, bid):
                lse = T.alloc_fragment([num_split], accum_dtype)
                lse_max = T.alloc_fragment([1], accum_dtype)
                lse_logsum = T.alloc_local([1], accum_dtype)
                o_accum = T.alloc_fragment([dim], accum_dtype)
                for k in T.Parallel(num_split):
                    lse[k] = glse[bid, kv_head, k, r]
                T.fill(lse_max, -T.infinity(accum_dtype))
                T.reduce_max(lse, lse_max, dim=0, clear=False)
                # Weights relative to the max keep the normalization term from rounding away.
                lse_logsum[0] = 0
                for k in T.serial(num_split):
                    lse_logsum[0] += T.exp2(glse[bid, kv_head, k, r] - lse_max[0])
                lse_logsum[0] = T.log2(lse_logsum[0]) + lse_max[0]
                T.clear(o_accum)
                for k in T.serial(num_split):
                    w = T.exp2(glse[bid, kv_head, k, r] - lse_logsum[0])
                    for d in T.Parallel(dim):
                        o_accum[d] += Output_partial[bid, kv_head, k, r, d] * w
                # A row no split saw a key for outputs zeros.
                for d in T.Parallel(dim):
                    Output[bid, r // group, kv_head * group + r % group, d] = T.if_then_else(
                        lse_max[0] < T.cast(_NO_KEY_LSE, accum_dtype), 0, o_accum[d]
                    )

        @T.prim_func
        def gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, max_pages_per_req], T.int32),
            glse: T.Tensor(shape_lse, accum_dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            split_length: T.Tensor([batch, num_split], "int32"),
            Output: T.Tensor(shape_q, dtype),
        ):
            _gqa_decode_split(
                Q, K, V, real_seqlen_kv, block_table, glse, Output_partial, split_length
            )
            combine(glse, Output_partial, Output)

        return gqa_decode_split

    return _func


def _gqa_decode_paged_no_split_run(
    batch: int,
    heads: int,
    heads_kv: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    max_pages_per_req: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Run the one-pass variant; ``Q`` is ``[batch, seqlen_q, heads, dim]`` or packed."""
    kernel = _gqa_decode_no_split_paged_kernel(
        batch,
        heads,
        heads_kv,
        seqlen_q,
        seqlen_kv,
        dim,
        page_size,
        max_pages_per_req,
        is_causal,
        sm_scale,
        softcap,
        dtype,
    )(block_M, block_N, num_stages, threads)
    q = Q.view(batch, seqlen_q, heads, dim)
    return kernel(q, K, V, real_seqlen_kv, block_table).view(Q.shape)


def _gqa_decode_paged_split_run(
    batch: int,
    heads: int,
    heads_kv: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    max_pages_per_req: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_M: int,
    block_N: int,
    num_split: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
    acc_split_length: torch.Tensor,
) -> torch.Tensor:
    kernel = _gqa_decode_split_paged_kernel(
        batch,
        heads,
        heads_kv,
        seqlen_q,
        seqlen_kv,
        dim,
        page_size,
        max_pages_per_req,
        is_causal,
        sm_scale,
        softcap,
        dtype,
    )(block_M, block_N, num_split, num_stages, threads)
    q = Q.view(batch, seqlen_q, heads, dim)
    out = kernel(q, K, V, real_seqlen_kv, block_table, glse, Output_partial, acc_split_length)
    return out.view(Q.shape)


def paged_decode_entry(cls: type, call: AttentionCall) -> Entry:
    """The entry for the paged-decode kernel.

    A one-token causal query sees the whole cache, so it builds the non-causal
    kernel. The device index is in the identity because the kernel is compiled
    for the architecture it is built on.
    """
    index = call.device.index if call.device is not None else None
    args = (
        call.batch,
        call.heads,
        call.heads_kv,
        call.max_seqlen_q,
        call.seqlen_kv,
        call.dim,
        call.page_size,
        call.max_pages_per_req,
        call.is_causal and call.max_seqlen_q > 1,
        call.dtype,
    )
    extra = dict(sm_scale=call.sm_scale, softcap=call.softcap)
    identity = (*args, *extra.values(), index)
    return identity, lambda: cls(*args, **extra, tune=call.tune, device_index=index)


class GQADecodePagedKernel(Kernel):
    """Paged decode for any head grouping and one query length shared by every request."""

    supported_archs: list[int] = [80, 89, 90]
    # The implementation behind the specialised ones for this key.
    general: bool = True

    @classmethod
    def applies(cls, call) -> bool:
        return cls._region_refusal(call) is None

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        return cls._region_refusal(call)

    @staticmethod
    def _region_refusal(call: AttentionCall) -> Optional[str]:
        """Why *call* is outside the decode region or no key tile covers its pages."""
        reason = paged_decode_refusal(call)
        if reason is not None:
            return reason
        try:
            gqa_decode_paged_block_ns(call.page_size)
        except ValueError as exc:
            return str(exc)
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        return paged_decode_entry(cls, call)

    def __init__(
        self,
        batch,
        heads,
        heads_kv,
        seqlen_q,
        seqlen_kv,
        dim,
        page_size,
        max_pages_per_req,
        is_causal,
        dtype="float16",
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune=False,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be a positive multiple of heads_kv")
        if seqlen_q <= 0:
            raise ValueError("seqlen_q must be positive")
        if seqlen_kv <= 0 or page_size <= 0 or seqlen_kv % page_size != 0:
            raise ValueError("seqlen_kv must be a positive multiple of page_size")
        if max_pages_per_req <= 0:
            raise ValueError("max_pages_per_req must be positive")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.max_pages_per_req = max_pages_per_req
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.rows = seqlen_q * (heads // heads_kv)
        self._supported_block_ns = gqa_decode_paged_block_ns(page_size)
        if config is not None:
            block_n = config.get("block_N")
            if block_n is not None and block_n not in self._supported_block_ns:
                raise ValueError(f"block_N={block_n} is not supported for page_size={page_size}")

        self._builder_args = (
            self.batch,
            self.heads,
            self.heads_kv,
            self.seqlen_q,
            self.seqlen_kv,
            self.dim,
            self.page_size,
            self.max_pages_per_req,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
        )
        # autotune targets the split kernel
        self.kernel = _gqa_decode_split_paged_kernel(*self._builder_args)
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    def _make_supply_prog(self):
        """Create a supply_prog that handles int32 tensor parameters for paged attention."""
        from tilelang.utils.tensor import get_tensor_supply as _get_tensor_supply

        default_supply = _get_tensor_supply(tilelang.TensorSupplyType.Auto)
        batch = self.batch
        width = self.max_pages_per_req
        pool_pages = self.seqlen_kv // self.page_size
        # Every request fills its table, whose entries name pages inside the pool.
        seqlen_kv = width * self.page_size

        def supply_prog(params):
            inputs = []
            for index, param in enumerate(params):
                if index == 3:
                    value = torch.full((batch,), seqlen_kv, dtype=torch.int32, device="cuda")
                elif index == 4:
                    pages = torch.arange(width, dtype=torch.int32, device="cuda") % pool_pages
                    value = pages.unsqueeze(0).expand(batch, -1).contiguous()
                elif index == 7:
                    num_split = param.shape[1]
                    base = seqlen_kv // num_split
                    value = torch.full(param.shape, base, dtype=torch.int32, device="cuda")
                    value[:, -1] += seqlen_kv % num_split
                    value = torch.cumsum(value, dim=1).to(torch.int32)
                else:
                    value = default_supply(param)
                inputs.append(value)
            return inputs

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    def _block_M_choices(self) -> list[int]:
        return [64] if self.rows <= 64 else [64, 128]

    @property
    def default_config(self) -> dict:
        return {
            "block_M": self._block_M_choices()[-1],
            "block_N": self._supported_block_ns[0],
            "num_split": 16,
            "num_stages": 2,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        keys = ("block_M", "block_N", "num_split", "num_stages", "threads")
        values = itertools.product(
            self._block_M_choices(), self._supported_block_ns, [2, 4, 8, 16], [1, 2, 3], [128]
        )
        return [dict(zip(keys, c, strict=True)) for c in values]

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ):
        """Attend ``Q``, ``[batch, seqlen_q, heads, dim]`` or packed, over the paged cache."""
        c = self.config
        args = (*self._builder_args, c["block_M"], c["block_N"])
        # A cache shorter than one tile per split is not worth splitting.
        real_max = int(real_seqlen_kv.max().item())
        num_split = c["num_split"]
        if real_max < num_split * c["block_N"]:
            return _gqa_decode_paged_no_split_run(
                *args, c["num_stages"], c["threads"], Q, K, V, real_seqlen_kv, block_table
            )

        chunk_size = real_max // (num_split * c["block_N"]) * c["block_N"]
        split_length = torch.full(
            (self.batch, num_split), chunk_size, dtype=torch.int32, device=Q.device
        )
        split_length[:, -1] = real_max - (num_split - 1) * chunk_size
        acc_split_length = torch.cumsum(split_length, dim=1).to(torch.int32)
        glse = torch.empty(
            (self.batch, self.heads_kv, num_split, self.rows), dtype=torch.float32, device=Q.device
        )
        Output_partial = torch.empty(
            (self.batch, self.heads_kv, num_split, self.rows, self.dim),
            dtype=self.dtype,
            device=Q.device,
        )
        return _gqa_decode_paged_split_run(
            *args,
            num_split,
            c["num_stages"],
            c["threads"],
            Q,
            K,
            V,
            real_seqlen_kv,
            block_table,
            glse,
            Output_partial,
            acc_split_length,
        )
