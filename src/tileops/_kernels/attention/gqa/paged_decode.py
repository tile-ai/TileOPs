"""Paged decode attention programs for any head grouping.

One CTA owns ``block_M`` query rows of one KV head. Row ``r`` of a request is
query position ``r // group`` of head ``r % group`` within the heads sharing
that KV head. MHA is the ``group == 1`` case; a one-token GQA decode is the
``seqlen_q == 1`` case. ``GQADecodePagedKernel`` serves MHA paged decode; the
batch-1 GQA paged kernel builds on this module's programs.
"""

import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import (
    AttentionCall,
    MHAPagedDecodeFwdInterface,
)
from tileops._kernels.attention.online_softmax import (
    make_apply_softcap,
    make_online_softmax,
    make_online_softmax_with_mask_guard,
    make_rescale,
)
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import LOG2E
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = ["GQADecodePagedKernel"]


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


def _softmax_scale(dim, sm_scale, softcap):
    """The score scale, the exp2-domain factor the softmax applies, and whether scores are 0.

    Negative scales are applied before masking; softmax then uses a positive exp2 factor.
    A zero scale makes every score zero. The kernel then zeroes the scores before the
    mask and applies a unit factor, so a masked key stays at -inf instead of -inf * 0.
    """
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    if softcap > 0.0:
        return score_scale, LOG2E, False
    if score_scale == 0.0:
        return score_scale, 1.0, True
    if score_scale < 0.0:
        return score_scale, LOG2E, False
    return score_scale, score_scale * LOG2E, False


def _tail_block_n(block_N: int) -> int:
    """The key tile height of the pass over the tile crossing the cache end."""
    return 16 if block_N % 16 == 0 else block_N


def _threads(block_M: int, dim: int) -> int:
    """Four warps, or eight where four would hold the output accumulator in more than three
    quarters of a thread's 255 registers. Eight warps split the head dim in two halves of
    whole 16-wide matrix-multiply-accumulate (MMA) slices, so the head dim is a multiple of
    32."""
    if block_M * dim // 128 <= 255 * 3 // 4 or dim % 32 != 0:
        return 128
    return 256


def _weights_alloc(threads: int):
    """Where the cast softmax weights live. Four warps split only the query rows, so each
    warp's weights feed its own value product from registers. Eight warps also split the
    key tile, so the weights go through shared memory to reach every warp."""
    return T.alloc_fragment if threads <= 128 else T.alloc_shared


def _default_configs(block_ns: tuple[int, ...], block_M: int, dim: int) -> list[dict]:
    """The untuned configs, widest first: two stages of the widest key tile; where shared
    memory cannot hold them (SM89 holds 99 KB), one stage of at most 64 keys, then 64 query
    rows, then narrower key tiles, and last the narrowest with four warps, whose weights
    take no shared memory. The split count is the most a config takes."""
    wide = {"block_M": block_M, "block_N": block_ns[0], "num_split": 16, "num_stages": 2}
    narrow = [n for n in block_ns if n <= 64]
    configs = [
        wide,
        {**wide, "block_N": narrow[0], "num_stages": 1},
        *({**wide, "block_M": 64, "block_N": n, "num_stages": 1} for n in narrow),
    ]
    configs = [{**c, "threads": _threads(c["block_M"], dim)} for c in configs]
    if configs[-1]["threads"] != 128:
        configs.append({**configs[-1], "threads": 128})
    return configs


def _shared_bytes(config: dict, dim: int, element: int, arch: int) -> int:
    """Shared memory either program allocates for *config* on *arch*: the query rows, the K
    and the V tiles of every stage, and the softmax weights where eight warps share them.

    SM90 copies through the tensor memory accelerator (TMA): it adds buffers of their own
    for the tail K and V tiles and starts every buffer on the 1024-byte alignment of its
    swizzled layout. Earlier architectures reuse the pipeline's space for the tail tiles.
    """
    block_M, block_N = config["block_M"], config["block_N"]
    pipeline = config["num_stages"] * block_N * dim * element
    tail_n = _tail_block_n(block_N)
    weights = (
        (block_M * block_N * element, block_M * tail_n * element)
        if _weights_alloc(config["threads"]) is T.alloc_shared
        else ()
    )
    if arch < 90:
        return block_M * dim * element + 2 * pipeline + sum(weights)
    buffers = (
        block_M * dim * element,
        pipeline,
        pipeline,
        *2 * (tail_n * dim * element,),
        *weights,
    )
    return sum(-(-size // 1024) * 1024 for size in buffers)


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
            elif softcap <= 0.0 and score_scale < 0.0:
                for i, j in T.Parallel(block_M, tile_n):
                    acc_s[i, j] *= score_scale
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


@target_cache(maxsize=32)
def gqa_decode_no_split_paged_kernel(
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
        alloc_weights = _weights_alloc(threads)
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
                acc_s_cast = alloc_weights([block_M, block_N], dtype)
                K_tail = T.alloc_shared([tail_n, dim], dtype)
                V_tail = T.alloc_shared([tail_n, dim], dtype)
                acc_tail = T.alloc_fragment([block_M, tail_n], accum_dtype)
                acc_tail_cast = alloc_weights([block_M, tail_n], dtype)
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


@target_cache(maxsize=32)
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
        alloc_weights = _weights_alloc(threads)
        load_q, full_tile, tail_tile, visible_end = _make_tile_steps(
            block_M, block_N, dim, dtype, page_size, group, seqlen_q, is_causal, sm_scale, softcap
        )
        # The table's capacity splits into equal runs of whole key tiles, the last split
        # taking the remainder.
        capacity_tiles = max_pages_per_req * page_size // block_N
        split_tiles = capacity_tiles // num_split

        @T.macro
        def _gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, max_pages_per_req], T.int32),
            glse: T.Tensor(shape_lse, accum_dtype),
            Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(rows, block_M), heads_kv * batch, num_split, threads=threads
            ) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = alloc_weights([block_M, block_N], dtype)
                K_tail = T.alloc_shared([tail_n, dim], dtype)
                V_tail = T.alloc_shared([tail_n, dim], dtype)
                acc_tail = T.alloc_fragment([block_M, tail_n], accum_dtype)
                acc_tail_cast = alloc_weights([block_M, tail_n], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                row0 = bx * block_M
                kv_head = by % heads_kv
                bid = by // heads_kv
                sid = bz
                kv_len = real_seqlen_kv[bid]

                load_q(Q, Q_shared, bid, kv_head, row0)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # Each split runs its tiles up to the last key a row of this block sees,
                # so a split past a short cache runs none.
                offset = sid * split_tiles
                blocks_in_split = T.if_then_else(
                    sid < num_split - 1, split_tiles, capacity_tiles - offset
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

        # Below this, no split of the row saw a key. A threshold, not an equality with -inf:
        # fast math folds comparisons with infinity away.
        no_key_lse = -1.0e30

        combine_threads = 128
        # A one-dimensional fragment has a layout only when every thread owns the same
        # number of its elements, so the accumulator spans the head dim rounded up to a
        # multiple of the thread count.
        combine_width = -(-dim // combine_threads) * combine_threads

        @T.macro
        def combine(
            glse: T.Tensor(shape_lse, accum_dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(rows, heads_kv, batch, threads=combine_threads) as (r, kv_head, bid):
                lse = T.alloc_fragment([num_split], accum_dtype)
                lse_max = T.alloc_fragment([1], accum_dtype)
                lse_logsum = T.alloc_local([1], accum_dtype)
                o_accum = T.alloc_fragment([combine_width], accum_dtype)
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
                    for d in T.Parallel(combine_width):
                        if d < dim:
                            o_accum[d] += Output_partial[bid, kv_head, k, r, d] * w
                # A row no split saw a key for outputs zeros.
                for d in T.Parallel(combine_width):
                    if d < dim:
                        Output[bid, r // group, kv_head * group + r % group, d] = T.if_then_else(
                            lse_max[0] < T.cast(no_key_lse, accum_dtype), 0, o_accum[d]
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
            Output: T.Tensor(shape_q, dtype),
        ):
            _gqa_decode_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return gqa_decode_split

    return _func


class GQADecodePagedKernel(Kernel, MHAPagedDecodeFwdInterface):
    """MHA paged decode for one query length shared by every request.

    Only ``MHAPagedFwdOp`` registers it, so every call it serves has one query head a KV head.
    """

    supported_archs: list[int] = [80, 89, 90]
    # The implementation behind the specialised ones for this key.
    general: bool = True

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why *call* is outside the decode region, its page size, or a head dim whose
        narrowest tiles overflow the device's shared memory."""
        reason = call.paged_decode_refusal or call.tensor_core_dim_refusal
        if reason is not None:
            return reason
        try:
            block_ns = gqa_decode_paged_block_ns(call.page_size)
        except ValueError as exc:
            return str(exc)
        narrowest = _default_configs(block_ns, 64, call.dim)[-1]
        need = _shared_bytes(narrowest, call.dim, call.dtype.itemsize, call.arch)
        if need > call.smem_budget:
            return (
                f"head dim {call.dim} needs {need} bytes of shared memory with the narrowest "
                f"tiles; a block has {call.smem_budget}"
            )
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """A one-token causal query sees the whole cache, so it builds the non-causal
        kernel. The device index is in the identity: the kernel is compiled for its
        architecture."""
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
        return identity, lambda: cls(*args, **extra, device_index=index)

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
        self.init_config(config)

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
                    pages = torch.arange(batch * width, dtype=torch.int32, device="cuda")
                    value = (pages % pool_pages).view(batch, width)
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
        candidates = _default_configs(
            self._supported_block_ns, self._block_M_choices()[-1], self.dim
        )
        cap = self.build_target.smem_budget
        for config in candidates:
            if self._shared_bytes(config) <= cap:
                # Splitting the keys pays until the grid fills the device once; a further
                # split only shortens every block's loop and adds partials to combine.
                blocks = -(-self.rows // config["block_M"]) * self.heads_kv * self.batch
                sm_count = self.build_target.sm_count
                num_split = config["num_split"]
                while num_split > 1 and blocks * num_split > sm_count:
                    num_split //= 2
                return {**config, "num_split": num_split}
        raise ValueError(
            f"{type(self).__name__} needs {self._shared_bytes(candidates[-1])} bytes of shared "
            f"memory at head dimension {self.dim}; the device grants {cap}"
        )

    def _shared_bytes(self, config: dict) -> int:
        element = getattr(torch, self.dtype_str).itemsize
        return _shared_bytes(config, self.dim, element, self.build_target.arch)

    def tuning_candidates(self, configs, untuned=None) -> list[dict]:
        candidates = super().tuning_candidates(configs, untuned)
        budget = self.build_target.smem_budget
        return [c for c in candidates if self._shared_bytes(c) <= budget]

    @property
    def autotune_configs(self) -> list[dict]:
        keys = ("block_M", "block_N", "num_split", "num_stages")
        values = itertools.product(
            self._block_M_choices(), self._supported_block_ns, [2, 4, 8, 16], [1, 2, 3]
        )
        configs = [dict(zip(keys, c, strict=True)) for c in values]
        return [{**c, "threads": _threads(c["block_M"], self.dim)} for c in configs]

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ):
        """Attend ``Q``, ``[batch, seqlen_q, heads, dim]``, over the paged cache."""
        c = self.config
        # Specialize from the static table capacity. Kernels bound each split by
        # the live device lengths, so a graph replay can grow or shrink the cache.
        num_split = c["num_split"]
        program = self.program(self._compile)
        q = Q.view(self.batch, self.seqlen_q, self.heads, self.dim)
        if self._unsplit(c):
            return program(q, K, V, real_seqlen_kv, block_table).view(Q.shape)
        glse = torch.empty(
            (self.batch, self.heads_kv, num_split, self.rows), dtype=torch.float32, device=Q.device
        )
        Output_partial = torch.empty(
            (self.batch, self.heads_kv, num_split, self.rows, self.dim),
            dtype=self.dtype,
            device=Q.device,
        )
        out = program(q, K, V, real_seqlen_kv, block_table, glse, Output_partial)
        return out.view(Q.shape)

    def _unsplit(self, cfg: dict) -> bool:
        """Whether *cfg* runs unsplit: one split, or a table holding fewer than
        ``num_split`` key tiles."""
        capacity = self.max_pages_per_req * self.page_size
        return cfg["num_split"] == 1 or capacity < cfg["num_split"] * cfg["block_N"]

    def _compile(self, cfg: dict):
        if self._unsplit(cfg):
            return gqa_decode_no_split_paged_kernel(*self._builder_args)(
                cfg["block_M"], cfg["block_N"], cfg["num_stages"], cfg["threads"]
            )
        return self.kernel(
            cfg["block_M"], cfg["block_N"], cfg["num_split"], cfg["num_stages"], cfg["threads"]
        )
