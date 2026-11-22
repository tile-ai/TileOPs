"""Pieces the GQA decode kernels share: the SM90 context-split schedule, the split combine,
the key partition and the query rotation."""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.online_softmax import (
    make_apply_softcap,
    make_dense_split_sink_merge,
)
from tileops._kernels.attention.varlen_rope import rope_channel_pair
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS

COMPILE_FLAGS = ["--use_fast_math", *CUDA_COMPILE_FLAGS]

# K/V slots of the context split's ring; the split's barrier parities and every loader's slot
# index assume this double-buffered schedule.
RING_DEPTH = 2


def allocate_partials(
    q: torch.Tensor,
    batch: int,
    heads: int,
    dim: int,
    ctx_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The float32 log-sum-exp and output each context split writes for the combine."""
    glse = torch.empty((batch, heads, ctx_splits), dtype=torch.float32, device=q.device)
    output_partial = torch.empty(
        (batch, heads, ctx_splits, dim),
        dtype=torch.float32,
        device=q.device,
    )
    return glse, output_partial


def balanced_split(seqlen_kv, block_n: int, num_split: int, sid):
    """Split *sid*'s first key and key count: whole key tiles split as evenly as possible,
    the first ``num_tiles % num_split`` splits taking one tile more than the rest."""
    num_tiles = T.ceildiv(seqlen_kv, block_n)
    base_tiles = num_tiles // num_split
    extra_tiles = num_tiles % num_split
    tiles_this_split = base_tiles + T.if_then_else(sid < extra_tiles, 1, 0)
    base = (sid * base_tiles + T.min(sid, extra_tiles)) * block_n
    return base, T.max(T.min(seqlen_kv - base, tiles_this_split * block_n), 0)


def rope_partner(rope_layout: str, rotary_dim: int, j):
    """The frequency of rotated channel *j*, the channel it rotates with, and the sign of
    that channel's term: channel ``j`` becomes ``x[j] * cos + sign * x[partner] * sin``."""
    half = rotary_dim // 2
    if rope_layout == "neox":
        return (
            T.if_then_else(j < half, j, j - half),
            T.if_then_else(j < half, j + half, j - half),
            T.if_then_else(j < half, -1.0, 1.0),
        )
    return (
        j // 2,
        T.if_then_else(j % 2 == 0, j + 1, j - 1),
        T.if_then_else(j % 2 == 0, -1.0, 1.0),
    )


def make_gqa_decode_bs1_split(
    batch: int,
    groups: int,
    block_m: int,
    block_n: int,
    dim: int,
    dtype: str,
    scale: float,
    kv_group_num: int,
    ctx_splits: int,
    threads: int,
    accum_dtype: str,
    load_kv,
    *,
    paged: bool,
    score_scale: float = 1.0,
    softcap: float = 0.0,
    fuse_rope: bool = False,
    rotary_dim: int = 0,
    rope_layout: str = "neox",
):
    """Create the context-split schedule around a layout-specific KV loader.

    A producer of ``threads - 128`` threads feeds one 128-thread WGMMA consumer warpgroup
    through a ring of ``RING_DEPTH`` slots. A dense cache splits whole key tiles as evenly
    as possible. A *paged* one reads each request's length from a buffer, gives every split
    the same whole tiles and the last the rest, and zeroes the value rows past the cache.
    RoPE makes the producer a warpgroup, whose first warp copies and whose threads all
    rotate K, so the rotation overlaps the consumer and the consumer reads K through WGMMA
    alone.
    """
    consumer_threads = 128
    # TileLang gives each thread range it inserts a sync for its own named barrier, counting
    # from 3 per kernel; this kernel's producer and consumer ranges take 3 and 4.
    consumer_barrier = 5
    producer_barrier = 6
    tma_threads = 32
    apply_softcap = (
        make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
        if softcap > 0.0
        else None
    )

    @T.macro
    def consumer(
        Q,
        bid,
        hid,
        sid,
        this_len,
        seqlen_kv,
        loop_range,
        rope_cos,
        rope_sin,
        Qs,
        Ks,
        Vs,
        Ps,
        ready,
        free,
        acc_s,
        acc_o,
        sm,
        smp,
        alpha,
        ss,
        logsum,
        glse,
        Output_partial,
    ):
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(sm, -T.infinity(accum_dtype))
        if fuse_rope:
            for i, j in T.Parallel(block_m, dim):
                if i < kv_group_num:
                    if j < rotary_dim:
                        freq, partner, sign = rope_partner(rope_layout, rotary_dim, j)
                        x = Q[bid, hid * kv_group_num + i, j]
                        x_partner = Q[bid, hid * kv_group_num + i, partner]
                        cos = rope_cos[seqlen_kv - 1, freq]
                        sin = rope_sin[seqlen_kv - 1, freq]
                        Qs[i, j] = x * cos + sign * x_partner * sin
                    else:
                        Qs[i, j] = Q[bid, hid * kv_group_num + i, j]
                else:
                    Qs[i, j] = 0
            T.sync_threads(consumer_barrier, consumer_threads)
        else:
            T.copy(
                Q[bid, hid * kv_group_num : hid * kv_group_num + kv_group_num, :],
                Qs[0:kv_group_num, :],
            )
        for k in T.serial(loop_range):
            T.mbarrier_wait_parity(ready[k % RING_DEPTH], (k // RING_DEPTH) % RING_DEPTH)
            # Nested: `and` between a Python bool and a TIR comparison is not TIR.
            if paged:  # noqa: SIM102
                if (k + 1) * block_n > this_len:
                    # A page holds whatever its slots past the cache last held, and a
                    # zero weight does not cancel a non-finite value in P @ V.
                    for j, d in T.Parallel(block_n, dim):
                        if k * block_n + j >= this_len:
                            Vs[k % RING_DEPTH, j, d] = T.cast(0, dtype)
                    T.fence_proxy_async()
                    T.sync_threads(consumer_barrier, consumer_threads)
            T.wgmma_gemm(
                Qs,
                Ks[k % RING_DEPTH, :, :],
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            if softcap > 0.0:
                apply_softcap(acc_s)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.if_then_else(
                    k * block_n + j < this_len,
                    acc_s[i, j],
                    -T.infinity(accum_dtype),
                )
            T.copy(sm, smp)
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            for i in T.Parallel(block_m):
                alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
            T.reduce_sum(acc_s, ss, dim=1)
            for i in T.Parallel(block_m):
                logsum[i] = logsum[i] * alpha[i] + ss[i]
            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] *= alpha[i]
            T.copy(acc_s, Ps)
            T.wgmma_gemm(
                Ps,
                Vs[k % RING_DEPTH, :, :],
                acc_o,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=False,
            )
            T.wait_wgmma(0)
            T.mbarrier_arrive(free[k % RING_DEPTH])
        for i, j in T.Parallel(block_m, dim):
            # A graph may replay with fewer live tiles than static splits.
            acc_o[i, j] /= T.if_then_else(this_len > 0, logsum[i], 1.0)
        for i in T.Parallel(block_m):
            if i < kv_group_num:
                glse[bid, hid * kv_group_num + i, sid] = T.if_then_else(
                    this_len > 0,
                    T.log2(logsum[i]) + sm[i] * scale,
                    -T.infinity(accum_dtype),
                )
        for i, j in T.Parallel(block_m, dim):
            if i < kv_group_num:
                Output_partial[bid, hid * kv_group_num + i, sid, j] = T.if_then_else(
                    this_len > 0,
                    acc_o[i, j],
                    0.0,
                )

    @T.macro
    def split(
        Q,
        K,
        V,
        kv_layout,
        real_seqlen_kv,
        rope_cos,
        rope_sin,
        glse,
        output_partial,
    ):
        with T.Kernel(batch, groups, ctx_splits, threads=threads) as (bid, hid, sid):
            qs = T.alloc_shared([block_m, dim], dtype)
            ks = T.alloc_shared([RING_DEPTH, block_n, dim], dtype)
            vs = T.alloc_shared([RING_DEPTH, block_n, dim], dtype)
            ps = T.alloc_shared([block_m, block_n], dtype)
            rope_pipeline = fuse_rope
            # TMA needs 16-byte table rows; narrower 16-bit tables are read from global memory.
            tma_rope_tables = fuse_rope and rotary_dim % 16 == 0
            if tma_rope_tables:
                rope_cos_s = T.alloc_shared([block_n, rotary_dim // 2], dtype)
                rope_sin_s = T.alloc_shared([block_n, rotary_dim // 2], dtype)
            T.annotate_layout(
                {
                    qs: tilelang.layout.make_swizzled_layout(qs),
                    ks: tilelang.layout.make_swizzled_layout(ks),
                    vs: tilelang.layout.make_swizzled_layout(vs),
                    ps: tilelang.layout.make_swizzled_layout(ps),
                }
            )
            producer_threads = threads - consumer_threads
            ready = T.alloc_barrier([producer_threads] * RING_DEPTH)
            free = T.alloc_barrier([consumer_threads] * RING_DEPTH)
            if rope_pipeline:
                loaded = T.alloc_barrier([tma_threads] * RING_DEPTH)
            load_ready = loaded if rope_pipeline else ready
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            sm = T.alloc_fragment([block_m], accum_dtype)
            smp = T.alloc_fragment([block_m], accum_dtype)
            alpha = T.alloc_fragment([block_m], accum_dtype)
            ss = T.alloc_fragment([block_m], accum_dtype)
            logsum = T.alloc_fragment([block_m], accum_dtype)

            if paged:
                seqlen_kv_b = real_seqlen_kv[bid]
                base_len = seqlen_kv_b // (ctx_splits * block_n) * block_n
                this_len = T.if_then_else(
                    sid == ctx_splits - 1,
                    seqlen_kv_b - (ctx_splits - 1) * base_len,
                    base_len,
                )
                base = base_len * sid
            else:
                seqlen_kv_b = real_seqlen_kv
                base, this_len = balanced_split(seqlen_kv_b, block_n, ctx_splits, sid)
            loop_range = T.ceildiv(this_len, block_n)
            tx = T.get_thread_binding()

            if tx >= consumer_threads:
                for k in T.serial(loop_range):
                    T.mbarrier_wait_parity(
                        free[k % RING_DEPTH],
                        ((k // RING_DEPTH) % RING_DEPTH) ^ 1,
                    )
                    if tma_rope_tables:
                        # One table buffer: every producer warp must have read the last tile's.
                        T.sync_threads(producer_barrier, producer_threads)
                    if tx < consumer_threads + tma_threads:
                        load_kv(
                            K,
                            V,
                            kv_layout,
                            bid,
                            hid,
                            base,
                            k,
                            ks,
                            vs,
                            load_ready,
                        )
                        if tma_rope_tables:
                            T.tma_copy(
                                rope_cos[
                                    base + k * block_n : base + (k + 1) * block_n,
                                    :,
                                ],
                                rope_cos_s,
                                barrier=loaded[k % RING_DEPTH],
                            )
                            T.tma_copy(
                                rope_sin[
                                    base + k * block_n : base + (k + 1) * block_n,
                                    :,
                                ],
                                rope_sin_s,
                                barrier=loaded[k % RING_DEPTH],
                            )
                        T.mbarrier_arrive(load_ready[k % RING_DEPTH])
                    if rope_pipeline:
                        T.mbarrier_wait_parity(
                            loaded[k % RING_DEPTH],
                            (k // RING_DEPTH) % RING_DEPTH,
                        )
                        for i, freq in T.Parallel(block_n, rotary_dim // 2):
                            if i < this_len - k * block_n:
                                d0, d1 = rope_channel_pair(rope_layout, rotary_dim // 2, freq)
                                x0 = ks[k % RING_DEPTH, i, d0]
                                x1 = ks[k % RING_DEPTH, i, d1]
                                if tma_rope_tables:
                                    cos = rope_cos_s[i, freq]
                                    sin = rope_sin_s[i, freq]
                                else:
                                    cos = rope_cos[base + k * block_n + i, freq]
                                    sin = rope_sin[base + k * block_n + i, freq]
                                ks[k % RING_DEPTH, i, d0] = x0 * cos - x1 * sin
                                ks[k % RING_DEPTH, i, d1] = x1 * cos + x0 * sin
                        T.mbarrier_arrive(ready[k % RING_DEPTH])
            else:
                consumer(
                    Q,
                    bid,
                    hid,
                    sid,
                    this_len,
                    seqlen_kv_b,
                    loop_range,
                    rope_cos,
                    rope_sin,
                    qs,
                    ks,
                    vs,
                    ps,
                    ready,
                    free,
                    acc_s,
                    acc_o,
                    sm,
                    smp,
                    alpha,
                    ss,
                    logsum,
                    glse,
                    output_partial,
                )

    return split


def make_gqa_decode_bs1_combine(
    batch: int,
    heads: int,
    ctx_splits: int,
    dim: int,
    dtype: str,
    accum_dtype: str,
    has_sinks: bool = False,
    sink_heads: int | None = None,
):
    """Create the paging-independent split-output combine macro.

    ``heads`` counts the output rows of one batch entry. ``sink_heads`` is the number of
    attention heads when those rows hold several query positions, row ``hq`` taking sink
    ``hq % sink_heads``; ``None`` means one row a head.
    """
    sink_merge = make_dense_split_sink_merge(accum_dtype)

    @T.macro
    def combine(glse, Output_partial, Output, sinks=None):
        with T.Kernel(heads, batch, threads=128) as (hq, bid):
            lse_vec = T.alloc_fragment([ctx_splits], accum_dtype)
            lse_max = T.alloc_fragment([1], accum_dtype)
            lse_logsum = T.alloc_local([1], accum_dtype)
            o_accum = T.alloc_fragment([dim], accum_dtype)

            for s in T.Parallel(ctx_splits):
                lse_vec[s] = glse[bid, hq, s]
            T.fill(lse_max, -T.infinity(accum_dtype))
            T.reduce_max(lse_vec, lse_max, dim=0, clear=False)
            lse_max[0] = T.if_then_else(lse_max[0] == -T.infinity(accum_dtype), 0.0, lse_max[0])
            lse_logsum[0] = 0
            if has_sinks:
                sink_merge(
                    sinks[hq if sink_heads is None else hq % sink_heads], lse_max, lse_logsum
                )
            for s in T.serial(ctx_splits):
                lse_logsum[0] += T.exp2(glse[bid, hq, s] - lse_max[0])
            if not has_sinks:
                lse_logsum[0] = T.log2(lse_logsum[0]) + lse_max[0]
            T.clear(o_accum)
            for s in T.serial(ctx_splits):
                if has_sinks:
                    # Keep the denominator separate: adding log2(sum) to a large maximum
                    # can round away the sink contribution.
                    partial_weight = T.exp2(glse[bid, hq, s] - lse_max[0]) / lse_logsum[0]
                else:
                    partial_weight = T.exp2(glse[bid, hq, s] - lse_logsum[0])
                weight = T.if_then_else(
                    glse[bid, hq, s] == -T.infinity(accum_dtype),
                    0.0,
                    partial_weight,
                )
                for j in T.Parallel(dim):
                    o_accum[j] += Output_partial[bid, hq, s, j] * weight
            for j in T.Parallel(dim):
                Output[bid, hq, j] = T.cast(o_accum[j], dtype)

    return combine
