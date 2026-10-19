"""SM90 warp-specialized grouped-query attention over a paged FP8 cache the kernel only reads.

A persistent CTA per SM runs a producer warpgroup and two consumer warpgroups. A work item is
one KV head and 128 query rows: ``128 // group`` query positions of each of the head's
``group`` query heads, so the cache crosses memory once per KV head.

The producer warpgroup has two roles. Its first warp claims items and issues TMA copies of
the stored e4m3 key and value tiles, a page or part of one per copy. Its other three warps
convert the landed tiles into the 16-bit operands the consumers multiply, which keeps the
conversion off the consumers' critical path; they also zero the value rows a page holds past
the cache end, whose stale bits a zero weight would not cancel. An e4m3 value converts
exactly into either 16-bit type.

An FP8 query multiplies the stored keys in e4m3 directly, and the values in float16; a 16-bit
query multiplies both converted tiles in its own dtype. The probabilities stay 16-bit. The
dequantization scales are per item: the query's and the keys' join the softmax's exponent
scale and the values' joins the output's row scale.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops.kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    AttentionCall,
    GQAPagedFwdInterface,
)
from tileops.kernels.attention.gqa.paged_ws import SPLIT_MIN_TILES
from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin, get_sm_count

__all__ = ["GQAPagedFP8WSKernel"]

_FP8 = "float8_e4m3fn"


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        # Row r of a tile stores query r % span of head r // span, which is injective in r;
        # the checker cannot prove it and reports the output store as a race.
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
    compile_flags=[
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-DNDEBUG",
        *csrc_include("fp8_paged_helper.h"),
    ],
)
def _gqa_paged_fp8_ws_kernel(
    batch, heads, heads_kv, dim, page_size, max_pages_per_req, is_causal, sm_scale, softcap,
    dtype, q_dtype, per_request_kv_scale, block_n, stages, num_ctas, split_small,
):  # fmt: skip
    """A persistent CTA per SM: a TMA warp and three converting warps feed two consumers."""
    q_fp8 = q_dtype != dtype
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else sm_scale * LOG2E
    group = heads // heads_kv
    accum = "float"
    # The 16-bit type the converted operands take: an FP8 query multiplies its values in
    # float16, whatever the output type, and a 16-bit query fixes it.
    wide = "float16" if q_fp8 else dtype
    # The query and the keys meet in e4m3 for an FP8 query, and in the 16-bit type otherwise.
    qk_dtype = _FP8 if q_fp8 else dtype
    half = 64  # rows per consumer warpgroup: one m64 WGMMA
    block_m = 2 * half
    consumers = 256  # threads of the two consumer warpgroups
    converters = 96  # the producer warpgroup's three converting warps
    meta_count = 7
    policy = T.GemmWarpPolicy.FullRow
    total_q = T.dynamic("total_q")
    pool_rows = T.dynamic("pool_rows")
    kv_scale_shape = (batch, heads_kv) if per_request_kv_scale else (1,)
    # Row r of a tile is query r % span of head kv_head * group + r // span.
    span = block_m // group
    tiling = GroupTiling(batch, span)
    # A key tile is part of one page or a run of whole pages: one TMA copy per page.
    box = min(block_n, page_size)
    boxes = block_n // box
    # A TMA box spans 128 bytes of a row, the widest that lands on the swizzled buffer.
    q_cols = 128 // (1 if q_fp8 else 2)
    kv_cols = 128
    # The C++ name of the converted type.
    wide_c = "half_t" if wide == "float16" else "bfloat16_t"
    # Marks the stage that ends the converting warps' loop.
    stop = -(1 << 30)
    # With split_small, a request whose rows fit one warpgroup, over at least this many key
    # tiles, splits its tiles between the two warpgroups, which merge at the end.
    split_min_tiles = SPLIT_MIN_TILES

    @T.macro
    def apply_softcap(acc_s):
        for i, j in T.Parallel(half, block_n):
            capped = T.cast(softcap, accum) * T.tanh(
                acc_s[i, j] * T.cast(sm_scale / softcap, accum)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum), -T.infinity(accum), capped
            )

    @T.macro
    def mask(acc_s, key0, q0, wg, align, kv_len):
        """Send the scores of keys past the cache end, or past a row's position, to -inf."""
        for i, j in T.Parallel(half, block_n):
            if is_causal:
                hidden = (key0 + j >= kv_len) | (key0 + j > q0 + (wg * half + i) % span + align)
            else:
                hidden = key0 + j >= kv_len
            acc_s[i, j] = T.if_then_else(hidden, -T.infinity(accum), acc_s[i, j])

    @T.macro
    def descale(acc_s, qk_descale):
        """Dequantize a score tile where a softcap reads it before the softmax does."""
        for i, j in T.Parallel(half, block_n):
            acc_s[i, j] *= qk_descale

    @T.macro
    def softmax_step(acc_s, sm, smp, alpha, ss, red, item_scale):
        """Fold one score tile into the running max, rescale factor, and row sums."""
        if use_softcap:
            apply_softcap(acc_s)
        T.copy(sm, smp)
        T.reduce_max(T.reshape(acc_s, [half, block_n // 8, 8]), red, dim=1, clear=True)
        T.reduce_max(red, sm, dim=1, clear=False, batch=2)
        for i in T.Parallel(half):
            sm[i] = T.if_then_else((sm[i] - smp[i]) * item_scale > 8.0, sm[i], smp[i])
            alpha[i] = 1.0
            if sm[i] != smp[i]:
                alpha[i] = T.exp2(smp[i] * item_scale - sm[i] * item_scale)
        for i, j in T.Parallel(half, block_n):
            acc_s[i, j] = T.exp2(acc_s[i, j] * item_scale - sm[i] * item_scale)
        T.reduce_sum(acc_s, ss, dim=1, batch=2)

    @T.macro
    def kv_step(
        q_tile, Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp, alpha, ss,
        red, logsum, my_bar, nxt_bar, k, n, q0, wg, align, kv_len, tail: bool, item_scale,
        qk_descale,
    ):  # fmt: skip
        """QK of key tile k overlapped with PV of tile k - 1; n counts tiles across items."""
        sk = n % stages
        svp = (n - 1) % stages
        T.sync_threads(my_bar, consumers)
        T.mbarrier_wait_parity(kready[sk], (n // stages) % 2)
        T.wgmma_gemm(q_tile, Ks[sk, :, :], acc_s, transpose_B=True, policy=policy, clear_accum=True)
        rescale = T.alloc_var("int32", init=0)
        for i in T.Parallel(half):
            if alpha[i] != 1.0:
                rescale = 1
        if rescale == 1:
            for i, j in T.Parallel(half, dim):
                acc_o[i, j] *= alpha[i]
        T.mbarrier_wait_parity(vready[svp], ((n - 1) // stages) % 2)
        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
        T.named_barrier_arrive(nxt_bar, consumers)
        T.wait_wgmma(1)
        T.mbarrier_arrive(kfree[sk])
        if use_softcap:
            descale(acc_s, qk_descale)
        if tail:
            mask(acc_s, k * block_n, q0, wg, align, kv_len)
        softmax_step(acc_s, sm, smp, alpha, ss, red, item_scale)
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        for i in T.Parallel(half):
            logsum[i] = logsum[i] * alpha[i] + ss[i]
        T.copy(acc_s, pcast)

    @T.macro
    def locate(work, CuQ, cache_seqlens, tile_cum, lo, hi, request, q_row, meta):
        """Fill meta (kv head, request, q_start, q_len, kv_len, q0, tiles) for *work*.

        Requests run in order, each from its last query tile back, so a causal request's
        longest tiles start first and the split one-tile requests at the packing's end fill
        the closing wave.
        """
        if split_small:
            tiling.decode(work // heads_kv, tile_cum, lo, hi, request, q_row)
            q_row[0] = (tile_cum[request[0] + 1] - tile_cum[request[0]] - 1) * span - q_row[0]
        else:
            tiling.decode(tile_cum[batch] - 1 - work // heads_kv, tile_cum, lo, hi, request, q_row)
        meta[0] = work % heads_kv
        meta[1] = request[0]
        meta[2] = CuQ[request[0]]
        meta[3] = CuQ[request[0] + 1] - meta[2]
        meta[4] = cache_seqlens[request[0]]
        meta[5] = q_row[0]
        if is_causal:
            # The tile's last row sees the most keys: query positions end at the cache end.
            last_q = T.min(meta[3] - 1, meta[5] + span - 1) + meta[4] - meta[3]
            key_end = T.min(meta[4], last_q + 1)
        else:
            key_end = meta[4]
        # At least one tile, so a cache shorter than its queries, outside the op's contract,
        # cannot leave the consumers waiting on a tile the producer never loads.
        meta[6] = T.max(1, T.ceildiv(key_end, block_n))

    @T.macro
    def fetch(Sched, lane, work):
        """Claim the next work item into work[0]: lane 0 claims, the warp shares it."""
        if lane == 0:
            work[0] = T.atomic_add(Sched[0], 1, return_prev=True)
        work[0] = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), work[0], 0, 32, 32)

    @T.macro
    def tile_rows(page_table, rows, request, key0, kv_len):
        """The pool row each box of the key tile at *key0* starts at; a page past the cache
        end repeats the last one, whose rows the mask and the value zeroing discard."""
        last_page = T.max(0, (kv_len - 1) // page_size)
        for b in T.unroll(boxes):
            key = key0 + b * box
            rows[b] = page_table[request, T.min(key // page_size, last_page)] * page_size + (
                key % page_size
            )

    @T.macro
    def load_tile(pool, dst, stage, bars, bar, rows, cv):
        """TMA one key tile of *pool* into stage *stage* of *dst*, landing on ``bars[bar]``."""
        for b in T.unroll(boxes):
            for c in T.unroll(dim // kv_cols):
                T.tma_copy(
                    pool[rows[b] : rows[b] + box, cv, c * kv_cols : (c + 1) * kv_cols],
                    dst[stage, b * box : (b + 1) * box, c * kv_cols : (c + 1) * kv_cols],
                    barrier=bars[bar],
                )

    @T.macro
    def load_q(Q, Qs, slot, q_bar, q_start, cv):
        """TMA each head's run of the tile's queries into the warpgroup that owns its rows; a
        head spans both warpgroups when the tile holds more queries than one has rows."""
        rows = min(half, span)
        for wg in T.unroll(2):
            for h in T.unroll(max(1, half // span)):
                head = cv * group + (wg * half) // span + h
                first = q_start + (wg * half) % span
                for c in T.unroll(dim // q_cols):
                    T.tma_copy(
                        Q[first : first + rows, head, c * q_cols : (c + 1) * q_cols],
                        Qs[slot, wg, h * rows : (h + 1) * rows, c * q_cols : (c + 1) * q_cols],
                        barrier=q_bar[slot],
                    )

    @T.macro
    def producer(
        Q, K, V, CuQ, cache_seqlens, page_table, Sched, Qs, Kr, Vr, item_slot, valid, q_bar,
        qfree, kr_ready, kr_free, vr_ready, vr_free, total, tile_cum, lo, hi, request, q_row,
        meta, lane,
    ):  # fmt: skip
        """Claim work items until none remain; load each one's Q, then its key tiles."""
        issued = T.alloc_var("int32", init=0)
        loaded = T.alloc_var("int32", init=0)
        work = T.alloc_local([1], "int32")
        rows = T.alloc_local([boxes], "int32")
        fetch(Sched, lane, work)
        while work[0] < total:
            locate(work[0], CuQ, cache_seqlens, tile_cum, lo, hi, request, q_row, meta)
            cv = meta[0]
            req = meta[1]
            kv_len = meta[4]
            slot = loaded % 2
            T.mbarrier_wait_parity(qfree[slot], ((loaded // 2) % 2) ^ 1)
            # The consumers read the item's geometry here rather than decode it again.
            for i in T.unroll(meta_count):
                item_slot[slot, i] = meta[i]
            item_slot[slot, meta_count] = work[0]
            # A split item's consumers gather its query rows themselves.
            if split_small:
                if (meta[3] * group > half) | (meta[6] < split_min_tiles):
                    load_q(Q, Qs, slot, q_bar, meta[2] + meta[5], cv)
            else:
                load_q(Q, Qs, slot, q_bar, meta[2] + meta[5], cv)
            T.mbarrier_arrive(q_bar[slot])
            for k in T.serial(meta[6]):
                n = issued + k
                s = n % stages
                key0 = k * block_n
                # Read before the wait, so the table's latency hides behind the consumers.
                tile_rows(page_table, rows, req, key0, kv_len)
                T.mbarrier_wait_parity(kr_free[s], ((n // stages) % 2) ^ 1)
                load_tile(K, Kr, s, kr_ready, s, rows, cv)
                T.mbarrier_arrive(kr_ready[s])
                T.mbarrier_wait_parity(vr_free[s], ((n // stages) % 2) ^ 1)
                # The rows of the tile that lie inside the cache; the converters zero the rest.
                if lane == 0:
                    valid[s] = kv_len - key0
                load_tile(V, Vr, s, vr_ready, s, rows, cv)
                T.mbarrier_arrive(vr_ready[s])
            issued += meta[6]
            loaded += 1
            fetch(Sched, lane, work)
        # Stop the converters and the consumers, then leave the counters zeroed.
        s_end = issued % stages
        T.mbarrier_wait_parity(vr_free[s_end], ((issued // stages) % 2) ^ 1)
        if lane == 0:
            valid[s_end] = stop
        T.mbarrier_arrive(vr_ready[s_end])
        if not q_fp8:
            T.mbarrier_wait_parity(kr_free[s_end], ((issued // stages) % 2) ^ 1)
            T.mbarrier_arrive(kr_ready[s_end])
        T.mbarrier_wait_parity(qfree[loaded % 2], ((loaded // 2) % 2) ^ 1)
        item_slot[loaded % 2, meta_count] = -1
        T.mbarrier_arrive(q_bar[loaded % 2])
        if lane == 0:
            finished = T.atomic_add(Sched[1], 1, return_prev=True)
            if finished == num_ctas - 1:
                Sched[0] = 0
                Sched[1] = 0

    @T.macro
    def convert_tile(src, dst, s, t, valid_rows):
        """Convert stage *s* of *src* into *dst*; a row at or past *valid_rows* converts to
        zero. 16 stored bytes a step, in the two buffers' swizzled layouts."""
        T.call_extern(
            "handle",
            f"tileops::fp8_tile_to_wide_sw128<{block_n}, {converters}, {wide_c}>",
            T.access_ptr(src[s, 0, 0], "r"),
            T.access_ptr(dst[s, 0, 0], "w"),
            t,
            valid_rows,
        )

    @T.macro
    def converter(Kr, Vr, Kc, Vc, valid, kr_ready, kr_free, kc_ready, kc_free, vr_ready,
                  vr_free, vc_ready, vc_free, t):  # fmt: skip
        """Convert each landed tile into the consumers' 16-bit stage until the stop mark."""
        n = T.alloc_var("int32", init=0)
        running = T.alloc_var("int32", init=1)
        while running == 1:
            s = n % stages
            phase = (n // stages) % 2
            if not q_fp8:
                T.mbarrier_wait_parity(kr_ready[s], phase)
            T.mbarrier_wait_parity(vr_ready[s], phase)
            rows_in = valid[s]
            if rows_in == stop:
                running = 0
            else:
                if not q_fp8:
                    T.mbarrier_wait_parity(kc_free[s], phase ^ 1)
                    convert_tile(Kr, Kc, s, t, block_n)
                    T.mbarrier_arrive(kr_free[s])
                    # The consumers read the tile through the async proxy.
                    T.fence_proxy_async()
                    T.mbarrier_arrive(kc_ready[s])
                T.mbarrier_wait_parity(vc_free[s], phase ^ 1)
                convert_tile(Vr, Vc, s, t, rows_in)
                T.mbarrier_arrive(vr_free[s])
                T.fence_proxy_async()
                T.mbarrier_arrive(vc_ready[s])
                n += 1

    @T.macro
    def split_item(
        wg: int, q_frag, Ks, Vs, O, kready, kfree, vready, vfree, mrg, mrg_o, acc_s, pcast,
        acc_o, sm, smp, alpha, ss, red, logsum, kv_head, q_start, q_len, kv_len, eff, done,
        item_scale, qk_descale, v_descale,
    ):  # fmt: skip
        """Both warpgroups attend one request's rows, at most 64, held position-major in
        ``q_frag``: warpgroup *wg* scans key tiles wg, wg + 2, ... and arrives on the others'
        barriers so that their phases stay shared; warpgroup 1 hands its normalized half to
        warpgroup 0, which merges."""
        rows = q_len * group
        align = kv_len - q_len
        for k in T.serial(eff):
            s = (done + k) % stages
            phase = ((done + k) // stages) % 2
            T.mbarrier_wait_parity(kready[s], phase)
            if k % 2 == wg:
                T.wgmma_gemm(
                    q_frag, Ks[s, :, :], acc_s, transpose_B=True, policy=policy, clear_accum=True
                )
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[s])
                if use_softcap:
                    descale(acc_s, qk_descale)
                key0 = k * block_n
                if (key0 + block_n > kv_len) | (key0 + block_n > align + 1 if is_causal else False):
                    for i, j in T.Parallel(half, block_n):
                        hidden = key0 + j >= kv_len
                        if is_causal:
                            hidden = hidden | (key0 + j > i // group + align)
                        acc_s[i, j] = T.if_then_else(hidden, -T.infinity(accum), acc_s[i, j])
                softmax_step(acc_s, sm, smp, alpha, ss, red, item_scale)
                for i in T.Parallel(half):
                    logsum[i] = logsum[i] * alpha[i] + ss[i]
                for i, d in T.Parallel(half, dim):
                    acc_o[i, d] *= alpha[i]
                T.copy(acc_s, pcast)
                T.mbarrier_wait_parity(vready[s], phase)
                T.wgmma_gemm(pcast, Vs[s, :, :], acc_o, policy=policy, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vfree[s])
            else:
                T.mbarrier_arrive(kfree[s])
                T.mbarrier_wait_parity(vready[s], phase)
                T.mbarrier_arrive(vfree[s])
        if wg == 1:
            for i, d in T.Parallel(half, dim):
                mrg_o[i, d] = T.cast(
                    T.if_then_else(logsum[i] > 0, acc_o[i, d] / logsum[i], 0), wide
                )
            for i in T.Parallel(half):
                mrg[0, i] = sm[i]
                mrg[1, i] = logsum[i]
        T.sync_threads(3, consumers)
        if wg == 0:
            for i in T.Parallel(half):
                peak = T.max(sm[i], mrg[0, i])
                own = T.exp2((sm[i] - peak) * item_scale)
                other = T.exp2((mrg[0, i] - peak) * item_scale) * mrg[1, i]
                total = logsum[i] * own + other
                alpha[i] = T.if_then_else(total > 0, own * v_descale / total, 0)
                ss[i] = T.if_then_else(total > 0, other * v_descale / total, 0)
            for i, d in T.Parallel(half, dim):
                if i < rows:
                    O[q_start + i // group, kv_head * group + i % group, d] = T.cast(
                        acc_o[i, d] * alpha[i] + T.cast(mrg_o[i, d], accum) * ss[i], dtype
                    )
        # Warpgroup 0 has read the handed-over half before the next item reuses it.
        T.sync_threads(3, consumers)

    @T.macro
    def consumer(
        wg: int, Q, Qs, Ks, Vs, O, QS, KS, VS, item_slot, q_bar, qfree, kready, kfree, vready,
        vfree, meta, mrg, mrg_o,
    ):  # fmt: skip
        """Consumer warpgroup *wg*: rows ``wg*64`` onward of each claimed item's tile."""
        # 2 x 128 x 224 + 128 x 56 registers: the 64512 the multiprocessor grants.
        T.set_max_nreg(224, 1)
        my_bar = 1 + wg
        nxt_bar = 2 - wg
        # The query tile is the register operand of every key product of the item: read from
        # shared memory once, it leaves the shared-memory port to the key and value tiles.
        q_frag = T.alloc_fragment([half, dim], qk_dtype)
        acc_s = T.alloc_fragment([half, block_n], accum)
        pcast = T.alloc_fragment([half, block_n], wide)
        acc_o = T.alloc_fragment([half, dim], accum)
        sm = T.alloc_fragment([half], accum)
        smp = T.alloc_fragment([half], accum)
        alpha = T.alloc_fragment([half], accum)
        ss = T.alloc_fragment([half], accum)
        red = T.alloc_fragment([half, 8], accum)
        logsum = T.alloc_fragment([half], accum)
        full = T.alloc_var("int32", init=0)
        # Finished key tiles and items: they carry the barrier phases.
        done = T.alloc_var("int32", init=0)
        served = T.alloc_var("int32", init=0)
        work = T.alloc_var("int32", init=0)
        qk_descale = T.alloc_var(accum, init=1.0)
        item_scale = T.alloc_var(accum, init=scale)
        v_descale = T.alloc_var(accum, init=1.0)
        T.mbarrier_wait_parity(q_bar[0], 0)
        work = item_slot[0, meta_count]
        while work >= 0:
            for i in T.unroll(meta_count):
                meta[i] = item_slot[served % 2, i]
            kv_head = meta[0]
            req = meta[1]
            q_start = meta[2]
            q_len = meta[3]
            kv_len = meta[4]
            # Query position p of the request is key align + p.
            align = kv_len - q_len
            q0 = meta[5]
            eff = meta[6]
            if per_request_kv_scale:
                qk_descale = KS[req, kv_head]
                v_descale = VS[req, kv_head]
            else:
                qk_descale = KS[0]
                v_descale = VS[0]
            if q_fp8:
                qk_descale = qk_descale * QS[req, kv_head]
            # A softcap reads the dequantized score; otherwise the descale joins the exponent.
            item_scale = scale if use_softcap else scale * qk_descale
            # A build without the split traces only the shared scan below.
            split = T.alloc_var("int32", init=0) if split_small else 0
            # Nested: `and` between a Python bool and a TIR comparison is not TIR.
            if split_small:  # noqa: SIM102
                if (q_len * group <= half) & (eff >= split_min_tiles):
                    split = 1
            slot = served % 2
            if split == 1:
                # Position-major rows, row r query r // group of head r % group, all in one
                # warpgroup's half; warpgroup 0 gathers them for both.
                if wg == 0:
                    for i, d in T.Parallel(half, dim):
                        r = T.min(i, q_len * group - 1)
                        Qs[slot, 0, i, d] = T.if_then_else(
                            i < q_len * group,
                            Q[q_start + r // group, kv_head * group + r % group, d],
                            T.cast(0, qk_dtype),
                        )
                T.sync_threads(3, consumers)
                T.copy(Qs[slot, 0, :, :], q_frag)
            else:
                T.copy(Qs[slot, wg, :, :], q_frag)
            # The query is in registers, so the producer may load the next item's.
            T.mbarrier_arrive(qfree[slot])
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(alpha, 1.0)
            T.fill(sm, -T.infinity(accum))
            if wg == 1 and served == 0:
                T.named_barrier_arrive(1, consumers)  # let warpgroup 0 go first
            if split == 1:
                split_item(
                    wg, q_frag, Ks, Vs, O, kready, kfree, vready, vfree, mrg, mrg_o, acc_s,
                    pcast, acc_o, sm, smp, alpha, ss, red, logsum, kv_head, q_start, q_len,
                    kv_len, eff, done, item_scale, qk_descale, v_descale,
                )  # fmt: skip
            else:
                # Key tile 0: QK and softmax only.
                s0 = done % stages
                T.sync_threads(my_bar, consumers)
                T.mbarrier_wait_parity(kready[s0], (done // stages) % 2)
                T.wgmma_gemm(
                    q_frag, Ks[s0, :, :], acc_s, transpose_B=True, policy=policy, clear_accum=True
                )
                T.named_barrier_arrive(nxt_bar, consumers)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[s0])
                # Tiles before `full` are inside every row's visible range and need no mask.
                if is_causal:
                    # The warpgroup's first row holds its earliest query.
                    first_q = q0 + (wg * half) % span + align
                    full = T.min(eff, T.max(0, (first_q + 1) // block_n))
                else:
                    full = T.min(eff, kv_len // block_n)
                if use_softcap:
                    descale(acc_s, qk_descale)
                if full < 1:
                    mask(acc_s, 0, q0, wg, align, kv_len)
                if use_softcap:
                    apply_softcap(acc_s)
                T.reduce_max(acc_s, sm, dim=1, clear=False)
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * item_scale - sm[i] * item_scale)
                T.reduce_sum(acc_s, ss, dim=1, batch=2)
                for i in T.Parallel(half):
                    logsum[i] = ss[i]
                T.copy(acc_s, pcast)

                step = (Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
                for k in T.serial(1, T.max(1, full)):
                    kv_step(
                        q_frag, *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                        k, done + k, q0, wg, align, kv_len, False, item_scale, qk_descale,
                    )  # fmt: skip
                for k in T.serial(T.max(1, full), eff):
                    kv_step(
                        q_frag, *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                        k, done + k, q0, wg, align, kv_len, True, item_scale, qk_descale,
                    )  # fmt: skip

                # PV of the last tile.
                svp = (done + eff - 1) % stages
                for i, j in T.Parallel(half, dim):
                    acc_o[i, j] *= alpha[i]
                T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
                T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vfree[svp])

                for i in T.Parallel(half):
                    alpha[i] = v_descale / logsum[i]
                for i, d in T.Parallel(half, dim):
                    r = wg * half + i
                    if q0 + r % span < q_len:
                        O[q_start + q0 + r % span, kv_head * group + r // span, d] = T.cast(
                            acc_o[i, d] * alpha[i], dtype
                        )
            done += eff
            served += 1
            T.mbarrier_wait_parity(q_bar[served % 2], (served // 2) % 2)
            work = item_slot[served % 2, meta_count]

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, heads, dim], q_dtype),
        K: T.Tensor([pool_rows, heads_kv, dim], _FP8),
        V: T.Tensor([pool_rows, heads_kv, dim], _FP8),
        cache_seqlens: T.Tensor([batch], "int32"),
        page_table: T.Tensor([batch, max_pages_per_req], "int32"),
        CuQ: T.Tensor([batch + 1], "int32"),
        QS: T.Tensor([batch, heads_kv], accum),
        KS: T.Tensor(kv_scale_shape, accum),
        VS: T.Tensor(kv_scale_shape, accum),
        O: T.Tensor([total_q, heads, dim], dtype),
        Sched: T.Tensor([2], "int32"),
    ):
        with T.Kernel(num_ctas, threads=384):
            Qs = T.alloc_shared([2, 2, half, dim], qk_dtype)
            Kr = T.alloc_shared([stages, block_n, dim], _FP8)
            Vr = T.alloc_shared([stages, block_n, dim], _FP8)
            # An FP8 query multiplies the stored keys; a 16-bit one, their conversion.
            Kc = Kr if q_fp8 else T.alloc_shared([stages, block_n, dim], wide)
            Vc = T.alloc_shared([stages, block_n, dim], wide)
            tile_cum = T.alloc_shared([batch + 1], "int32")
            item_slot = T.alloc_shared([2, meta_count + 1], "int32")
            valid = T.alloc_shared([stages], "int32")
            # A split item's warpgroup 1 hands warpgroup 0 its normalized rows, row maxima and
            # sums here.
            mrg_o = T.alloc_shared([half, dim], wide)
            mrg = T.alloc_shared([2, half], accum)
            layouts = {
                Qs: make_swizzled_layout(Qs),
                Kr: make_swizzled_layout(Kr),
                Vr: make_swizzled_layout(Vr),
                Vc: make_swizzled_layout(Vc),
            }
            if not q_fp8:
                layouts[Kc] = make_swizzled_layout(Kc)
            T.annotate_layout(layouts)
            q_bar = T.alloc_barrier([32, 32])
            qfree = T.alloc_barrier([consumers, consumers])
            kr_ready = T.alloc_barrier([32] * stages)
            # The stored keys are released by whoever reads them.
            kr_free = T.alloc_barrier([consumers if q_fp8 else converters] * stages)
            vr_ready = T.alloc_barrier([32] * stages)
            vr_free = T.alloc_barrier([converters] * stages)
            kc_ready = kr_ready if q_fp8 else T.alloc_barrier([converters] * stages)
            kc_free = kr_free if q_fp8 else T.alloc_barrier([consumers] * stages)
            vc_ready = T.alloc_barrier([converters] * stages)
            vc_free = T.alloc_barrier([consumers] * stages)
            lo = T.alloc_local([1], "int32")
            hi = T.alloc_local([1], "int32")
            q_row = T.alloc_local([1], "int32")
            request = T.alloc_local([1], "int32")
            meta = T.alloc_local([meta_count], "int32")

            tiling.cumsum_offsets(CuQ, tile_cum)
            T.sync_threads()
            tx = T.get_thread_binding()
            if tx >= consumers:
                T.set_max_nreg(56, 0)  # the producer warpgroup only copies and converts
            if tx >= consumers and tx < consumers + 32:
                producer(
                    Q, K, V, CuQ, cache_seqlens, page_table, Sched, Qs, Kr, Vr, item_slot,
                    valid, q_bar, qfree, kr_ready, kr_free, vr_ready, vr_free,
                    tile_cum[batch] * heads_kv, tile_cum, lo, hi, request, q_row, meta,
                    tx - consumers,
                )  # fmt: skip
            if tx >= consumers + 32:
                converter(
                    Kr, Vr, Kc, Vc, valid, kr_ready, kr_free, kc_ready, kc_free, vr_ready,
                    vr_free, vc_ready, vc_free, tx - consumers - 32,
                )  # fmt: skip
            args = (Q, Qs, Kc, Vc, O, QS, KS, VS, item_slot, q_bar, qfree, kc_ready, kc_free,
                    vc_ready, vc_free, meta, mrg, mrg_o)  # fmt: skip
            with T.ws(0):
                consumer(0, *args)
            with T.ws(1):
                consumer(1, *args)

    return main


class GQAPagedFP8WSKernel(Kernel, GQAPagedFwdInterface):
    """SM90 warp-specialized paged attention over an FP8 cache for packings whose requests
    fill query tiles.

    It serves a float16 or bfloat16 output over an FP8 cache, the query 16-bit in that dtype
    or FP8, at head dimension 128, over pages of a multiple of 16 rows, causal or
    bidirectional, with a positive scale and an optional softcap, when the packed query rows
    average at least one 128-row tile a request. Windows, sinks and RoPE stay on
    ``GQAPagedFwdKernel``.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_paged_varlen_kernel", "gqa_paged_fp8_decode_kernel"})
    _DIMS: tuple[int, ...] = (128,)
    # A tile's 128 rows are whole heads' query runs, each at least the 8 rows of a swizzle
    # atom so that its TMA box starts on one.
    _GROUPS: tuple[int, ...] = (1, 2, 4, 8, 16)
    # Key tiles in flight: three where the rings fit, which lets the converters run two tiles
    # ahead of the consumers; two otherwise.
    _STAGE_CHOICES: tuple[int, ...] = (3, 2)
    _TILE_ROWS: int = 128
    # The per-request tile prefix shares shared memory with the tiles; 4096 requests take
    # 16 KB of what they leave.
    _MAX_BATCH: int = 4096

    @staticmethod
    def block_n_for(page_size: int) -> Optional[int]:
        """The widest key tile up to 128 rows that is part of one page or a run of whole
        pages, a multiple of the 16 rows one WGMMA step reduces over; ``None`` when none is."""
        for block_n in (128, 112, 96, 80, 64, 48, 32, 16):
            if page_size % block_n == 0 or block_n % page_size == 0 and page_size % 16 == 0:
                return block_n
        return None

    @classmethod
    def shared_bytes(cls, dim: int, block_n: int, stages: int, q_fp8: bool, batch: int) -> int:
        """Shared memory the program allocates: the two query slots, the stored key and value
        rings, the converted rings, and the per-request tile offsets."""
        q = 2 * cls._TILE_ROWS * dim * (1 if q_fp8 else 2)
        stored = 2 * stages * block_n * dim
        converted = (1 if q_fp8 else 2) * stages * block_n * dim * 2
        # A split item's handover: 64 normalized 16-bit rows, their maxima and sums.
        merge = cls._TILE_ROWS // 2 * (dim * 2 + 8)
        return q + stored + converted + merge + 4 * (batch + 1) + 256

    @classmethod
    def tiling_for(cls, call: AttentionCall) -> "tuple[int, int] | None":
        """The key tile and ring depth: the page's tile at the deepest ring that fits, else
        the tile halved down to 64 rows."""
        block_n = cls.block_n_for(call.page_size)
        if block_n is None:
            return None
        q_fp8 = call.query_dtype == torch.float8_e4m3fn
        cap = get_shared_memory_optin(call.device.index if call.device is not None else None)
        while True:
            for stages in cls._STAGE_CHOICES:
                if cls.shared_bytes(call.dim, block_n, stages, q_fp8, call.batch) <= cap:
                    return block_n, stages
            if block_n <= 64:
                return block_n, cls._STAGE_CHOICES[-1]
            block_n //= 2

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why *call* is outside this implementation's region, or ``None`` when it is inside."""
        if call.dtype not in ATTENTION_DTYPES:
            return "requires a float16 or bfloat16 output"
        if call.cache_dtype != torch.float8_e4m3fn:
            return "requires an FP8 cache"
        if call.query_dtype not in (None, call.dtype, torch.float8_e4m3fn):
            return "requires a query of the output dtype or FP8"
        if call.dim not in cls._DIMS:
            return f"requires head dimension in {cls._DIMS}"
        if call.heads_kv <= 0 or call.heads // call.heads_kv not in cls._GROUPS:
            return f"requires a query group of {cls._GROUPS} heads"
        if call.sm_scale is not None and call.sm_scale <= 0.0:
            return "requires a positive scale"
        if call.page_size <= 0 or cls.block_n_for(call.page_size) is None:
            return "requires a page size that is a multiple of 16 rows"
        if call.uses_sliding_window or call.has_sinks or call.fuse_rope:
            return "does not serve windows, sinks or RoPE"
        if call.max_seqlen_q * (call.heads // call.heads_kv) < cls._TILE_ROWS * call.batch:
            return "requires packed query rows averaging one 128-row tile a request"
        if call.batch > cls._MAX_BATCH:
            return f"serves at most {cls._MAX_BATCH} requests"
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The packed totals are read from the tensors, so one object serves every packing of
        these call facts; the device index is in the identity, as the SM count is."""
        index = call.device.index if call.device is not None else None
        args = dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            dim=call.dim,
            page_size=call.page_size,
            max_pages_per_req=call.max_pages_per_req,
            is_causal=call.is_causal,
            dtype=call.dtype,
            query_dtype=call.query_dtype or call.dtype,
            per_request_kv_scale=call.per_request_kv_scale,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            tiling=cls.tiling_for(call),
        )
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        page_size: int,
        max_pages_per_req: int,
        is_causal: bool,
        dtype: torch.dtype = torch.float16,
        query_dtype: Optional[torch.dtype] = None,
        per_request_kv_scale: bool = False,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        tiling: "tuple[int, int]" = (128, 2),
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.page_size = page_size
        self.max_pages_per_req = max_pages_per_req
        self.is_causal = is_causal
        self.dtype = dtype
        self.query_dtype = query_dtype or dtype
        self.per_request_kv_scale = per_request_kv_scale
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.block_n, self.stages = tiling
        # Whether a request's key tiles split between the warpgroups; a table too short for
        # SPLIT_MIN_TILES tiles builds without the split.
        self.split_small = max_pages_per_req * page_size >= SPLIT_MIN_TILES * self.block_n
        device = torch.device("cuda", self.device_index)
        self._unit_q_scale = torch.ones((batch, heads_kv), dtype=torch.float32, device=device)
        # One counter per stream: concurrent launches on one counter steal each other's items.
        self._counters: dict = {}
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def kernel(self):
        return _gqa_paged_fp8_ws_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.page_size,
            self.max_pages_per_req,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            str(self.query_dtype).removeprefix("torch."),
            self.per_request_kv_scale,
            self.block_n,
            self.stages,
            get_sm_count(self.device_index),
            self.split_small,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q.shape[0] == 0:
            return torch.empty(q.shape, dtype=self.dtype, device=q.device)
        if torch.cuda.is_current_stream_capturing():
            # Each captured graph owns a counter from its private pool.
            counter = torch.zeros(2, dtype=torch.int32, device=q.device)
        else:
            stream = torch.cuda.current_stream(q.device).cuda_stream
            counter = self._counters.get(stream)
            if counter is None:
                counter = torch.zeros(2, dtype=torch.int32, device=q.device)
                self._counters[stream] = counter
        out = torch.empty(q.shape, dtype=self.dtype, device=q.device)
        self.kernel(
            q,
            k_pool,
            v_pool,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            q_scale if q_scale is not None else self._unit_q_scale,
            k_scale,
            v_scale,
            out,
            counter,
        )
        return out
