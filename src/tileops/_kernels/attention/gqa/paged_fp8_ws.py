"""SM90 warp-specialized GQA over a read-only paged FP8 cache.

A persistent CTA uses a producer warpgroup and two consumer warpgroups. The
producer copies FP8 K/V tiles and turns them into the consumers' operands,
zeroing stale V rows before they are consumed. Per-item scales are folded into
the score and output rescale; long small-row requests may split across consumer
warpgroups and FP8 runs.

An FP8 query multiplies the stored keys and values in e4m3, the probabilities
rounded to e4m3 in the registers that hold the scores: e4m3 WGMMA reads its B
operand K-major only, so the converting warps transpose each value tile,
ordering its keys so that each thread's scores are the probabilities it
supplies. Its row sums are per-thread partial sums of the unrounded
probabilities. A 16-bit query multiplies both tiles converted to its dtype.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops._kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    AttentionCall,
    GQAPagedFwdInterface,
)
from tileops._kernels.attention.gqa.paged_ws import scheduler_counter
from tileops._kernels.constants import LOG2E
from tileops._kernels.grouped_tiling import GroupTiling
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._utils import get_shared_memory_optin, get_sm_count

__all__ = ["GQAPagedFP8WSKernel"]


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        # Row r of a tile stores query r % span of head r // span, which is injective in r;
        # the checker cannot prove it and reports the output store as a race.
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        # The planner reads the warp roles' buffers as live at different times and gave the
        # split handover the addresses of the stored value ring, which copies still land in.
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
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
        *csrc_include("fp8_paged_helper.h", "warp_uniform.h"),
    ],
)
def _gqa_paged_fp8_ws_kernel(
    batch, heads, heads_kv, dim, page_size, max_pages_per_req, is_causal, sm_scale, softcap,
    dtype, q_dtype, per_request_kv_scale, block_n, stages, num_ctas, split_small,
    split_chunks, item_meta_words, split_min_tiles, fp8_pv_keys,
):  # fmt: skip
    """A persistent CTA per SM: a TMA warp and three converting warps feed two consumers."""
    fp8_dtype = "float8_e4m3fn"
    q_fp8 = q_dtype != dtype
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else sm_scale * LOG2E
    group = heads // heads_kv
    accum = "float"
    # The 16-bit type of a split request's partials and of a 16-bit query's converted
    # operands: float16 for an FP8 query, whatever the output type; a 16-bit query fixes it.
    wide = "float16" if q_fp8 else dtype
    # The query and the keys meet in e4m3 for an FP8 query, and in the 16-bit type otherwise.
    qk_dtype = fp8_dtype if q_fp8 else dtype
    # An FP8 query's value product reduces fp8_pv_keys key slots whatever the key tile:
    # slots past it hold zero probabilities and zero values.
    pv_keys = fp8_pv_keys if q_fp8 else block_n
    # The running row maximum moves only when a score passes it by more than 2^lazy, so a
    # probability reaches 2^lazy times 2^p_shift. An e4m3 probability takes 2^8 of the
    # 448 its range holds, which keeps the small ones out of its subnormals, and the
    # maximum moves with every larger score.
    lazy = 0.0 if q_fp8 else 8.0
    p_shift = 8.0 if q_fp8 else 0.0
    # An FP8 query scans an item's key tiles from the last back, the masked ones first: an
    # e4m3 probability rounds against the running maximum, so the keys nearest a causal
    # row's position round against their own maximum rather than flush below one that
    # earlier keys raised. A 16-bit probability has the range to scan forward.
    backward = q_fp8
    half = 64  # rows per consumer warpgroup: one m64 WGMMA
    block_m = 2 * half
    consumers = 256  # threads of the two consumer warpgroups
    converters = 96  # the producer warpgroup's three converting warps
    # A program that keeps one run a request builds the single-run scan alone.
    runs = split_chunks > 1
    # kv head, request, q_start, q_len, kv_len, q0, key tiles; with runs, also first key tile,
    # chunks, chunk
    meta_count = item_meta_words if runs else 7
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

    def request_chunks(q_len, eff):
        """Items a split request's key tiles divide into: up to split_chunks runs of at least
        split_min_tiles tiles; zero for a request that does not split."""
        small = (q_len > 0) & (q_len * group <= half) & (eff >= split_min_tiles)
        return T.if_then_else(small, T.min(split_chunks, eff // split_min_tiles), 0)

    def request_tiles(src, g):
        """Work items of request *g*: its key-tile runs if it splits, else its query tiles."""
        cu_q, kv_lens = src
        q_len = cu_q[g + 1] - cu_q[g]
        chunks = request_chunks(q_len, T.max(1, T.ceildiv(kv_lens[g], block_n)))
        return T.if_then_else(chunks > 0, chunks, T.ceildiv(q_len, span))

    request_prefix = tiling.cumsum_by(request_tiles)
    # The split runs' workspace: placeholders where a request keeps one run.
    part_shape = [batch, heads_kv] if split_chunks > 1 else [1, 1]

    # An FP8 query's unsplit items sum their probabilities into lane-local partial sums,
    # reduced once the item ends.
    partial_sums = q_fp8
    threads = consumers + 128

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

    def tile_of(k, tiles):
        """The key tile, of *tiles*, that an item's scan reaches at its step *k*."""
        return tiles - 1 - k if backward else k

    @T.macro
    def exponentiate(acc_s, sm, smp, item_scale):
        """Turn the scores into probabilities against the running maximum sm. With an FP8
        query, the maximum comes off each score before the scale: a large maximum, scaled
        first, rounds by more than the 2^8 of e4m3 range that p_shift leaves. The backward
        scan meets rows with no visible key yet, whose maximum is replaced by zero so that
        exp2(-inf) = 0 rather than NaN. That maximum goes in smp, whose previous maximum
        nothing reads once softmax_step has the rescale factor."""
        if q_fp8:
            for i in T.Parallel(half):
                smp[i] = T.if_then_else(sm[i] == -T.infinity(accum), 0.0, sm[i])
            for i, j in T.Parallel(half, block_n):
                acc_s[i, j] = T.exp2((acc_s[i, j] - smp[i]) * item_scale + p_shift)
        else:
            for i, j in T.Parallel(half, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * item_scale - sm[i] * item_scale)

    @T.macro
    def softmax_step(acc_s, sm, smp, alpha, ss, red, item_scale, row_sums=True):
        """Fold one score tile into the running max, rescale factor, and row sums."""
        if use_softcap:
            apply_softcap(acc_s)
        T.copy(sm, smp)
        T.reduce_max(T.reshape(acc_s, [half, block_n // 8, 8]), red, dim=1, clear=True)
        T.reduce_max(red, sm, dim=1, clear=False, batch=2)
        for i in T.Parallel(half):
            sm[i] = T.if_then_else((sm[i] - smp[i]) * item_scale > lazy, sm[i], smp[i])
            alpha[i] = 1.0
            if sm[i] != smp[i]:
                alpha[i] = T.exp2(smp[i] * item_scale - sm[i] * item_scale)
        exponentiate(acc_s, sm, smp, item_scale)
        if row_sums:
            T.reduce_sum(acc_s, ss, dim=1, batch=2)

    @T.macro
    def add_partial_sums(acc_s, lp):
        """Add a tile's probabilities to lp: column j sums score columns j, j + 16, ...,
        which one thread holds."""
        for t in T.unroll(block_n // 16):
            for i, j in T.Parallel(half, 16):
                lp[i, j] += acc_s[i, j + t * 16]

    @T.macro
    def cast_probs(acc_s, pcast):
        """Round the probabilities to the value product's A operand; in e4m3, in the key
        order of the transposed values, from the scores each thread holds."""
        if q_fp8:
            T.call_extern("handle", f"tileops::fp8_pack_probs<{block_n}>", acc_s.data, pcast.data)
        else:
            T.copy(acc_s, pcast)

    @T.macro
    def value_product(pcast, Vs, s, acc_o):
        """Issue acc_o += P V over stage *s*, its values transposed for an FP8 query."""
        T.wgmma_gemm(pcast, Vs[s, :, :], acc_o, transpose_B=q_fp8, policy=policy, clear_accum=False)

    @T.macro
    def kv_step(
        q_tile, Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp, alpha, ss,
        red, logsum, my_bar, nxt_bar, k, n, q0, wg, align, kv_len, masked: bool, item_scale,
        qk_descale, acc_l,
    ):  # fmt: skip
        """QK of key tile k overlapped with PV of the tile scanned before it; n counts tiles
        across items."""
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
        value_product(pcast, Vs, svp, acc_o)
        T.named_barrier_arrive(nxt_bar, consumers)
        T.wait_wgmma(1)
        T.mbarrier_arrive(kfree[sk])
        if use_softcap:
            descale(acc_s, qk_descale)
        if masked:
            mask(acc_s, k * block_n, q0, wg, align, kv_len)
        softmax_step(acc_s, sm, smp, alpha, ss, red, item_scale, not partial_sums)
        if partial_sums:
            for i, j in T.Parallel(half, 16):
                acc_l[i, j] *= alpha[i]
            add_partial_sums(acc_s, acc_l)
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        if not partial_sums:
            for i in T.Parallel(half):
                logsum[i] = logsum[i] * alpha[i] + ss[i]
        cast_probs(acc_s, pcast)

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
        if runs:
            meta[7] = 0
            meta[8] = 0
            meta[9] = 0
            chunks = T.alloc_var("int32", init=request_chunks(meta[3], meta[6]))
            if chunks > 0:
                # A split request's rows are all of its queries; q0 indexes its key-tile run.
                tiles = T.alloc_var("int32", init=meta[6])
                chunk = T.alloc_var("int32", init=meta[5] // span)
                meta[5] = 0
                meta[7] = chunk * tiles // chunks
                meta[6] = (chunk + 1) * tiles // chunks - meta[7]
                meta[8] = chunks
                meta[9] = chunk

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
            if runs:
                if meta[8] == 0:
                    load_q(Q, Qs, slot, q_bar, meta[2] + meta[5], cv)
            elif split_small:
                if (meta[3] * group > half) | (meta[6] < split_min_tiles):
                    load_q(Q, Qs, slot, q_bar, meta[2] + meta[5], cv)
            else:
                load_q(Q, Qs, slot, q_bar, meta[2] + meta[5], cv)
            T.mbarrier_arrive(q_bar[slot])
            for k in T.serial(meta[6]):
                n = issued + k
                s = n % stages
                key0 = ((meta[7] if runs else 0) + tile_of(k, meta[6])) * block_n
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
        """Convert stage *s* of *src* into the same stage of *dst*, zeroing rows at or past
        *valid_rows*."""
        T.call_extern(
            "handle",
            f"tileops::fp8_tile_to_wide_sw128<{block_n}, {converters}, {wide_c}>",
            T.access_ptr(src[s, 0, 0], "r"),
            T.access_ptr(dst[s, 0, 0], "w"),
            t,
            valid_rows,
        )

    @T.macro
    def transpose_tile(src, dst, s, t, valid_rows):
        """Transpose value stage *s* of *src* into the same stage of the K-major e4m3
        operand *dst*; a key at or past *valid_rows* transposes to zero."""
        T.call_extern(
            "handle",
            f"tileops::fp8_tile_transpose_sw128<{block_n}, {converters}>",
            T.access_ptr(src[s, 0, 0], "r"),
            T.access_ptr(dst[s, 0, 0], "w"),
            t,
            valid_rows,
        )

    @T.macro
    def converter(Kr, Vr, Kc, Vc, valid, kr_ready, kr_free, kc_ready, kc_free, vr_ready,
                  vr_free, vc_ready, vc_free, t):  # fmt: skip
        """Turn each landed tile into the consumers' stage until the stop mark: a 16-bit
        query's keys and values convert to its dtype, an FP8 query's values transpose."""
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
                if q_fp8:
                    transpose_tile(Vr, Vc, s, t, rows_in)
                else:
                    convert_tile(Vr, Vc, s, t, rows_in)
                T.mbarrier_arrive(vr_free[s])
                T.fence_proxy_async()
                T.mbarrier_arrive(vc_ready[s])
                n += 1

    @T.macro
    def split_item(
        wg: int, q_frag, Ks, Vs, O, kready, kfree, vready, vfree, mrg, mrg_o, acc_s, pcast,
        acc_o, sm, smp, alpha, ss, red, logsum, kv_head, q_start, q_len, kv_len, eff, done,
        item_scale, qk_descale, v_descale, req, first, chunks, chunk, Part, PartML, Arrive,
        item_slot,
    ):  # fmt: skip
        """Both warpgroups attend one request's rows, at most 64, held position-major in
        ``q_frag``, over one run of its key tiles: warpgroup *wg* scans tiles wg, wg + 2, ...
        and arrives on the others' barriers so that their phases stay shared; warpgroup 1
        hands its normalized half to warpgroup 0, which merges. A request split into several
        runs leaves each run's partial in global memory, and the last run to arrive combines
        them."""
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
                # Under partial sums a warp of padding rows skips the softmax.
                live = (T.get_thread_binding() % 128) // 32 * 16 < rows if partial_sums else True
                if live:
                    if use_softcap:
                        descale(acc_s, qk_descale)
                    key0 = (first + tile_of(k, eff)) * block_n
                    if (key0 + block_n > kv_len) | (
                        key0 + block_n > align + 1 if is_causal else False
                    ):
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
                cast_probs(acc_s, pcast)
                T.mbarrier_wait_parity(vready[s], phase)
                value_product(pcast, Vs, s, acc_o)
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
            combine = (chunks > 1) if runs else False
            # The run's partial leaves out the value descale; the combine applies it once.
            unit = T.if_then_else(combine, 1.0, v_descale) if runs else v_descale
            for i in T.Parallel(half):
                peak = T.max(sm[i], mrg[0, i])
                own = T.exp2((sm[i] - peak) * item_scale)
                other = T.exp2((mrg[0, i] - peak) * item_scale) * mrg[1, i]
                total = logsum[i] * own + other
                alpha[i] = T.if_then_else(total > 0, own * unit / total, 0)
                ss[i] = T.if_then_else(total > 0, other * unit / total, 0)
                if runs:
                    sm[i] = peak
                    logsum[i] = total
            if combine:
                for i, d in T.Parallel(half, dim):
                    Part[req, kv_head, chunk, i, d] = T.cast(
                        acc_o[i, d] * alpha[i] + T.cast(mrg_o[i, d], accum) * ss[i], wide
                    )
                for i in T.Parallel(half):
                    PartML[req, kv_head, chunk, 0, i] = sm[i]
                    PartML[req, kv_head, chunk, 1, i] = logsum[i]
                # The barrier orders every writer's partial before the count's release, and the
                # count's acquire before the combine's reads.
                T.sync_threads(4, half * 2)
                if T.get_thread_binding() == 0:
                    item_slot[0, meta_count + 1] = T.atomic_add(
                        Arrive[req * heads_kv + kv_head],
                        1,
                        memory_order="acq_rel",
                        return_prev=True,
                    )
                T.sync_threads(4, half * 2)
                if item_slot[0, meta_count + 1] == chunks - 1:
                    # The last run of the request and KV head combines all of them.
                    for i in T.Parallel(half):
                        smp[i] = -T.infinity(accum)
                        ss[i] = 0
                    for c in T.unroll(split_chunks):
                        if c < chunks:
                            for i in T.Parallel(half):
                                smp[i] = T.max(smp[i], PartML[req, kv_head, c, 0, i])
                    for c in T.unroll(split_chunks):
                        if c < chunks:
                            for i in T.Parallel(half):
                                ss[i] = (
                                    ss[i]
                                    + T.exp2((PartML[req, kv_head, c, 0, i] - smp[i]) * item_scale)
                                    * PartML[req, kv_head, c, 1, i]
                                )
                    T.fill(acc_o, 0)
                    for c in T.unroll(split_chunks):
                        if c < chunks:
                            for i in T.Parallel(half):
                                alpha[i] = (
                                    T.exp2((PartML[req, kv_head, c, 0, i] - smp[i]) * item_scale)
                                    * PartML[req, kv_head, c, 1, i]
                                )
                            for i, d in T.Parallel(half, dim):
                                acc_o[i, d] += alpha[i] * T.cast(Part[req, kv_head, c, i, d], accum)
                    for i in T.Parallel(half):
                        alpha[i] = T.if_then_else(ss[i] > 0, v_descale / ss[i], 0)
                    for i, d in T.Parallel(half, dim):
                        if i < rows:
                            O[q_start + i // group, kv_head * group + i % group, d] = T.cast(
                                acc_o[i, d] * alpha[i], dtype
                            )
                    if T.get_thread_binding() == 0:
                        Arrive[req * heads_kv + kv_head] = 0
            else:
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
        vfree, meta, mrg, mrg_o, Part, PartML, Arrive,
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
        pcast = T.alloc_fragment([half, pv_keys], fp8_dtype if q_fp8 else wide)
        if pv_keys > block_n:
            for i, k in T.Parallel(half, pv_keys - block_n):
                pcast[i, block_n + k] = T.cast(0, fp8_dtype)
        acc_o = T.alloc_fragment([half, dim], accum)
        acc_l = T.alloc_fragment([half, 16], accum) if partial_sums else None
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
                if q_fp8:
                    # Lane 0's value keeps the derived WGMMA descriptors in uniform registers.
                    meta[i] = T.call_extern(
                        "int32", "tileops::warp_uniform_int", item_slot[served % 2, i]
                    )
                else:
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
                if (meta[8] > 0) if runs else (q_len * group <= half) & (eff >= split_min_tiles):
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
            if partial_sums:
                T.fill(acc_l, 0)
            T.fill(logsum, 0)
            T.fill(alpha, 1.0)
            T.fill(sm, -T.infinity(accum))
            if wg == 1 and served == 0:
                T.named_barrier_arrive(1, consumers)  # let warpgroup 0 go first
            if split == 1:
                split_item(
                    wg, q_frag, Ks, Vs, O, kready, kfree, vready, vfree, mrg, mrg_o, acc_s,
                    pcast, acc_o, sm, smp, alpha, ss, red, logsum, kv_head, q_start, q_len,
                    kv_len, eff, done, item_scale, qk_descale, v_descale, req,
                    meta[7] if runs else 0, meta[8] if runs else 1, meta[9] if runs else 0, Part,
                    PartML, Arrive, item_slot,
                )  # fmt: skip
            else:
                # The first tile of the scan: QK and softmax only.
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
                if (full < eff) if backward else (full < 1):
                    mask(acc_s, tile_of(0, eff) * block_n, q0, wg, align, kv_len)
                if use_softcap:
                    apply_softcap(acc_s)
                T.reduce_max(acc_s, sm, dim=1, clear=False)
                exponentiate(acc_s, sm, smp, item_scale)
                if partial_sums:
                    add_partial_sums(acc_s, acc_l)
                else:
                    T.reduce_sum(acc_s, ss, dim=1, batch=2)
                    for i in T.Parallel(half):
                        logsum[i] = ss[i]
                cast_probs(acc_s, pcast)

                step = (Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
                # Tiles at or past `full` take the mask: the scan's first ones going backward,
                # its last ones going forward.
                for k in T.serial(1, T.max(1, eff - full if backward else full)):
                    kv_step(
                        q_frag, *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                        tile_of(k, eff), done + k, q0, wg, align, kv_len, backward, item_scale,
                        qk_descale, acc_l,
                    )  # fmt: skip
                for k in T.serial(T.max(1, eff - full if backward else full), eff):
                    kv_step(
                        q_frag, *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                        tile_of(k, eff), done + k, q0, wg, align, kv_len, not backward,
                        item_scale, qk_descale, acc_l,
                    )  # fmt: skip

                # PV of the last tile.
                svp = (done + eff - 1) % stages
                for i, j in T.Parallel(half, dim):
                    acc_o[i, j] *= alpha[i]
                T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
                value_product(pcast, Vs, svp, acc_o)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vfree[svp])
                if partial_sums:
                    T.reduce_sum(acc_l, logsum, dim=1)

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
        K: T.Tensor([pool_rows, heads_kv, dim], fp8_dtype),
        V: T.Tensor([pool_rows, heads_kv, dim], fp8_dtype),
        cache_seqlens: T.Tensor([batch], "int32"),
        page_table: T.Tensor([batch, max_pages_per_req], "int32"),
        CuQ: T.Tensor([batch + 1], "int32"),
        QS: T.Tensor([batch, heads_kv], accum),
        KS: T.Tensor(kv_scale_shape, accum),
        VS: T.Tensor(kv_scale_shape, accum),
        O: T.Tensor([total_q, heads, dim], dtype),
        Sched: T.Tensor([2], "int32"),
        Part: T.Tensor(part_shape + [split_chunks, half, dim], wide),
        PartML: T.Tensor(part_shape + [split_chunks, 2, half], accum),
        Arrive: T.Tensor([part_shape[0] * part_shape[1]], "int32"),
    ):
        with T.Kernel(num_ctas, threads=threads):
            # With shared-memory reuse off, the planner places the buffers in the order of
            # their names; the prefixes keep the layout the 16-bit query runs fastest at.
            sa_tile_cum = T.alloc_shared([batch + 1], "int32")
            # Per slot: the item's geometry and its claim number; [0, meta_count + 1] holds a
            # split run's arrival count.
            sb_item_slot = T.alloc_shared([2, meta_count + (2 if runs else 1)], "int32")
            # A split item's warpgroup 1 hands warpgroup 0 its row maxima and sums, and its
            # normalized rows, in sc_mrg and sg_mrg_o.
            sc_mrg = T.alloc_shared([2, half], accum)
            sd_valid = T.alloc_shared([stages], "int32")
            se_Qs = T.alloc_shared([2, 2, half, dim], qk_dtype)
            sf_Kr = T.alloc_shared([stages, block_n, dim], fp8_dtype)
            sg_mrg_o = T.alloc_shared([half, dim], wide)
            sh_Vr = T.alloc_shared([stages, block_n, dim], fp8_dtype)
            # An FP8 query multiplies the stored keys; a 16-bit one, their conversion.
            si_Kc = sf_Kr if q_fp8 else T.alloc_shared([stages, block_n, dim], wide)
            # An FP8 query's values transposed: dim rows of pv_keys key slots.
            sj_Vc = (
                T.alloc_shared([stages, dim, pv_keys], fp8_dtype)
                if q_fp8
                else T.alloc_shared([stages, block_n, dim], wide)
            )
            layouts = {
                se_Qs: make_swizzled_layout(se_Qs),
                sf_Kr: make_swizzled_layout(sf_Kr),
                sh_Vr: make_swizzled_layout(sh_Vr),
                sj_Vc: make_swizzled_layout(sj_Vc),
            }
            if not q_fp8:
                layouts[si_Kc] = make_swizzled_layout(si_Kc)
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

            if split_chunks > 1:
                request_prefix((CuQ, cache_seqlens), sa_tile_cum)
            else:
                tiling.cumsum_offsets(CuQ, sa_tile_cum)
            tx = T.get_thread_binding()
            if pv_keys > block_n:
                # The slots past the key tile, which no transpose writes; a stale NaN there
                # would survive its zero probability.
                pad = pv_keys - block_n
                for u in T.serial(T.ceildiv(stages * dim * pad, threads)):
                    e = u * threads + tx
                    if e < stages * dim * pad:
                        sj_Vc[e // (dim * pad), e // pad % dim, block_n + e % pad] = T.cast(
                            0, fp8_dtype
                        )
                # The consumers read the zeroed slots through the async proxy.
                T.fence_proxy_async()
            T.sync_threads()
            if tx >= consumers:
                T.set_max_nreg(56, 0)  # the producer warpgroup only copies, converts and transposes
            if tx >= consumers and tx < consumers + 32:
                producer(
                    Q, K, V, CuQ, cache_seqlens, page_table, Sched, se_Qs, sf_Kr, sh_Vr,
                    sb_item_slot, sd_valid, q_bar, qfree, kr_ready, kr_free, vr_ready, vr_free,
                    sa_tile_cum[batch] * heads_kv, sa_tile_cum, lo, hi, request, q_row, meta,
                    tx - consumers,
                )  # fmt: skip
            if tx >= consumers + 32:
                converter(
                    sf_Kr, sh_Vr, si_Kc, sj_Vc, sd_valid, kr_ready, kr_free, kc_ready, kc_free,
                    vr_ready, vr_free, vc_ready, vc_free, tx - consumers - 32,
                )  # fmt: skip
            args = (Q, se_Qs, si_Kc, sj_Vc, O, QS, KS, VS, sb_item_slot, q_bar, qfree, kc_ready,
                    kc_free, vc_ready, vc_free, meta, sc_mrg, sg_mrg_o, Part, PartML,
                    Arrive)  # fmt: skip
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
    ``GQAPagedFwdKernel``. The partials of a split request's key-tile runs take up to
    _SPLIT_WORKSPACE bytes per kernel instance and stream, allocated on its first call.
    """

    # Host eligibility and the device split schedule must use the same threshold.
    _SPLIT_MIN_TILES = 8

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_paged_varlen_kernel", "gqa_paged_fp8_decode_kernel"})
    _DIMS: tuple[int, ...] = (128,)
    # A tile's 128 rows are whole heads' query runs, each at least the 8 rows of a swizzle
    # atom so that its TMA box starts on one.
    _GROUPS: tuple[int, ...] = (1, 2, 4, 8, 16)
    # Key tiles in flight for a 16-bit query: three where the rings fit, which lets the
    # converters run two tiles ahead of the consumers; two otherwise.
    _STAGE_CHOICES: tuple[int, ...] = (3, 2)
    # Key tiles in flight for an FP8 query: two. A third leaves the consumers'
    # long-scoreboard stalls unchanged, so two already keep the loads ahead of them, and it
    # divides every slot and parity index by three (3.7% more instructions at 128 keys).
    _FP8_STAGE_CHOICES: tuple[int, ...] = (2,)
    _TILE_ROWS: int = 128
    # Key slots of an FP8 query's transposed value tile: its rows are 128 e4m3 bytes, one
    # swizzle span, whatever the key tile.
    _PV_KEYS: int = 128
    # Allocation and capacity checks share this kernel's split-item layout.
    _ITEM_META_WORDS: int = 10
    # The per-request tile prefix shares shared memory with the tiles; 4096 requests take
    # 16 KB of what they leave.
    _MAX_BATCH: int = 4096
    # Bytes the partials of split requests' key-tile runs may take.
    _SPLIT_WORKSPACE: int = 64 << 20

    @staticmethod
    def block_n_for(page_size: int) -> Optional[int]:
        """The widest key tile up to 128 rows that is part of one page or a run of whole
        pages, a multiple of the 16 rows one WGMMA step reduces over; ``None`` when none is."""
        for block_n in (128, 112, 96, 80, 64, 48, 32, 16):
            if page_size % block_n == 0 or block_n % page_size == 0 and page_size % 16 == 0:
                return block_n
        return None

    @classmethod
    def shared_bytes(
        cls,
        dim: int,
        block_n: int,
        stages: int,
        q_fp8: bool,
        batch: int,
    ) -> int:
        """Shared memory the program allocates, with shared-memory reuse off.

        It counts the two query slots, the stored key and value rings, the converted rings,
        the split handover, and the per-request tile offsets, item slots, handover maxima and
        sums, and ring flags, each small buffer padded to 16 bytes and all of them together to
        the kilobyte the query slots start on, plus the barriers' static shared memory.
        """

        def padded(nbytes: int) -> int:
            return -(-nbytes // 16) * 16

        q = 2 * cls._TILE_ROWS * dim * (1 if q_fp8 else 2)
        stored = 2 * stages * block_n * dim
        # An FP8 query's transposed values are dim rows of _PV_KEYS key slots; a 16-bit query
        # converts both tiles.
        converted = stages * dim * cls._PV_KEYS if q_fp8 else 2 * stages * block_n * dim * 2
        # A split item's handover: 64 normalized 16-bit rows.
        merge = cls._TILE_ROWS // 2 * dim * 2
        # Tile offsets; two item slots with two control words each; handover maxima and sums;
        # ring flags, one int32 a stage. They come first, and the query slots after them start
        # on a kilobyte.
        small = (
            padded(4 * (batch + 1))
            + padded(2 * (cls._ITEM_META_WORDS + 2) * 4)
            + padded(cls._TILE_ROWS // 2 * 8)
            + padded(4 * stages)
        )
        flags = -(-small // 1024) * 1024
        # The barriers are static shared memory, which the build reserves as one kilobyte.
        barriers = 1024
        return q + stored + converted + merge + flags + barriers

    @classmethod
    def tiling_for(cls, call: AttentionCall) -> "tuple[int, int] | None":
        """The key tile and the depth of the stored and converted rings: the page's tile at
        the deepest ring that fits, else the tile halved down to 64 rows."""
        block_n = cls.block_n_for(call.page_size)
        if block_n is None:
            return None
        q_fp8 = call.query_dtype == torch.float8_e4m3fn
        cap = get_shared_memory_optin(call.device.index if call.device is not None else None)

        def fits(block_n: int, stages: int) -> bool:
            return cls.shared_bytes(call.dim, block_n, stages, q_fp8, call.batch) <= cap

        choices = cls._FP8_STAGE_CHOICES if q_fp8 else cls._STAGE_CHOICES
        while True:
            for stages in choices:
                if fits(block_n, stages):
                    return block_n, stages
            if block_n <= 64:
                return block_n, choices[-1]
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
        # _SPLIT_MIN_TILES tiles builds without the split.
        self.split_small = max_pages_per_req * page_size >= self._SPLIT_MIN_TILES * self.block_n
        self.split_chunks = self.split_chunks_for(
            batch,
            heads_kv,
            dim,
            max_pages_per_req * page_size,
            self.block_n,
            self.query_dtype == torch.float8_e4m3fn,
        )
        device = torch.device("cuda", self.device_index)
        self._unit_q_scale = torch.ones((batch, heads_kv), dtype=torch.float32, device=device)
        self._counters: dict = {}
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {}

    @classmethod
    def build_program(
        cls,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        page_size: int,
        max_pages_per_req: int,
        is_causal: bool,
        sm_scale: float,
        softcap: float,
        dtype: str,
        query_dtype: str,
        per_request_kv_scale: bool,
        tiling: "tuple[int, int]",
        device_index: Optional[int],
    ):
        """The persistent program for one geometry; a request's key tiles split between the
        warpgroups where the table holds _SPLIT_MIN_TILES tiles."""
        block_n, stages = tiling
        return _gqa_paged_fp8_ws_kernel(
            batch,
            heads,
            heads_kv,
            dim,
            page_size,
            max_pages_per_req,
            is_causal,
            sm_scale,
            softcap,
            dtype,
            query_dtype,
            per_request_kv_scale,
            block_n,
            stages,
            get_sm_count(device_index),
            max_pages_per_req * page_size >= cls._SPLIT_MIN_TILES * block_n,
            cls.split_chunks_for(
                batch,
                heads_kv,
                dim,
                max_pages_per_req * page_size,
                block_n,
                query_dtype == "float8_e4m3fn",
            ),
            cls._ITEM_META_WORDS,
            cls._SPLIT_MIN_TILES,
            cls._PV_KEYS,
        )

    @classmethod
    def split_chunks_for(
        cls, batch: int, heads_kv: int, dim: int, table_rows: int, block_n: int, q_fp8: bool
    ) -> int:
        """Runs a split request's key tiles divide into.

        An FP8 query takes two runs when the table holds two runs of
        _SPLIT_MIN_TILES tiles and the partials of every request and KV head fit
        _SPLIT_WORKSPACE bytes. Every other call takes one run.
        """
        # Two runs let the consumer warpgroups overlap long FP8 requests.
        split_chunks = 2
        partial = cls._TILE_ROWS // 2 * (dim * 2 + 8)
        long_table = table_rows >= split_chunks * cls._SPLIT_MIN_TILES * block_n
        fits = batch * heads_kv * split_chunks * partial <= cls._SPLIT_WORKSPACE
        # A 16-bit query keeps one run: its program runs slower with the combine built in.
        return split_chunks if q_fp8 and long_table and fits else 1

    @staticmethod
    def split_workspace(
        counters: dict,
        chunks: int,
        batch: int,
        heads_kv: int,
        dim: int,
        wide: torch.dtype,
        device,
    ) -> tuple:
        """The split runs' partials and arrival counts of the stream's launches: the counts
        return to zero as each request and KV head combines, so they are zeroed once. A
        program that keeps one run a request takes placeholders it never reads."""
        if chunks == 1:
            batch = heads_kv = 1
        rows = GQAPagedFP8WSKernel._TILE_ROWS // 2
        key = ("split", torch.cuda.current_stream(device).cuda_stream, wide)
        workspace = None if torch.cuda.is_current_stream_capturing() else counters.get(key)
        if workspace is None:
            workspace = (
                torch.empty((batch, heads_kv, chunks, rows, dim), dtype=wide, device=device),
                torch.empty((batch, heads_kv, chunks, 2, rows), dtype=torch.float32, device=device),
                torch.zeros(batch * heads_kv, dtype=torch.int32, device=device),
            )
            if not torch.cuda.is_current_stream_capturing():
                counters[key] = workspace
        return workspace

    @staticmethod
    def launch(
        program,
        counters: dict,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        dtype: torch.dtype,
        split_chunks: int,
    ) -> torch.Tensor:
        """Run *program* over the pools, claiming work from the stream's counter in
        *counters*: a new output of q's shape in *dtype*. *split_chunks* is the program's
        split_chunks_for, which sizes its workspace."""
        out = torch.empty(q.shape, dtype=dtype, device=q.device)
        program(
            q,
            k_pool,
            v_pool,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            q_scale,
            k_scale,
            v_scale,
            out,
            scheduler_counter(counters, q.device),
            *GQAPagedFP8WSKernel.split_workspace(
                counters,
                split_chunks,
                page_table.shape[0],
                k_pool.shape[1],
                q.shape[2],
                torch.float16 if q.dtype == torch.float8_e4m3fn else dtype,
                q.device,
            ),
        )
        return out

    @property
    def kernel(self):
        """The persistent program, which profiling reads."""
        return self.program(self._compile)

    def _compile(self, cfg: dict):
        """The persistent program for this geometry."""
        return self.build_program(
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
            (self.block_n, self.stages),
            self.device_index,
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
        return self.launch(
            self.program(self._compile),
            self._counters,
            q,
            k_pool,
            v_pool,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            q_scale if q_scale is not None else self._unit_q_scale,
            k_scale,
            v_scale,
            self.dtype,
            self.split_chunks,
        )
