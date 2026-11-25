"""SM90 warp-specialized GQA over a read-only paged KV cache.

A persistent CTA has one TMA producer warp and two consumer warpgroups. Work
is 128 query rows of one head group, query heads that share a KV head; wide query
groups over one KV head use a flat Q layout, whose rows run on three consumer
warpgroups, 192 rows an item, at dimension 64. The producer zeroes stale V rows
in the final partial cache page before consumers read them.
"""

import math

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
from tileops._kernels.attention.gqa.paged_rope_prepare import _gqa_paged_rope_prepare_kernel
from tileops._kernels.attention.online_softmax import make_apply_softcap, make_sink_scale
from tileops._kernels.attention.stream_counter import stream_counter
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import LOG2E, WGMMA_ROWS
from tileops._kernels.grouped_tiling import GroupTiling
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel

__all__ = ["GQAPagedFwdWSKernel"]


@T.macro
def claim_work(Sched, lane, work):
    """Claim the next work item of a persistent launch into work[0]: lane 0 claims, the warp
    shares it."""
    if lane == 0:
        work[0] = T.atomic_add(Sched[0], 1, return_prev=True)
    work[0] = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), work[0], 0, 32, 32)


@target_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        # Row r of a tile stores query r % span of head r // span, which is injective in r;
        # the checker cannot prove it and reports the output store as a race.
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
    compile_flags=[
        *CUDA_COMPILE_FLAGS,
        *csrc_include("value_wait_anchor.h"),
    ],
)
def _gqa_paged_ws_kernel(
    batch, heads, heads_kv, dim, page_size, max_pages_per_req, is_causal, sm_scale, softcap,
    dtype, block_n, stages, num_ctas, window_left, window_right, has_sinks, split_small,
    split_min_tiles, flat, tile_heads, rope_cols=0, rope_rows=0,
):  # fmt: skip
    """A persistent CTA per SM: a TMA producer warp claims work, consumer warpgroups run it.

    A work item is a tile of one head group's query rows: *tile_heads* query heads of one KV
    head (``GQAPagedFwdWSKernel._tile_heads``), each owning ``span`` rows. A KV head's query
    group holds ``slices`` head groups, and row r of a tile is query q0 + r % span of head
    ``group_id * tile_heads + r // span``. *flat* rows (``GQAPagedFwdWSKernel._flat``) cover
    every query head of the one KV head: row q0 + r is query (q0 + r) // tile_heads of head
    (q0 + r) % tile_heads.

    With *rope_cols*, a key's first rope_cols channels come from KR (rope_rows rows a request)."""
    has_window = window_left >= 0 or window_right >= 0
    meta_count = 8 if has_window else 7
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else sm_scale * LOG2E
    head_groups = heads // tile_heads
    slices = head_groups // heads_kv
    accum = "float"
    half = WGMMA_ROWS  # rows per consumer warpgroup: one m64 WGMMA
    # Flat rows at dimension 64 (their items never split) take three consumer warpgroups.
    wgs = 3 if flat and dim <= 64 else 2
    block_m = wgs * half
    consumers = wgs * 128  # threads of the consumer warpgroups
    # Threads of a pingpong barrier: the two warpgroups passing the turn.
    turn = 256
    policy = T.GemmWarpPolicy.FullRow
    total_q = T.dynamic("total_q")
    pool_rows = T.dynamic("pool_rows")
    rotated_rows = T.dynamic("rotated_rows")
    span = block_m // tile_heads
    sink_scale = make_sink_scale(scale, accum, half, rows_per_head=span)
    tiling = (
        GroupTiling(batch, block_m, rows_per_offset=tile_heads)
        if flat
        else GroupTiling(batch, span)
    )

    # Where requests may split, a CTA ranks up to 128 of them by cache length and claims the
    # longest first, so the closing wave runs the short and split ones; the ranking compares
    # every pair. Without the split, items run from the packing's last tile back.
    lpt = split_small and batch <= 128
    # Rows a tile of a request's own enumeration covers, in the units decode returns.
    rows_per_tile = block_m if flat else span

    def position(q0, r):
        """The query position, within its request, of row r of the tile at q0."""
        return (q0 + r) // tile_heads if flat else q0 + r % span

    def q_first(meta):
        """The first row of Q the item's tile reads: a flat row, or a query position."""
        return meta[2] * tile_heads + meta[5] if flat else meta[2] + meta[5]

    # A key tile is part of one page or a run of whole pages: one TMA copy per page.
    box = min(block_n, page_size)
    boxes = block_n // box
    # 16-byte chunks of one V row, the unit the producer zeroes.
    chunks = dim // 8
    # Query tiles in flight: a second slot lets the producer load the next item's queries
    # while the consumers scan; at dimension 256 the key and value rings leave room for one.
    q_slots = 2 if dim <= 128 else 1

    apply_softcap = (
        make_apply_softcap(sm_scale, softcap, accum, half, block_n) if use_softcap else None
    )

    @T.macro
    def mask(acc_s, key0, q0, wg, align, kv_len):
        """Send the scores of keys past the cache end, or past a row's position, to -inf."""
        for i, j in T.Parallel(half, block_n):
            if is_causal:
                hidden = (key0 + j >= kv_len) | (key0 + j > position(q0, wg * half + i) + align)
            else:
                hidden = key0 + j >= kv_len
            if window_left >= 0:
                hidden = hidden | (key0 + j < position(q0, wg * half + i) + align - window_left)
            if window_right >= 0:
                hidden = hidden | (key0 + j > position(q0, wg * half + i) + align + window_right)
            acc_s[i, j] = T.if_then_else(hidden, -T.infinity(accum), acc_s[i, j])

    @T.macro
    def softmax_step(acc_s, sm, smp, alpha, ss, red):
        """Fold one score tile into the running max, rescale factor, and row sums."""
        if use_softcap:
            apply_softcap(acc_s)
        T.copy(sm, smp)
        T.reduce_max(T.reshape(acc_s, [half, block_n // 8, 8]), red, dim=1, clear=True)
        T.reduce_max(red, sm, dim=1, clear=False, batch=2)
        for i in T.Parallel(half):
            sm[i] = T.if_then_else((sm[i] - smp[i]) * scale > 8.0, sm[i], smp[i])
            alpha[i] = 1.0
            if sm[i] != smp[i]:
                alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
        for i, j in T.Parallel(half, block_n):
            maximum = T.if_then_else(sm[i] == -T.infinity(accum), 0, sm[i]) if has_window else sm[i]
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - maximum * scale)
        T.reduce_sum(acc_s, ss, dim=1, batch=2)

    @T.macro
    def kv_step(
        q_tile, Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp, alpha, ss,
        red, logsum, my_bar, nxt_bar, k, n, q0, wg, align, kv_len, tail: bool, anchor,
    ):  # fmt: skip
        """QK of key tile k overlapped with PV of tile k - 1; n counts tiles across items."""
        sk = n % stages
        svp = (n - 1) % stages
        T.sync_threads(my_bar, turn)
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
        T.named_barrier_arrive(nxt_bar, turn)
        T.wait_wgmma(1)
        T.mbarrier_arrive(kfree[sk])
        if tail:
            mask(acc_s, k * block_n, q0, wg, align, kv_len)
        softmax_step(acc_s, sm, smp, alpha, ss, red)
        if anchor is not None:
            T.call_extern(
                "handle", "tileops::value_wait_anchor_slot", T.access_ptr(anchor[wg * 128], "w")
            )
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        for i in T.Parallel(half):
            logsum[i] = logsum[i] * alpha[i] + ss[i]
        T.copy(acc_s, pcast)

    # With the ranking, tile_cum holds three runs: the tile prefix over the ranked order
    # (batch + 1), the order (batch), and scratch (batch). The producer reads the prefix and
    # the order while the consumers write their split handover.
    order0 = batch + 1
    lens0 = 2 * batch + 1

    @T.macro
    def rank_requests(CuQ, cache_seqlens, tile_cum):
        """Order the requests longest cache first, ties by index, and build the tile prefix
        over that order. Every thread of the block takes part."""
        tx = T.get_thread_binding()
        for i in T.serial(T.ceildiv(batch, 384)):
            r = i * 384 + tx
            if r < batch:
                tile_cum[lens0 + r] = cache_seqlens[r]
        T.sync_threads()
        for i in T.serial(T.ceildiv(batch, 384)):
            r = i * 384 + tx
            if r < batch:
                rank = T.alloc_var("int32", init=0)
                for j in T.serial(batch):
                    if (tile_cum[lens0 + j] > tile_cum[lens0 + r]) | (
                        (tile_cum[lens0 + j] == tile_cum[lens0 + r]) & (j < r)
                    ):
                        rank += 1
                tile_cum[order0 + rank] = r
        T.sync_threads()
        for i in T.serial(T.ceildiv(batch, 384)):
            r = i * 384 + tx
            if r < batch:
                request = tile_cum[order0 + r]
                rows = (CuQ[request + 1] - CuQ[request]) * (tile_heads if flat else 1)
                tile_cum[lens0 + r] = T.ceildiv(rows, rows_per_tile)
        T.sync_threads()
        if tx == 0:
            tile_cum[0] = 0
            for i in T.serial(batch):
                tile_cum[i + 1] = tile_cum[i] + tile_cum[lens0 + i]
        T.sync_threads()

    @T.macro
    def locate(work, CuQ, cache_seqlens, tile_cum, lo, hi, request, q_row, meta):
        """Fill meta (head group, request, q_start, q_len, kv_len, q0, tiles) for *work*.

        Items run from the last query tile back, so a causal request's longest tiles start
        first. Ranked requests run longest first; past 128, where requests may split, they run
        in order, so the split one-tile requests at the packing's end fill the closing wave.
        """
        if lpt:
            tiling.decode(work // head_groups, tile_cum, lo, hi, request, q_row)
            q_row[0] = (
                tile_cum[request[0] + 1] - tile_cum[request[0]] - 1
            ) * rows_per_tile - q_row[0]
            request[0] = tile_cum[order0 + request[0]]
        elif split_small:
            tiling.decode(work // head_groups, tile_cum, lo, hi, request, q_row)
            q_row[0] = (tile_cum[request[0] + 1] - tile_cum[request[0]] - 1) * span - q_row[0]
        else:
            tiling.decode(
                tile_cum[batch] - 1 - work // head_groups, tile_cum, lo, hi, request, q_row
            )
        meta[0] = work % head_groups
        meta[1] = request[0]
        meta[2] = CuQ[request[0]]
        meta[3] = CuQ[request[0] + 1] - meta[2]
        meta[4] = cache_seqlens[request[0]]
        meta[5] = q_row[0]
        if is_causal:
            # The tile's last row sees the most keys: query positions end at the cache end.
            last_q = T.min(meta[3] - 1, position(meta[5], block_m - 1)) + meta[4] - meta[3]
            key_end = T.min(meta[4], last_q + 1)
        else:
            key_end = meta[4]
        # At least one tile, so a cache shorter than its queries, outside the op's contract,
        # cannot leave the consumers waiting on a tile the producer never loads. The scan
        # stays unconditional: under a branch ptxas keeps the WGMMA descriptors out of
        # uniform registers.
        if window_right >= 0:
            last_q = T.min(meta[3] - 1, meta[5] + span - 1) + meta[4] - meta[3]
            key_end = T.min(key_end, last_q + window_right + 1)
        if has_window:
            first_key = T.alloc_var("int32", init=0)
            if window_left >= 0:
                first_key = T.max(0, meta[5] + meta[4] - meta[3] - window_left)
            meta[7] = first_key // block_n
            meta[6] = T.max(1, T.ceildiv(key_end, block_n) - meta[7])
        else:
            meta[6] = T.max(1, T.ceildiv(key_end, block_n))

    @T.macro
    def tile_rows(page_table, rows, rotated, request, key0, kv_len):
        """The pool row (and with rope_cols the KR row) each box of the key tile at *key0*
        starts at; a page past the cache end repeats the last one."""
        last_page = T.max(0, (kv_len - 1) // page_size)
        for b in T.unroll(boxes):
            key = key0 + b * box
            page = T.min(key // page_size, last_page)
            rows[b] = page_table[request, page] * page_size + key % page_size
            if rope_cols > 0:
                rotated[b] = request * rope_rows + page * page_size + key % page_size

    @T.macro
    def load_tile(pool, dst, stage, bars, bar, rows, cv, cols=0, front=None, front_rows=None):
        """TMA one key tile of *pool* into stage *stage* of *dst*, landing on ``bars[bar]``:
        a box a page and a 128-byte column, the widest box that lands on a row range of the
        swizzled buffer. The first *cols* columns come from *front* at *front_rows*."""
        split = cols // 64
        for b in T.unroll(boxes):
            if split > 0:
                for c in T.unroll(split):
                    T.tma_copy(
                        front[front_rows[b] : front_rows[b] + box, cv, c * 64 : (c + 1) * 64],
                        dst[stage, b * box : (b + 1) * box, c * 64 : (c + 1) * 64],
                        barrier=bars[bar],
                    )
            for c in T.unroll(split, dim // 64):
                T.tma_copy(
                    pool[rows[b] : rows[b] + box, cv, c * 64 : (c + 1) * 64],
                    dst[stage, b * box : (b + 1) * box, c * 64 : (c + 1) * 64],
                    barrier=bars[bar],
                )

    @T.macro
    def load_q(Q, Qs, slot, q_bar, q_start, group_id):
        """TMA each head's run of the tile's queries into the warpgroup that owns its rows; a
        head spans both warpgroups when the tile holds more queries than one has rows. Flat
        rows are 64 consecutive rows of Q a warpgroup, from flat row *q_start*."""
        if flat:
            for wg in T.unroll(wgs):
                for c in T.unroll(dim // 64):
                    T.tma_copy(
                        Q[q_start + wg * half : q_start + (wg + 1) * half, c * 64 : (c + 1) * 64],
                        Qs[slot, wg, :, c * 64 : (c + 1) * 64],
                        barrier=q_bar[slot],
                    )
        else:
            rows = min(half, span)
            for wg in T.unroll(wgs):
                for h in T.unroll(max(1, half // span)):
                    head = group_id * tile_heads + (wg * half) // span + h
                    first = q_start + (wg * half) % span
                    for c in T.unroll(dim // 64):
                        T.tma_copy(
                            Q[first : first + rows, head, c * 64 : (c + 1) * 64],
                            Qs[slot, wg, h * rows : (h + 1) * rows, c * 64 : (c + 1) * 64],
                            barrier=q_bar[slot],
                        )

    @T.macro
    def producer(
        Q, K, V, KR, CuQ, cache_seqlens, page_table, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree,
        kready, kfree, vready, vfree, vcut, total, tile_cum, lo, hi, request, q_row, meta, lane,
    ):  # fmt: skip
        """Claim work items until none remain; load each one's Q, then its key tiles."""
        issued = T.alloc_var("int32", init=0)
        loaded = T.alloc_var("int32", init=0)
        cuts = T.alloc_var("int32", init=0)
        work = T.alloc_local([1], "int32")
        rows = T.alloc_local([boxes], "int32")
        rotated = T.alloc_local([boxes], "int32") if rope_cols > 0 else None
        claim_work(Sched, lane, work)
        while work[0] < total:
            locate(work[0], CuQ, cache_seqlens, tile_cum, lo, hi, request, q_row, meta)
            cv = meta[0] // slices
            req = meta[1]
            kv_len = meta[4]
            slot = loaded % q_slots
            T.mbarrier_wait_parity(qfree[slot], ((loaded // q_slots) % 2) ^ 1)
            # The consumers read the item's geometry here rather than decode it again.
            for i in T.unroll(meta_count):
                item_slot[slot, i] = meta[i]
            item_slot[slot, meta_count] = work[0]
            if split_small:
                # A split item's consumers gather its query rows themselves.
                # Small requests split alternate key tiles across the two warpgroups.
                if (meta[3] * tile_heads > half) | (meta[6] < split_min_tiles):
                    load_q(Q, Qs, slot, q_bar, q_first(meta), meta[0])
            else:
                load_q(Q, Qs, slot, q_bar, q_first(meta), meta[0])
            T.mbarrier_arrive(q_bar[slot])
            for k in T.serial(meta[6]):
                n = issued + k
                s = n % stages
                key0 = (k + (meta[7] if has_window else 0)) * block_n
                # Read before the wait, so the table's latency hides behind the consumers.
                tile_rows(page_table, rows, rotated, req, key0, kv_len)
                T.mbarrier_wait_parity(kfree[s], ((n // stages) % 2) ^ 1)
                load_tile(K, Ks, s, kready, s, rows, cv, rope_cols, KR, rotated)
                T.mbarrier_arrive(kready[s])
                T.mbarrier_wait_parity(vfree[s], ((n // stages) % 2) ^ 1)
                if key0 + block_n > kv_len:
                    # NaN or infinity in a stale V row would survive its zero weight in the
                    # value sum, so the tile lands on a barrier of the producer's own and
                    # its stale rows are zeroed before the consumers are released.
                    load_tile(V, Vs, s, vcut, 0, rows, cv)
                    T.mbarrier_arrive(vcut[0])
                    T.mbarrier_wait_parity(vcut[0], cuts % 2)
                    cuts += 1
                    valid = kv_len - key0
                    for c in T.serial(T.ceildiv(block_n * chunks, 32)):
                        e = c * 32 + lane
                        j = e // chunks
                        if j < block_n and j >= valid:
                            for v in T.vectorized(8):
                                Vs[s, j, (e % chunks) * 8 + v] = T.cast(0, dtype)
                    # The consumers read V through the async proxy.
                    T.fence_proxy_async()
                else:
                    load_tile(V, Vs, s, vready, s, rows, cv)
                T.mbarrier_arrive(vready[s])
            issued += meta[6]
            loaded += 1
            claim_work(Sched, lane, work)
        # Stop the consumers, then leave the counters zeroed for the next launch.
        T.mbarrier_wait_parity(qfree[loaded % q_slots], ((loaded // q_slots) % 2) ^ 1)
        item_slot[loaded % q_slots, meta_count] = -1
        T.mbarrier_arrive(q_bar[loaded % q_slots])
        if lane == 0:
            finished = T.atomic_add(Sched[1], 1, return_prev=True)
            if finished == num_ctas - 1:
                Sched[0] = 0
                Sched[1] = 0

    @T.macro
    def split_item(
        wg: int, Q, Qs, Ks, Vs, O, qfree, kready, kfree, vready, vfree, mrg, acc_s, pcast,
        acc_o, sm, smp, alpha, ss, red, logsum, slot, group_id, q_start, q_len, kv_len, eff, done,
    ):  # fmt: skip
        """Both warpgroups attend one request's rows, at most 64: warpgroup *wg* scans key
        tiles wg, wg + 2, ... and arrives on the others' barriers so that their phases stay
        shared; warpgroup 1 hands its normalized half to warpgroup 0, which merges.

        Rows are position-major here, row r query r // tile_heads of head r % tile_heads, so they
        sit in one warpgroup's half of the tile.
        """
        rows = q_len * tile_heads
        align = kv_len - q_len
        if wg == 0:
            for i, d in T.Parallel(half, dim):
                r = T.min(i, rows - 1)
                Qs[slot, 0, i, d] = T.if_then_else(
                    i < rows,
                    Q[q_start + r // tile_heads, group_id * tile_heads + r % tile_heads, d],
                    T.cast(0, dtype),
                )  # fmt: skip
            # The key products read the rows through the async proxy.
            T.fence_proxy_async()
        T.sync_threads(3, consumers)
        for k in T.serial(eff):
            s = (done + k) % stages
            phase = ((done + k) // stages) % 2
            T.mbarrier_wait_parity(kready[s], phase)
            if k % 2 == wg:
                T.wgmma_gemm(
                    Qs[slot, 0, :, :], Ks[s, :, :], acc_s, transpose_B=True, policy=policy,
                    clear_accum=True,
                )  # fmt: skip
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[s])
                key0 = k * block_n
                if (key0 + block_n > kv_len) | (key0 + block_n > align + 1 if is_causal else False):
                    for i, j in T.Parallel(half, block_n):
                        hidden = key0 + j >= kv_len
                        if is_causal:
                            hidden = hidden | (key0 + j > i // tile_heads + align)
                        acc_s[i, j] = T.if_then_else(hidden, -T.infinity(accum), acc_s[i, j])
                softmax_step(acc_s, sm, smp, alpha, ss, red)
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
                Qs[slot, 1, i, d] = T.cast(
                    T.if_then_else(logsum[i] > 0, acc_o[i, d] / logsum[i], 0), dtype
                )
            for i in T.Parallel(half):
                mrg[0, i] = sm[i]
                mrg[1, i] = logsum[i]
        T.sync_threads(3, consumers)
        if wg == 0:
            for i in T.Parallel(half):
                peak = T.max(sm[i], mrg[0, i])
                own = T.exp2((sm[i] - peak) * scale)
                other = T.exp2((mrg[0, i] - peak) * scale) * mrg[1, i]
                total = logsum[i] * own + other
                alpha[i] = T.if_then_else(total > 0, own / total, 0)
                ss[i] = T.if_then_else(total > 0, other / total, 0)
            for i, d in T.Parallel(half, dim):
                if i < rows:
                    O[q_start + i // tile_heads, group_id * tile_heads + i % tile_heads, d] = (
                        T.cast(
                            acc_o[i, d] * alpha[i] + T.cast(Qs[slot, 1, i, d], accum) * ss[i], dtype
                        )
                    )
        # Warpgroup 0 has read the handed-over half; the slot may be reloaded by TMA.
        T.sync_threads(3, consumers)
        T.fence_proxy_async()
        T.mbarrier_arrive(qfree[slot])

    @T.macro
    def consumer(
        wg: int, Q, Qs, Ks, Vs, O, Sinks, item_slot, q_bar, qfree, kready, kfree, vready, vfree,
        meta, mrg, anchor,
    ):  # fmt: skip
        """Consumer warpgroup *wg*: rows ``wg*64`` onward of each claimed item's tile."""
        # 64512 registers: 240 a thread for two consumer warpgroups, 160 for three.
        T.set_max_nreg(240 if wgs == 2 else 160, 1)
        my_bar = 1 + wg
        nxt_bar = 1 + (wg + 1) % wgs
        acc_s = T.alloc_fragment([half, block_n], accum)
        pcast = T.alloc_fragment([half, block_n], dtype)
        acc_o = T.alloc_fragment([half, dim], accum)
        sm = T.alloc_fragment([half], accum)
        smp = T.alloc_fragment([half], accum)
        alpha = T.alloc_fragment([half], accum)
        ss = T.alloc_fragment([half], accum)
        red = T.alloc_fragment([half, 8], accum)
        logsum = T.alloc_fragment([half], accum)
        full = T.alloc_var("int32", init=0)
        if has_window:
            first_full = T.alloc_var("int32", init=0)
        # Finished key tiles and items: they carry the barrier phases.
        done = T.alloc_var("int32", init=0)
        served = T.alloc_var("int32", init=0)
        work = T.alloc_var("int32", init=0)
        T.mbarrier_wait_parity(q_bar[0], 0)
        work = item_slot[0, meta_count]
        while work >= 0:
            for i in T.unroll(meta_count):
                meta[i] = item_slot[served % q_slots, i]
            group_id = meta[0]
            q_start = meta[2]
            q_len = meta[3]
            kv_len = meta[4]
            # Query position p of the request is key align + p.
            align = kv_len - q_len
            q0 = meta[5]
            eff = meta[6]
            first_tile = meta[7] if has_window else 0
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(alpha, 1.0)
            T.fill(sm, -T.infinity(accum))
            if wg == wgs - 1 and served == 0:
                T.named_barrier_arrive(1, turn)  # let warpgroup 0 go first
            # A build without the split traces only the shared scan below.
            split = T.alloc_var("int32", init=0) if split_small else 0
            # Nested: `and` between a Python bool and a TIR comparison is not TIR.
            if split_small:  # noqa: SIM102
                if (q_len * tile_heads <= half) & (eff >= split_min_tiles):
                    split = 1
            if split == 1:
                split_item(
                    wg, Q, Qs, Ks, Vs, O, qfree, kready, kfree, vready, vfree, mrg, acc_s,
                    pcast, acc_o, sm, smp, alpha, ss, red, logsum, served % q_slots, group_id,
                    q_start, q_len, kv_len, eff, done,
                )  # fmt: skip
            else:
                # Key tile 0: QK and softmax only.
                s0 = done % stages
                T.sync_threads(my_bar, turn)
                T.mbarrier_wait_parity(kready[s0], (done // stages) % 2)
                T.wgmma_gemm(
                    Qs[served % q_slots, wg, :, :], Ks[s0, :, :], acc_s, transpose_B=True,
                    policy=policy, clear_accum=True,
                )  # fmt: skip
                T.named_barrier_arrive(nxt_bar, turn)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[s0])
                # Tiles before `full` are inside every row's visible range and need no mask.
                if is_causal:
                    # The warpgroup's first row holds its earliest query.
                    first_q = position(q0, wg * half) + align
                    full = T.min(eff, T.max(0, (first_q + 1) // block_n - first_tile))
                else:
                    full = T.min(eff, kv_len // block_n - first_tile)
                if has_window:
                    first_full = 0
                    if window_left >= 0:
                        # Every row must see the entire tile, including the latest query.
                        last_q = q0 + T.min(span - 1, (wg * half) % span + half - 1) + align
                        first_full = T.min(
                            eff, T.max(0, T.ceildiv(last_q - window_left, block_n) - first_tile)
                        )
                    if window_right >= 0:
                        first_q = q0 + (wg * half) % span + align
                        full = T.min(
                            full, T.max(0, (first_q + window_right + 1) // block_n - first_tile)
                        )
                    # When no tile is fully visible, keep the masked ranges disjoint.
                    full = T.max(first_full, full)
                if full < 1 or (first_full > 0 if has_window else False):
                    mask(acc_s, first_tile * block_n, q0, wg, align, kv_len)
                if use_softcap:
                    apply_softcap(acc_s)
                T.reduce_max(acc_s, sm, dim=1, clear=False)
                for i, j in T.Parallel(half, block_n):
                    # Preserve -inf in the running max until a visible key arrives.
                    maximum = (
                        T.if_then_else(sm[i] == -T.infinity(accum), 0, sm[i])
                        if has_window
                        else sm[i]
                    )
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - maximum * scale)
                T.reduce_sum(acc_s, ss, dim=1, batch=2)
                for i in T.Parallel(half):
                    logsum[i] = ss[i]
                T.copy(acc_s, pcast)

                step = (Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
                if has_window:
                    for k in T.serial(1, T.max(1, first_full)):
                        kv_step(
                            Qs[served % q_slots, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                            k + first_tile, done + k, q0, wg, align, kv_len, True, anchor,
                        )  # fmt: skip
                for k in T.serial(T.max(1, first_full) if has_window else 1, T.max(1, full)):
                    kv_step(
                        Qs[served % q_slots, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                        k + first_tile, done + k, q0, wg, align, kv_len, False, anchor,
                    )  # fmt: skip
                for k in T.serial(T.max(1, full), eff):
                    kv_step(
                        Qs[served % q_slots, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                        k + first_tile, done + k, q0, wg, align, kv_len, True, anchor,
                    )  # fmt: skip
                # Every QK is done, so the producer may reload Q.
                T.mbarrier_arrive(qfree[served % q_slots])

                # PV of the last tile.
                svp = (done + eff - 1) % stages
                for i, j in T.Parallel(half, dim):
                    acc_o[i, j] *= alpha[i]
                T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
                T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vfree[svp])

                if has_sinks:
                    sink_scale(sm, logsum, Sinks, group_id * tile_heads + wg * half // span, alpha)
                else:
                    # Windowed padding rows can have no visible key.
                    for i in T.Parallel(half):
                        if has_window:
                            alpha[i] = T.if_then_else(logsum[i] > 0, 1.0 / logsum[i], 0)
                        else:
                            alpha[i] = 1.0 / logsum[i]
                for i, d in T.Parallel(half, dim):
                    r = wg * half + i
                    if flat:
                        if q0 + r < q_len * tile_heads:
                            O[q_start * tile_heads + q0 + r, d] = T.cast(
                                acc_o[i, d] * alpha[i], dtype
                            )
                    elif q0 + r % span < q_len:
                        O[q_start + q0 + r % span, group_id * tile_heads + r // span, d] = T.cast(
                            acc_o[i, d] * alpha[i], dtype
                        )
            done += eff
            served += 1
            T.mbarrier_wait_parity(q_bar[served % q_slots], (served // q_slots) % 2)
            work = item_slot[served % q_slots, meta_count]

    # Flat rows read Q and write O as (total_q * heads, dim) matrices.
    q_shape = [total_q * heads, dim] if flat else [total_q, heads, dim]

    @T.macro
    def cta(Q, K, V, KR, cache_seqlens, page_table, CuQ, O, Sched, Sinks):
        """One persistent CTA: the producer warp and the consumer warpgroups."""
        Qs = T.alloc_shared([q_slots, wgs, half, dim], dtype)
        Ks = T.alloc_shared([stages, block_n, dim], dtype)
        Vs = T.alloc_shared([stages, block_n, dim], dtype)
        tile_cum = T.alloc_shared([3 * batch + 1 if lpt else batch + 1], "int32")
        item_slot = T.alloc_shared([q_slots, meta_count + 1], "int32")
        # A split item's warpgroup 1 hands warpgroup 0 its row maxima and sums here.
        # mrg and anchor, which the consumers write while the producer still reads tile_cum,
        # are static shared memory: the dynamic planner orders lifetimes by program text, where
        # the producer's branch comes first, and would give them tile_cum's bytes.
        mrg = T.alloc_shared([2, half], accum, scope="shared") if split_small else None
        # Above dimension 128 a store here keeps the exponentials ahead of the value wait.
        anchor = T.alloc_shared([consumers], accum, scope="shared") if dim > 128 else None
        T.annotate_layout(
            {
                Qs: make_swizzled_layout(Qs),
                Ks: make_swizzled_layout(Ks),
                Vs: make_swizzled_layout(Vs),
            }
        )
        q_bar = T.alloc_barrier([32] * q_slots)
        qfree = T.alloc_barrier([consumers] * q_slots)
        kready = T.alloc_barrier([32] * stages)
        kfree = T.alloc_barrier([consumers] * stages)
        vready = T.alloc_barrier([32] * stages)
        vfree = T.alloc_barrier([consumers] * stages)
        vcut = T.alloc_barrier([32])
        lo = T.alloc_local([1], "int32")
        hi = T.alloc_local([1], "int32")
        q_row = T.alloc_local([1], "int32")
        request = T.alloc_local([1], "int32")
        meta = T.alloc_local([meta_count], "int32")

        if lpt:
            rank_requests(CuQ, cache_seqlens, tile_cum)
        else:
            tiling.cumsum_offsets(CuQ, tile_cum)
            T.sync_threads()
        tx = T.get_thread_binding()
        if tx >= consumers:
            T.set_max_nreg(24, 0)  # the producer only issues TMA
        if tx >= consumers and tx < consumers + 32:
            producer(
                Q, K, V, KR, CuQ, cache_seqlens, page_table, Sched, Qs, Ks, Vs, item_slot, q_bar,
                qfree, kready, kfree, vready, vfree, vcut, tile_cum[batch] * head_groups,
                tile_cum, lo, hi, request, q_row, meta, tx - consumers,
            )  # fmt: skip
        args = (
            Q,
            Qs,
            Ks,
            Vs,
            O,
            Sinks,
            item_slot,
            q_bar,
            qfree,
            kready,
            kfree,
            vready,
            vfree,
            meta,
            mrg,
            anchor,
        )
        with T.ws(0):
            consumer(0, *args)
        with T.ws(1):
            consumer(1, *args)
        if wgs == 3:
            with T.ws(2):
                consumer(2, *args)

    if rope_cols > 0:

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor([pool_rows, heads_kv, dim], dtype),
            V: T.Tensor([pool_rows, heads_kv, dim], dtype),
            KR: T.Tensor([rotated_rows, heads_kv, rope_cols], dtype),
            cache_seqlens: T.Tensor([batch], "int32"),
            page_table: T.Tensor([batch, max_pages_per_req], "int32"),
            CuQ: T.Tensor([batch + 1], "int32"),
            O: T.Tensor(q_shape, dtype),
            Sched: T.Tensor([2], "int32"),
            Sinks: T.Tensor([heads], "float32"),
        ):
            with T.Kernel(num_ctas, threads=consumers + 128):
                cta(Q, K, V, KR, cache_seqlens, page_table, CuQ, O, Sched, Sinks)

    else:

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor([pool_rows, heads_kv, dim], dtype),
            V: T.Tensor([pool_rows, heads_kv, dim], dtype),
            cache_seqlens: T.Tensor([batch], "int32"),
            page_table: T.Tensor([batch, max_pages_per_req], "int32"),
            CuQ: T.Tensor([batch + 1], "int32"),
            O: T.Tensor(q_shape, dtype),
            Sched: T.Tensor([2], "int32"),
            Sinks: T.Tensor([heads], "float32"),
        ):
            with T.Kernel(num_ctas, threads=consumers + 128):
                cta(Q, K, V, None, cache_seqlens, page_table, CuQ, O, Sched, Sinks)

    return main


class GQAPagedFwdWSKernel(Kernel, GQAPagedFwdInterface):
    """SM90 warp-specialized paged attention for packings whose requests fill query tiles.

    It serves float16 and bfloat16 heads of dimension 64, 128 or 256 over pages of a multiple of
    16 rows, causal or bidirectional, with a positive scale and an optional softcap, when the
    packed query rows average at least one 128-row tile a request. It does not split a
    request's keys across CTAs, so a decode packing, whose tiles fall short of the
    multiprocessors, stays on ``GQAPagedFwdKernel``'s split scan. A query group split into head
    groups is served at dimension 128 or 256 only where its packing is dense enough
    (``_split_group_refusal``): ``_LEAST_TILE_FILL`` bounds the empty rows of its head-group
    tiles; the wave fills bound the multiprocessors a partial last wave leaves idle, lowest
    with RoPE, whose fused rotation slows the split scan more than this kernel's RoPE
    preparation costs, and lower at dimension 256, where the split scan is slowest;
    ``_LEAST_TILE_FILL_OVER_WAVES`` keeps thin tiles at dimension 128 off several waves,
    whose per-item costs match the split scan's over short caches. At dimension 64 the split
    scan keeps every such group.
    Windows bound the scan and sinks add denominator-only mass; without RoPE it serves no
    window.
    RoPE preprocesses private request pages without mutating the cache: over a 16-bit cache
    they hold the rotated leading key channels, the rest of each key and the values coming from
    the cache; an FP8 query or cache is served only with RoPE, whose preparation dequantizes
    all of K and V into those 16-bit pages. It serves any query group; over one KV head, more
    query heads than a tile holds 8-row swizzle atoms are served without windows, sinks or RoPE.
    """

    # Host eligibility and the device split schedule must use the same threshold.
    _SPLIT_MIN_TILES = 8

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_paged_varlen_kernel"})
    _DIMS: tuple[int, ...] = (64, 128, 256)
    _STAGES: int = 2
    # Rows of one work item: two consumer warpgroups of one m64 WGMMA each.
    _TILE_ROWS: int = 2 * WGMMA_ROWS
    _LEAST_TILE_FILL: float = 1 / 3
    _LEAST_TILE_FILL_OVER_WAVES: float = 2 / 3
    _LEAST_WAVE_FILL: float = 0.8
    _LEAST_WAVE_FILL_DIM_256: float = 0.7
    _LEAST_WAVE_FILL_ROPE: float = 0.5
    # The per-request tile prefix shares shared memory with the Q, K and V buffers, which take
    # 192 of the 227 KB at dimension 128; 4096 requests take 16 KB of the rest.
    _MAX_BATCH: int = 4096
    # Bound temporary storage to 512 MiB per call, including each captured graph pool.
    _MAX_ROPE_SCRATCH_BYTES: int = 512 * 1024 * 1024

    @staticmethod
    def block_n_for(page_size: int, dim: int) -> int | None:
        """The widest key tile up to 128 rows that is part of one page or a run of whole
        pages, a multiple of the 16 rows one WGMMA step reduces over; ``None`` when none is.
        At dimension 256 a tile holds at most 64 rows: the query tile and two stages of key
        and value tiles then fill the 227 KB of shared memory.

        A box smaller than a page costs the producer a table read and a TMA copy per box, so a
        tile of whole pages wins over a wider tile of boxes cut from pages.
        """
        widest = 128 if dim <= 128 else 64
        for block_n in (128, 112, 96, 80, 64, 48, 32, 16):
            if block_n > widest:
                continue
            if page_size % block_n == 0 or block_n % page_size == 0 and page_size % 16 == 0:
                return block_n
        return None

    @staticmethod
    def rope_pages_for(width: int, page_size: int, positions: int) -> int:
        """The rotary table bounds live cache lengths without reading device metadata."""
        return min(width, tilelang.cdiv(positions, page_size))

    @staticmethod
    def rope_cols_for(dim: int, rotary_dim: int, cache_fp8: bool) -> int:
        """Rotated key channels RoPE preparation writes over a 16-bit cache, in 64-channel
        columns; zero over an FP8 cache, whose preparation writes all of K and V."""
        return 0 if cache_fp8 else min(dim, tilelang.cdiv(rotary_dim, 64) * 64)

    @classmethod
    def _flat(cls, heads: int, heads_kv: int) -> bool:
        """Whether a tile's rows are flat rows of Q: over one KV head, more query heads than a
        tile holds 8-row swizzle atoms, whose queries are then one (total_q * heads, dim)
        matrix."""
        return heads_kv == 1 and heads > cls._TILE_ROWS // 8

    @classmethod
    def _tile_heads(cls, heads: int, heads_kv: int) -> int:
        """Query heads one work item's rows cover: every head when the rows are flat, else a head
        group, the largest power of two that divides the query group and gives each head whole
        8-row swizzle atoms of a tile."""
        if cls._flat(heads, heads_kv):
            return heads
        return math.gcd(heads // heads_kv, cls._TILE_ROWS // 8)

    @classmethod
    def _split_group_refusal(cls, call: AttentionCall) -> str | None:
        """Why the head groups of *call* would run slower here than on the split scan, or
        ``None`` when its query group does not split or its packing is dense enough. Tiles
        and work items are counted as if every request held the average query length; an
        empty packing has nothing to count."""
        tile_heads = cls._tile_heads(call.heads, call.heads_kv)
        if tile_heads == call.heads // call.heads_kv or call.batch == 0:
            return None
        if call.dim == 64:
            return "serves a split query group at dimension 128 or 256"
        rows = call.max_seqlen_q * tile_heads / call.batch
        tiles = math.ceil(rows / cls._TILE_ROWS)
        fill = rows / (tiles * cls._TILE_ROWS)
        if fill < cls._LEAST_TILE_FILL:
            return "requires a split query group's head-group tiles fuller"
        items = call.batch * tiles * (call.heads // tile_heads)
        waves = -(-items // call.sm_count)
        busy = items / (waves * call.sm_count)
        if call.fuse_rope:
            least = cls._LEAST_WAVE_FILL_ROPE
        elif call.dim == 256:
            least = cls._LEAST_WAVE_FILL_DIM_256
        else:
            least = cls._LEAST_WAVE_FILL
            if waves > 1 and fill < cls._LEAST_TILE_FILL_OVER_WAVES:
                return "requires a split query group's tiles over several waves fuller"
        if busy < least:
            return "requires a split query group's work items to fill their waves of SMs"
        return None

    @classmethod
    def refusal(cls, call: AttentionCall) -> str | None:
        """Why *call* is outside this implementation's region, or ``None`` when it is inside."""
        if call.dtype not in ATTENTION_DTYPES:
            return "requires a float16 or bfloat16 output"
        if call.is_fp8 and not call.fuse_rope:
            # RoPE preparation writes 16-bit pages; without it the tiles are read as stored.
            return "serves FP8 only with RoPE"
        if call.cache_dtype not in (call.dtype, torch.float8_e4m3fn):
            return "requires a cache of the output dtype or FP8"
        if call.query_dtype not in (None, call.dtype, call.cache_dtype):
            return "requires a query of the output dtype or, over an FP8 cache, FP8"
        if call.dim not in cls._DIMS:
            return f"requires head dimension in {cls._DIMS}"
        if call.heads_kv <= 0 or call.heads % call.heads_kv != 0:
            return "requires heads a multiple of heads_kv"
        if cls._flat(call.heads, call.heads_kv) and (
            call.fuse_rope or call.uses_sliding_window or call.has_sinks
        ):
            return "serves flat query rows over one KV head without RoPE, windows or sinks"
        if call.sm_scale is not None and call.sm_scale <= 0.0:
            return "requires a positive scale"
        if call.page_size <= 0 or cls.block_n_for(call.page_size, call.dim) is None:
            return "requires a page size that is a multiple of 16 rows"
        if not call.fuse_rope and call.uses_sliding_window:
            return "serves windows only with RoPE"
        if call.max_seqlen_q * (call.heads // call.heads_kv) < cls._TILE_ROWS * call.batch:
            return "requires packed query rows averaging one 128-row tile a request"
        split = cls._split_group_refusal(call)
        if split is not None:
            return split
        if call.fuse_rope:
            cols = cls.rope_cols_for(
                call.dim, call.rotary_dim or call.dim, call.cache_dtype == torch.float8_e4m3fn
            )
            kv_elements = (
                call.batch
                * cls.rope_pages_for(call.max_pages_per_req, call.page_size, call.max_position)
                * call.page_size
                * call.heads_kv
                * (cols or 2 * call.dim)
            )
            q_elements = call.max_seqlen_q * call.heads * call.dim
            if cols:
                kv_elements += call.heads_kv * call.dim
            if (kv_elements + q_elements) * call.dtype.itemsize > cls._MAX_ROPE_SCRATCH_BYTES:
                return f"RoPE preparation exceeds the {cls._MAX_ROPE_SCRATCH_BYTES} byte scratch budget"
        if call.batch > cls._MAX_BATCH:
            return f"serves at most {cls._MAX_BATCH} requests"
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The packed totals are read from the tensors, so one object serves every packing of
        these call facts; the device index is in the identity, as the SM count is."""
        index = call.device_index
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
            cache_dtype=call.cache_dtype or call.dtype,
            per_request_kv_scale=call.per_request_kv_scale,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            window_size_left=call.window_size_left,
            window_size_right=call.window_size_right,
            has_sinks=call.has_sinks,
            **call.rope_args,
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
        query_dtype: torch.dtype | None = None,
        cache_dtype: torch.dtype | None = None,
        per_request_kv_scale: bool = False,
        sm_scale: float | None = None,
        softcap: float = 0.0,
        window_size_left: int = -1,
        window_size_right: int = -1,
        has_sinks: bool = False,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        config: dict | None = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be a positive multiple of heads_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.page_size = page_size
        self.max_pages_per_req = max_pages_per_req
        self.is_causal = is_causal
        self.dtype = dtype
        self.query_dtype = query_dtype or dtype
        self.cache_dtype = cache_dtype or dtype
        self.per_request_kv_scale = per_request_kv_scale
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.window_size_left, self.window_size_right = window_size_left, window_size_right
        self.has_sinks = has_sinks
        self.fuse_rope, self.max_position = fuse_rope, max_position
        self.rotary_dim, self.rope_layout = rotary_dim or dim, rope_layout
        device = torch.device("cuda", self.device_index)
        self._unused_sinks = torch.empty(heads, dtype=torch.float32, device=device)
        # Stand-ins for the scales of a 16-bit operand, which preparation never reads.
        kv_scale_shape = (batch, heads_kv) if per_request_kv_scale else (1,)
        self._unit_scales = (
            torch.ones((batch, heads_kv), dtype=torch.float32, device=device),
            torch.ones(kv_scale_shape, dtype=torch.float32, device=device),
            torch.ones(kv_scale_shape, dtype=torch.float32, device=device),
        )
        self._rope_pages = (
            self.rope_pages_for(max_pages_per_req, page_size, max_position)
            if fuse_rope
            else max_pages_per_req
        )
        self._rope_cols = (
            self.rope_cols_for(dim, self.rotary_dim, self.cache_dtype == torch.float8_e4m3fn)
            if fuse_rope
            else 0
        )
        # Pages that hold whole keys replace the cache, and the table names them in order.
        self._rope_table = (
            torch.arange(batch * self._rope_pages, dtype=torch.int32, device=device).reshape(
                batch, self._rope_pages
            )
            if fuse_rope and not self._rope_cols
            else None
        )
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
        sm_count: int,
        window_size_left: int = -1,
        window_size_right: int = -1,
        has_sinks: bool = False,
        rope_cols: int = 0,
        rope_rows: int = 0,
        key_pages: int | None = None,
    ):
        """The persistent program for one geometry over a table *max_pages_per_req* wide;
        *key_pages* bounds a request's key pages (the table width by default)."""
        split_small = cls.splits_small(
            heads, heads_kv, dim, page_size, key_pages or max_pages_per_req, window_size_left,
            window_size_right, has_sinks,
        )  # fmt: skip
        return _gqa_paged_ws_kernel(
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
            cls.block_n_for(page_size, dim),
            cls._STAGES,
            sm_count,
            window_size_left,
            window_size_right,
            has_sinks,
            split_small,
            cls._SPLIT_MIN_TILES,
            cls._flat(heads, heads_kv),
            cls._tile_heads(heads, heads_kv),
            rope_cols,
            rope_rows,
        )

    @classmethod
    def splits_small(
        cls,
        heads: int,
        heads_kv: int,
        dim: int,
        page_size: int,
        max_pages_per_req: int,
        window_size_left: int,
        window_size_right: int,
        has_sinks: bool,
    ) -> bool:
        """Whether a request's key tiles split between the warpgroups. Windows, sinks and flat
        rows keep the shared scan; a table too short for _SPLIT_MIN_TILES tiles builds without
        the split and claims requests in reverse."""
        return (
            not cls._flat(heads, heads_kv)
            and window_size_left < 0
            and window_size_right < 0
            and not has_sinks
            and max_pages_per_req * page_size
            >= cls._SPLIT_MIN_TILES * cls.block_n_for(page_size, dim)
        )

    @property
    def split_small(self) -> bool:
        return self.splits_small(
            self.heads, self.heads_kv, self.dim, self.page_size, self._rope_pages,
            self.window_size_left, self.window_size_right, self.has_sinks,
        )  # fmt: skip

    @classmethod
    def launch(
        cls,
        program,
        counters: dict,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        sinks: torch.Tensor,
        k_rotated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run *program* over the pools, claiming work from the stream's counter in
        *counters*: a new output of q's shape and dtype. *k_rotated* feeds a rope_cols build."""
        heads, dim = q.shape[1], q.shape[2]
        out = torch.empty_like(q)
        if cls._flat(heads, k_pool.shape[1]):
            q, out = q.view(-1, dim), out.view(-1, dim)
        rotated = () if k_rotated is None else (k_rotated,)
        program(
            q,
            k_pool,
            v_pool,
            *rotated,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            out,
            stream_counter(counters, q.device),
            sinks,
        )
        return out.view(-1, heads, dim)

    @property
    def kernel(self):
        """The persistent program, which profiling reads."""
        return self.program(self._compile)[0]

    def _compile(self, cfg: dict) -> tuple:
        """The persistent program, and the RoPE preparation ahead of it or ``None``."""
        # Whole rotated pages replace the cache and its table; rotated leading channels keep both.
        attention = self.build_program(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.page_size,
            self.max_pages_per_req if self._rope_cols else self._rope_pages,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self.build_target.sm_count,
            self.window_size_left,
            self.window_size_right,
            self.has_sinks,
            self._rope_cols,
            self._rope_pages * self.page_size if self._rope_cols else 0,
            self._rope_pages,
        )
        if not self.fuse_rope:
            return CompiledProgram(attention), None
        prepare = _gqa_paged_rope_prepare_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.page_size,
            self.max_pages_per_req,
            self._rope_pages,
            self.build_target.sm_count,
            self.max_position,
            self.rotary_dim,
            self.rope_layout,
            self.dtype_str,
            str(self.query_dtype).removeprefix("torch."),
            str(self.cache_dtype).removeprefix("torch."),
            self.per_request_kv_scale,
            self._rope_cols or self.dim,
            not self._rope_cols,
            self.window_size_left,
            self.block_n_for(self.page_size, self.dim),
        )
        return CompiledProgram(attention), CompiledProgram(prepare)

    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.shape[0] == 0:
            return torch.empty(q.shape, dtype=self.dtype, device=q.device)
        attention, prepare = self.program(self._compile)
        if prepare is not None:
            q, k_rotated, v_rotated = prepare(
                q,
                k_pool,
                v_pool,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                *(
                    t if t is not None else unit
                    for t, unit in zip((q_scale, k_scale, v_scale), self._unit_scales, strict=True)
                ),
                rope_cos,
                rope_sin,
            )
            if not self._rope_cols:
                k_pool, v_pool, k_rotated = k_rotated, v_rotated, None
                page_table = self._rope_table
        else:
            k_rotated = None
        return self.launch(
            attention,
            self._counters,
            q,
            k_pool,
            v_pool,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            sinks if self.has_sinks else self._unused_sinks,
            k_rotated,
        )
