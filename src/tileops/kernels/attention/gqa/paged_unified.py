"""One-launch paged attention: persistent packed/prefill work and inline split reduction."""

import functools

import tilelang
import tilelang.language as T
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops.kernels.attention.online_softmax import (
    make_online_softmax_with_mask_guard,
    make_rescale,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling

_PAGED_HELPER_FLAGS = csrc_include("paged_attention.h")


def _make_prefill(
    batch, heads, heads_kv, dim, is_causal, sm_scale, softcap, dtype, block_n, stages, num_ctas, page_size, max_pages_per_req,
    load_mode="paged", producer_regs=24,
    min_query_length=0, splits=1,
):  # fmt: skip
    """A persistent CTA per SM: producers claim/load work, two consumer warpgroups run it."""
    assert dim in (64, 128)
    assert page_size % 16 == 0
    assert (
        block_n % page_size == 0 or page_size % block_n == 0 or (page_size == 48 and block_n == 128)
    )
    score_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = heads // heads_kv
    accum = "float"
    half = 64  # rows per consumer warpgroup: one m64 WGMMA
    block_m = 2 * half
    consumers = 256  # threads of the two consumer warpgroups
    assert load_mode in ("paged", "async_grouped_ahead")
    async_load = load_mode == "async_grouped_ahead"
    lookahead = async_load
    producers = 128 if async_load else 32
    policy = T.GemmWarpPolicy.FullRow
    q_tiling = GroupTiling(batch, 1)
    cut = max(0, min_query_length - 1)
    padded_rows = max(128, ((cut * groups + 127) // 128) * 128)

    @T.macro
    def prefetch_grouped(Table, offsets, parity, tile, request, kv_len, lane):
        # Each group cooperates on one D-wide row. Its lanes hold the addresses
        # of successive rows, matching FlashInfer's strided group partition.
        threads_per_row = dim // 8
        row_stride = producers // threads_per_row
        logical = tile * block_n + lane // threads_per_row + lane % threads_per_row * row_stride
        offsets[parity] = 0
        if logical < kv_len:
            offsets[parity] = Table[request, logical // page_size] * page_size + logical % page_size

    @T.macro
    def load_grouped(Source, Shared, Ready, offsets, parity, slot, tile, kv_head, kv_len, lane):
        threads_per_row = dim // 8
        row_stride = producers // threads_per_row
        for chunk in T.unroll(block_n // row_stride):
            row = lane // threads_per_row + chunk * row_stride
            column = lane % threads_per_row * 8
            source_lane = lane % 32 // threads_per_row * threads_per_row + chunk
            physical = T.tvm_warp_shuffle(
                T.uint32(0xFFFFFFFF), offsets[parity], source_lane, 32, 32
            )
            T.ptx_cp_async(
                T.access_ptr(Shared[slot, row, column], "w", extent=8),
                T.access_ptr(Source[physical, kv_head, column], "r", extent=8),
                8,
                tile * block_n + row < kv_len,
            )
        T.cp_async_barrier_noinc(Ready[slot])

    @T.macro
    def load_async(Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, lane):
        rows_per_pass = producers // (dim // 8)
        for chunk in T.serial(block_n // rows_per_pass):
            row = chunk * rows_per_pass + lane // (dim // 8)
            column = lane % (dim // 8) * 8
            logical = tile * block_n + row
            physical = T.alloc_var("int32", init=0)
            if logical < kv_len:
                physical = Table[request, logical // page_size] * page_size + logical % page_size
            T.ptx_cp_async(
                T.access_ptr(Shared[slot, row, column], "w", extent=8),
                T.access_ptr(Source[physical, kv_head, column], "r", extent=8),
                8,
                logical < kv_len,
            )
        T.cp_async_barrier_noinc(Ready[slot])

    @T.macro
    def load_piece(
        Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, start, length
    ):
        logical = tile * block_n + start
        page = T.min(logical // page_size, (kv_len - 1) // page_size)
        physical = Table[request, page] * page_size + logical % page_size
        for column in T.unroll(dim // 64):
            T.tma_copy(
                Source[physical : physical + length, kv_head, column * 64 : (column + 1) * 64],
                Shared[slot, start : start + length, column * 64 : (column + 1) * 64],
                barrier=Ready[slot],
            )

    @T.macro
    def load_paged(Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len):
        if page_size == 48 and block_n == 128:
            if tile % 3 == 0:
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 0, 48
                )
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 48, 48
                )
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 96, 32
                )
            elif tile % 3 == 1:
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 0, 16
                )
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 16, 48
                )
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 64, 48
                )
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 112, 16
                )
            else:
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 0, 32
                )
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 32, 48
                )
                load_piece(
                    Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 80, 48
                )
        elif block_n % page_size == 0:
            for piece in T.unroll(block_n // page_size):
                load_piece(
                    Source,
                    Table,
                    Shared,
                    Ready,
                    slot,
                    tile,
                    request,
                    kv_head,
                    kv_len,
                    piece * page_size,
                    page_size,
                )
        else:
            load_piece(
                Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, 0, block_n
            )

    @T.macro
    def load_tma_or_tail(Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, lane):
        # A page's unused suffix may contain NaN/Inf. Zero-fill partial tiles.
        if (tile + 1) * block_n <= kv_len:
            load_paged(Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len)
            T.mbarrier_arrive(Ready[slot])
        else:
            load_async(Source, Table, Shared, Ready, slot, tile, request, kv_head, kv_len, lane)

    @T.macro
    def apply_softcap(acc_s):
        for i, j in T.Parallel(half, block_n):
            capped = T.cast(softcap, accum) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum), -T.infinity(accum), capped
            )

    @T.macro
    def row_max(acc_s, red, sm):
        """sm = max(sm, row max): per-thread partials over the 16 column groups, then one
        cross-thread reduce."""
        T.reduce_max(T.reshape(acc_s, [half, block_n // 8, 8]), red, dim=1, clear=True)
        T.reduce_max(red, sm, dim=1, clear=False, batch=2)

    @T.macro
    def softmax_step(acc_s, sm, smp, alpha, ss, red):
        """Fold one score tile into the running max, rescale factor, and row sums."""
        if use_softcap:
            apply_softcap(acc_s)
        T.copy(sm, smp)
        row_max(acc_s, red, sm)
        for i in T.Parallel(half):
            sm[i] = T.if_then_else((sm[i] - smp[i]) * scale > 8.0, sm[i], smp[i])
            alpha[i] = 1.0
            if sm[i] != smp[i]:
                alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
            # An all-masked row keeps its running max at -inf. Subtract a
            # finite value only for exp2, so its probabilities are zero.
            smp[i] = T.if_then_else(sm[i] == -T.infinity(accum), 0.0, sm[i])
        for i, j in T.Parallel(half, block_n):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - smp[i] * scale)
        T.reduce_sum(acc_s, ss, dim=1, batch=2)

    @T.macro
    def kv_step(
        q_tile,
        Ks,
        Vs,
        kready,
        kfree,
        vready,
        vfree,
        acc_s,
        pcast,
        acc_o,
        sm,
        smp,
        alpha,
        ss,
        red,
        logsum,
        my_bar,
        nxt_bar,
        k,
        n,
        row,
        causal_offset,
        kv_len,
        pack,
        tail: bool,
    ):
        """QK of KV tile k overlapped with PV of tile k - 1; n counts tiles across work items."""
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
        if tail:
            if is_causal:
                limit = causal_offset - k * block_n
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(
                        limit + T.if_then_else(pack == 1, row + i, (row + i) // groups) >= j,
                        acc_s[i, j],
                        -T.infinity(accum),
                    )
            else:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(
                        k * block_n + j < kv_len, acc_s[i, j], -T.infinity(accum)
                    )
        softmax_step(acc_s, sm, smp, alpha, ss, red)
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        for i in T.Parallel(half):
            logsum[i] = logsum[i] * alpha[i] + ss[i]
        T.copy(acc_s, pcast)

    @T.macro
    def locate(work, CuQ, CacheLengths, tile_cum, lo, hi, request, q_row, meta):
        """Fill meta (head, q_start, request_id, q_len, kv_len, q0, eff) for work item *work*."""
        q_tiling.decode(tile_cum[batch] - 1 - work, tile_cum, lo, hi, request, q_row)
        meta[1] = CuQ[request[0]]
        meta[2] = request[0]
        meta[3] = CuQ[request[0] + 1] - meta[1]
        meta[4] = CacheLengths[request[0]]
        packed = meta[3] <= cut
        meta[7] = T.if_then_else(packed, groups, 1)
        if packed:
            meta[0] = (q_row[0] % heads_kv) * groups
            meta[8] = q_row[0] // heads_kv % splits
            meta[5] = q_row[0] // (heads_kv * splits) * block_m
        else:
            meta[0] = q_row[0] % heads
            meta[8] = 0
            meta[5] = q_row[0] // heads * block_m
        end = T.alloc_var("int32", init=T.max(0, T.ceildiv(meta[4], block_n)))
        if is_causal:
            end = T.max(
                0,
                T.min(
                    end,
                    T.ceildiv(
                        T.if_then_else(
                            packed, (meta[5] + block_m - 1) // groups, meta[5] + block_m - 1
                        )
                        + 1
                        + meta[4]
                        - meta[3],
                        block_n,
                    ),
                ),
            )
        chunk = T.if_then_else(packed, T.ceildiv(end, splits), end)
        meta[9] = meta[8] * chunk
        meta[6] = T.max(1, T.min(chunk, end - meta[9]))

    @T.macro
    def fetch(Sched, lane, work, work_shared):
        """Claim the next work item into work[0]: lane 0 claims, the warp shares it."""
        if async_load:
            if lane == 0:
                work_shared[0] = T.atomic_add(Sched[0], 1, return_prev=True)
            T.sync_threads(15, producers)
            work[0] = work_shared[0]
            T.sync_threads(15, producers)
        else:
            if lane == 0:
                work[0] = T.atomic_add(Sched[0], 1, return_prev=True)
            work[0] = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), work[0], 0, 32, 32)

    @T.macro
    def producer(
        Q, K, V, CuQ, CacheLengths, PageTable, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree, kready, kfree, vready,
        vfree, total, tile_cum, lo, hi, request, q_row, meta, lane, work_shared,
    ):  # fmt: skip
        """Claim work items until none remain; load each one's Q, then its KV tiles."""
        issued = T.alloc_var("int32", init=0)
        loaded = T.alloc_var("int32", init=0)
        work = T.alloc_local([1], "int32")
        offsets = T.alloc_local([2], "int32")
        fetch(Sched, lane, work, work_shared)
        while work[0] < total:
            locate(work[0], CuQ, CacheLengths, tile_cum, lo, hi, request, q_row, meta)
            head = meta[0]
            q_start = meta[1] + meta[5]
            request_id = meta[2]
            cv = head // groups
            slot = loaded % 2
            T.mbarrier_wait_parity(qfree[slot], ((loaded // 2) % 2) ^ 1)
            # The consumers read the item's geometry here rather than decode it again.
            if lane < 32:
                for i in T.unroll(10):
                    item_slot[slot, i] = meta[i]
                item_slot[slot, 10] = work[0]
                if meta[3] <= cut:
                    # Pack a KV head's query heads into rows, including Q=1.
                    for piece in T.serial(block_m * dim // (32 * 8)):
                        r = (piece * 32 + lane) // (dim // 8)
                        d = (piece * 32 + lane) % (dim // 8) * 8
                        qr = (meta[5] + r) // groups
                        qh = head + (meta[5] + r) % groups
                        T.ptx_cp_async(
                            T.access_ptr(Qs[slot, r // half, r % half, d], "w", extent=8),
                            T.access_ptr(Q[meta[1] + T.min(qr, meta[3] - 1), qh, d], "r", extent=8),
                            8,
                            qr < meta[3],
                        )
                    T.cp_async_barrier_noinc(q_bar[slot])
                else:
                    T.tma_copy(
                        Q[q_start : q_start + half, head, :], Qs[slot, 0, :, :], barrier=q_bar[slot]
                    )
                    T.tma_copy(
                        Q[q_start + half : q_start + block_m, head, :],
                        Qs[slot, 1, :, :],
                        barrier=q_bar[slot],
                    )
                    T.mbarrier_arrive(q_bar[slot])

            for k in T.serial(meta[6]):
                n = issued + k
                s = n % stages
                if async_load:
                    prefetch_grouped(
                        PageTable, offsets, k % 2, k + meta[9], request_id, meta[4], lane
                    )
                T.mbarrier_wait_parity(kfree[s], ((n // stages) % 2) ^ 1)
                if async_load:
                    load_grouped(K, Ks, kready, offsets, k % 2, s, k + meta[9], cv, meta[4], lane)
                else:
                    load_tma_or_tail(
                        K, PageTable, Ks, kready, s, k + meta[9], request_id, cv, meta[4], lane
                    )
                if not lookahead or k > 0:
                    v_tile = k - int(lookahead)
                    v_n = issued + v_tile
                    v_s = v_n % stages
                    T.mbarrier_wait_parity(vfree[v_s], ((v_n // stages) % 2) ^ 1)
                    if async_load:
                        load_grouped(
                            V,
                            Vs,
                            vready,
                            offsets,
                            v_tile % 2,
                            v_s,
                            v_tile + meta[9],
                            cv,
                            meta[4],
                            lane,
                        )
                    else:
                        load_tma_or_tail(
                            V,
                            PageTable,
                            Vs,
                            vready,
                            v_s,
                            v_tile + meta[9],
                            request_id,
                            cv,
                            meta[4],
                            lane,
                        )
            if lookahead:
                last = issued + meta[6] - 1
                s_last = last % stages
                T.mbarrier_wait_parity(vfree[s_last], ((last // stages) % 2) ^ 1)
                if async_load:
                    load_grouped(
                        V,
                        Vs,
                        vready,
                        offsets,
                        (meta[6] - 1) % 2,
                        s_last,
                        meta[6] - 1 + meta[9],
                        cv,
                        meta[4],
                        lane,
                    )
                else:
                    load_tma_or_tail(
                        V,
                        PageTable,
                        Vs,
                        vready,
                        s_last,
                        meta[6] - 1 + meta[9],
                        request_id,
                        cv,
                        meta[4],
                        lane,
                    )
            issued += meta[6]
            loaded += 1
            fetch(Sched, lane, work, work_shared)
        # Stop the consumers, then leave the counters zeroed for the next launch.
        T.mbarrier_wait_parity(qfree[loaded % 2], ((loaded // 2) % 2) ^ 1)
        if lane < 32:
            item_slot[loaded % 2, 10] = -1
            T.mbarrier_arrive(q_bar[loaded % 2])
        if lane == 0:
            finished = T.atomic_add(Sched[1], 1, return_prev=True)
            if finished == num_ctas - 1:
                Sched[0] = 0
                Sched[1] = 0

    @T.macro
    def consumer(
        wg: int, Qs, Ks, Vs, Os, O, Part, LSE, Sched, Merge, item_slot, q_bar, qfree, kready, kfree, vready, vfree, meta,
    ):  # fmt: skip
        """Consumer warpgroup *wg*: rows ``q0 + wg*64`` onward of each claimed query tile."""
        # A 384-thread CTA has 168 registers/thread in the initial pool.
        # Keep all three warpgroups within 504 * 128 registers when redistributing.
        T.set_max_nreg((504 - producer_regs) // 16 * 8, 1)
        my_bar = 1 + wg
        nxt_bar = 2 - wg
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
        # Finished KV tiles and items: they carry the barrier phases.
        done = T.alloc_var("int32", init=0)
        served = T.alloc_var("int32", init=0)
        work = T.alloc_var("int32", init=0)
        T.mbarrier_wait_parity(q_bar[0], 0)
        work = item_slot[0, 10]
        while work >= 0:
            for i in T.unroll(10):
                meta[i] = item_slot[served % 2, i]
            head = meta[0]
            q_start = meta[1]
            q_len = meta[3]
            kv_len = meta[4]
            causal_offset = meta[4] - meta[3]
            row = meta[5] + wg * half
            eff = meta[6]
            pack = meta[7]
            first = meta[9]
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(alpha, 1.0)
            T.fill(sm, -T.infinity(accum))
            if wg == 1 and served == 0:
                T.named_barrier_arrive(1, consumers)  # let warpgroup 0 go first

            # KV tile 0: QK and softmax only.
            s0 = done % stages
            T.sync_threads(my_bar, consumers)
            T.mbarrier_wait_parity(kready[s0], (done // stages) % 2)
            T.wgmma_gemm(
                Qs[served % 2, wg, :, :],
                Ks[s0, :, :],
                acc_s,
                transpose_B=True,
                policy=policy,
                clear_accum=True,
            )
            T.named_barrier_arrive(nxt_bar, consumers)
            T.wait_wgmma(0)
            T.mbarrier_arrive(kfree[s0])
            if is_causal:
                limit = causal_offset - first * block_n
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(
                        limit + T.if_then_else(pack == 1, row + i, (row + i) // groups) >= j,
                        acc_s[i, j],
                        -T.infinity(accum),
                    )
            elif not is_causal:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(
                        first * block_n + j < kv_len, acc_s[i, j], -T.infinity(accum)
                    )
            if use_softcap:
                apply_softcap(acc_s)
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            for i in T.Parallel(half):
                smp[i] = T.if_then_else(sm[i] == -T.infinity(accum), 0.0, sm[i])
            for i, j in T.Parallel(half, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - smp[i] * scale)
            T.reduce_sum(acc_s, ss, dim=1, batch=2)
            for i in T.Parallel(half):
                logsum[i] = ss[i]
            T.copy(acc_s, pcast)

            # Tiles 1 .. full - 1 need no mask; full .. eff - 1 do.
            if is_causal:
                full = T.max(
                    1,
                    T.min(
                        eff,
                        T.floordiv(
                            T.if_then_else(pack == 1, row, row // groups) + causal_offset + 1,
                            block_n,
                        )
                        - first,
                    ),
                )
            else:
                full = T.max(1, T.min(eff, T.floordiv(kv_len, block_n) - first))
            step = (Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
            for k in T.serial(1, full):
                kv_step(
                    Qs[served % 2, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar, first + k,
                    done + k, row, causal_offset, kv_len, pack, False,
                )  # fmt: skip
            for k in T.serial(full, eff):
                kv_step(
                    Qs[served % 2, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar, first + k,
                    done + k, row, causal_offset, kv_len, pack, True,
                )  # fmt: skip
            # Every QK is done, so the producer may reload Q.
            T.mbarrier_arrive(qfree[served % 2])

            # PV of the last tile, then normalize and store.
            svp = (done + eff - 1) % stages
            for i, j in T.Parallel(half, dim):
                acc_o[i, j] *= alpha[i]
            T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
            T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
            T.wait_wgmma(0)
            T.mbarrier_arrive(vfree[svp])
            if splits > 1 and q_len <= cut:
                # Publish one consumer's packed rows; the last split merges them.
                for i, d in T.Parallel(half, dim):
                    Part[meta[2], head // groups, meta[8], row + i, d] = T.if_then_else(
                        logsum[i] > 0, acc_o[i, d] / logsum[i], 0.0
                    )
                for i in T.Parallel(half):
                    LSE[meta[2], head // groups, meta[8], row + i] = T.if_then_else(
                        logsum[i] > 0,
                        T.log2(T.max(logsum[i], 1e-30)) + sm[i] * scale,
                        -T.infinity(accum),
                    )
                T.sync_threads(3 + wg, 128)
                counter = (
                    2 + (meta[2] * heads_kv + head // groups) * (padded_rows // half) + row // half
                )
                if T.get_thread_binding() == wg * 128:
                    Merge[wg] = T.call_extern(
                        "int32",
                        "tileops::paged_arrive",
                        T.access_ptr(Sched[counter], "rw", extent=1),
                    )
                T.sync_threads(3 + wg, 128)
                if Merge[wg] == splits - 1:
                    for i, d in T.Parallel(half, dim):
                        if (row + i) // groups < q_len:
                            peak = T.alloc_var("float32", init=-T.infinity(accum))
                            den = T.alloc_var("float32", init=0.0)
                            val = T.alloc_var("float32", init=0.0)
                            for sid in T.unroll(splits):
                                peak = T.max(peak, LSE[meta[2], head // groups, sid, row + i])
                            peak = T.if_then_else(peak == -T.infinity(accum), 0.0, peak)
                            for sid in T.unroll(splits):
                                weight = T.exp2(LSE[meta[2], head // groups, sid, row + i] - peak)
                                den += weight
                                val += weight * Part[meta[2], head // groups, sid, row + i, d]
                            O[q_start + (row + i) // groups, head + (row + i) % groups, d] = (
                                T.if_then_else(den > 0, val / den, 0.0)
                            )
                    T.sync_threads(3 + wg, 128)
                    if T.get_thread_binding() == wg * 128:
                        Sched[counter] = 0
            else:
                # Every row sees a key: causal needs kv_len >= q_len, non-causal kv_len > 0.
                if q_len > cut and row + half <= q_len and kv_len >= (q_len if is_causal else 1):
                    for i in T.Parallel(half):
                        alpha[i] = 1.0 / logsum[i]
                    for i, j in T.Parallel(half, dim):
                        acc_o[i, j] *= alpha[i]
                    # The warpgroup has stored the previous item before Os is reused.
                    T.sync_threads(3 + wg, 128)
                    T.copy(acc_o, Os[wg, :, :])
                    T.sync_threads(3 + wg, 128)
                    T.copy(Os[wg, :, :], O[q_start + row : q_start + row + half, head, :])
                else:
                    # A partial tile, or rows with no visible key (written as zero).
                    for i, j in T.Parallel(half, dim):
                        if T.if_then_else(pack == 1, row + i, (row + i) // groups) < q_len:
                            O[
                                q_start + T.if_then_else(pack == 1, row + i, (row + i) // groups),
                                head + T.if_then_else(pack == 1, 0, (row + i) % groups),
                                j,
                            ] = T.if_then_else(
                                logsum[i] > 0,
                                T.cast(acc_o[i, j] / logsum[i], dtype),
                                T.cast(0, dtype),
                            )
            done += eff
            served += 1
            T.mbarrier_wait_parity(q_bar[served % 2], (served // 2) % 2)
            work = item_slot[served % 2, 10]

    @T.macro
    def launch(Q, K, V, CuQ, CacheLengths, PageTable, O, Sched, Part, LSE, bx, by, bz):
        """One persistent CTA: the TMA producer warp, then the two consumer warpgroups."""
        Qs = T.alloc_shared([2, 2, half, dim], dtype)
        Ks = T.alloc_shared([stages, block_n, dim], dtype)
        Vs = T.alloc_shared([stages, block_n, dim], dtype)
        Os = T.alloc_shared([2, half, dim], dtype)
        tile_cum = T.alloc_shared([batch + 1], "int32")
        item_slot = T.alloc_shared([2, 11], "int32")
        Merge = T.alloc_shared([2], "int32")
        T.annotate_layout(
            {
                Qs: make_swizzled_layout(Qs),
                Ks: make_swizzled_layout(Ks),
                Vs: make_swizzled_layout(Vs),
                Os: make_swizzled_layout(Os),
            }
        )
        q_bar = T.alloc_barrier([32, 32])
        qfree = T.alloc_barrier([consumers, consumers])
        kready = T.alloc_barrier([producers] * stages)
        kfree = T.alloc_barrier([consumers] * stages)
        vready = T.alloc_barrier([producers] * stages)
        vfree = T.alloc_barrier([consumers] * stages)
        lo = T.alloc_local([1], "int32")
        hi = T.alloc_local([1], "int32")
        q_row = T.alloc_local([1], "int32")
        request = T.alloc_local([1], "int32")
        meta = T.alloc_local([10], "int32")
        work_shared = T.alloc_shared([1], "int32")

        tile_cum[0] = 0
        for request_id in T.serial(batch):
            length = CuQ[request_id + 1] - CuQ[request_id]
            count = T.if_then_else(
                length <= cut,
                T.ceildiv(length * groups, block_m) * heads_kv * splits,
                T.ceildiv(length, block_m) * heads,
            )
            tile_cum[request_id + 1] = tile_cum[request_id] + count
        T.sync_threads()
        tx = T.get_thread_binding()
        if tx >= consumers:
            T.set_max_nreg(producer_regs, 0)
        if tx >= consumers and tx < consumers + producers:
            producer(
                Q, K, V, CuQ, CacheLengths, PageTable, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree, kready, kfree,
                vready, vfree, tile_cum[batch], tile_cum, lo, hi, request, q_row, meta,
                tx - consumers, work_shared,
            )  # fmt: skip
        args = (
            Qs,
            Ks,
            Vs,
            Os,
            O,
            Part,
            LSE,
            Sched,
            Merge,
            item_slot,
            q_bar,
            qfree,
            kready,
            kfree,
            vready,
            vfree,
            meta,
        )
        with T.ws(0):
            consumer(0, *args)
        with T.ws(1):
            consumer(1, *args)

    return launch


def _make_decode(
    batch,
    heads,
    heads_kv,
    dim,
    page_size,
    width,
    causal,
    scale,
    softcap,
    dtype,
    splits=1,
    window_left=-1,
    window_right=-1,
):
    block_m, block_n, stages, threads = 16, 16, 1, 128
    group = heads // heads_kv
    assert group <= block_m and page_size % block_n == 0
    warps = threads // 32
    # Adjacent warps traverse adjacent window tiles; a split still owns a
    # disjoint contiguous interval. This improves the sparse window scan.
    stride = warps if window_left >= 0 else 1
    # Folding cap into exp2's scale saves an elementwise multiply. Avoid
    # underflow of that scale and loss of significant logits when hardware
    # tanh flushes subnormal inputs; extreme caps keep the ordinary formula.
    fast_cap = 2**-126 <= softcap <= 2**100
    sf = softcap * LOG2E if fast_cap else LOG2E if softcap else scale * LOG2E

    def make_step(n):
        softmax = make_online_softmax_with_mask_guard(sf, "float", block_m, n)
        rescale = make_rescale(block_m, dim)

        @T.macro
        def step(
            K,
            V,
            Table,
            Qs,
            Ks,
            Vs,
            S,
            P,
            O,
            sm,
            smp,
            alpha,
            ss,
            ls,
            req,
            head,
            warp,
            key0,
            kv_len,
            end,
            lower,
            tail: bool,
        ):
            if page_size % n == 0:
                # Small windows can leave most warps empty. Bound the
                # address for TileLang index legalization in that regime;
                # every executed tile is already inside these bounds.
                address_key = (
                    T.min(T.max(key0, 0), width * page_size - n)
                    if 0 <= window_left < block_n * warps * splits
                    else key0
                )
                base = Table[req, address_key // page_size] * page_size + address_key % page_size
                T.copy(
                    K[base : base + n, head, :],
                    Ks,
                    eviction_policy="evict_first" if window_left >= 0 else None,
                )
                T.copy(
                    V[base : base + n, head, :],
                    Vs,
                    eviction_policy="evict_first" if window_left >= 0 else None,
                )
            else:
                for j, d in T.Parallel(n, dim):
                    key = T.min(key0 + j, kv_len - 1)
                    row = Table[req, key // page_size] * page_size + key % page_size
                    Ks[j, d] = K[row, head, d]
                    Vs[j, d] = V[row, head, d]
            if tail and key0 + n > kv_len:
                for j, d in T.Parallel(n, dim):
                    if key0 + j >= kv_len:
                        Vs[j, d] = T.cast(0, dtype)
            T.clear(S)
            T.gemm(Qs, Ks, S, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            if softcap > 0:
                # The MMA tile pads the head group to 16 rows. Only real
                # heads need tanh; padded rows never reach the output.
                for i, j in T.Parallel(block_m, n):
                    if i < group:
                        if fast_cap:
                            S[i, j] = T.call_pure_extern(
                                "float32", "tileops::paged_tanh", S[i, j] * (scale / softcap)
                            )
                        elif softcap > 2**100:
                            # tanh(x) rounds to x at this relative precision.
                            # Taking that limit before division preserves
                            # logits whose ratio to cap would be subnormal.
                            logit = S[i, j] * scale
                            S[i, j] = T.if_then_else(
                                T.abs(logit) < softcap * 2**-12,
                                logit,
                                softcap * T.tanh(logit / softcap),
                            )
                        else:
                            S[i, j] = softcap * T.tanh(S[i, j] * (scale / softcap))
            if key0 + n > end or key0 < lower:
                for i, j in T.Parallel(block_m, n):
                    S[i, j] = T.if_then_else(
                        (key0 + j < end) & (key0 + j >= lower), S[i, j], -T.infinity("float")
                    )
            softmax(S, sm, smp, alpha, ss, ls)
            rescale(O, alpha)
            T.copy(S, P)
            T.gemm(P, Vs, O, policy=T.GemmWarpPolicy.FullRow)

        return step

    full_step, tail_step = make_step(block_n), make_step(16)

    @T.macro
    def consumer(Q, K, V, CuQ, Lengths, Table, Qs, Ks, Vs, Kt, Vt, WL, WO, qid, head, sid, warp):
        St = T.alloc_fragment((block_m, 16), "float")
        Pt = T.alloc_fragment((block_m, 16), dtype)
        S = T.alloc_fragment((block_m, block_n), "float")
        P = T.alloc_fragment((block_m, block_n), dtype)
        O = T.alloc_fragment((block_m, dim), "float")
        sm = T.alloc_fragment((block_m,), "float")
        smp = T.alloc_fragment((block_m,), "float")
        alpha = T.alloc_fragment((block_m,), "float")
        ss = T.alloc_fragment((block_m,), "float")
        ls = T.alloc_fragment((block_m,), "float")
        lo = T.alloc_var("int32", init=0)
        hi = T.alloc_var("int32", init=batch - 1)
        for _ in T.serial(max(1, (batch - 1).bit_length())):
            mid = (lo + hi) // 2
            if CuQ[mid + 1] <= qid:
                lo = mid + 1
            else:
                hi = mid
        req = lo
        kv_len = Lengths[req]
        pos = kv_len - (CuQ[req + 1] - qid)
        causal_end = T.max(0, T.min(kv_len, pos + 1)) if causal else kv_len
        lower = T.max(0, pos - window_left) if window_left >= 0 else 0
        end = (
            T.max(0, T.min(causal_end, pos + window_right + 1)) if window_right >= 0 else causal_end
        )
        first_tile = lower // block_n
        chunk = T.ceildiv(T.max(0, T.ceildiv(end, block_n) - first_tile), splits * warps)
        begin = (
            (first_tile + sid * warps * chunk + warp) * block_n
            if window_left >= 0
            else (first_tile + (sid * warps + warp) * chunk) * block_n
        )
        count = T.max(0, T.min(chunk, T.ceildiv(end - begin, block_n * stride)))
        T.clear(O)
        T.clear(ls)
        T.fill(sm, -T.infinity("float"))
        if window_left >= 0:
            full_count = T.max(
                0, T.min(count, T.ceildiv(kv_len // block_n - begin // block_n, stride))
            )
            tail_start = begin + full_count * block_n * stride
            tail_end = T.if_then_else(
                count > full_count, T.min(end, tail_start + block_n), tail_start
            )
        else:
            full_count = T.max(0, T.min(count, kv_len // block_n - begin // block_n))
            tail_start = begin + full_count * block_n
            tail_end = T.min(end, begin + count * block_n)
        for t in T.serial(T.max(0, T.ceildiv(tail_end - tail_start, 16))):
            tail_step(
                K,
                V,
                Table,
                Qs,
                Kt,
                Vt,
                St,
                Pt,
                O,
                sm,
                smp,
                alpha,
                ss,
                ls,
                req,
                head,
                warp,
                tail_start + t * 16,
                kv_len,
                end,
                lower,
                True,
            )
        for t in T.Pipelined(full_count, num_stages=stages):
            full_step(
                K,
                V,
                Table,
                Qs,
                Ks,
                Vs,
                S,
                P,
                O,
                sm,
                smp,
                alpha,
                ss,
                ls,
                req,
                head,
                warp,
                begin + t * block_n * stride,
                kv_len,
                end,
                lower,
                False,
            )
        for i, d in T.Parallel(block_m, dim):
            O[i, d] = T.if_then_else(ls[i] > 0, O[i, d] / ls[i], 0.0)
        for i in T.Parallel(block_m):
            ls[i] = T.if_then_else(
                ls[i] > 0, T.log2(T.max(ls[i], 1e-30)) + sm[i] * sf, -T.infinity("float")
            )
        for i in T.Parallel(block_m):
            if i < group:
                WL[warp, i] = ls[i]
        for i, d in T.Parallel(block_m, dim):
            if i < group:
                WO[warp, i, d] = O[i, d]

    @T.macro
    def launch(Q, K, V, CuQ, Lengths, Table, Output, Counter, Part, LSE, qid, sid, head):
        K0 = T.alloc_shared((block_n, dim), dtype)
        V0 = T.alloc_shared((block_n, dim), dtype)
        K1 = T.alloc_shared((block_n, dim), dtype)
        V1 = T.alloc_shared((block_n, dim), dtype)
        K2 = T.alloc_shared((block_n, dim), dtype)
        V2 = T.alloc_shared((block_n, dim), dtype)
        K3 = T.alloc_shared((block_n, dim), dtype)
        V3 = T.alloc_shared((block_n, dim), dtype)
        Qs = T.alloc_shared((block_m, dim), dtype)
        # Only the warp owning the final partial KV tile enters the
        # tail scan, so all four warps can share this separate buffer.
        Kt = T.alloc_shared((16, dim), dtype)
        Vt = T.alloc_shared((16, dim), dtype)
        WL = T.alloc_shared((warps, group), "float")
        WO = T.alloc_shared((warps, group, dim), "float")
        for i, d in T.Parallel(block_m, dim):
            Qs[i, d] = T.if_then_else(i < group, Q[qid, head * group + T.min(i, group - 1), d], 0)
        T.sync_threads()
        tx = T.get_thread_binding()
        if tx >= 0 and tx < 32:
            with T.attr(0, "warp_specialize", 1):
                consumer(
                    Q,
                    K,
                    V,
                    CuQ,
                    Lengths,
                    Table,
                    Qs,
                    K0,
                    V0,
                    Kt,
                    Vt,
                    WL,
                    WO,
                    qid,
                    head,
                    sid,
                    0,
                )
        if tx >= 32 and tx < 64:
            with T.attr(0, "warp_specialize", 1):
                consumer(
                    Q,
                    K,
                    V,
                    CuQ,
                    Lengths,
                    Table,
                    Qs,
                    K1,
                    V1,
                    Kt,
                    Vt,
                    WL,
                    WO,
                    qid,
                    head,
                    sid,
                    1,
                )
        if tx >= 64 and tx < 96:
            with T.attr(0, "warp_specialize", 1):
                consumer(
                    Q,
                    K,
                    V,
                    CuQ,
                    Lengths,
                    Table,
                    Qs,
                    K2,
                    V2,
                    Kt,
                    Vt,
                    WL,
                    WO,
                    qid,
                    head,
                    sid,
                    2,
                )
        if tx >= 96 and tx < 128:
            with T.attr(0, "warp_specialize", 1):
                consumer(
                    Q,
                    K,
                    V,
                    CuQ,
                    Lengths,
                    Table,
                    Qs,
                    K3,
                    V3,
                    Kt,
                    Vt,
                    WL,
                    WO,
                    qid,
                    head,
                    sid,
                    3,
                )
        T.sync_threads()
        # Reuse each warp-reduction weight across four adjacent output channels.
        for h, vector in T.Parallel(group, dim // 4):
            peak = T.alloc_var("float32", init=-T.infinity("float32"))
            denominator = T.alloc_var("float32", init=0.0)
            values4 = T.alloc_local((4,), "float32")
            for init_d in T.vectorized(4):
                values4[init_d] = 0.0
            for w in T.unroll(warps):
                peak = T.max(peak, WL[w, h])
            peak = T.if_then_else(peak == -T.infinity("float32"), 0.0, peak)
            for w in T.unroll(warps):
                weight = T.exp2(WL[w, h] - peak)
                denominator += weight
                for d in T.vectorized(4):
                    values4[d] += weight * WO[w, h, vector * 4 + d]
            for d in T.vectorized(4):
                value = T.if_then_else(denominator > 0, values4[d] / denominator, 0.0)
                if splits == 1:
                    Output[qid, head * group + h, vector * 4 + d] = value
                else:
                    Part[qid, head, sid, h, vector * 4 + d] = value
            if splits > 1 and vector == 0:
                LSE[qid, head, sid, h] = T.if_then_else(
                    denominator > 0,
                    T.log2(T.max(denominator, 1e-30)) + peak,
                    -T.infinity("float32"),
                )
        if splits > 1:
            ready = T.alloc_shared((1,), "int32")
            # The barrier orders every partial writer before leader publication.
            T.sync_threads()
            if tx == 0:
                ready[0] = T.call_extern(
                    "int32",
                    "tileops::paged_arrive",
                    T.access_ptr(Counter[qid * heads_kv + head], "rw", extent=1),
                )
            T.sync_threads()
            if ready[0] == splits - 1:
                for h, vector in T.Parallel(group, dim // 4):
                    peak = T.alloc_var("float32", init=-T.infinity("float32"))
                    denominator = T.alloc_var("float32", init=0.0)
                    values4 = T.alloc_local((4,), "float32")
                    for init_d in T.vectorized(4):
                        values4[init_d] = 0.0
                    for split in T.unroll(splits):
                        peak = T.max(peak, LSE[qid, head, split, h])
                    peak = T.if_then_else(peak == -T.infinity("float32"), 0.0, peak)
                    for split in T.unroll(splits):
                        weight = T.exp2(LSE[qid, head, split, h] - peak)
                        denominator += weight
                        for d in T.vectorized(4):
                            values4[d] += weight * Part[qid, head, split, h, vector * 4 + d]
                    for d in T.vectorized(4):
                        Output[qid, head * group + h, vector * 4 + d] = T.if_then_else(
                            denominator > 0, values4[d] / denominator, 0.0
                        )
                T.sync_threads()
                if tx == 0:
                    Counter[qid * heads_kv + head] = 0

    return launch


# Both dtypes, load modes, split counts, and window/softcap specializations.
@functools.lru_cache(maxsize=64)
def paged_unified_kernel(
    batch,
    heads,
    heads_kv,
    dim,
    page_size,
    width,
    causal,
    scale,
    softcap,
    dtype,
    decode,
    splits,
    num_ctas,
    short_q,
    load_mode,
    window_left=-1,
    window_right=-1,
):
    """Build one CUDA kernel; packed query count selects its launch geometry."""

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: not decode,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: decode,
            # Producer and consumers overlap. Reusing their shared allocations
            # based on textual lifetimes corrupts the live request queue.
            tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: decode,
        },
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            *_PAGED_HELPER_FLAGS,
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DNDEBUG",
        ],
    )
    def build():
        total_q, pool_rows = T.dynamic("total_q"), T.dynamic("pool_rows")
        rows = 16 if decode else max(128, ((short_q * (heads // heads_kv) + 127) // 128) * 128)
        owners = total_q if decode else batch
        counters = total_q * heads_kv if decode else 2 + batch * heads_kv * (rows // 64)
        grid = total_q if decode else num_ctas
        grid_s = splits if decode else 1
        grid_h = heads_kv if decode else 1
        threads = 128 if decode else 384
        if decode:
            compute = _make_decode(
                batch,
                heads,
                heads_kv,
                dim,
                page_size,
                width,
                causal,
                scale,
                softcap,
                dtype,
                splits,
                window_left,
                window_right,
            )
        else:
            compute = _make_prefill(
                batch,
                heads,
                heads_kv,
                dim,
                causal,
                scale,
                softcap,
                dtype,
                128,
                2,
                num_ctas,
                page_size,
                width,
                load_mode,
                88 if load_mode == "async_grouped_ahead" else 40,
                short_q + 1,
                splits,
            )

        @T.prim_func
        def main(
            Q: T.Tensor((total_q, heads, dim), dtype),
            K: T.Tensor((pool_rows, heads_kv, dim), dtype),
            V: T.Tensor((pool_rows, heads_kv, dim), dtype),
            CuQ: T.Tensor((batch + 1,), "int32"),
            Lengths: T.Tensor((batch,), "int32"),
            Table: T.Tensor((batch, width), "int32"),
            Output: T.Tensor((total_q, heads, dim), dtype),
            Counter: T.Tensor((counters,), "int32"),
            Part: T.Tensor((owners, heads_kv, splits, rows, dim), dtype),
            LSE: T.Tensor((owners, heads_kv, splits, rows), "float32"),
        ):
            with T.Kernel(grid, grid_s, grid_h, threads=threads) as (bx, by, bz):
                compute(Q, K, V, CuQ, Lengths, Table, Output, Counter, Part, LSE, bx, by, bz)

        return main

    return build()
