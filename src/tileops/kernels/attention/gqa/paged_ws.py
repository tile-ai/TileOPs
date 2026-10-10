"""SM90 warp-specialized grouped-query attention over a paged KV cache the kernel only reads.

A persistent CTA per SM runs a TMA producer warp and two consumer warpgroups. A work item is
one KV head and 128 query rows: ``128 // group`` query positions of each of the head's
``group`` query heads, so the cache crosses memory once per KV head, and each query head's
positions are one contiguous TMA box of Q.

A key tile is whole pages or part of one page, so each page it touches is one TMA copy. The
rows a page holds past the cache end are stale and may hold any bits; the producer zeroes V's
in the one tile the cache end cuts before the consumers read it.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    AttentionCall,
    GQAPagedFwdInterface,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_sm_count

__all__ = ["GQAPagedFwdWSKernel"]


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
    ],
)
def _gqa_paged_ws_kernel(
    batch, heads, heads_kv, dim, page_size, max_pages_per_req, is_causal, sm_scale, softcap,
    dtype, block_n, stages, num_ctas,
):  # fmt: skip
    """A persistent CTA per SM: a TMA producer warp claims work, two consumer warpgroups run it."""
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else sm_scale * LOG2E
    group = heads // heads_kv
    accum = "float"
    half = 64  # rows per consumer warpgroup: one m64 WGMMA
    block_m = 2 * half
    consumers = 256  # threads of the two consumer warpgroups
    policy = T.GemmWarpPolicy.FullRow
    total_q = T.dynamic("total_q")
    pool_rows = T.dynamic("pool_rows")
    # Row r of a tile is query r % span of head kv_head * group + r // span.
    span = block_m // group
    tiling = GroupTiling(batch, span)
    # A key tile is part of one page or a run of whole pages: one TMA copy per page.
    box = min(block_n, page_size)
    boxes = block_n // box
    # 16-byte chunks of one V row, the unit the producer zeroes.
    chunks = dim // 8

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
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
        T.reduce_sum(acc_s, ss, dim=1, batch=2)

    @T.macro
    def kv_step(
        q_tile, Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp, alpha, ss,
        red, logsum, my_bar, nxt_bar, k, n, q0, wg, align, kv_len, tail: bool,
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
        if tail:
            mask(acc_s, k * block_n, q0, wg, align, kv_len)
        softmax_step(acc_s, sm, smp, alpha, ss, red)
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        for i in T.Parallel(half):
            logsum[i] = logsum[i] * alpha[i] + ss[i]
        T.copy(acc_s, pcast)

    @T.macro
    def locate(work, CuQ, cache_seqlens, tile_cum, lo, hi, request, q_row, meta):
        """Fill meta (kv head, request, q_start, q_len, kv_len, q0, tiles) for *work*.

        Items run from the last query tile back, so a causal request's longest tiles start
        first.
        """
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
        # cannot leave the consumers waiting on a tile the producer never loads. The scan
        # stays unconditional: under a branch ptxas keeps the WGMMA descriptors out of
        # uniform registers.
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
        end repeats the last one, whose rows the mask and the V zeroing discard."""
        last_page = T.max(0, (kv_len - 1) // page_size)
        for b in T.unroll(boxes):
            key = key0 + b * box
            rows[b] = page_table[request, T.min(key // page_size, last_page)] * page_size + (
                key % page_size
            )

    @T.macro
    def load_tile(pool, dst, stage, bars, bar, rows, cv):
        """TMA one key tile of *pool* into stage *stage* of *dst*, landing on ``bars[bar]``:
        a box a page and a 128-byte column, the widest box that lands on a row range of the
        swizzled buffer."""
        for b in T.unroll(boxes):
            for c in T.unroll(dim // 64):
                T.tma_copy(
                    pool[rows[b] : rows[b] + box, cv, c * 64 : (c + 1) * 64],
                    dst[stage, b * box : (b + 1) * box, c * 64 : (c + 1) * 64],
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
                for c in T.unroll(dim // 64):
                    T.tma_copy(
                        Q[first : first + rows, head, c * 64 : (c + 1) * 64],
                        Qs[slot, wg, h * rows : (h + 1) * rows, c * 64 : (c + 1) * 64],
                        barrier=q_bar[slot],
                    )

    @T.macro
    def producer(
        Q, K, V, CuQ, cache_seqlens, page_table, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree,
        kready, kfree, vready, vfree, vcut, total, tile_cum, lo, hi, request, q_row, meta, lane,
    ):  # fmt: skip
        """Claim work items until none remain; load each one's Q, then its key tiles."""
        issued = T.alloc_var("int32", init=0)
        loaded = T.alloc_var("int32", init=0)
        cuts = T.alloc_var("int32", init=0)
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
            for i in T.unroll(7):
                item_slot[slot, i] = meta[i]
            item_slot[slot, 7] = work[0]
            load_q(Q, Qs, slot, q_bar, meta[2] + meta[5], cv)
            T.mbarrier_arrive(q_bar[slot])
            for k in T.serial(meta[6]):
                n = issued + k
                s = n % stages
                key0 = k * block_n
                # Read before the wait, so the table's latency hides behind the consumers.
                tile_rows(page_table, rows, req, key0, kv_len)
                T.mbarrier_wait_parity(kfree[s], ((n // stages) % 2) ^ 1)
                load_tile(K, Ks, s, kready, s, rows, cv)
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
            fetch(Sched, lane, work)
        # Stop the consumers, then leave the counters zeroed for the next launch.
        T.mbarrier_wait_parity(qfree[loaded % 2], ((loaded // 2) % 2) ^ 1)
        item_slot[loaded % 2, 7] = -1
        T.mbarrier_arrive(q_bar[loaded % 2])
        if lane == 0:
            finished = T.atomic_add(Sched[1], 1, return_prev=True)
            if finished == num_ctas - 1:
                Sched[0] = 0
                Sched[1] = 0

    @T.macro
    def consumer(
        wg: int, Qs, Ks, Vs, O, item_slot, q_bar, qfree, kready, kfree, vready, vfree, meta,
    ):  # fmt: skip
        """Consumer warpgroup *wg*: rows ``wg*64`` onward of each claimed item's tile."""
        T.set_max_nreg(240, 1)
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
        # Finished key tiles and items: they carry the barrier phases.
        done = T.alloc_var("int32", init=0)
        served = T.alloc_var("int32", init=0)
        work = T.alloc_var("int32", init=0)
        T.mbarrier_wait_parity(q_bar[0], 0)
        work = item_slot[0, 7]
        while work >= 0:
            for i in T.unroll(7):
                meta[i] = item_slot[served % 2, i]
            kv_head = meta[0]
            q_start = meta[2]
            q_len = meta[3]
            kv_len = meta[4]
            # Query position p of the request is key align + p.
            align = kv_len - q_len
            q0 = meta[5]
            eff = meta[6]
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(alpha, 1.0)
            T.fill(sm, -T.infinity(accum))
            if wg == 1 and served == 0:
                T.named_barrier_arrive(1, consumers)  # let warpgroup 0 go first

            # Key tile 0: QK and softmax only.
            s0 = done % stages
            T.sync_threads(my_bar, consumers)
            T.mbarrier_wait_parity(kready[s0], (done // stages) % 2)
            T.wgmma_gemm(
                Qs[served % 2, wg, :, :], Ks[s0, :, :], acc_s, transpose_B=True,
                policy=policy, clear_accum=True,
            )  # fmt: skip
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
            if full < 1:
                mask(acc_s, 0, q0, wg, align, kv_len)
            if use_softcap:
                apply_softcap(acc_s)
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            for i, j in T.Parallel(half, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
            T.reduce_sum(acc_s, ss, dim=1, batch=2)
            for i in T.Parallel(half):
                logsum[i] = ss[i]
            T.copy(acc_s, pcast)

            step = (Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
            for k in T.serial(1, T.max(1, full)):
                kv_step(
                    Qs[served % 2, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                    k, done + k, q0, wg, align, kv_len, False,
                )  # fmt: skip
            for k in T.serial(T.max(1, full), eff):
                kv_step(
                    Qs[served % 2, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                    k, done + k, q0, wg, align, kv_len, True,
                )  # fmt: skip
            # Every QK is done, so the producer may reload Q.
            T.mbarrier_arrive(qfree[served % 2])

            # PV of the last tile.
            svp = (done + eff - 1) % stages
            for i, j in T.Parallel(half, dim):
                acc_o[i, j] *= alpha[i]
            T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
            T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
            T.wait_wgmma(0)
            T.mbarrier_arrive(vfree[svp])

            # Every row sees key 0, so its sum is positive.
            for i in T.Parallel(half):
                alpha[i] = 1.0 / logsum[i]
            for i, d in T.Parallel(half, dim):
                r = wg * half + i
                if q0 + r % span < q_len:
                    O[q_start + q0 + r % span, kv_head * group + r // span, d] = T.cast(
                        acc_o[i, d] * alpha[i], dtype
                    )
            done += eff
            served += 1
            T.mbarrier_wait_parity(q_bar[served % 2], (served // 2) % 2)
            work = item_slot[served % 2, 7]

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, heads, dim], dtype),
        K: T.Tensor([pool_rows, heads_kv, dim], dtype),
        V: T.Tensor([pool_rows, heads_kv, dim], dtype),
        cache_seqlens: T.Tensor([batch], "int32"),
        page_table: T.Tensor([batch, max_pages_per_req], "int32"),
        CuQ: T.Tensor([batch + 1], "int32"),
        O: T.Tensor([total_q, heads, dim], dtype),
        Sched: T.Tensor([2], "int32"),
    ):
        with T.Kernel(num_ctas, threads=384):
            Qs = T.alloc_shared([2, 2, half, dim], dtype)
            Ks = T.alloc_shared([stages, block_n, dim], dtype)
            Vs = T.alloc_shared([stages, block_n, dim], dtype)
            tile_cum = T.alloc_shared([batch + 1], "int32")
            item_slot = T.alloc_shared([2, 8], "int32")
            T.annotate_layout(
                {
                    Qs: make_swizzled_layout(Qs),
                    Ks: make_swizzled_layout(Ks),
                    Vs: make_swizzled_layout(Vs),
                }
            )
            q_bar = T.alloc_barrier([32, 32])
            qfree = T.alloc_barrier([consumers, consumers])
            kready = T.alloc_barrier([32] * stages)
            kfree = T.alloc_barrier([consumers] * stages)
            vready = T.alloc_barrier([32] * stages)
            vfree = T.alloc_barrier([consumers] * stages)
            vcut = T.alloc_barrier([32])
            lo = T.alloc_local([1], "int32")
            hi = T.alloc_local([1], "int32")
            q_row = T.alloc_local([1], "int32")
            request = T.alloc_local([1], "int32")
            meta = T.alloc_local([7], "int32")

            tiling.cumsum_offsets(CuQ, tile_cum)
            T.sync_threads()
            tx = T.get_thread_binding()
            if tx >= consumers:
                T.set_max_nreg(24, 0)  # the producer only issues TMA
            if tx >= consumers and tx < consumers + 32:
                producer(
                    Q, K, V, CuQ, cache_seqlens, page_table, Sched, Qs, Ks, Vs, item_slot, q_bar,
                    qfree, kready, kfree, vready, vfree, vcut, tile_cum[batch] * heads_kv,
                    tile_cum, lo, hi, request, q_row, meta, tx - consumers,
                )  # fmt: skip
            args = (Qs, Ks, Vs, O, item_slot, q_bar, qfree, kready, kfree, vready, vfree, meta)
            with T.ws(0):
                consumer(0, *args)
            with T.ws(1):
                consumer(1, *args)

    return main


class GQAPagedFwdWSKernel(Kernel, GQAPagedFwdInterface):
    """SM90 warp-specialized paged attention for packings whose requests fill query tiles.

    It serves float16 and bfloat16 heads of dimension 64 or 128 over pages of a multiple of 16
    rows, causal or bidirectional, with a positive scale and an optional softcap, when the
    packed query rows average at least one 128-row tile a request. It does not split a
    request's keys across CTAs, so a decode packing, whose tiles fall short of the
    multiprocessors, stays on ``GQAPagedFwdKernel``'s split scan; so do a window, RoPE and FP8.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_paged_varlen_kernel"})
    _DIMS: tuple[int, ...] = (64, 128)
    # A tile's 128 rows are whole heads' query runs, each at least the 8 rows of a swizzle
    # atom so that its TMA box starts on one.
    _GROUPS: tuple[int, ...] = (1, 2, 4, 8, 16)
    _STAGES: int = 2
    # Rows of one work item: two consumer warpgroups of one m64 WGMMA each.
    _TILE_ROWS: int = 128
    # The per-request tile prefix shares shared memory with the Q, K and V buffers, which take
    # 192 of the 227 KB at dimension 128; 4096 requests take 16 KB of the rest.
    _MAX_BATCH: int = 4096

    @staticmethod
    def block_n_for(page_size: int) -> Optional[int]:
        """The widest key tile up to 128 rows that is part of one page or a run of whole
        pages, a multiple of the 16 rows one WGMMA step reduces over; ``None`` when none is.

        A box smaller than a page costs the producer a table read and a TMA copy per box: at
        a 48-row page, 128-row tiles of 16-row boxes measured slower than 96-row tiles of
        whole pages.
        """
        for block_n in (128, 112, 96, 80, 64, 48, 32, 16):
            if page_size % block_n == 0 or block_n % page_size == 0 and page_size % 16 == 0:
                return block_n
        return None

    @classmethod
    def applies(cls, call: AttentionCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why *call* is outside this implementation's region, or ``None`` when it is inside."""
        if call.dtype not in ATTENTION_DTYPES or call.cache_dtype != call.dtype:
            return "requires float16 or bfloat16 Q and KV of one dtype"
        if call.is_fp8:
            return "does not serve FP8"
        if call.dim not in cls._DIMS:
            return f"requires head dimension in {cls._DIMS}"
        if call.heads_kv <= 0 or call.heads // call.heads_kv not in cls._GROUPS:
            return f"requires a query group of {cls._GROUPS} heads"
        if call.uses_sliding_window:
            return "does not serve a window"
        if call.fuse_rope:
            return "does not serve RoPE"
        if call.sm_scale is not None and call.sm_scale <= 0.0:
            return "requires a positive scale"
        if call.page_size <= 0 or cls.block_n_for(call.page_size) is None:
            return "requires a page size that is a multiple of 16 rows"
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
            sm_scale=call.sm_scale,
            softcap=call.softcap,
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
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
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
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        # One counter per stream: concurrent launches on one counter steal each other's items.
        self._counters: dict = {}
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def kernel(self):
        return _gqa_paged_ws_kernel(
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
            self.block_n_for(self.page_size),
            self._STAGES,
            get_sm_count(self.device_index),
        )

    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            # Each captured graph owns a counter from its private pool.
            counter = torch.zeros(2, dtype=torch.int32, device=q.device)
        else:
            stream = torch.cuda.current_stream(q.device).cuda_stream
            counter = self._counters.get(stream)
            if counter is None:
                counter = torch.zeros(2, dtype=torch.int32, device=q.device)
                self._counters[stream] = counter
        out = torch.empty_like(q)
        self.kernel(q, k_pool, v_pool, cache_seqlens, page_table, cu_seqlens_q, out, counter)
        return out
