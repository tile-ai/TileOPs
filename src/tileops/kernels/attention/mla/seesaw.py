"""The SM90 staggered-seesaw program for Multi-Head Latent Attention (MLA) decode, absorbed form.

A request's keys are its cache rows, whole, and its values their first ``dim_v``
columns. A block owns 64 query rows of one request (row ``s * heads + h``).

Two consumer warpgroups are staggered over pairs of 64-key tiles (a, b). Warpgroup 0
scores a, warpgroup 1 scores b, and the running max passes one way, m -> m_a -> m_b ->
the next pair, so each warpgroup's softmax runs while the other's matrix products do.
Warpgroup 0 holds O's left value half and warpgroup 1 the right; each multiplies both
probability tiles into its half, its own from registers and the other's from shared
memory, where it sits in the rope slot of its tile once that tile is scored. A producer
warp loads the tiles by TMA, or the producer warpgroup converts an FP8 cache into the
query dtype, and asks L2 for the tile after the one it loads.

The cache is either paged, one pool of ``dim``-wide rows reached through a block table,
or dense, each request's keys a slab of its own split into a ``dim_v``-wide latent part
and the rope part.
"""

import functools

import tilelang
import tilelang.language as T
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops.kernels.constants import LOG2E

__all__: list[str] = []


def _mla_seesaw_program(
    heads,
    s_q,
    dim,
    dim_v,
    page_size,
    is_causal,
    sm_scale,
    dtype,
    cache_dtype,
    num_splits,
    partial_dtype,
):
    """Staggered seesaw: see the module docstring.

    ``page_size`` None reads a dense cache: ``Q`` and ``K`` hold the latent columns,
    ``Qpe`` and ``Kpe`` the rope ones, and every request has ``seqlen`` keys. Otherwise
    ``Q`` and ``KV`` hold whole rows, reached through ``Table`` up to ``Lens``.

    With ``num_splits`` above one, block z of a request takes the z-th of that many equal
    runs of its tiles and writes a partial output in ``partial_dtype`` and a float32
    log-sum-exp, which ``_mla_seesaw_combine_kernel`` merges.
    """
    dense = page_size is None
    rows = s_q * heads
    batch = T.dynamic("batch")
    half_v = dim_v // 2
    pe = dim - dim_v
    fp8 = cache_dtype == "float8_e4m3fn"
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    accum = "float"
    # Rows a block owns: one m64 WGMMA per warpgroup.
    block_h = 64
    # Keys a tile holds: two 64-key tiles and the query fill the 227 KB an SM90 block may
    # use. A 32-key tile reads more shared memory per score product.
    block_n = 64
    consumers = 256
    # A key row's bytes in the cache: an L2 prefetch moves whole rows.
    row_bytes = dim * (1 if fp8 else 2)
    # The most pages a tile spans, for the L2 prefetch: a 16-bit cache's tile sits in one
    # page; an FP8 cache's spans several, one more where a page boundary falls inside it.
    if not dense:
        aligned = block_n % page_size == 0 or page_size % block_n == 0
        pages_per_tile = max(1, block_n // page_size) if aligned else (block_n - 2) // page_size + 2
    # A dense request read by one head block streams its tiles at the DRAM rate, and an
    # L2 prefetch there only adds L2 traffic; where several head blocks read each tile, it
    # hides the latency of the tile all of them wait for.
    prefetching = not dense or rows > block_h
    if num_splits == 1:
        out_shape, out_dtype, lse_shape = (batch, rows, dim_v), dtype, (batch, rows)
    else:
        out_shape, out_dtype = (batch, num_splits, rows, dim_v), partial_dtype
        lse_shape = (batch, num_splits, rows)

    def tile_row(Table, b, t, r):
        """The cache row holding key ``t * block_n + r``; past the table it is any valid row."""
        key = t * block_n + r
        return Table[b, T.min(key // page_size, Table.shape[1] - 1)] * page_size + key % page_size

    @T.macro
    def score(Ql, Qr, Qp, Kl, Kr, Kp, kready, s, lt, acc_s, scale):
        T.mbarrier_wait_parity(kready[s], (lt // 2) % 2)
        T.wgmma_gemm(Ql, Kl[s, :, :], acc_s, transpose_B=True, clear_accum=True)
        T.wgmma_gemm(Qr, Kr[s, :, :], acc_s, transpose_B=True, clear_accum=False)
        T.wgmma_gemm(Qp, Kp[s, :, :], acc_s, transpose_B=True, clear_accum=False)
        T.wait_wgmma(0)
        # Scaled before the max: the max of the scaled scores is the softmax's for a scale
        # of any sign, zero included.
        for r, j in T.Parallel(block_h, block_n):
            acc_s[r, j] *= scale

    @T.macro
    def mask(acc_s, Kl, Kr, s, t, L, hb, own_bar):
        """Hide the keys a row may not see, and zero the values past the request's end.

        A paged cache row past ``L`` may hold anything, NaN included, and a zero
        probability times NaN is NaN, so its value columns are cleared before any product
        reads them. A dense tile's rows past ``L`` lie past the slab, where TMA reads zero.
        """
        last_visible = L - s_q if is_causal else L - 1
        if (t + 1) * block_n > last_visible + 1:
            for r, j in T.Parallel(block_h, block_n):
                last = L - s_q + (hb * block_h + r) // heads if is_causal else L - 1
                acc_s[r, j] = T.if_then_else(
                    t * block_n + j > last, -T.infinity(accum), acc_s[r, j]
                )
        # The FP8 producer clears these rows as it converts the tile. Here the warpgroup's
        # threads clear rows others may read: every store lands before any product.
        if not fp8 and not dense and (t + 1) * block_n > L:
            for r, c in T.Parallel(block_n, half_v):
                if t * block_n + r >= L:
                    Kl[s, r, c] = T.cast(0, dtype)
                    Kr[s, r, c] = T.cast(0, dtype)
            T.fence_proxy_async()
            T.sync_threads(own_bar, 128)

    @T.macro
    def rescale(acc_o, al):
        """Multiply the rows whose max moved; skip the product when no row of this thread did."""
        moved = T.alloc_var("int32", init=0)
        for r in T.Parallel(block_h):
            if al[r] != 1.0:
                moved = 1
        if moved == 1:
            for r, c in T.Parallel(block_h, half_v):
                acc_o[r, c] *= al[r]

    @T.macro
    def lazy_max(m, rowmax, al):
        """Keep a stale max while the scores stay within 2^8 of it: probabilities stay
        below 256 and O is not rescaled."""
        for r in T.Parallel(block_h):
            al[r] = 1.0
            if rowmax[r] - m[r] > 8.0:
                al[r] = T.exp2(m[r] - rowmax[r])
                m[r] = rowmax[r]

    @T.macro
    def store(acc_o, O, K, b, z, hb, w, scale_out, l, own_bar):
        """Write this warpgroup's value half of O.

        A 16-bit O is staged in this warpgroup's key buffer ``K``, which only its own
        products read once the other warpgroup has scored its last tile, and leaves in
        whole lines by TMA; a float32 O is stored from registers.
        """
        r0 = hb * block_h
        if out_dtype == accum:
            # A row that saw no key in this run (a run past its last visible key) has max
            # -inf, so its probabilities and l are NaN; every later tile of the run hides it
            # too, and it is written as zero with log-sum-exp -inf.
            for r, c in T.Parallel(block_h, half_v):
                if r0 + r < rows:
                    O[b, z, r0 + r, w * half_v + c] = T.if_then_else(
                        l[r] > 0, acc_o[r, c] * scale_out / l[r], 0.0
                    )
        else:
            # The 16-bit O is in the key tiles' dtype: one tile slot holds it.
            stage = K[0, :, :]
            for r, c in T.Parallel(block_h, half_v):
                K[0, r, c] = T.cast(
                    T.if_then_else(l[r] > 0, acc_o[r, c] * scale_out / l[r], 0.0), out_dtype
                )
            T.fence_proxy_async()
            T.sync_threads(own_bar, 128)
            # TMA drops the rows past the last head.
            if num_splits == 1:
                T.copy(stage, O[b, r0 : r0 + block_h, w * half_v : (w + 1) * half_v])
            else:
                T.copy(stage, O[b, z, r0 : r0 + block_h, w * half_v : (w + 1) * half_v])

    @T.macro
    def wg0(
        Ql,
        Qr,
        Qp,
        Kl,
        Kr,
        Kp,
        ms,
        ss,
        lfin,
        qbar,
        kready,
        kfree,
        O,
        Lse,
        b,
        z,
        hb,
        L,
        t0,
        NI,
        scale,
        scale_out,
    ):
        T.set_max_nreg(208 if fp8 else 240, 1)
        acc_s = T.alloc_fragment([block_h, block_n], accum)
        pcast = T.alloc_fragment([block_h, block_n], dtype)
        acc_o = T.alloc_fragment([block_h, half_v], accum)
        m = T.alloc_fragment([block_h], accum)
        mn = T.alloc_fragment([block_h], accum)
        al = T.alloc_fragment([block_h], accum)
        sm = T.alloc_fragment([block_h], accum)
        l = T.alloc_fragment([block_h], accum)
        T.fill(acc_o, 0)
        T.fill(l, 0)
        T.fill(m, -T.infinity(accum))
        T.mbarrier_wait_parity(qbar[0], 0)
        for i in T.serial(T.ceildiv(NI, 2)):
            ta = 2 * i
            ph = i % 2
            score(Ql, Qr, Qp, Kl, Kr, Kp, kready, 0, ta, acc_s, scale)
            mask(acc_s, Kl, Kr, 0, t0 + ta, L, hb, 12)
            T.reduce_max(acc_s, mn, dim=1, clear=True)
            lazy_max(m, mn, al)
            for r, j in T.Parallel(block_h, block_n):
                acc_s[r, j] = T.exp2(acc_s[r, j] - m[r])
            T.reduce_sum(acc_s, sm, dim=1)
            T.copy(acc_s, pcast)
            T.copy(pcast, Kp[0, :, :])
            for r in T.Parallel(block_h):
                ms[ph, 0, r] = m[r]
            T.fence_proxy_async()
            T.named_barrier_arrive(10, consumers)  # P_a, m_a
            for r in T.Parallel(block_h):
                l[r] = l[r] * al[r] + sm[r]
            rescale(acc_o, al)
            T.wgmma_gemm(pcast, Kl[0, :, :], acc_o, clear_accum=False)
            T.sync_threads(11, consumers)  # P_b, m_b, sum_b
            for r in T.Parallel(block_h):
                mn[r] = ms[ph, 1, r]
                al[r] = 1.0
                if mn[r] != m[r]:
                    al[r] = T.exp2(m[r] - mn[r])
                    m[r] = mn[r]
                l[r] = l[r] * al[r] + ss[ph, r]
            T.wait_wgmma(0)
            T.mbarrier_arrive(kfree[0])
            rescale(acc_o, al)
            if ta + 1 < NI:
                T.wgmma_gemm(Kp[1, :, :], Kl[1, :, :], acc_o, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[1])
        for r in T.Parallel(block_h):
            lfin[r] = l[r]
        T.named_barrier_arrive(14, consumers)
        store(acc_o, O, Kl, b, z, hb, 0, scale_out, l, 12)
        # log-sum-exp in the natural base; -inf for a row that saw no key in this run.
        for r in T.Parallel(block_h):
            if hb * block_h + r < rows:
                lse = T.if_then_else(l[r] > 0, T.log(l[r]) + m[r] / LOG2E, -T.infinity(accum))
                if num_splits == 1:
                    Lse[b, hb * block_h + r] = lse
                else:
                    Lse[b, z, hb * block_h + r] = lse

    @T.macro
    def wg1(
        Ql,
        Qr,
        Qp,
        Kl,
        Kr,
        Kp,
        ms,
        ss,
        lfin,
        qbar,
        kready,
        kfree,
        O,
        b,
        z,
        hb,
        L,
        t0,
        NI,
        scale,
        scale_out,
    ):
        T.set_max_nreg(208 if fp8 else 240, 1)
        acc_s = T.alloc_fragment([block_h, block_n], accum)
        pcast = T.alloc_fragment([block_h, block_n], dtype)
        acc_o = T.alloc_fragment([block_h, half_v], accum)
        m = T.alloc_fragment([block_h], accum)
        mn = T.alloc_fragment([block_h], accum)
        al = T.alloc_fragment([block_h], accum)
        sm = T.alloc_fragment([block_h], accum)
        lf = T.alloc_fragment([block_h], accum)
        T.fill(acc_o, 0)
        T.fill(m, -T.infinity(accum))
        T.mbarrier_wait_parity(qbar[0], 0)
        for i in T.serial(T.ceildiv(NI, 2)):
            tb = 2 * i + 1
            ph = i % 2
            if tb < NI:
                score(Ql, Qr, Qp, Kl, Kr, Kp, kready, 1, tb, acc_s, scale)
                mask(acc_s, Kl, Kr, 1, t0 + tb, L, hb, 13)
                T.reduce_max(acc_s, sm, dim=1, clear=True)
            T.sync_threads(10, consumers)  # P_a, m_a
            # O's right half moves to the basis of m_a and takes P_a.
            for r in T.Parallel(block_h):
                mn[r] = ms[ph, 0, r]
                al[r] = 1.0
                if mn[r] != m[r]:
                    al[r] = T.exp2(m[r] - mn[r])
                    m[r] = mn[r]
            rescale(acc_o, al)
            T.wgmma_gemm(Kp[0, :, :], Kr[0, :, :], acc_o, clear_accum=False)
            if tb < NI:
                lazy_max(m, sm, al)
                for r, j in T.Parallel(block_h, block_n):
                    acc_s[r, j] = T.exp2(acc_s[r, j] - m[r])
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, pcast)
                T.copy(pcast, Kp[1, :, :])
            else:
                T.fill(al, 1.0)
                T.fill(sm, 0)
            for r in T.Parallel(block_h):
                ms[ph, 1, r] = m[r]
                ss[ph, r] = sm[r]
            T.fence_proxy_async()
            T.named_barrier_arrive(11, consumers)  # P_b, m_b, sum_b
            T.wait_wgmma(0)
            T.mbarrier_arrive(kfree[0])
            if tb < NI:
                rescale(acc_o, al)
                T.wgmma_gemm(pcast, Kr[1, :, :], acc_o, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[1])
        T.sync_threads(14, consumers)
        for r in T.Parallel(block_h):
            lf[r] = lfin[r]
        store(acc_o, O, Kr, b, z, hb, 1, scale_out, lf, 13)

    @T.macro
    def load_q(Q, QP, Ql, Qr, Qp, qbar, b, hb):
        """The block's query rows; a paged query holds its rope columns after the latent."""
        r0 = hb * block_h
        pe0 = 0 if dense else dim_v
        T.tma_copy(Q[b, r0 : r0 + block_h, 0:half_v], Ql, barrier=qbar[0])
        T.tma_copy(Q[b, r0 : r0 + block_h, half_v:dim_v], Qr, barrier=qbar[0])
        T.tma_copy(QP[b, r0 : r0 + block_h, pe0 : pe0 + pe], Qp, barrier=qbar[0])
        T.mbarrier_arrive(qbar[0])

    @T.macro
    def prefetch_rows(base, row0, count, width: int):
        """Ask L2 for ``count`` rows of ``width`` 16-bit columns, from row ``row0``."""
        T.evaluate(
            T.call_extern(
                "handle", "tileops::bulk_prefetch_l2", base.data,
                T.Cast("int64", row0) * width * 2, T.Cast("uint32", count * width * 2),
            )
        )  # fmt: skip

    @T.macro
    def prefetch(KV, KP, Table, b, t, L):
        """Ask L2 for the rows of tile t: a dense slab's at once, a paged cache's page by page."""
        if dense:
            row0 = b * L + t * block_n
            count = T.min(block_n, L - t * block_n)
            prefetch_rows(KV, row0, count, dim_v)
            prefetch_rows(KP, row0, count, pe)
        elif aligned:
            piece = min(block_n, page_size)
            for j in T.unroll(pages_per_tile):
                k0 = tile_row(Table, b, t, j * piece)
                T.evaluate(
                    T.call_extern(
                        "handle", "tileops::bulk_prefetch_l2", KV.data,
                        T.Cast("int64", k0) * row_bytes, T.uint32(piece * row_bytes),
                    )
                )  # fmt: skip
        else:
            for j in T.unroll(pages_per_tile):
                # The tile's keys on its j-th page: [lo, hi).
                page = t * block_n // page_size + j
                lo = T.max(t * block_n, page * page_size)
                hi = T.min(t * block_n + block_n, page * page_size + page_size)
                if lo < hi:
                    k0 = tile_row(Table, b, t, lo - t * block_n)
                    T.evaluate(
                        T.call_extern(
                            "handle", "tileops::bulk_prefetch_l2", KV.data,
                            T.Cast("int64", k0) * row_bytes,
                            T.Cast("uint32", (hi - lo) * row_bytes),
                        )
                    )  # fmt: skip

    @T.macro
    def load_tile(KV, KP, Table, Kl, Kr, Kp, kready, s, b, t):
        """The tile's latent halves and rope part: from the slab, or from its page."""
        if dense:
            k0 = t * block_n
            T.tma_copy(KV[b, k0 : k0 + block_n, 0:half_v], Kl[s, :, :], barrier=kready[s])
            T.tma_copy(KV[b, k0 : k0 + block_n, half_v:dim_v], Kr[s, :, :], barrier=kready[s])
            T.tma_copy(KP[b, k0 : k0 + block_n, 0:pe], Kp[s, :, :], barrier=kready[s])
        else:
            k0 = tile_row(Table, b, t, 0)
            T.tma_copy(KV[k0 : k0 + block_n, 0:half_v], Kl[s, :, :], barrier=kready[s])
            T.tma_copy(KV[k0 : k0 + block_n, half_v:dim_v], Kr[s, :, :], barrier=kready[s])
            T.tma_copy(KV[k0 : k0 + block_n, dim_v:dim], Kp[s, :, :], barrier=kready[s])

    def load_tiles(KV, KP, Table, Kl, Kr, Kp, kready, s, b, t):
        """The builder traces both producers; an FP8 cache has no TMA path into the 16-bit
        tiles, so this one emits nothing for it."""
        if not fp8:
            load_tile(KV, KP, Table, Kl, Kr, Kp, kready, s, b, t)

    @T.macro
    def tma_producer(
        Q, QP, KV, KP, Table, Ql, Qr, Qp, Kl, Kr, Kp, qbar, kready, kfree, b, hb, L, t0, NI
    ):
        load_q(Q, QP, Ql, Qr, Qp, qbar, b, hb)
        for lt in T.serial(NI):
            s = lt % 2
            T.mbarrier_wait_parity(kfree[s], ((lt // 2) % 2) ^ 1)
            load_tiles(KV, KP, Table, Kl, Kr, Kp, kready, s, b, t0 + lt)
            T.mbarrier_arrive(kready[s])
            if prefetching and lt + 1 < NI:
                prefetch(KV, KP, Table, b, t0 + lt + 1, L)

    @T.macro
    def read_rows(KV, Table, buf, b, t, k0, col, cols: int):
        """Tile t's key rows, columns ``col : col + cols``, into registers: one contiguous
        block when a page holds the whole tile, else row by row through the table."""
        if page_size % block_n == 0:
            T.copy(KV[k0 : k0 + block_n, col : col + cols], buf)
        else:
            for r, c in T.Parallel(block_n, cols):
                buf[r, c] = KV[tile_row(Table, b, t, r), col + c]

    @T.macro
    def fp8_producer(
        Q, KV, Table, Ql, Qr, Qp, Kl, Kr, Kp, qbar, kready, kfree, b, hb, L, t0, NI, tx
    ):
        """The producer warpgroup reads each FP8 tile and stores it in the query dtype.

        E4M3 values are exact in either 16-bit dtype, so the cache scale stays out of the
        stored tile and multiplies the scores and the output instead. Each 64 x 128 chunk
        is read whole into registers before any of it is converted, so its loads are in
        flight together. A value row past the request's end is then cleared.
        """
        chunk = T.alloc_fragment([block_n, 128], cache_dtype)
        tail = T.alloc_fragment([block_n, pe], cache_dtype)
        if tx < consumers + 32:
            load_q(Q, Q, Ql, Qr, Qp, qbar, b, hb)
        for lt in T.serial(NI):
            s = lt % 2
            t = t0 + lt
            T.mbarrier_wait_parity(kfree[s], ((lt // 2) % 2) ^ 1)
            k0 = tile_row(Table, b, t, 0)
            for q in T.unroll(half_v // 128):
                read_rows(KV, Table, chunk, b, t, k0, q * 128, 128)
                for r, c in T.Parallel(block_n, 128):
                    Kl[s, r, q * 128 + c] = T.cast(chunk[r, c], dtype)
            for q in T.unroll(half_v // 128):
                read_rows(KV, Table, chunk, b, t, k0, half_v + q * 128, 128)
                for r, c in T.Parallel(block_n, 128):
                    Kr[s, r, q * 128 + c] = T.cast(chunk[r, c], dtype)
            read_rows(KV, Table, tail, b, t, k0, dim_v, pe)
            for r, c in T.Parallel(block_n, pe):
                Kp[s, r, c] = T.cast(tail[r, c], dtype)
            if (t + 1) * block_n > L:
                # Other threads converted these rows: all of them land first.
                T.sync_threads(15, 128)
                for r, c in T.Parallel(block_n, half_v):
                    if t * block_n + r >= L:
                        Kl[s, r, c] = T.cast(0, dtype)
                        Kr[s, r, c] = T.cast(0, dtype)
            T.fence_proxy_async()
            T.mbarrier_arrive(kready[s])
            if tx == consumers and lt + 1 < NI:
                prefetch(KV, KV, Table, b, t + 1, L)

    @T.macro
    def attend(Q, QP, KV, KP, Table, Lens, KvScale, O, Lse):
        """One block: the query rows of head block hb of request b, over run z of its tiles.

        A dense cache passes every request's length as ``Lens`` and no scale.
        """
        with T.Kernel(T.ceildiv(rows, block_h), batch, num_splits, threads=384) as (hb, b, z):
            Ql = T.alloc_shared([block_h, half_v], dtype)
            Qr = T.alloc_shared([block_h, half_v], dtype)
            Qp = T.alloc_shared([block_h, pe], dtype)
            Kl = T.alloc_shared([2, block_n, half_v], dtype)
            Kr = T.alloc_shared([2, block_n, half_v], dtype)
            Kp = T.alloc_shared([2, block_n, pe], dtype)
            ms = T.alloc_shared([2, 2, block_h], accum)
            ss = T.alloc_shared([2, block_h], accum)
            lfin = T.alloc_shared([block_h], accum)
            T.annotate_layout(
                {
                    Ql: make_swizzled_layout(Ql),
                    Qr: make_swizzled_layout(Qr),
                    Qp: make_swizzled_layout(Qp),
                    Kl: make_swizzled_layout(Kl),
                    Kr: make_swizzled_layout(Kr),
                    Kp: make_swizzled_layout(Kp),
                }
            )
            qbar = T.alloc_barrier([32])
            # A TMA tile is announced by the producer warp, a converted one by the warpgroup.
            kready = T.alloc_barrier([128 if fp8 else 32] * 2)
            kfree = T.alloc_barrier([consumers] * 2)
            L = Lens if dense else Lens[b]
            # This block's run of tiles: t0 onward, NI of them (none for a run past the end).
            per_split = T.ceildiv(T.ceildiv(L, block_n), num_splits)
            t0 = z * per_split
            NI = T.max(0, T.min(per_split, T.ceildiv(L, block_n) - t0))
            tx = T.get_thread_binding()
            # The cache scale multiplies every score and every output value.
            kv_scale = KvScale[0] if fp8 else 1.0
            scale = score_scale * LOG2E * kv_scale
            # The register file holds 384 x 168: the consumers take 2 x 128 x 240 and leave
            # the TMA warp 24, or 2 x 128 x 208 and leave the converting warpgroup the 80 its
            # 64 x 128 register chunk needs.
            if tx >= consumers:
                T.set_max_nreg(80 if fp8 else 24, 0)
            if fp8:
                with T.ws(2):
                    fp8_producer(
                        Q,
                        KV,
                        Table,
                        Ql,
                        Qr,
                        Qp,
                        Kl,
                        Kr,
                        Kp,
                        qbar,
                        kready,
                        kfree,
                        b,
                        hb,
                        L,
                        t0,
                        NI,
                        tx,
                    )
            if not fp8 and tx >= consumers and tx < consumers + 32:
                tma_producer(
                    Q,
                    QP,
                    KV,
                    KP,
                    Table,
                    Ql,
                    Qr,
                    Qp,
                    Kl,
                    Kr,
                    Kp,
                    qbar,
                    kready,
                    kfree,
                    b,
                    hb,
                    L,
                    t0,
                    NI,
                )
            with T.ws(0):
                wg0(
                    Ql,
                    Qr,
                    Qp,
                    Kl,
                    Kr,
                    Kp,
                    ms,
                    ss,
                    lfin,
                    qbar,
                    kready,
                    kfree,
                    O,
                    Lse,
                    b,
                    z,
                    hb,
                    L,
                    t0,
                    NI,
                    scale,
                    kv_scale,
                )
            with T.ws(1):
                wg1(
                    Ql,
                    Qr,
                    Qp,
                    Kl,
                    Kr,
                    Kp,
                    ms,
                    ss,
                    lfin,
                    qbar,
                    kready,
                    kfree,
                    O,
                    b,
                    z,
                    hb,
                    L,
                    t0,
                    NI,
                    scale,
                    kv_scale,
                )

    if dense:
        seqlen = T.dynamic("seqlen")

        @T.prim_func
        def main(
            Q: T.Tensor((batch, rows, dim_v), dtype),  # type: ignore
            Qpe: T.Tensor((batch, rows, pe), dtype),  # type: ignore
            K: T.Tensor((batch, seqlen, dim_v), dtype),  # type: ignore
            Kpe: T.Tensor((batch, seqlen, pe), dtype),  # type: ignore
            O: T.Tensor(out_shape, out_dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum),  # type: ignore
        ):
            attend(Q, Qpe, K, Kpe, None, seqlen, None, O, Lse)

        return main

    pool_rows = T.dynamic("pool_rows")
    width = T.dynamic("width")

    @T.prim_func
    def main(
        Q: T.Tensor((batch, rows, dim), dtype),  # type: ignore
        KV: T.Tensor((pool_rows, dim), cache_dtype),  # type: ignore
        Table: T.Tensor((batch, width), "int32"),  # type: ignore
        Lens: T.Tensor((batch,), "int32"),  # type: ignore
        KvScale: T.Tensor((1,), accum),  # type: ignore
        O: T.Tensor(out_shape, out_dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum),  # type: ignore
    ):
        attend(Q, Q, KV, KV, Table, Lens, KvScale, O, Lse)

    return main


_SEESAW_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}
_SEESAW_FLAGS = ["-O3", "-DNDEBUG", *csrc_include("bulk_prefetch.h")]


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[5, 6], pass_configs=_SEESAW_PASS_CONFIGS, compile_flags=_SEESAW_FLAGS)
def _mla_seesaw_paged_kernel(
    heads, s_q, dim, dim_v, page_size, is_causal, sm_scale, dtype, cache_dtype, num_splits
):
    """The program over a paged cache: ``Q, KV, Table, Lens, KvScale`` in, ``O, Lse`` out."""
    return _mla_seesaw_program(
        heads,
        s_q,
        dim,
        dim_v,
        page_size,
        is_causal,
        sm_scale,
        dtype,
        cache_dtype,
        num_splits,
        "float",
    )


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[4, 5], pass_configs=_SEESAW_PASS_CONFIGS, compile_flags=_SEESAW_FLAGS)
def _mla_seesaw_dense_kernel(heads, dim, dim_v, dtype, num_splits):
    """The program over a dense cache, one query token a request, no mask:
    ``Q, Qpe, K, Kpe`` in, ``O, Lse`` out. A split request's runs leave their partial
    outputs in ``dtype``, half the bytes the merge reads."""
    return _mla_seesaw_program(
        heads, 1, dim, dim_v, None, False, None, dtype, dtype, num_splits, dtype
    )


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[2, 3], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _mla_seesaw_combine_kernel(rows, dim_v, num_splits, dtype, partial_dtype):
    """Merge each row's runs by their log-sum-exp: o = sum_z e^(lse_z - lse) o_z."""
    batch = T.dynamic("batch")
    accum = "float"

    @T.prim_func
    def main(
        Opart: T.Tensor((batch, num_splits, rows, dim_v), partial_dtype),  # type: ignore
        Lpart: T.Tensor((batch, num_splits, rows), accum),  # type: ignore
        O: T.Tensor((batch, rows, dim_v), dtype),  # type: ignore
        Lse: T.Tensor((batch, rows), accum),  # type: ignore
    ):
        with T.Kernel(rows, batch, threads=128) as (r, b):
            acc = T.alloc_fragment([dim_v], accum)
            top = T.alloc_var(accum)
            total = T.alloc_var(accum)
            lse = T.alloc_var(accum)
            top = -T.infinity(accum)
            for z in T.serial(num_splits):
                top = T.max(top, Lpart[b, z, r])
            # Every row sees at least one key, so some run's log-sum-exp is finite.
            total = 0.0
            for z in T.serial(num_splits):
                total += T.exp(Lpart[b, z, r] - top)
            lse = T.log(total) + top
            T.clear(acc)
            for z in T.serial(num_splits):
                for c in T.Parallel(dim_v):
                    acc[c] += T.exp(Lpart[b, z, r] - lse) * Opart[b, z, r, c]
            for c in T.Parallel(dim_v):
                O[b, r, c] = T.cast(acc[c], dtype)
            Lse[b, r] = lse

    return main
