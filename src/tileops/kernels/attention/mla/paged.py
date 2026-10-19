"""Multi-Head Latent Attention (MLA) decode over a paged latent cache, absorbed form.

A request's keys are its cache rows, whole, and its values their first ``dim_v``
columns. A block owns 64 query rows of one request (row ``s * heads + h``).

``MLAPagedSeesawKernel`` (SM90) staggers two consumer warpgroups over pairs of 64-key
tiles (a, b). Warpgroup 0 scores a, warpgroup 1 scores b, and the running max passes one
way, m -> m_a -> m_b -> the next pair, so each warpgroup's softmax runs while the other's
matrix products do. Warpgroup 0 holds O's left value half and warpgroup 1 the right; each
multiplies both probability tiles into its half, its own from registers and the other's
from shared memory, where it sits in the rope slot of its tile once that tile is scored.
A producer warp loads the tiles by TMA, or the producer warpgroup converts an FP8 cache
into the query dtype, and asks L2 for the tile after the one it loads.

``MLAPagedFwdKernel`` serves every other call: one block of query rows walks the tiles
in a plain loop, gathering each key row through the block table.
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
    MLAPagedCall,
    MLAPagedFwdInterface,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin, get_sm_count

__all__ = ["MLAPagedFwdKernel", "MLAPagedSeesawKernel"]


def _pad16(width: int) -> int:
    return -(-width // 16) * 16


def _score_scale(dim: int, sm_scale: Optional[float]) -> float:
    return dim**-0.5 if sm_scale is None else sm_scale


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    out_idx=[5, 6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
    compile_flags=["-O3", "-DNDEBUG", *csrc_include("bulk_prefetch.h")],
)
def _mla_paged_seesaw_kernel(
    heads, s_q, dim, dim_v, page_size, is_causal, sm_scale, dtype, cache_dtype, num_splits
):
    """Staggered seesaw: see the module docstring.

    With ``num_splits`` above one, block z of a request takes the z-th of that many equal
    runs of its tiles and writes a float32 partial output and log-sum-exp, which
    ``_mla_paged_combine_kernel`` merges.
    """
    rows = s_q * heads
    batch = T.dynamic("batch")
    pool_rows = T.dynamic("pool_rows")
    width = T.dynamic("width")
    half_v = dim_v // 2
    pe = dim - dim_v
    fp8 = cache_dtype == "float8_e4m3fn"
    score_scale = _score_scale(dim, sm_scale)
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
    aligned = block_n % page_size == 0 or page_size % block_n == 0
    pages_per_tile = max(1, block_n // page_size) if aligned else (block_n - 2) // page_size + 2
    if num_splits == 1:
        out_shape, out_dtype, lse_shape = (batch, rows, dim_v), dtype, (batch, rows)
    else:
        out_shape, out_dtype = (batch, num_splits, rows, dim_v), accum
        lse_shape = (batch, num_splits, rows)

    def tile_row(Table, b, t, r):
        """The cache row holding key ``t * block_n + r``; past the table it is any valid row."""
        key = t * block_n + r
        return Table[b, T.min(key // page_size, width - 1)] * page_size + key % page_size

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

        A cache row past ``L`` may hold anything, NaN included, and a zero probability
        times NaN is NaN, so its value columns are cleared before any product reads them.
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
        if not fp8 and (t + 1) * block_n > L:
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
    def store(acc_o, O, b, z, hb, w, scale_out, l):
        # A row that saw no key in this run (a run past its last visible key) has max -inf,
        # so its probabilities and l are NaN; every later tile of the run hides it too, and
        # it is written as zero with log-sum-exp -inf.
        for r, c in T.Parallel(block_h, half_v):
            if hb * block_h + r < rows:
                value = T.if_then_else(l[r] > 0, acc_o[r, c] * scale_out / l[r], 0.0)
                if num_splits == 1:
                    O[b, hb * block_h + r, w * half_v + c] = T.cast(value, dtype)
                else:
                    O[b, z, hb * block_h + r, w * half_v + c] = value

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
        store(acc_o, O, b, z, hb, 0, scale_out, l)
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
        store(acc_o, O, b, z, hb, 1, scale_out, lf)

    @T.macro
    def load_q(Q, Ql, Qr, Qp, qbar, b, hb):
        r0 = hb * block_h
        T.tma_copy(Q[b, r0 : r0 + block_h, 0:half_v], Ql, barrier=qbar[0])
        T.tma_copy(Q[b, r0 : r0 + block_h, half_v:dim_v], Qr, barrier=qbar[0])
        T.tma_copy(Q[b, r0 : r0 + block_h, dim_v:dim], Qp, barrier=qbar[0])
        T.mbarrier_arrive(qbar[0])

    @T.macro
    def prefetch(KV, Table, b, t):
        """Ask L2 for the rows of tile t, page by page."""
        if aligned:
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
    def load_page(KV, Table, Kl, Kr, Kp, kready, s, b, t, rs: int):
        k0 = tile_row(Table, b, t, rs)
        T.tma_copy(KV[k0 : k0 + block_n, 0:half_v], Kl[s, :, :], barrier=kready[s])
        T.tma_copy(KV[k0 : k0 + block_n, half_v:dim_v], Kr[s, :, :], barrier=kready[s])
        T.tma_copy(KV[k0 : k0 + block_n, dim_v:dim], Kp[s, :, :], barrier=kready[s])

    def load_pages(KV, Table, Kl, Kr, Kp, kready, s, b, t):
        """The tile's three column ranges from its page.

        The builder traces both producers; an FP8 cache has no TMA path into the 16-bit
        tiles, so this one emits nothing for it.
        """
        if not fp8:
            load_page(KV, Table, Kl, Kr, Kp, kready, s, b, t, 0)

    @T.macro
    def tma_producer(Q, KV, Table, Ql, Qr, Qp, Kl, Kr, Kp, qbar, kready, kfree, b, hb, t0, NI):
        load_q(Q, Ql, Qr, Qp, qbar, b, hb)
        for lt in T.serial(NI):
            s = lt % 2
            T.mbarrier_wait_parity(kfree[s], ((lt // 2) % 2) ^ 1)
            load_pages(KV, Table, Kl, Kr, Kp, kready, s, b, t0 + lt)
            T.mbarrier_arrive(kready[s])
            if lt + 1 < NI:
                prefetch(KV, Table, b, t0 + lt + 1)

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
            load_q(Q, Ql, Qr, Qp, qbar, b, hb)
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
                prefetch(KV, Table, b, t + 1)

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
            L = Lens[b]
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
                    Q, KV, Table, Ql, Qr, Qp, Kl, Kr, Kp, qbar, kready, kfree, b, hb, t0, NI
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

    return main


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[2, 3], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _mla_paged_combine_kernel(rows, dim_v, num_splits, dtype):
    """Merge each row's runs by their log-sum-exp: o = sum_z e^(lse_z - lse) o_z."""
    batch = T.dynamic("batch")
    accum = "float"

    @T.prim_func
    def main(
        Opart: T.Tensor((batch, num_splits, rows, dim_v), accum),  # type: ignore
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


@functools.lru_cache(maxsize=32)
def _mla_paged_kernel(heads, s_q, dim, dim_v, page_size, is_causal, sm_scale, dtype, cache_dtype):
    """One block of query rows walks the tiles, gathering each key row through the table."""
    rows = s_q * heads
    pe = dim - dim_v
    # The products contract in 16-wide steps: a narrower width is padded with zero columns.
    # The value tile is read by both products, whose layouts for it agree from 32 columns.
    dv_pad, pe_pad = max(32, _pad16(dim_v)), _pad16(pe)
    padded = (dv_pad, pe_pad) != (dim_v, pe)
    fp8 = cache_dtype == "float8_e4m3fn"
    score_scale = _score_scale(dim, sm_scale)
    accum = "float"

    @tilelang.jit(out_idx=[5, 6], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _func(block_h, block_n, threads):
        batch = T.dynamic("batch")
        pool_rows = T.dynamic("pool_rows")
        width = T.dynamic("width")

        @T.prim_func
        def main(
            Q: T.Tensor((batch, rows, dim), dtype),  # type: ignore
            KV: T.Tensor((pool_rows, dim), cache_dtype),  # type: ignore
            Table: T.Tensor((batch, width), "int32"),  # type: ignore
            Lens: T.Tensor((batch,), "int32"),  # type: ignore
            KvScale: T.Tensor((1,), accum),  # type: ignore
            O: T.Tensor((batch, rows, dim_v), dtype),  # type: ignore
            Lse: T.Tensor((batch, rows), accum),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_h), batch, threads=threads) as (hb, b):
                Q_s = T.alloc_shared([block_h, dv_pad], dtype)
                Qp_s = T.alloc_shared([block_h, pe_pad], dtype)
                K_s = T.alloc_shared([block_n, dv_pad], dtype)
                Kp_s = T.alloc_shared([block_n, pe_pad], dtype)
                S_s = T.alloc_shared([block_h, block_n], dtype)
                acc_s = T.alloc_fragment([block_h, block_n], accum)
                acc_o = T.alloc_fragment([block_h, dv_pad], accum)
                m = T.alloc_fragment([block_h], accum)
                m_prev = T.alloc_fragment([block_h], accum)
                alpha = T.alloc_fragment([block_h], accum)
                ssum = T.alloc_fragment([block_h], accum)
                l = T.alloc_fragment([block_h], accum)
                r0 = hb * block_h
                if padded:
                    for i, c in T.Parallel(block_h, dv_pad):
                        Q_s[i, c] = T.if_then_else(
                            c < dim_v, Q[b, T.min(r0 + i, rows - 1), T.min(c, dim_v - 1)], 0
                        )
                    for i, c in T.Parallel(block_h, pe_pad):
                        Qp_s[i, c] = T.if_then_else(
                            c < pe, Q[b, T.min(r0 + i, rows - 1), dim_v + T.min(c, pe - 1)], 0
                        )
                else:
                    T.copy(Q[b, r0 : r0 + block_h, 0:dim_v], Q_s)
                    T.copy(Q[b, r0 : r0 + block_h, dim_v:dim], Qp_s)
                T.fill(acc_o, 0)
                T.fill(l, 0)
                T.fill(m, -T.infinity(accum))
                L = Lens[b]
                kv_scale = KvScale[0] if fp8 else 1.0
                scale = score_scale * LOG2E * kv_scale
                # A plain loop: the row gather's addresses depend on the table, so the
                # pipeliner would turn it into cp.async of single elements, which PTX refuses.
                for k in T.serial(T.ceildiv(L, block_n)):
                    # A value row past the request's end may hold NaN, which a zero
                    # probability would carry into the output: it is stored as zero.
                    for r, c in T.Parallel(block_n, dv_pad):
                        key = k * block_n + r
                        row = (
                            Table[b, T.min(key // page_size, width - 1)] * page_size
                            + key % page_size
                        )
                        # A padded column repeats a real one: the query's padded columns
                        # are zero, and the output drops the value's.
                        K_s[r, c] = T.if_then_else(
                            key < L, T.cast(KV[row, T.min(c, dim_v - 1)], dtype), T.cast(0, dtype)
                        )
                    for r, c in T.Parallel(block_n, pe_pad):
                        key = k * block_n + r
                        row = (
                            Table[b, T.min(key // page_size, width - 1)] * page_size
                            + key % page_size
                        )
                        Kp_s[r, c] = T.cast(KV[row, dim_v + T.min(c, pe - 1)], dtype)
                    T.clear(acc_s)
                    T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(Qp_s, Kp_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    # Scaled before the max, so the max is the softmax's for any sign of scale.
                    for i, j in T.Parallel(block_h, block_n):
                        last = L - s_q + (r0 + i) // heads if is_causal else L - 1
                        acc_s[i, j] = T.if_then_else(
                            k * block_n + j > last, -T.infinity(accum), acc_s[i, j] * scale
                        )
                    T.copy(m, m_prev)
                    T.reduce_max(acc_s, m, dim=1, clear=False)
                    for i in T.Parallel(block_h):
                        alpha[i] = T.exp2(m_prev[i] - m[i])
                    for i, j in T.Parallel(block_h, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] - m[i])
                    T.reduce_sum(acc_s, ssum, dim=1)
                    T.copy(acc_s, S_s)
                    for i in T.Parallel(block_h):
                        l[i] = l[i] * alpha[i] + ssum[i]
                    for i, j in T.Parallel(block_h, dv_pad):
                        acc_o[i, j] *= alpha[i]
                    T.gemm(S_s, K_s, acc_o, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(block_h, dim_v):
                    if r0 + i < rows:
                        O[b, r0 + i, j] = T.cast(acc_o[i, j] * kv_scale / l[i], dtype)
                for i in T.Parallel(block_h):
                    if r0 + i < rows:
                        Lse[b, r0 + i] = T.log(l[i]) + m[i] / LOG2E

        return main

    return _func


def _kernel_args(call: MLAPagedCall) -> dict:
    return dict(
        heads=call.heads,
        s_q=call.s_q,
        dim=call.dim,
        dim_v=call.dim_v,
        page_size=call.page_size,
        is_causal=call.is_causal,
        sm_scale=call.sm_scale,
        dtype=call.dtype,
        cache_dtype=call.cache_dtype,
    )


class _MLAPagedBase(Kernel, MLAPagedFwdInterface):
    """The call geometry, the cache's unit scale, and the call wrapper both schedules share."""

    @classmethod
    def entry_for(cls, call: MLAPagedCall) -> Entry:
        args = _kernel_args(call)
        index = call.device.index if call.device is not None else None
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def _setup(self, heads, s_q, dim, dim_v, page_size, is_causal, sm_scale, dtype, cache_dtype):
        self.heads, self.s_q, self.dim, self.dim_v = heads, s_q, dim, dim_v
        self.page_size, self.is_causal, self.sm_scale = page_size, is_causal, sm_scale
        self.dtype, self.cache_dtype = dtype, cache_dtype
        self.quantized = cache_dtype == torch.float8_e4m3fn
        # A 16-bit cache passes no scale; the kernel still takes one and multiplies by it.
        device = (
            torch.device("cuda", self.device_index)
            if self.device_index is not None
            else torch.device("cuda")
        )
        self._unit = torch.ones(1, dtype=torch.float32, device=device)
        return (
            heads, s_q, dim, dim_v, page_size, is_causal, sm_scale,
            self.dtype_str, self.dtype_to_str(cache_dtype),
        )  # fmt: skip

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each request's query rows to its cache; see the interface for the contract."""
        batch = q.shape[0]
        if batch == 0:
            return (
                q.new_empty((0, self.s_q, self.heads, self.dim_v)),
                q.new_empty((0, self.s_q, self.heads), dtype=torch.float32),
            )
        o, lse = self._program()(
            q.view(batch, self.s_q * self.heads, self.dim),
            kv_cache.view(-1, self.dim),
            block_table,
            cache_seqlens,
            self._unit if kv_scale is None else kv_scale,
        )
        return o.view(batch, self.s_q, self.heads, self.dim_v), lse.view(
            batch, self.s_q, self.heads
        )


class MLAPagedSeesawKernel(_MLAPagedBase):
    """SM90 staggered seesaw over 64-key tile pairs; see the module docstring."""

    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: MLAPagedCall) -> Optional[str]:
        """Why *call* is outside the shapes this schedule serves."""
        if call.dtype not in ATTENTION_DTYPES:
            return f"serves float16 and bfloat16 queries, got {call.dtype}"
        if (call.dim_v, call.dim - call.dim_v) != (512, 64):
            # The query, two key tiles and the exchange buffers fill the shared memory of an
            # SM90 block at a 512-wide value and a 64-wide rope part.
            return f"its buffers are sized for a 512 + 64 latent row, got {call.dim_v} + {call.dim - call.dim_v}"
        if not call.quantized and call.page_size % 64 != 0:
            # TMA fills a 64-key tile from one page: a swizzled tile takes no partial rows.
            return f"loads a 64-key tile from one page, got page size {call.page_size}"
        return super().refusal(call)

    def __init__(
        self,
        heads: int,
        s_q: int,
        dim: int,
        dim_v: int,
        page_size: int,
        is_causal: bool,
        sm_scale: Optional[float],
        dtype: torch.dtype,
        cache_dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        """Build the program for one model's decode geometry.

        Args:
            heads: Query heads.
            s_q: Query tokens of a request.
            dim: Width of a cache row, the key.
            dim_v: Width of the value, the row's first columns.
            page_size: Rows of a page.
            is_causal: Whether a query row sees only keys at or before its position.
            sm_scale: Score scale, or None for ``dim ** -0.5``.
            dtype: Query and output dtype.
            cache_dtype: Cache dtype; float8_e4m3fn dequantizes by the call's scale.
            config: Unused; the schedule's geometry is fixed.
            device_index: The device the program is built for, or None for the current one.
        """
        super().__init__(device_index=device_index)
        self._args = self._setup(
            heads, s_q, dim, dim_v, page_size, is_causal, sm_scale, dtype, cache_dtype
        )
        self.sm_count = get_sm_count(device_index)
        self.kernel = _mla_paged_seesaw_kernel(*self._args, 1)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {}

    def num_splits(self, batch: int, width: int) -> int:
        """Runs a request's tiles are cut into: enough blocks to fill the SMs once.

        Read from the batch and the block table's width, never from the lengths, which
        are on the device. A power of two keeps the builds few; a run keeps at least two
        tiles of the widest request the table can hold.
        """
        blocks = batch * -(-self.s_q * self.heads // 64)
        tiles = -(-width * self.page_size // 64)
        splits = 1
        while blocks * splits * 2 <= self.sm_count and tiles >= splits * 4 and splits < 32:
            splits *= 2
        return splits

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each request's query rows to its cache; see the interface for the contract."""
        batch = q.shape[0]
        splits = self.num_splits(batch, block_table.shape[1])
        if batch == 0 or splits == 1:
            return super().forward(q, kv_cache, block_table, cache_seqlens, kv_scale)
        rows = self.s_q * self.heads
        o_part, lse_part = _mla_paged_seesaw_kernel(*self._args, splits)(
            q.view(batch, rows, self.dim),
            kv_cache.view(-1, self.dim),
            block_table,
            cache_seqlens,
            self._unit if kv_scale is None else kv_scale,
        )
        o, lse = _mla_paged_combine_kernel(rows, self.dim_v, splits, self.dtype_str)(
            o_part, lse_part
        )
        return o.view(batch, self.s_q, self.heads, self.dim_v), lse.view(
            batch, self.s_q, self.heads
        )

    def _program(self):
        return self.kernel


class MLAPagedFwdKernel(_MLAPagedBase):
    """One block of query rows a request, the key tiles gathered row by row."""

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True
    # (rows a block, keys a tile, threads), widest first: the first one the device's shared
    # memory holds is the default.
    _CANDIDATES = ((64, 64, 256), (64, 32, 256), (32, 32, 128), (16, 16, 128))

    @staticmethod
    def _shared_bytes(dim: int, dim_v: int, itemsize: int, config) -> int:
        block_h, block_n, _ = config
        padded = max(32, _pad16(dim_v)) + _pad16(dim - dim_v)
        return (block_h * padded + block_n * padded + block_h * block_n) * itemsize

    @classmethod
    def refusal(cls, call: MLAPagedCall) -> Optional[str]:
        """Why *call* is outside the shapes this schedule serves."""
        if call.dtype not in ATTENTION_DTYPES:
            return f"serves float16 and bfloat16 queries, got {call.dtype}"
        need = cls._shared_bytes(call.dim, call.dim_v, call.dtype.itemsize, cls._CANDIDATES[-1])
        if call.smem_budget and need > call.smem_budget:
            return f"needs {need} bytes of shared memory at its narrowest tile, over {call.smem_budget}"
        return super().refusal(call)

    def __init__(
        self,
        heads: int,
        s_q: int,
        dim: int,
        dim_v: int,
        page_size: int,
        is_causal: bool,
        sm_scale: Optional[float],
        dtype: torch.dtype,
        cache_dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        """Build the program for one model's decode geometry; arguments as the seesaw's.

        Args:
            heads: Query heads.
            s_q: Query tokens of a request.
            dim: Width of a cache row, the key.
            dim_v: Width of the value, the row's first columns.
            page_size: Rows of a page.
            is_causal: Whether a query row sees only keys at or before its position.
            sm_scale: Score scale, or None for ``dim ** -0.5``.
            dtype: Query and output dtype.
            cache_dtype: Cache dtype; float8_e4m3fn dequantizes by the call's scale.
            config: Tile sizes, or None for the widest the device holds.
            device_index: The device the program is built for, or None for the current one.
        """
        super().__init__(device_index=device_index)
        self._args = self._setup(
            heads, s_q, dim, dim_v, page_size, is_causal, sm_scale, dtype, cache_dtype
        )
        self.kernel = _mla_paged_kernel(*self._args)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        budget = get_shared_memory_optin(self.device_index)
        for config in self._CANDIDATES:
            if self._shared_bytes(self.dim, self.dim_v, self.dtype.itemsize, config) <= budget:
                break
        block_h, block_n, threads = config
        return {"block_h": block_h, "block_n": block_n, "threads": threads}

    def _program(self):
        return self.kernel(**self.config)
