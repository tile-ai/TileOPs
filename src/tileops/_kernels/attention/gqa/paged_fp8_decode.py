"""SM90 grouped-query attention over a paged FP8 cache, for packings of few query rows a request.

One CTA owns one KV head, a tile of one request's packed rows and one chunk of its key range, so
a key tile is read once per request, KV head and chunk whatever the row count of a request whose
rows fit the tile; the tile is sized from the rows a request averages, and a longer request's
further row tiles read its keys again. Row ``r`` of a request is query position ``r // group`` of
head ``kv_head * group + r % group``. A
warp copies each FP8 key and value tile into shared memory with TMA, one copy a page.

- A tile of 64 rows fills the M extent of a WGMMA, which holds the rows: ``S = Q K^T`` and
  ``O += P V`` with the query and the probabilities in registers. A producer warpgroup's other
  three warps widen each key tile into 16-bit shared memory, and each row's softmax stays in one
  warp.
- A narrower tile is the N extent of products that hold the keys on M: ``S^T = K Q^T`` with the
  keys widened into registers, and ``O^T = V^T P^T`` with the probabilities staged in shared
  memory. Their cost follows the rows the tile holds; one copying warp sits beside the consumer.

In both, the consumer warpgroup widens each value tile into shared memory while its key product
runs. Chunks of one row tile combine by their log-sum-exp in a second kernel. The launch bound
asks for the CTAs a multiprocessor's shared memory and registers hold, and the chunk count is
chosen against those waves.
"""

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
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import LOG2E, WARPGROUP_THREADS, WGMMA_ROWS
from tileops._kernels.grouped_tiling import GroupTiling
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = ["GQAPagedFP8DecodeKernel"]

# Shared memory one SM90 multiprocessor holds for its resident CTAs.
_SM_SHARED_BYTES = 228 * 1024
# Registers a consumer thread keeps beyond its tiles: indices, addresses and loop state.
_ADDRESS_REGS = 32


def _value_stages(block_m: int) -> int:
    """Widened value stages a program of *block_m*-row tiles holds: two with the rows on M, one
    with the keys on M."""
    return 2 if block_m == WGMMA_ROWS else 1


@target_cache(maxsize=32)
def _gqa_paged_fp8_decode_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    max_pages_per_req: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    q_dtype: str,
    per_request_kv_scale: bool,
):
    """Build the split decode program for one fixed set of call facts."""
    fp8_dtype = "float8_e4m3fn"
    accum = "float"
    group = heads // heads_kv
    q_fp8 = q_dtype != dtype
    # The type the tensor cores multiply in. An e4m3 value converts exactly into either
    # 16-bit type, and into float16 in one instruction, so an FP8 query multiplies in float16
    # whatever the output type; a 16-bit query fixes it.
    wide = "float16" if q_fp8 else dtype
    wide_c = "half_t" if wide == "float16" else "bfloat16_t"
    # The widening helper of the head dimension's shared layouts.
    widen_form = "sw128" if dim == 128 else "d64"
    use_softcap = softcap > 0.0
    # exp2 of a score times this is exp of the scaled score; a softcap applies the scale first.
    softmax_scale = LOG2E if use_softcap else sm_scale * LOG2E
    # Below this a chunk saw no key. A threshold, not an equality with -inf: fast math folds
    # comparisons with infinity away.
    no_key = -1.0e30
    consumers = WARPGROUP_THREADS
    policy = T.GemmWarpPolicy.FullRow

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            # Row i of the tile is query i // group of head i % group, which is injective
            # in i; the checker cannot prove it and reports the output store as a race.
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
            tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        },
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            *csrc_include("fp8_paged_helper.h"),
        ],
    )
    def _func(
        block_m: int,
        block_n: int,
        num_split: int,
        stages: int,
        cstages: int,
        producer_regs: int,
        resident: int,
    ):
        # A tile of 64 rows fills the M extent of a WGMMA, which then holds the rows; a narrower
        # tile is the N extent of products that hold the keys on M, so their cost follows the
        # rows the tile holds.
        rows_on_m = block_m == WGMMA_ROWS
        # Value stages: with rows on M one tile's value product runs while the next one's values
        # widen; with keys on M the consumer waits for it, and one stage lets more CTAs share a
        # multiprocessor.
        vstages = _value_stages(block_m)
        # With rows on M the producer warpgroup's first warp copies and the others widen keys;
        # keys on M widen in the consumer's registers, one copying warp beside it.
        converters = WARPGROUP_THREADS - 32
        threads = consumers + (WARPGROUP_THREADS if rows_on_m else 32)
        # Registers the consumer takes from the producers' share of the launch bound.
        launch_regs = 65536 // (resident * threads) // 8 * 8
        spare = launch_regs * threads - producer_regs * (threads - consumers)
        consumer_regs = min(240, spare // consumers // 8 * 8)
        total_q = T.dynamic("total_q")
        pool_rows = T.dynamic("pool_rows")
        tiles = T.dynamic("tiles")
        shape_q = (total_q, heads, dim)
        shape_kv = (pool_rows, heads_kv, dim)
        kv_scale_shape = (batch, heads_kv) if per_request_kv_scale else (1,)
        shape_lse = (tiles, heads_kv, num_split, block_m)
        shape_partial = (tiles, heads_kv, num_split, block_m, dim)
        tiling = GroupTiling(batch, block_m, rows_per_offset=group)
        # A key tile is part of one page or a run of whole pages: one copy per page.
        box = min(block_n, page_size)
        boxes = block_n // box
        split = num_split > 1

        def hidden(key, q_pos, kv_len):
            """Whether *key* is outside what the row at *q_pos* sees."""
            out = key >= kv_len
            if is_causal:
                out = out | (key > q_pos)
            if window_size_left >= 0:
                out = out | (key < q_pos - window_size_left)
            if window_size_right >= 0:
                out = out | (key > q_pos + window_size_right)
            return out

        def cut_tile(key0, kv_len, first_q, last_q):
            """Whether the key tile at *key0* reaches past some row's visible range: its upper
            bound is set by the tile's first row and its lower by its last."""
            cut = key0 + block_n > kv_len
            if is_causal:
                cut = cut | (key0 + block_n > first_q + 1)
            elif window_size_right >= 0:
                cut = cut | (key0 + block_n > first_q + window_size_right + 1)
            if window_size_left >= 0:
                cut = cut | (key0 < last_q - window_size_left)
            return cut

        @T.macro
        def advance_max(m, m_prev, rescale, item_scale):
            """Keep each row's maximum until it grows by 2^8, so its probabilities stay below
            256, exact in either 16-bit type, and most tiles skip the rescale; rescale holds the
            factor that brings earlier tiles to the kept maximum."""
            for i in T.Parallel(block_m):
                m[i] = T.max(m[i], T.cast(-1e38, accum))
                m[i] = T.if_then_else((m[i] - m_prev[i]) * item_scale > 8.0, m[i], m_prev[i])
                rescale[i] = T.exp2((m_prev[i] - m[i]) * item_scale)

        @T.macro
        def load_tile(pool, dst, s, bars, page_table, request, key0, last_page, kv_head):
            """TMA the key tile at *key0* into stage *s* of *dst*, one copy a page; a page
            past the cache end repeats the last one, whose rows the mask and the value
            zeroing discard."""
            for b in T.unroll(boxes):
                key = key0 + b * box
                row = page_table[request, T.min(key // page_size, last_page)] * page_size + (
                    key % page_size
                )
                T.tma_copy(
                    pool[row : row + box, kv_head, :],
                    dst[s, b * box : (b + 1) * box, :],
                    barrier=bars[s],
                )

        @T.macro
        def widen(src, dst, s, c, t, valid_rows, threads_, form):
            """Widen stored stage *s* of *src* into stage *c* of *dst*, zeroing rows at or past
            *valid_rows*."""
            T.call_extern(
                "handle",
                f"tileops::fp8_tile_to_wide_{widen_form}{form}<{block_n}, {threads_}, {wide_c}>",
                T.access_ptr(src[s, 0, 0], "r"),
                T.access_ptr(dst[c, 0, 0], "w"),
                t,
                valid_rows,
            )

        @T.macro
        def key_product(q_f, k_c, acc_s, kc_ready, t):
            """Issue the key product of tile *t* once its keys are widened."""
            T.mbarrier_wait_parity(kc_ready[t % cstages], (t // cstages) % 2)
            T.wgmma_gemm(
                q_f, k_c[t % cstages, :, :], acc_s, transpose_B=True, policy=policy,
                clear_accum=True,
            )  # fmt: skip

        @T.macro
        def widen_values(v_r, v_c, valid, vr_ready, vr_free, t, tx):
            """Widen tile *t*'s values into stage t % vstages, which no value product in flight
            reads, and publish them to the consumer's async proxy."""
            s = t % stages
            T.mbarrier_wait_parity(vr_ready[s], (t // stages) % 2)
            form = "_prefetched" if dim == 128 else ""
            widen(v_r, v_c, s, t % vstages, tx, valid[s], consumers, form)
            T.mbarrier_arrive(vr_free[s])
            T.fence_proxy_async()
            T.sync_threads(1, consumers)

        @T.macro
        def softmax(
            acc_s, red, m, m_prev, rescale, lp, key0, kv_len, first_q, last_q, row0, align,
            qk_descale, item_scale,
        ):  # fmt: skip
            """Fold the rows-by-keys score tile at *key0* into the running maxima and sums,
            leaving its probabilities in acc_s and the factor that rescales earlier tiles in
            rescale."""
            if use_softcap:
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.cast(softcap, accum) * T.tanh(
                        acc_s[i, j] * qk_descale * T.cast(sm_scale / softcap, accum)
                    )
            if cut_tile(key0, kv_len, first_q, last_q):
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(
                        hidden(key0 + j, (row0 + i) // group + align, kv_len),
                        -T.infinity(accum),
                        acc_s[i, j],
                    )
            T.copy(m, m_prev)
            # Two levels, so each thread's maximum is a tree, not one chain of block_n.
            T.reduce_max(T.reshape(acc_s, [block_m, block_n // 8, 8]), red, dim=1, clear=True)
            T.reduce_max(red, m, dim=1, clear=False)
            advance_max(m, m_prev, rescale, item_scale)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * item_scale - m[i] * item_scale)
            # Each thread sums the columns it holds; the rows reduce once, after the last tile.
            for i, j in T.Parallel(block_m, 16):
                lp[i, j] *= rescale[i]
            for u in T.unroll(block_n // 16):
                for i, j in T.Parallel(block_m, 16):
                    lp[i, j] += acc_s[i, j + u * 16]

        @T.macro
        def rescale_rows(acc_o, rescale):
            """Scale the output rows whose maximum moved; a thread skips it when none of its
            rows did."""
            moved = T.alloc_var("int32", init=0)
            for i in T.Parallel(block_m):
                if rescale[i] != 1.0:
                    moved = 1
            if moved == 1:
                for i, d in T.Parallel(block_m, dim):
                    acc_o[i, d] *= rescale[i]

        @T.macro
        def gather_query(Q, dst, q_start, row0, rows, kv_head):
            """The tile's query rows into *dst*, zero past the request's rows."""
            for i, d in T.Parallel(block_m, dim):
                r = T.min(row0 + i, rows - 1)
                x = Q[q_start + r // group, kv_head * group + r % group, d]
                dst[i, d] = T.if_then_else(
                    row0 + i < rows, T.cast(x, wide) if q_fp8 else x, T.cast(0, wide)
                )

        @T.macro
        def store(acc, keys_on_m: bool, m, logsum, row_sum, glse, partial, Output, item_scale,
                  v_descale, q_start, row0, rows, bx, by, bz):  # fmt: skip
            """Normalize the tile's output, held rows-by-dims or (keys on M) dims-by-rows, and
            store it: the chunk's partial and log-sum-exp when the range is split, else the
            output rows."""
            for i in T.Parallel(block_m):
                row_sum[i] = T.if_then_else(logsum[i] == 0, 0, v_descale / logsum[i])
            if split:
                for i, d in T.Parallel(block_m, dim):
                    held = acc[d, i] if keys_on_m else acc[i, d]
                    partial[bx, by, bz, i, d] = held * row_sum[i]
                for i in T.Parallel(block_m):
                    # The row's log2-sum-exp, -inf where it saw no key.
                    glse[bx, by, bz, i] = T.if_then_else(
                        logsum[i] == 0,
                        -T.infinity(accum),
                        T.log2(T.if_then_else(logsum[i] == 0, 1, logsum[i])) + m[i] * item_scale,
                    )
            elif keys_on_m:
                for d, i in T.Parallel(dim, block_m):
                    if row0 + i < rows:
                        r = row0 + i
                        Output[q_start + r // group, by * group + r % group, d] = T.cast(
                            acc[d, i] * row_sum[i], dtype
                        )
            else:
                for i, d in T.Parallel(block_m, dim):
                    if row0 + i < rows:
                        r = row0 + i
                        Output[q_start + r // group, by * group + r % group, d] = T.cast(
                            acc[i, d] * row_sum[i], dtype
                        )

        @T.macro
        def scan(
            Q: T.Tensor(shape_q, q_dtype),
            K: T.Tensor(shape_kv, fp8_dtype),
            V: T.Tensor(shape_kv, fp8_dtype),
            cache_seqlens: T.Tensor([batch], T.int32),
            page_table: T.Tensor([batch, max_pages_per_req], T.int32),
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),
            QS: T.Tensor([batch, heads_kv], accum),
            KS: T.Tensor(kv_scale_shape, accum),
            VS: T.Tensor(kv_scale_shape, accum),
            glse: T.Tensor(shape_lse, accum),
            partial: T.Tensor(shape_partial, accum),
            Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(tiles, heads_kv, num_split, threads=threads) as (bx, by, bz):
                T.annotate_min_blocks_per_sm(resident)
                tile_cum = T.alloc_shared([batch + 1], "int32")
                valid = T.alloc_shared([stages], "int32")
                k_r = T.alloc_shared([stages, block_n, dim], fp8_dtype)
                v_r = T.alloc_shared([stages, block_n, dim], fp8_dtype)
                v_c = T.alloc_shared([vstages, block_n, dim], wide)
                layouts = {
                    k_r: make_swizzled_layout(k_r),
                    v_r: make_swizzled_layout(v_r),
                    v_c: make_swizzled_layout(v_c),
                }
                if rows_on_m:
                    k_c = T.alloc_shared([cstages, block_n, dim], wide)
                    layouts[k_c] = make_swizzled_layout(k_c)
                else:
                    # The products' B operands: the query rows and the probabilities.
                    q_s = T.alloc_shared([block_m, dim], wide)
                    p_s = T.alloc_shared([block_m, block_n], wide)
                    layouts[q_s] = make_swizzled_layout(q_s)
                    layouts[p_s] = make_swizzled_layout(p_s)
                T.annotate_layout(layouts)
                kr_ready = T.alloc_barrier([32] * stages)
                kr_free = T.alloc_barrier([converters if rows_on_m else consumers] * stages)
                vr_ready = T.alloc_barrier([32] * stages)
                vr_free = T.alloc_barrier([consumers] * stages)
                if rows_on_m:
                    kc_ready = T.alloc_barrier([converters] * cstages)
                    kc_free = T.alloc_barrier([consumers] * cstages)
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                found = T.alloc_local([1], "int32")
                first_row = T.alloc_local([1], "int32")

                tiling.cumsum_offsets(cu_seqlens_q, tile_cum)
                tx = T.get_thread_binding()
                if bx < tile_cum[batch]:
                    tiling.decode(bx, tile_cum, lo, hi, found, first_row)
                    request = found[0]
                    row0 = first_row[0]
                    q_start = cu_seqlens_q[request]
                    q_len = cu_seqlens_q[request + 1] - q_start
                    kv_len = cache_seqlens[request]
                    rows = q_len * group
                    # Queries sit at the end of the cache: position i is key kv_len - q_len + i.
                    align = kv_len - q_len
                    first_q = row0 // group + align
                    last_q = T.min(q_len - 1, (row0 + block_m - 1) // group) + align
                    key_end = T.alloc_var("int32", init=kv_len)
                    if is_causal:
                        key_end = T.min(kv_len, last_q + 1)
                    elif window_size_right >= 0:
                        key_end = T.min(kv_len, last_q + 1 + window_size_right)
                    key_start = T.alloc_var("int32", init=0)
                    if window_size_left >= 0:
                        key_start = T.max(0, first_q - window_size_left)
                    tile0 = key_start // block_n
                    span = T.max(0, T.ceildiv(key_end, block_n) - tile0)
                    # Whole key tiles per chunk, so a chunk boundary never splits one.
                    per_split = T.ceildiv(span, num_split)
                    mine = T.max(0, T.min(per_split, span - bz * per_split))
                    first = tile0 + bz * per_split
                    last_page = T.max(0, (kv_len - 1) // page_size)
                    kv_at = (request, by) if per_request_kv_scale else (0,)
                    qk_descale = T.alloc_var(
                        accum, init=KS[kv_at] * QS[request, by] if q_fp8 else KS[kv_at]
                    )
                    v_descale = T.alloc_var(accum, init=VS[kv_at])
                    # A softcap reads the dequantized score; otherwise the descale joins the
                    # exponent.
                    item_scale = T.alloc_var(
                        accum, init=softmax_scale if use_softcap else softmax_scale * qk_descale
                    )

                    if rows_on_m and tx >= consumers:
                        T.set_max_nreg(producer_regs, 0)
                    if tx >= consumers and tx < consumers + 32:
                        for t in T.serial(mine):
                            s = t % stages
                            phase = (t // stages) % 2
                            key0 = (first + t) * block_n
                            T.mbarrier_wait_parity(kr_free[s], phase ^ 1)
                            load_tile(K, k_r, s, kr_ready, page_table, request, key0, last_page, by)
                            T.mbarrier_arrive(kr_ready[s])
                            T.mbarrier_wait_parity(vr_free[s], phase ^ 1)
                            if tx == consumers:
                                valid[s] = kv_len - key0
                            load_tile(V, v_r, s, vr_ready, page_table, request, key0, last_page, by)
                            T.mbarrier_arrive(vr_ready[s])
                    if rows_on_m and tx >= consumers + 32:
                        for t in T.serial(mine):
                            s = t % stages
                            phase = (t // stages) % 2
                            c = t % cstages
                            cphase = (t // cstages) % 2
                            ct = tx - consumers - 32
                            T.mbarrier_wait_parity(kr_ready[s], phase)
                            T.mbarrier_wait_parity(kc_free[c], cphase ^ 1)
                            widen(k_r, k_c, s, c, ct, block_n, converters, "")
                            T.mbarrier_arrive(kr_free[s])
                            # The consumer reads the tile through the async proxy.
                            T.fence_proxy_async()
                            T.mbarrier_arrive(kc_ready[c])
                    with T.ws(0):
                        if rows_on_m:
                            T.set_max_nreg(consumer_regs, 1)
                            q_f = T.alloc_fragment([block_m, dim], wide)
                            acc_s = T.alloc_fragment([block_m, block_n], accum)
                            p_f = T.alloc_fragment([block_m, block_n], wide)
                            acc_o = T.alloc_fragment([block_m, dim], accum)
                            m = T.alloc_fragment([block_m], accum)
                            red = T.alloc_fragment([block_m, 8], accum)
                            lp = T.alloc_fragment([block_m, 16], accum)
                            m_prev = T.alloc_fragment([block_m], accum)
                            rescale = T.alloc_fragment([block_m], accum)
                            row_sum = T.alloc_fragment([block_m], accum)
                            logsum = T.alloc_fragment([block_m], accum)
                            # The query is the register operand of every key product.
                            gather_query(Q, q_f, q_start, row0, rows, by)
                            T.clear(acc_o)
                            T.clear(lp)
                            T.fill(m, -T.infinity(accum))
                            if mine > 0:
                                # Tile 0: key product, value widening and softmax.
                                key_product(q_f, k_c, acc_s, kc_ready, 0)
                                widen_values(v_r, v_c, valid, vr_ready, vr_free, 0, tx)
                                T.wait_wgmma(0)
                                T.mbarrier_arrive(kc_free[0])
                                softmax(
                                    acc_s, red, m, m_prev, rescale, lp, first * block_n, kv_len,
                                    first_q, last_q, row0, align, qk_descale, item_scale,
                                )  # fmt: skip
                                T.copy(acc_s, p_f)
                                # Tile t's key product and value widening run beside tile t - 1's
                                # value product; tile t's softmax follows while the latter ends.
                                for t in T.serial(1, mine):
                                    key_product(q_f, k_c, acc_s, kc_ready, t)
                                    rescale_rows(acc_o, rescale)
                                    T.wgmma_gemm(
                                        p_f, v_c[(t - 1) % vstages, :, :], acc_o, policy=policy,
                                        clear_accum=False,
                                    )  # fmt: skip
                                    widen_values(v_r, v_c, valid, vr_ready, vr_free, t, tx)
                                    T.wait_wgmma(1)
                                    T.mbarrier_arrive(kc_free[t % cstages])
                                    softmax(
                                        acc_s, red, m, m_prev, rescale, lp, (first + t) * block_n,
                                        kv_len, first_q, last_q, row0, align, qk_descale, item_scale,
                                    )  # fmt: skip
                                    T.wait_wgmma(0)
                                    T.copy(acc_s, p_f)
                                # The value product of the last tile.
                                rescale_rows(acc_o, rescale)
                                T.wgmma_gemm(
                                    p_f, v_c[(mine - 1) % vstages, :, :], acc_o, policy=policy,
                                    clear_accum=False,
                                )  # fmt: skip
                                T.wait_wgmma(0)
                            T.reduce_sum(lp, logsum, dim=1)
                            store(
                                acc_o, False, m, logsum, row_sum, glse, partial, Output, item_scale,
                                v_descale, q_start, row0, rows, bx, by, bz,
                            )  # fmt: skip
                        else:
                            k_f = T.alloc_fragment([block_n, dim], wide)
                            st = T.alloc_fragment([block_n, block_m], accum)
                            lt = T.alloc_fragment([block_n, block_m], accum)
                            acc_ot = T.alloc_fragment([dim, block_m], accum)
                            m = T.alloc_fragment([block_m], accum)
                            m_prev = T.alloc_fragment([block_m], accum)
                            rescale = T.alloc_fragment([block_m], accum)
                            row_sum = T.alloc_fragment([block_m], accum)
                            logsum = T.alloc_fragment([block_m], accum)
                            gather_query(Q, q_s, q_start, row0, rows, by)
                            T.fence_proxy_async()
                            T.sync_threads(1, consumers)
                            T.clear(acc_ot)
                            T.clear(lt)
                            T.fill(m, -T.infinity(accum))
                            # Tile t's key product and value widening run beside tile t - 1's
                            # value product.
                            for t in T.serial(mine):
                                s = t % stages
                                # The keys widen into the register operand of their product.
                                T.mbarrier_wait_parity(kr_ready[s], (t // stages) % 2)
                                for j, d in T.Parallel(block_n, dim):
                                    k_f[j, d] = T.cast(k_r[s, j, d], wide)
                                T.mbarrier_arrive(kr_free[s])
                                T.wgmma_gemm(
                                    k_f, q_s, st, transpose_B=True, policy=policy, clear_accum=True
                                )  # fmt: skip
                                # The previous value product reads the one value stage; the wait
                                # holds the key product just issued too.
                                T.wait_wgmma(0)
                                widen_values(v_r, v_c, valid, vr_ready, vr_free, t, tx)
                                T.wait_wgmma(0)
                                # The softmax over the keys, the tile's first axis; the sums stay
                                # per element until the last tile.
                                key0 = (first + t) * block_n
                                if use_softcap:
                                    for j, i in T.Parallel(block_n, block_m):
                                        st[j, i] = T.cast(softcap, accum) * T.tanh(
                                            st[j, i]
                                            * qk_descale
                                            * T.cast(sm_scale / softcap, accum)
                                        )
                                if cut_tile(key0, kv_len, first_q, last_q):
                                    for j, i in T.Parallel(block_n, block_m):
                                        st[j, i] = T.if_then_else(
                                            hidden(key0 + j, (row0 + i) // group + align, kv_len),
                                            -T.infinity(accum),
                                            st[j, i],
                                        )
                                T.copy(m, m_prev)
                                T.reduce_max(st, m, dim=0, clear=False)
                                advance_max(m, m_prev, rescale, item_scale)
                                for j, i in T.Parallel(block_n, block_m):
                                    st[j, i] = T.exp2(st[j, i] * item_scale - m[i] * item_scale)
                                    lt[j, i] = lt[j, i] * rescale[i] + st[j, i]
                                for d, i in T.Parallel(dim, block_m):
                                    acc_ot[d, i] *= rescale[i]
                                for j, i in T.Parallel(block_n, block_m):
                                    p_s[i, j] = T.cast(st[j, i], wide)
                                T.fence_proxy_async()
                                T.sync_threads(1, consumers)
                                T.wgmma_gemm(
                                    v_c[t % vstages, :, :], p_s, acc_ot, transpose_A=True,
                                    transpose_B=True, policy=policy, clear_accum=False,
                                )  # fmt: skip
                            T.wait_wgmma(0)
                            T.reduce_sum(lt, logsum, dim=0)
                            store(
                                acc_ot, True, m, logsum, row_sum, glse, partial, Output, item_scale,
                                v_descale, q_start, row0, rows, bx, by, bz,
                            )  # fmt: skip

        @T.macro
        def combine(
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),
            glse: T.Tensor(shape_lse, accum),
            partial: T.Tensor(shape_partial, accum),
            Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(tiles, heads_kv, threads=WARPGROUP_THREADS) as (bx, by):
                tile_cum = T.alloc_shared([batch + 1], "int32")
                lse_max = T.alloc_local([1], accum)
                total = T.alloc_local([1], accum)
                o_acc = T.alloc_fragment([dim], accum)
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                found = T.alloc_local([1], "int32")
                first_row = T.alloc_local([1], "int32")

                tiling.cumsum_offsets(cu_seqlens_q, tile_cum)
                if bx < tile_cum[batch]:
                    tiling.decode(bx, tile_cum, lo, hi, found, first_row)
                    request = found[0]
                    row0 = first_row[0]
                    q_start = cu_seqlens_q[request]
                    rows = (cu_seqlens_q[request + 1] - q_start) * group
                    for i in T.serial(block_m):
                        if row0 + i < rows:
                            lse_max[0] = -T.infinity(accum)
                            for k in T.serial(num_split):
                                lse_max[0] = T.max(lse_max[0], glse[bx, by, k, i])
                            total[0] = 0
                            for k in T.serial(num_split):
                                total[0] += T.exp2(glse[bx, by, k, i] - lse_max[0])
                            total[0] = T.log2(total[0]) + lse_max[0]
                            T.clear(o_acc)
                            for k in T.serial(num_split):
                                weight = T.exp2(glse[bx, by, k, i] - total[0])
                                for d in T.Parallel(dim):
                                    o_acc[d] += partial[bx, by, k, i, d] * weight
                            r = row0 + i
                            for d in T.Parallel(dim):
                                Output[q_start + r // group, by * group + r % group, d] = (
                                    T.if_then_else(lse_max[0] < T.cast(no_key, accum), 0, o_acc[d])
                                )

        @T.prim_func
        def gqa_paged_fp8_decode(
            Q: T.Tensor(shape_q, q_dtype),
            K: T.Tensor(shape_kv, fp8_dtype),
            V: T.Tensor(shape_kv, fp8_dtype),
            cache_seqlens: T.Tensor([batch], T.int32),
            page_table: T.Tensor([batch, max_pages_per_req], T.int32),
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),
            QS: T.Tensor([batch, heads_kv], accum),
            KS: T.Tensor(kv_scale_shape, accum),
            VS: T.Tensor(kv_scale_shape, accum),
            glse: T.Tensor(shape_lse, accum),
            partial: T.Tensor(shape_partial, accum),
            Output: T.Tensor(shape_q, dtype),
        ):
            scan(
                Q, K, V, cache_seqlens, page_table, cu_seqlens_q, QS, KS, VS, glse, partial, Output
            )  # fmt: skip
            if split:
                combine(cu_seqlens_q, glse, partial, Output)

        return gqa_paged_fp8_decode

    return _func


class GQAPagedFP8DecodeKernel(Kernel, GQAPagedFwdInterface):
    """SM90 paged attention over an FP8 cache for packings of at most 64 rows a request.

    It serves a float16 or bfloat16 output over an FP8 cache, the query 16-bit in that dtype
    or FP8, at head dimension 64 or 128, over pages of a multiple or a divisor of 64 rows,
    causal or bidirectional, with windows and a softcap, and a positive scale. RoPE and sinks
    stay on ``GQAPagedFwdKernel``. A tile of 64 rows holds them on the WGMMA M extent; a
    narrower one holds the keys there.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_paged_varlen_kernel"})
    _DIMS: tuple[int, ...] = (64, 128)
    # Query rows a tile holds at most: one WGMMA's M extent.
    _MAX_ROWS: int = WGMMA_ROWS

    @staticmethod
    def block_n_for(page_size: int) -> int | None:
        """The key tile: one WGMMA's 64 rows, part of one page or a run of whole pages."""
        if page_size % WGMMA_ROWS == 0 or WGMMA_ROWS % page_size == 0:
            return WGMMA_ROWS
        return None

    @classmethod
    def rows_for(cls, call: AttentionCall) -> int:
        """Rows a tile holds: the packed rows a request averages, in steps of eight."""
        rows = -(-call.max_seqlen_q * (call.heads // call.heads_kv) // max(1, call.batch))
        return max(8, -(-rows // 8) * 8)

    @classmethod
    def refusal(cls, call: AttentionCall) -> str | None:
        """Why *call* is outside this implementation's region, or ``None`` when it is inside."""
        if call.dtype not in ATTENTION_DTYPES:
            return "requires a float16 or bfloat16 output"
        if call.cache_dtype != torch.float8_e4m3fn:
            return "requires an FP8 cache"
        if call.query_dtype not in (None, call.dtype, torch.float8_e4m3fn):
            return "requires a query of the output dtype or FP8"
        if call.dim not in cls._DIMS:
            return f"requires head dimension in {cls._DIMS}"
        if call.heads_kv <= 0 or call.heads % call.heads_kv != 0:
            return "requires heads to be a multiple of heads_kv"
        if call.sm_scale is not None and call.sm_scale <= 0.0:
            return "requires a positive scale"
        if call.page_size <= 0 or cls.block_n_for(call.page_size) is None:
            return f"requires a page size that is a multiple or a divisor of {WGMMA_ROWS} rows"
        if call.fuse_rope:
            return "does not serve RoPE"
        if call.has_sinks:
            return "does not serve sinks"
        if cls.rows_for(call) > cls._MAX_ROWS:
            return f"serves packings of at most {cls._MAX_ROWS} rows a request"
        if call.batch > cls._MAX_BATCH:
            return (
                f"holds the per-request tile prefix in shared memory, so batch <= {cls._MAX_BATCH}"
            )
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The device index is in the identity: the kernel is compiled for its architecture,
        and the split count reads its multiprocessors."""
        index = call.device.index if call.device is not None else None
        # A query position plus the right window is 32-bit; a window reaching the table's last
        # key hides none, so it is dropped before the sum can overflow.
        last_key = call.max_pages_per_req * call.page_size - 1
        args = dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            dim=call.dim,
            page_size=call.page_size,
            max_pages_per_req=call.max_pages_per_req,
            is_causal=call.is_causal,
            window_size_left=call.window_size_left,
            window_size_right=call.window_size_right if call.window_size_right < last_key else -1,
            dtype=call.dtype,
            query_dtype=call.query_dtype or call.dtype,
            per_request_kv_scale=call.per_request_kv_scale,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            block_m=cls.rows_for(call),
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
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        query_dtype: torch.dtype | None = None,
        per_request_kv_scale: bool = False,
        sm_scale: float | None = None,
        softcap: float = 0.0,
        block_m: int = WGMMA_ROWS,
        config: dict | None = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.page_size = page_size
        self.max_pages_per_req = max_pages_per_req
        self.dtype = dtype
        self.query_dtype = query_dtype or dtype
        self.block_m = block_m
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        device = torch.device("cuda", self.device_index)
        self._unit_q_scale = torch.ones((batch, heads_kv), dtype=torch.float32, device=device)
        self._processors = self.build_target.sm_count
        self._builder_args = (
            batch,
            heads,
            heads_kv,
            dim,
            page_size,
            max_pages_per_req,
            is_causal,
            window_size_left,
            window_size_right,
            self.sm_scale,
            softcap,
            self.dtype_str,
            str(self.query_dtype).removeprefix("torch."),
            per_request_kv_scale,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Zero reads the chunk count from the launch geometry; a positive value pins it. With
        # rows on M, two widened key stages; with keys on M the keys widen in registers and the
        # one stage is unused.
        rows_on_m = self.block_m == WGMMA_ROWS
        return {
            "block_n": self.block_n_for(self.page_size),
            "num_split": 0,
            "stages": 2,
            "cstages": 2 if rows_on_m else 1,
            "producer_regs": 56,
        }

    def resident_ctas(self, cfg: dict) -> int:
        """CTAs a multiprocessor holds: as many as both its shared memory and its registers
        fit. The launch bound asks ptxas for the same count.

        Every shared buffer starts on a kilobyte; beside the rings a CTA holds the tile offsets,
        the value row counts and, with keys on M, the reduction workspace, then a kilobyte of
        barriers and the kilobyte the multiprocessor reserves a block. A thread's registers are
        its share of the tiles the consumer holds, 32-bit or packed 16-bit pairs, and
        _ADDRESS_REGS more; with rows on M the producer warpgroup keeps producer_regs.
        """
        block_n, stages, rows, dim = cfg["block_n"], cfg["stages"], self.block_m, self.dim

        def kb(nbytes: int) -> int:
            return -(-nbytes // 1024) * 1024

        tile = block_n * dim
        rings = 2 * kb(stages * tile) + kb(_value_stages(rows) * tile * 2)
        if rows == WGMMA_ROWS:
            rings += kb(cfg["cstages"] * tile * 2)
            small = 2
            # Scores, row sums and output in float; query and probabilities in 16-bit pairs.
            held = (rows * block_n + rows * 16 + rows * dim) + (rows * dim + rows * block_n) // 2
            regs = (held // WARPGROUP_THREADS + _ADDRESS_REGS + cfg["producer_regs"]) // 2
            threads = 2 * WARPGROUP_THREADS
        else:
            rings += kb(rows * dim * 2) + kb(rows * block_n * 2)
            small = 3
            # Scores, their sums and the output in float; the widened keys in 16-bit pairs.
            held = 2 * block_n * rows + dim * rows + tile // 2
            regs = held // WARPGROUP_THREADS + _ADDRESS_REGS
            threads = WARPGROUP_THREADS + 32
        footprint = rings + kb(4 * (self.batch + 1)) + (small - 1) * 1024 + 2 * 1024
        by_registers = 64 * 1024 // (threads * regs)
        return max(1, min(_SM_SHARED_BYTES // footprint, by_registers))

    def _splits_for(self, tiles: int) -> int:
        """Chunks to cut each row tile's key range into: the count that minimizes the launch's
        waves times one CTA's key tiles plus the ring it fills before its first tile.

        A wave is the CTAs the multiprocessors hold at once; the key range is the table's, the
        longest any request can read. A chunk moves the same bytes as the whole range; what it
        buys is CTAs, and a partial last wave costs as much as a full one.
        """
        cfg = self.config
        slots = self.resident_ctas(cfg) * self._processors
        items = tiles * self.heads_kv
        span = max(1, -(-self.max_pages_per_req * self.page_size // cfg["block_n"]))

        def cost(splits: int) -> int:
            return -(-items * splits // slots) * (-(-span // splits) + cfg["stages"])

        # A chunk runs more key tiles than its ring fills and drains, so it spends longer
        # streaming than starting and stopping.
        steady = [n for n in range(2, span + 1) if -(-span // n) > 2 * cfg["stages"]]
        return min([1, *steady], key=lambda n: (cost(n), n))

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
        c = self.config
        group = self.heads // self.heads_kv
        # Row tiles this packing can need, at most one partial tile per request; read from
        # the shapes, because summing the offsets would synchronize with the device.
        tiles = q.shape[0] * group // self.block_m + self.batch
        # A request owning tokens owns at least one tile, and no packing owns fewer than its
        # rows need: exact for packings whose requests fit one tile.
        estimate = max(min(self.batch, q.shape[0]), -(-q.shape[0] * group // self.block_m), 1)
        splits = c["num_split"] or self._splits_for(estimate)
        program = self.program(self._compile, splits)
        glse = torch.empty(
            (tiles, self.heads_kv, splits, self.block_m), dtype=torch.float32, device=q.device
        )
        partial = torch.empty(
            (tiles, self.heads_kv, splits, self.block_m, self.dim),
            dtype=torch.float32,
            device=q.device,
        )
        return program(
            q,
            k_pool,
            v_pool,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            q_scale if q_scale is not None else self._unit_q_scale,
            k_scale,
            v_scale,
            glse,
            partial,
        )

    def _compile(self, cfg: dict, splits: int):
        """The program for *cfg* with the key range cut into *splits* chunks."""
        return _gqa_paged_fp8_decode_kernel(*self._builder_args)(
            self.block_m,
            cfg["block_n"],
            splits,
            cfg["stages"],
            cfg["cstages"],
            cfg["producer_regs"],
            self.resident_ctas(cfg),
        )
