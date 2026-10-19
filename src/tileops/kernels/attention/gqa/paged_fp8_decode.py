"""SM90 grouped-query attention over a paged FP8 cache, for packings of few query rows a request.

One CTA owns one KV head, ``block_m`` of that head's query rows and one chunk of the key range;
row ``r`` of a request is query position ``r // group`` of head ``kv_head * group + r % group``.
A key tile lands in shared memory as stored and converts to the 16-bit compute type in
registers, so the 16-bit tile never returns to shared memory, where a round trip would move
five times the cache bytes per key tile. The products swap
their operands so the converted tile is the register operand of the WGMMA:

    S^T = K Q^T    K the register operand, Q in shared memory
    O^T = V^T P^T  V^T the register operand, P^T staged in shared memory

The score tile is then keys by rows, and the softmax reduces over its first axis. Chunks of
one row tile combine by their log-sum-exp in a second kernel.
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
from tileops.kernels.constants import LOG2E, WARPGROUP_THREADS, WGMMA_ROWS
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["GQAPagedFP8DecodeKernel"]

_FP8 = "float8_e4m3fn"


@functools.lru_cache(maxsize=32)
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
    accum = "float"
    group = heads // heads_kv
    q_fp8 = q_dtype != dtype
    # The type the tensor cores multiply in. An e4m3 value converts exactly into either
    # 16-bit type, and into float16 in one instruction, so an FP8 query multiplies in float16
    # whatever the output type; a 16-bit query fixes it.
    mma = "float16" if q_fp8 else dtype
    use_softcap = softcap > 0.0
    # exp2 of a score times this is exp of the scaled score; a softcap applies the scale first.
    softmax_scale = LOG2E if use_softcap else sm_scale * LOG2E
    # Below this a chunk saw no key. A threshold, not an equality with -inf: fast math folds
    # comparisons with infinity away.
    no_key = -1.0e30

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            # One warpgroup issues the copies it waits for: an inferred producer warpgroup
            # takes registers to 240 a thread and leaves one CTA a multiprocessor.
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            # Row i of the tile is query i // group of head i % group, which is injective
            # in i; the checker cannot prove it and reports the output store as a race.
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_m: int, block_n: int, num_split: int, num_stages: int):
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

        @T.macro
        def load_tile(pool, dst, page_table, request, key0, last_page, kv_head):
            for p in T.unroll(boxes):
                key = key0 + p * box
                row = page_table[request, T.min(key // page_size, last_page)] * page_size
                T.copy(
                    pool[row + key % page_size : row + key % page_size + box, kv_head, :],
                    dst[p * box : (p + 1) * box, :],
                )

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

        @T.macro
        def scan(
            Q: T.Tensor(shape_q, q_dtype),
            K: T.Tensor(shape_kv, _FP8),
            V: T.Tensor(shape_kv, _FP8),
            cache_seqlens: T.Tensor([batch], T.int32),
            page_table: T.Tensor([batch, max_pages_per_req], T.int32),
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),
            QS: T.Tensor([batch, heads_kv], accum),
            KS: T.Tensor(kv_scale_shape, accum),
            VS: T.Tensor(kv_scale_shape, accum),
            glse: T.Tensor(shape_lse, accum),
            partial: T.Tensor(shape_partial, dtype),
        ):
            with T.Kernel(tiles, heads_kv, num_split, threads=WARPGROUP_THREADS) as (bx, by, bz):
                q_s = T.alloc_shared([block_m, dim], mma)
                k_s = T.alloc_shared([block_n, dim], _FP8)
                v_s = T.alloc_shared([block_n, dim], _FP8)
                p_s = T.alloc_shared([block_m, block_n], mma)
                tile_cum = T.alloc_shared([batch + 1], "int32")
                k_f = T.alloc_fragment([block_n, dim], mma)
                vt_f = T.alloc_fragment([dim, block_n], mma)
                st = T.alloc_fragment([block_n, block_m], accum)
                acc_ot = T.alloc_fragment([dim, block_m], accum)
                m = T.alloc_fragment([block_m], accum)
                m_prev = T.alloc_fragment([block_m], accum)
                rescale = T.alloc_fragment([block_m], accum)
                row_sum = T.alloc_fragment([block_m], accum)
                logsum = T.alloc_fragment([block_m], accum)
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                found = T.alloc_local([1], "int32")
                first_row = T.alloc_local([1], "int32")
                # A row of 128 bytes swizzled at 128 bytes: the register loads of eight
                # consecutive rows then fall in distinct banks.
                T.annotate_layout({k_s: make_swizzled_layout(k_s), v_s: make_swizzled_layout(v_s)})

                tiling.cumsum_offsets(cu_seqlens_q, tile_cum)
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
                    kv_at = (request, by) if per_request_kv_scale else (0,)
                    qk_descale = T.alloc_var(
                        accum, init=KS[kv_at] * QS[request, by] if q_fp8 else KS[kv_at]
                    )
                    v_descale = T.alloc_var(accum, init=VS[kv_at])

                    for i, d in T.Parallel(block_m, dim):
                        r = T.min(row0 + i, rows - 1)
                        x = Q[q_start + r // group, by * group + r % group, d]
                        q_s[i, d] = T.if_then_else(
                            row0 + i < rows, T.cast(x, mma) if q_fp8 else x, T.cast(0, mma)
                        )
                    T.clear(acc_ot)
                    T.clear(logsum)
                    T.fill(m, -T.infinity(accum))

                    first_q = row0 // group + align
                    last_q = T.min(q_len - 1, (row0 + block_m - 1) // group) + align
                    key_end = kv_len
                    if is_causal:
                        key_end = T.min(kv_len, last_q + 1)
                    elif window_size_right >= 0:
                        key_end = T.min(kv_len, last_q + 1 + window_size_right)
                    key_start = 0
                    if window_size_left >= 0:
                        key_start = T.max(0, first_q - window_size_left)
                    tile0 = key_start // block_n
                    span = T.max(0, T.ceildiv(key_end, block_n) - tile0)
                    # Whole key tiles per chunk, so a chunk boundary never splits one.
                    per_split = T.ceildiv(span, num_split)
                    mine = T.max(0, T.min(per_split, span - bz * per_split))
                    first = tile0 + bz * per_split
                    last_page = T.max(0, (kv_len - 1) // page_size)

                    for t in T.Pipelined(mine, num_stages=num_stages):
                        key0 = (first + t) * block_n
                        load_tile(K, k_s, page_table, request, key0, last_page, by)
                        load_tile(V, v_s, page_table, request, key0, last_page, by)
                        for j, d in T.Parallel(block_n, dim):
                            k_f[j, d] = T.cast(k_s[j, d], mma)
                        T.gemm(k_f, q_s, st, transpose_B=True, clear_accum=True)
                        # A tile wholly inside every row's visible range keeps every score;
                        # its upper bound is set by the first row and its lower by the last.
                        cut = key0 + block_n > kv_len
                        if is_causal:
                            cut = cut | (key0 + block_n > first_q + 1)
                        elif window_size_right >= 0:
                            cut = cut | (key0 + block_n > first_q + window_size_right + 1)
                        if window_size_left >= 0:
                            cut = cut | (key0 < last_q - window_size_left)
                        for j, i in T.Parallel(block_n, block_m):
                            st[j, i] *= qk_descale
                        if use_softcap:
                            for j, i in T.Parallel(block_n, block_m):
                                st[j, i] = T.cast(softcap, accum) * T.tanh(
                                    st[j, i] * T.cast(sm_scale / softcap, accum)
                                )
                        if cut:
                            for j, i in T.Parallel(block_n, block_m):
                                st[j, i] = T.if_then_else(
                                    hidden(key0 + j, (row0 + i) // group + align, kv_len)
                                    | (row0 + i >= rows),
                                    -T.infinity(accum),
                                    st[j, i],
                                )
                        T.copy(m, m_prev)
                        T.reduce_max(st, m, dim=0, clear=False)
                        for i in T.Parallel(block_m):
                            m[i] = T.max(T.max(m[i], m_prev[i]), T.cast(-1e38, accum))
                            rescale[i] = T.exp2((m_prev[i] - m[i]) * softmax_scale)
                        for j, i in T.Parallel(block_n, block_m):
                            st[j, i] = T.exp2((st[j, i] - m[i]) * softmax_scale)
                        T.reduce_sum(st, row_sum, dim=0)
                        for i in T.Parallel(block_m):
                            logsum[i] = logsum[i] * rescale[i] + row_sum[i]
                        for d, i in T.Parallel(dim, block_m):
                            acc_ot[d, i] *= rescale[i]
                        for j, i in T.Parallel(block_n, block_m):
                            p_s[i, j] = T.cast(st[j, i], mma)
                        # Rows past the cache end are stale and may hold NaN, which a zero
                        # weight does not cancel; the tile the cache end cuts drops them as it
                        # converts.
                        if key0 + block_n > kv_len:
                            for d, j in T.Parallel(dim, block_n):
                                vt_f[d, j] = T.if_then_else(
                                    key0 + j < kv_len, T.cast(v_s[j, d], mma), T.cast(0, mma)
                                )
                        else:
                            for d, j in T.Parallel(dim, block_n):
                                vt_f[d, j] = T.cast(v_s[j, d], mma)
                        T.gemm(vt_f, p_s, acc_ot, transpose_B=True)

                    for i in T.Parallel(block_m):
                        row_sum[i] = T.if_then_else(logsum[i] == 0, 0, v_descale / logsum[i])
                    for i, d in T.Parallel(block_m, dim):
                        partial[bx, by, bz, i, d] = T.cast(acc_ot[d, i] * row_sum[i], dtype)
                    for i in T.Parallel(block_m):
                        glse[bx, by, bz, i] = T.if_then_else(
                            logsum[i] == 0,
                            -T.infinity(accum),
                            T.log2(T.if_then_else(logsum[i] == 0, 1, logsum[i]))
                            + m[i] * softmax_scale,
                        )

        @T.macro
        def combine(
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),
            glse: T.Tensor(shape_lse, accum),
            partial: T.Tensor(shape_partial, dtype),
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
            K: T.Tensor(shape_kv, _FP8),
            V: T.Tensor(shape_kv, _FP8),
            cache_seqlens: T.Tensor([batch], T.int32),
            page_table: T.Tensor([batch, max_pages_per_req], T.int32),
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),
            QS: T.Tensor([batch, heads_kv], accum),
            KS: T.Tensor(kv_scale_shape, accum),
            VS: T.Tensor(kv_scale_shape, accum),
            glse: T.Tensor(shape_lse, accum),
            partial: T.Tensor(shape_partial, dtype),
            Output: T.Tensor(shape_q, dtype),
        ):
            scan(
                Q, K, V, cache_seqlens, page_table, cu_seqlens_q, QS, KS, VS, glse, partial
            )  # fmt: skip
            combine(cu_seqlens_q, glse, partial, Output)

        return gqa_paged_fp8_decode

    return _func


class GQAPagedFP8DecodeKernel(Kernel, GQAPagedFwdInterface):
    """SM90 paged attention over an FP8 cache for packings of at most 64 rows a request.

    It serves a float16 or bfloat16 output over an FP8 cache, the query 16-bit in that dtype
    or FP8, at head dimension 64 or 128, over pages of a multiple or a divisor of 64 rows,
    causal or bidirectional, with windows and a softcap, and a positive scale. RoPE and sinks
    stay on ``GQAPagedFwdKernel``.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_paged_varlen_kernel"})
    _DIMS: tuple[int, ...] = (64, 128)
    # Query rows a tile holds: the N extent of the swapped products, at most what the
    # combine buffers are laid out for.
    _MAX_ROWS: int = 64

    @staticmethod
    def block_n_for(page_size: int) -> Optional[int]:
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
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The device index is in the identity: the kernel is compiled for its architecture,
        and the split count reads its multiprocessors."""
        index = call.device.index if call.device is not None else None
        args = dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            dim=call.dim,
            page_size=call.page_size,
            max_pages_per_req=call.max_pages_per_req,
            is_causal=call.is_causal,
            window_size_left=call.window_size_left,
            window_size_right=call.window_size_right,
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
        query_dtype: Optional[torch.dtype] = None,
        per_request_kv_scale: bool = False,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        block_m: int = 8,
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
        self.dtype = dtype
        self.query_dtype = query_dtype or dtype
        self.block_m = block_m
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        device = torch.device("cuda", self.device_index)
        self._unit_q_scale = torch.ones((batch, heads_kv), dtype=torch.float32, device=device)
        self._processors = torch.cuda.get_device_properties(device).multi_processor_count
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
        # Zero reads the chunk count from the launch geometry; a positive value pins it.
        return {"block_n": self.block_n_for(self.page_size), "num_split": 0, "num_stages": 2}

    def _splits_for(self, tiles: int) -> int:
        """Chunks to cut the key range into: as many as keep the launch within one wave of
        four CTAs a multiprocessor, the residency the program's registers allow.

        A chunk moves the same bytes as the whole range; what it buys is CTAs, which a launch
        of a few row tiles per KV head lacks. A launch that spills into a partial second wave
        is slower than either full one.
        """
        return max(1, 4 * self._processors // (tiles * self.heads_kv))

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
        c = self.config
        group = self.heads // self.heads_kv
        # Row tiles this packing can need, at most one partial tile per request; read from
        # the shapes, because summing the offsets would synchronize with the device.
        tiles = q.shape[0] * group // self.block_m + self.batch
        # A request owning tokens owns at least one tile, and no packing owns fewer than its
        # rows need: exact for one-token requests, where the split decides anything.
        estimate = max(min(self.batch, q.shape[0]), -(-q.shape[0] * group // self.block_m), 1)
        splits = c["num_split"] or self._splits_for(estimate)
        program = _gqa_paged_fp8_decode_kernel(*self._builder_args)(
            self.block_m, c["block_n"], splits, c["num_stages"]
        )
        glse = torch.empty(
            (tiles, self.heads_kv, splits, self.block_m), dtype=torch.float32, device=q.device
        )
        partial = torch.empty(
            (tiles, self.heads_kv, splits, self.block_m, self.dim),
            dtype=self.dtype,
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
