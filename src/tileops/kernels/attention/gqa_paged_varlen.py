"""Packed grouped-query attention over a paged KV cache the kernel only reads.

One CTA owns one KV head and ``block_M`` of that head's query rows, where row ``r`` of a
request is query position ``r // group`` of head ``kv_head * group + r % group``. Folding the
KV group into the row axis is what makes a one-token request a single tile: the cache then
crosses memory once per KV head instead of once per query head, which is a factor of ``group``
on every request short enough for its rows to fit one tile.

Request lengths are a runtime input, so the row tiles are enumerated per request and the grid
takes an upper bound; a tile past the count the kernel computes exits, and a request with no
query tokens owns no tile.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    AttentionCall,
    GQAPagedFwdInterface,
)
from tileops.kernels.attention.online_softmax import (
    make_apply_softcap,
    make_online_softmax_with_mask_guard,
    make_rescale,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin

__all__ = ["GQAPagedVarlenFwdKernel"]


@functools.lru_cache(maxsize=32)
def _gqa_paged_varlen_kernel(
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
):
    """Build the paged packed-query attention program for one fixed set of call facts."""
    accum_dtype = "float"
    group = heads // heads_kv
    # A zero score scale makes every score zero. The scores are zeroed before the mask and a
    # unit exp2 factor applied, so a masked key stays at -inf instead of -inf * 0.
    zero_scores = softcap <= 0.0 and sm_scale == 0.0
    softmax_scale = LOG2E if softcap > 0.0 or zero_scores else sm_scale * LOG2E

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            # Row i of the tile is query i // group of head i % group, which is injective
            # in i; the checker cannot prove it and reports the output store as a race.
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_M: int, block_N: int, num_stages: int, threads: int):
        total_q = T.dynamic("total_q")
        pool_rows = T.dynamic("pool_rows")
        shape_q = (total_q, heads, dim)
        shape_kv = (pool_rows, heads_kv, dim)
        tiling = GroupTiling(batch, block_M, rows_per_offset=group)
        online_softmax = make_online_softmax_with_mask_guard(
            softmax_scale, accum_dtype, block_M, block_N
        )
        apply_softcap = (
            make_apply_softcap(sm_scale, softcap, accum_dtype, block_M, block_N)
            if softcap > 0.0
            else None
        )
        rescale = make_rescale(block_M, dim)

        # A tile starts at a multiple of block_N, so a page at least that long holds one
        # whole. That is the only case a contiguous copy serves: a tile spanning several
        # pages lands on rows the table scatters, and copying each page into its slice of the
        # tile costs more per page than gathering the whole tile by row. FlashAttention-3
        # draws the same line, taking its TMA path only for page_size % kBlockN == 0.
        one_page_holds_tile = page_size % block_N == 0

        @T.macro
        def load_kv(K, V, page_table, k_shared, v_shared, request, kv_head, key0, kv_len):
            """Read one key tile through the page table.

            Rows past the cache are not guarded: they land in a page the table names, so they
            are cached values rather than unmapped memory, the mask sends their scores to
            -inf, and the zero weight that follows removes them from the value sum. A guard
            here would cost the whole tile its vector width.
            """
            if one_page_holds_tile:
                base = page_table[request, key0 // page_size] * page_size + key0 % page_size
                T.copy(K[base : base + block_N, kv_head, :], k_shared)
                T.copy(V[base : base + block_N, kv_head, :], v_shared)
            else:
                for j, d in T.Parallel(block_N, dim):
                    key = T.min(key0 + j, kv_len - 1)
                    row = page_table[request, key // page_size] * page_size + key % page_size
                    k_shared[j, d] = K[row, kv_head, d]
                    v_shared[j, d] = V[row, kv_head, d]

        def masked(key, q_pos, kv_len):
            """Whether *key* is outside what the row at *q_pos* sees.

            A tile starts at or after key zero, so only the upper end needs a bound.
            """
            out = key >= kv_len
            if is_causal:
                out = out | (key > q_pos)
            if window_size_left >= 0:
                out = out | (key < q_pos - window_size_left)
            if window_size_right >= 0:
                out = out | (key > q_pos + window_size_right)
            return out

        @T.macro
        def apply_mask(acc_s, key0, row0, rows, align, kv_len):
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    masked(key0 + j, (row0 + i) // group + align, kv_len) | (row0 + i >= rows),
                    -T.infinity(accum_dtype),
                    acc_s[i, j],
                )

        @T.prim_func
        def gqa_paged_varlen(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            cache_seqlens: T.Tensor([batch], T.int32),
            page_table: T.Tensor([batch, max_pages_per_req], T.int32),
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),
            Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(tiling.tile_upper_bound(total_q * group), heads_kv, threads=threads) as (
                bx,
                by,
            ):
                q_shared = T.alloc_shared([block_M, dim], dtype)
                k_shared = T.alloc_shared([block_N, dim], dtype)
                v_shared = T.alloc_shared([block_N, dim], dtype)
                tile_cum = T.alloc_shared([batch + 1], "int32")
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                row_scale = T.alloc_fragment([block_M], accum_dtype)
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                found = T.alloc_local([1], "int32")
                first_row = T.alloc_local([1], "int32")

                tiling.cumsum_offsets(cu_seqlens_q, tile_cum)
                if bx < tile_cum[batch]:
                    tiling.decode(bx, tile_cum, lo, hi, found, first_row)
                    # Bind the search results: warp specialization hands the TMA producer
                    # uninitialized copies of the local buffers, and it hangs.
                    request = found[0]
                    row0 = first_row[0]

                    q_start = cu_seqlens_q[request]
                    q_len = cu_seqlens_q[request + 1] - q_start
                    kv_len = cache_seqlens[request]
                    rows = q_len * group
                    # Queries sit at the end of the cache, so position i of the request is
                    # key index i + kv_len - q_len.
                    align = kv_len - q_len

                    for i, d in T.Parallel(block_M, dim):
                        r = T.min(row0 + i, rows - 1)
                        q_shared[i, d] = T.if_then_else(
                            row0 + i < rows,
                            Q[q_start + r // group, by * group + r % group, d],
                            T.cast(0, dtype),
                        )
                    T.clear(acc_o)
                    T.clear(logsum)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    last_q = T.min(q_len - 1, (row0 + block_M - 1) // group) + align
                    if is_causal:
                        key_end = T.min(kv_len, last_q + 1)
                    elif window_size_right >= 0:
                        key_end = T.min(kv_len, last_q + 1 + window_size_right)
                    else:
                        key_end = kv_len
                    if window_size_left >= 0:
                        key_start = T.max(0, row0 // group + align - window_size_left)
                    else:
                        key_start = 0

                    tile0 = key_start // block_N
                    for t in T.Pipelined(
                        T.max(0, T.ceildiv(key_end, block_N) - tile0), num_stages=num_stages
                    ):
                        key0 = (tile0 + t) * block_N
                        load_kv(K, V, page_table, k_shared, v_shared, request, by, key0, kv_len)
                        T.clear(acc_s)
                        # The GEMM runs even for zero scores: it fixes the layout the row
                        # statistics share.
                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        if zero_scores:
                            T.clear(acc_s)
                        if softcap > 0.0:
                            apply_softcap(acc_s)
                        apply_mask(acc_s, key0, row0, rows, align, kv_len)
                        online_softmax(
                            acc_s,
                            scores_max,
                            scores_max_prev,
                            scores_scale,
                            scores_sum,
                            logsum,
                        )
                        T.copy(acc_s, acc_s_cast)
                        rescale(acc_o, scores_scale)
                        T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    # One reciprocal per row: a per-element divide by a row scalar is read
                    # as a divide by a varying value.
                    for i in T.Parallel(block_M):
                        row_scale[i] = T.if_then_else(logsum[i] == 0, 0, 1.0 / logsum[i])
                    for i, d in T.Parallel(block_M, dim):
                        if row0 + i < rows:
                            r = row0 + i
                            Output[q_start + r // group, by * group + r % group, d] = (
                                acc_o[i, d] * row_scale[i]
                            )

        return gqa_paged_varlen

    return _func


class GQAPagedVarlenFwdKernel(Kernel, GQAPagedFwdInterface):
    """Paged attention over packed queries of any per-request length, or a restricted window."""

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def applies(cls, call: AttentionCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why *call* is outside this implementation's region, or ``None`` when it is inside.

        It serves every call whose query and cache are float16 or bfloat16 of the same dtype:
        any mix of per-request query lengths, any positive page size, both window bounds,
        causal and bidirectional. An FP8 tensor and a fused rotation are refused.
        """
        if call.dtype not in ATTENTION_DTYPES:
            return "requires float16 or bfloat16 Q"
        if call.cache_dtype != call.dtype:
            return "requires Q and KV to share a dtype"
        if call.is_fp8:
            return "does not serve FP8"
        if call.fuse_rope:
            return "does not serve RoPE"
        if call.page_size <= 0:
            return "requires a positive page size"
        if call.tensor_core_dim_refusal is not None:
            return call.tensor_core_dim_refusal
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The packed totals are absent: the kernel reads them from the tensors it is handed,
        so one object serves every packing of these call facts. The device index is in the
        identity, because the kernel is compiled for its architecture."""
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
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            rows_fill_tile=call.is_uniform
            and call.max_seqlen_q * call.heads // call.heads_kv >= 128,
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
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        rows_fill_tile: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be a positive multiple of heads_kv")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if max_pages_per_req <= 0:
            raise ValueError("max_pages_per_req must be positive")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.page_size = page_size
        self.max_pages_per_req = max_pages_per_req
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.rows_fill_tile = rows_fill_tile
        self.kernel = _gqa_paged_varlen_kernel(
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
        )
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    def _shared_bytes(self, config: dict) -> int:
        """Shared memory the program allocates for *config*: the query rows, a K and a V tile
        per stage, and the per-request tile offsets."""
        rows = config["block_M"] + 2 * config["num_stages"] * config["block_N"]
        return rows * self.dim * self.dtype.itemsize + 4 * (self.batch + 1)

    @property
    def default_config(self) -> dict:
        """The 128-row tile where every request fills one, the 64-row tile otherwise, over the
        widest key tile up to 64 rows that the page divides or is divided by.

        Fitted over the entry's workload rows at head dimension 64 and 128. A request shorter
        than the tile pads it, and a packing holding even a few of them is faster on the
        narrow tile although its long requests are not; the wide tile pays only where every
        request fills it. A key tile the page neither holds nor fills costs the load its
        contiguous form. Re-fit by sweeping ``autotune_configs`` on those rows.
        """
        page = self.page_size
        held = [n for n in (128, 64, 48, 32) if page % n == 0]
        block_n = held[0] if held else 64
        tile = (
            {"block_M": 128, "block_N": block_n, "num_stages": 3, "threads": 256}
            if self.rows_fill_tile
            else {"block_M": 64, "block_N": block_n, "num_stages": 2, "threads": 128}
        )
        candidates = [
            tile,
            {**tile, "num_stages": 1},
            {**tile, "block_N": min(block_n, 16), "num_stages": 1},
        ]
        cap = get_shared_memory_optin(self.device_index)
        for config in candidates:
            if self._shared_bytes(config) <= cap:
                return config
        raise ValueError(
            f"{type(self).__name__} needs {self._shared_bytes(candidates[-1])} bytes of shared "
            f"memory at head dimension {self.dim}; the device grants {cap}"
        )

    @property
    def autotune_configs(self) -> list[dict]:
        # A warpgroup owns 64 rows of the score tile, so 256 threads need 128 of them.
        cap = get_shared_memory_optin(self.device_index)
        page = self.page_size
        configs = [
            {"block_M": block_m, "block_N": block_n, "num_stages": stages, "threads": threads}
            for block_m, threads in ((64, 128), (128, 128), (128, 256))
            # A key tile the page neither holds nor fills is read row by row.
            for block_n in (16, 32, 48, 64, 96, 128)
            if block_n % 16 == 0 and (page % block_n == 0 or block_n % page == 0 or block_n == 64)
            for stages in (1, 2, 3, 4)
        ]
        return [c for c in configs if self._shared_bytes(c) <= cap]

    def _make_supply_prog(self):
        """Supply one complete packed call while autotuning.

        Every tensor is built here: the query and the two pools carry a packed total this
        kernel reads at run time, which the automatic supplier cannot draw from a symbolic
        extent, and the lengths and the page table decide how many key tiles run at all.
        """
        batch, width, page_size = self.batch, self.max_pages_per_req, self.page_size
        heads, heads_kv, dim, dtype = self.heads, self.heads_kv, self.dim, self.dtype
        # One row tile of query tokens per request, against the cache its table spans. A
        # synthetic tuning point, not a claim that one config is best for every packing.
        tokens_per_request = 8
        cache_len = width * page_size

        def supply_prog(params):
            if len(params) != 6:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects q, the two pools, the cache "
                    f"lengths, the page table and the query offsets, got {len(params)} "
                    f"parameters"
                )
            device = torch.cuda.current_device()
            total_q = batch * tokens_per_request
            pool = torch.randn(cache_len, heads_kv, dim, dtype=dtype, device=device)
            return [
                torch.randn(total_q, heads, dim, dtype=dtype, device=device),
                pool,
                pool.clone(),
                torch.full((batch,), cache_len, dtype=torch.int32, device=device),
                torch.arange(width, dtype=torch.int32, device=device)
                .unsqueeze(0)
                .expand(batch, -1)
                .contiguous(),
                torch.arange(
                    0,
                    total_q + tokens_per_request,
                    tokens_per_request,
                    dtype=torch.int32,
                    device=device,
                ),
            ]

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.config
        program = self.kernel(c["block_M"], c["block_N"], c["num_stages"], c["threads"])
        return program(q, k_pool, v_pool, cache_seqlens, page_table, cu_seqlens_q)
