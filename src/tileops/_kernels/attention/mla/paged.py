"""Multi-Head Latent Attention (MLA) decode over a paged latent cache, absorbed form.

A request's keys are its cache rows, whole, and its values their first ``dim_v``
columns. A block owns 64 query rows of one request (row ``s * heads + h``).

``MLAPagedSeesawKernel`` (SM90) runs the staggered seesaw of ``mla.seesaw`` over the
pages. ``MLAPagedFwdKernel`` serves every other call: one block of query rows walks the
tiles in a plain loop, gathering each key row through the block table.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    MLAPagedCall,
    MLAPagedFwdInterface,
)
from tileops._kernels.attention.mla.seesaw import (
    _mla_seesaw_combine_kernel,
    _mla_seesaw_paged_kernel,
)
from tileops._kernels.constants import LOG2E
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._utils import get_shared_memory_optin, get_sm_count

__all__ = ["MLAPagedFwdKernel", "MLAPagedSeesawKernel"]


def _pad16(width: int) -> int:
    return -(-width // 16) * 16


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
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
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
            Lse: T.Tensor((batch, heads, s_q), accum),  # type: ignore
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
                        Lse[b, (r0 + i) % heads, (r0 + i) // heads] = T.log(l[i]) + m[i] / LOG2E

        return main

    return _func


class _MLAPagedBase(Kernel, MLAPagedFwdInterface):
    """The call geometry, the cache's unit scale, and the call wrapper both schedules share."""

    @classmethod
    def entry_for(cls, call: MLAPagedCall) -> Entry:
        args = dict(
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
                q.new_empty((0, self.heads, self.s_q), dtype=torch.float32),
            )
        o, lse = self.program(self._compile)(
            q.view(batch, self.s_q * self.heads, self.dim),
            kv_cache.view(-1, self.dim),
            block_table,
            cache_seqlens,
            self._unit if kv_scale is None else kv_scale,
        )
        return o.view(batch, self.s_q, self.heads, self.dim_v), lse


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
        self.init_config(config)
        self.program(self._compile)

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
        main, combine = self.program(self._compile_split, splits)
        o_part, lse_part = main(
            q.view(batch, rows, self.dim),
            kv_cache.view(-1, self.dim),
            block_table,
            cache_seqlens,
            self._unit if kv_scale is None else kv_scale,
        )
        o, lse = combine(o_part, lse_part)
        return o.view(batch, self.s_q, self.heads, self.dim_v), lse

    def _compile(self, cfg: dict):
        """The program over one run a request; the schedule reads no configuration."""
        return _mla_seesaw_paged_kernel(*self._args, 1)

    def _compile_split(self, cfg: dict, splits: int):
        """The program over *splits* runs a request, and the merge of their runs."""
        main = _mla_seesaw_paged_kernel(*self._args, splits)
        combine = _mla_seesaw_combine_kernel(
            self.heads, self.s_q, self.dim_v, splits, self.dtype_str, "float"
        )
        return CompiledProgram(main), CompiledProgram(combine)


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

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)
