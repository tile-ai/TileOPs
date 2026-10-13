"""Coalesced paged-cache gather with an optional FP8 dequantization multiply."""

import dataclasses
import functools
from abc import abstractmethod
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = ["PagedKVCacheGatherCall", "PagedKVCacheGatherFwdInterface", "PagedKVCacheGatherKernel"]


@dataclasses.dataclass(frozen=True)
class PagedKVCacheGatherCall(CallSpec):
    """The cache layout and optional metadata of one gather; lengths remain device data."""

    batch: int = 0
    num_pages: int = 0
    page_size: int = 0
    table_width: int = 0
    row_width: int = 0
    dtype: Optional[torch.dtype] = None
    out_dtype: Optional[torch.dtype] = None
    has_starts: bool = False


class PagedKVCacheGatherFwdInterface(KernelInterface):
    """Write contiguous packed rows from the logical ranges named by the page table."""

    request = PagedKVCacheGatherCall

    @abstractmethod
    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None) -> None:
        """Gather on call.device; all tensors are contiguous.

        Args:
            dst: Write-only [total_tokens, row_width] in call.out_dtype; returns no tensor.
            cache: Read-only [num_pages * page_size, row_width] in call.dtype.
            block_table: Physical page ids, int32 [batch, table_width].
            cu_seq_lens: Packed row offsets, int32 [batch + 1], ending at total_tokens.
            seq_starts: Optional int32 [batch] logical start positions; absent means zero.
            scale: FP32 [1] multiplier, present exactly when call.dtype is FP8 E4M3.
        """


@functools.lru_cache(maxsize=32)
def _paged_cache_gather_kernel(page_size, row_width, dtype, out_dtype, has_starts):
    quantized = dtype == "float8_e4m3fn"

    @tilelang.jit(out_idx=[])
    def _func(block_rows, block_cols, threads):
        total = T.dynamic("total")
        batch = T.dynamic("batch")
        num_pages = T.dynamic("num_pages")
        table_width = T.dynamic("table_width")
        cache_shape = (num_pages * page_size, row_width)

        @T.prim_func
        def gather(
            Dst: T.Tensor((total, row_width), out_dtype),
            Cache: T.Tensor(cache_shape, dtype),
            Table: T.Tensor((batch, table_width), "int32"),
            Cu: T.Tensor((batch + 1,), "int32"),
            Starts: T.Tensor((batch,) if has_starts else (batch + 1,), "int32"),
            Scale: T.Tensor((1,) if quantized else cache_shape, "float32" if quantized else dtype),
        ):
            with T.Kernel(
                T.ceildiv(total, block_rows), T.ceildiv(row_width, block_cols), threads=threads
            ) as (bx, by):
                slots = T.alloc_shared((block_rows,), "int32")
                # Search packed offsets directly: O(log batch) work per row, shared by
                # every column. Repeated offsets naturally skip empty requests.
                for r in T.Parallel(block_rows):
                    row = bx * block_rows + r
                    if row < total:
                        lo = T.alloc_var("int32", init=0)
                        hi = T.alloc_var("int32", init=batch)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if Cu[mid + 1] <= row:
                                lo = mid + 1
                            else:
                                hi = mid
                        pos = row - Cu[lo] + (Starts[lo] if has_starts else 0)
                        slots[r] = Table[lo, pos // page_size] * page_size + pos % page_size
                T.sync_threads()
                for r, c in T.Parallel(block_rows, block_cols):
                    row = bx * block_rows + r
                    col = by * block_cols + c
                    if row < total and col < row_width:
                        if quantized:
                            Dst[bx * block_rows + r, by * block_cols + c] = (
                                T.cast(Cache[slots[r], col], "float32") * Scale[0]
                            )
                        else:
                            Dst[bx * block_rows + r, by * block_cols + c] = Cache[slots[r], col]

        return gather

    return _func


class PagedKVCacheGatherKernel(Kernel, PagedKVCacheGatherFwdInterface):
    """One gather and optional dequantization stage, tiled over packed output rows."""

    @classmethod
    def entry_for(cls, call: PagedKVCacheGatherCall) -> Entry:
        args = dict(
            batch=call.batch,
            num_pages=call.num_pages,
            page_size=call.page_size,
            table_width=call.table_width,
            row_width=call.row_width,
            dtype=call.dtype,
            out_dtype=call.out_dtype,
            has_starts=call.has_starts,
        )
        index = call.device.index if call.device is not None else None
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self,
        batch,
        num_pages,
        page_size,
        table_width,
        row_width,
        dtype,
        out_dtype,
        has_starts=False,
        config=None,
        device_index=None,
    ):
        super().__init__(device_index=device_index)
        self.batch, self.num_pages, self.page_size = batch, num_pages, page_size
        self.table_width, self.row_width = table_width, row_width
        self.dtype, self.out_dtype, self.has_starts = dtype, out_dtype, has_starts
        self.kernel = _paged_cache_gather_kernel(
            page_size,
            row_width,
            self.dtype_str,
            self.dtype_to_str(out_dtype),
            has_starts,
        )
        self.init_config(config)

    @property
    def default_config(self):
        # Narrow column tiles avoid mostly padded CTAs on 576-wide MLA rows.
        # The same tile sustains the 16-bit copies at width 1024.
        return {"block_rows": 16, "block_cols": 128, "threads": 128}

    @property
    def autotune_configs(self):
        return [
            dict(block_rows=r, block_cols=c, threads=t)
            for r, c, t in [
                (16, 128, 128),
                (32, 64, 128),
                (4, 256, 128),
                (16, 256, 128),
                (8, 512, 128),
                (16, 512, 128),
                (32, 256, 128),
                (32, 512, 256),
            ]
        ]

    @property
    def autotune_supply_prog(self):
        def supply(params):
            device = torch.device("cuda", self.device_index)
            length = min(32, self.table_width * self.page_size)
            cache = torch.zeros(
                (self.num_pages * self.page_size, self.row_width), dtype=self.dtype, device=device
            )
            cu = torch.arange(self.batch + 1, device=device, dtype=torch.int32) * length
            table = (
                torch.arange(self.batch * self.table_width, device=device, dtype=torch.int32)
                % self.num_pages
            ).view(self.batch, self.table_width)
            return [
                torch.empty(
                    (self.batch * length, self.row_width), dtype=self.out_dtype, device=device
                ),
                cache,
                table,
                cu,
                torch.zeros(self.batch, dtype=torch.int32, device=device)
                if self.has_starts
                else cu,
                torch.ones(1, dtype=torch.float32, device=device)
                if self.dtype == torch.float8_e4m3fn
                else cache,
            ]

        return supply

    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None) -> None:
        # Absent optional arguments reuse existing tensors as unread placeholders.
        self.kernel(**self.config)(
            dst,
            cache,
            block_table,
            cu_seq_lens,
            seq_starts if self.has_starts else cu_seq_lens,
            scale if scale is not None else cache,
        )
