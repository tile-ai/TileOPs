"""Coalesced paged-cache gather with an optional FP8 dequantization multiply."""

import dataclasses
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = [
    "PagedKVCacheGatherCall",
    "PagedKVCacheGatherFwdInterface",
    "PagedKVCacheGatherKernel",
    "PagedKVCacheGatherRowsKernel",
    "PagedKVCacheGatherShortKernel",
]


@dataclasses.dataclass(frozen=True)
class PagedKVCacheGatherCall(CallSpec):
    """The cache layout and optional metadata of one gather; lengths remain device data."""

    batch: int = 0
    num_pages: int = 0
    page_size: int = 0
    table_width: int = 0
    row_width: int = 0
    dtype: torch.dtype | None = None
    out_dtype: torch.dtype | None = None
    has_starts: bool = False
    total_tokens: int = 0


class PagedKVCacheGatherFwdInterface(KernelInterface):
    """Write contiguous packed rows from the logical ranges named by the page table."""

    request = PagedKVCacheGatherCall

    @abstractmethod
    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None) -> None:
        """Gather on call.device; all tensors are contiguous.

        Args:
            dst: Write-only [total_tokens, *row_shape] in call.out_dtype; returns no tensor.
            cache: Read-only [num_pages, page_size, *row_shape] in call.dtype.
            block_table: Physical page ids, int32 [batch, table_width].
            cu_seq_lens: Packed row offsets, int32 [batch + 1], ending at total_tokens.
            seq_starts: Optional int32 [batch] logical start positions; absent means zero.
            scale: FP32 [1] multiplier, present exactly when call.dtype is FP8 E4M3.
        """


# Elements per vector of the row kernel: one 16-byte store of 16-bit output, so each store
# instruction of a warp covers whole sectors. Its builder, refusal and config agree on it.
_ROW_VECTOR = 8


@T.macro
def _row_slots(
    slots, Cu, Table, Starts, first, total, batch, *, block_rows, page_size, small_batch, has_starts
):
    """Write the physical cache slot of packed rows first .. first + block_rows into *slots*.

    The request search is O(log batch) per row, or a sum of comparisons for a small batch.
    Repeated offsets skip empty requests.
    """
    for r in T.Parallel(block_rows):
        row = first + r
        if row < total:
            lo = T.alloc_var("int32", init=0)
            if small_batch:
                for b in T.unroll(1, small_batch):
                    lo += T.cast(Cu[b] <= row, "int32")
            else:
                hi = T.alloc_var("int32", init=batch)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if Cu[mid + 1] <= row:
                        lo = mid + 1
                    else:
                        hi = mid
            pos = T.cast(row - Cu[lo], "int64") + (Starts[lo] if has_starts else 0)
            slots[r] = T.cast(Table[lo, pos // page_size], "int64") * page_size + pos % page_size


@target_cache(maxsize=32)
def _paged_cache_gather_kernel(page_size, row_width, dtype, out_dtype, has_starts, small_batch):
    quantized = dtype == "float8_e4m3fn"

    @tilelang.jit(out_idx=[])
    def _func(block_rows, block_cols, threads):
        total = T.dynamic("total")
        batch = T.dynamic("batch")
        # Both the flattened extent and physical slots can exceed int32 in large pools.
        num_pages = T.dynamic("num_pages", "int64")
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
                slots = T.alloc_shared((block_rows,), "int64")
                # The slots are shared within this CTA; other column CTAs repeat the lookup.
                _row_slots(
                    slots,
                    Cu,
                    Table,
                    Starts,
                    bx * block_rows,
                    total,
                    batch,
                    block_rows=block_rows,
                    page_size=page_size,
                    small_batch=small_batch,
                    has_starts=has_starts,
                )
                T.sync_threads()
                for r, c in T.Parallel(block_rows, block_cols):
                    row = bx * block_rows + r
                    col = by * block_cols + c
                    if row < total and col < row_width:
                        if quantized:
                            Dst[row, col] = T.cast(Cache[slots[r], col], "float32") * Scale[0]
                        else:
                            Dst[row, col] = Cache[slots[r], col]

        return gather

    return _func


@target_cache(maxsize=32)
def _paged_cache_gather_rows_kernel(
    page_size, row_width, dtype, out_dtype, has_starts, small_batch
):
    quantized = dtype == "float8_e4m3fn"
    # A 16-byte FP8 load would leave two stores per thread, each filling half of every sector
    # it touches.
    vec = _ROW_VECTOR
    lanes = row_width // vec

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            # The block table's contract keeps every slot inside the cache; the guard would
            # cost two 64-bit compares per vector.
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        },
        compile_flags=[*CUDA_COMPILE_FLAGS, *csrc_include("row_gather.h")],
    )
    def _func(block_rows, threads):
        total = T.dynamic("total")
        batch = T.dynamic("batch")
        num_pages = T.dynamic("num_pages", "int64")
        table_width = T.dynamic("table_width")
        cache_shape = (num_pages * page_size, row_width)
        vectors = block_rows * lanes
        steps = -(-vectors // threads)

        @T.prim_func
        def gather(
            Dst: T.Tensor((total, row_width), out_dtype),
            Cache: T.Tensor(cache_shape, dtype),
            Table: T.Tensor((batch, table_width), "int32"),
            Cu: T.Tensor((batch + 1,), "int32"),
            Starts: T.Tensor((batch,) if has_starts else (batch + 1,), "int32"),
            Scale: T.Tensor((1,) if quantized else cache_shape, "float32" if quantized else dtype),
        ):
            with T.Kernel(T.ceildiv(total, block_rows), threads=threads) as bx:
                slots = T.alloc_shared((block_rows,), "int64")
                tx = T.get_thread_binding()
                _row_slots(
                    slots,
                    Cu,
                    Table,
                    Starts,
                    bx * block_rows,
                    total,
                    batch,
                    block_rows=block_rows,
                    page_size=page_size,
                    small_batch=small_batch,
                    has_starts=has_starts,
                )
                T.sync_threads()
                # Every load of a thread is issued before its first store.
                held = T.alloc_local((steps, vec), dtype)
                for s in T.unroll(steps):
                    v = s * threads + tx
                    if v < vectors and bx * block_rows + v // lanes < total:
                        for i in T.vectorized(vec):
                            held[s, i] = Cache[slots[v // lanes], v % lanes * vec + i]
                for s in T.unroll(steps):
                    v = s * threads + tx
                    row = bx * block_rows + v // lanes
                    if v < vectors and row < total:
                        if quantized:
                            wide = T.alloc_local((vec,), out_dtype)
                            T.call_extern(
                                "handle",
                                "tileops::dequant8<__half>"
                                if out_dtype == "float16"
                                else "tileops::dequant8<__nv_bfloat16>",
                                T.address_of(wide[0]),
                                T.address_of(held[s, 0]),
                                Scale[0],
                            )
                            for i in T.vectorized(vec):
                                Dst[row, v % lanes * vec + i] = wide[i]
                        else:
                            for i in T.vectorized(vec):
                                Dst[row, v % lanes * vec + i] = held[s, i]

        return gather

    return _func


@target_cache(maxsize=32)
def _paged_cache_gather_short_kernel(page_size: int, row_width: int, dtype: str, single_page: bool):
    threads = ((row_width + 255) // 256) * 32
    index_dtype = "int32"

    @tilelang.jit
    def build():
        total = T.dynamic("total")
        num_pages = T.dynamic("num_pages", "int64")
        table_width = T.dynamic("table_width")

        @T.prim_func
        def gather(
            dst: T.Tensor[(total, row_width), dtype],
            cache: T.Tensor[(num_pages * page_size, row_width), dtype],
            table: T.Tensor[(2, table_width), index_dtype],
            offsets: T.Tensor[(3,), index_dtype],
        ):
            with T.Kernel(total, threads=threads) as row:
                # With two requests, the first packed boundary identifies the owning row.
                # For a single page of packed rows, every request fits on its first page.
                boundary = offsets[1]
                request = T.cast(row >= boundary, "int32")
                start = T.if_then_else(row >= boundary, boundary, 0)
                position = row - start
                page = 0 if single_page else position // page_size
                offset = position if single_page else position % page_size
                slot = T.cast(table[request, page], "int64") * page_size + offset
                lane = T.get_thread_binding()
                if lane < row_width // 8:
                    for i in T.vectorized(8):
                        column = lane * 8 + i
                        dst[row, column] = cache[slot, column]

        return gather

    return build


class PagedKVCacheGatherShortKernel(Kernel, PagedKVCacheGatherFwdInterface):
    """Gather two short requests by assigning one block and only one page lookup per row."""

    supported_archs = [90]
    preferred_over = frozenset({"paged_cache_gather_rows_kernel"})

    @classmethod
    def refusal(cls, call: PagedKVCacheGatherCall) -> str | None:
        if call.batch != 2 or call.has_starts or not (0 < call.total_tokens <= 2 * call.page_size):
            return "requires two zero-start requests packing at most two pages of rows"
        if call.dtype not in (torch.float16, torch.bfloat16) or call.out_dtype != call.dtype:
            return "serves same-dtype fp16 or bf16 copies"
        if not (128 <= call.row_width <= 1024 and call.row_width % 8 == 0):
            return "requires a row width of 128–1024 divisible by eight elements"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: PagedKVCacheGatherCall) -> Entry:
        index = call.device_index
        args = (call.page_size, call.row_width, call.dtype, call.total_tokens <= call.page_size)
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(self, page_size, row_width, dtype, single_page, *, device_index=None):
        super().__init__(device_index=device_index)
        self.row_width = row_width
        self.dtype = dtype
        self.kernel = _paged_cache_gather_short_kernel(
            page_size, row_width, self.dtype_str, single_page
        )
        self.init_config()

    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None) -> None:
        self.program(lambda config: self.kernel(**config))(
            dst.view(dst.shape[0], self.row_width),
            cache.view(cache.shape[0] * cache.shape[1], self.row_width),
            block_table,
            cu_seq_lens,
        )


class PagedKVCacheGatherRowsKernel(Kernel, PagedKVCacheGatherFwdInterface):
    """Gather whole rows of 8-element vectors, a few rows per block, dequantizing FP8 in registers."""

    supported_archs = [90]

    @classmethod
    def refusal(cls, call: PagedKVCacheGatherCall) -> str | None:
        if call.row_width % _ROW_VECTOR:
            return f"requires rows of whole {_ROW_VECTOR}-element vectors"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: PagedKVCacheGatherCall) -> Entry:
        args = dict(
            page_size=call.page_size,
            row_width=call.row_width,
            dtype=call.dtype,
            out_dtype=call.out_dtype,
            has_starts=call.has_starts,
            small_batch=call.batch if 2 <= call.batch <= 8 else 0,
        )
        index = call.device_index
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self, page_size, row_width, dtype, out_dtype, has_starts, small_batch, *, device_index=None
    ):
        super().__init__(device_index=device_index)
        self.row_width, self.dtype = row_width, dtype
        self.kernel = _paged_cache_gather_rows_kernel(
            page_size,
            row_width,
            self.dtype_str,
            self.dtype_to_str(out_dtype),
            has_starts,
            small_batch,
        )
        self.init_config()

    @property
    def default_config(self):
        # Eight rows per block, and 32 bytes of loads in flight per thread below the 1024-thread
        # cap: two 16-bit or four FP8 vectors. Fewer bytes per thread leave the copies short of
        # bandwidth, and wider blocks hold fewer blocks per SM.
        rows = 8
        loads = 32 // (_ROW_VECTOR * self.dtype.itemsize)
        vectors = rows * (self.row_width // _ROW_VECTOR)
        return {"block_rows": rows, "threads": min(1024, vectors // loads)}

    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None) -> None:
        flat_cache = cache.view(cache.shape[0] * cache.shape[1], self.row_width)
        # Absent optional arguments reuse existing tensors as unread placeholders.
        self.program(lambda config: self.kernel(**config))(
            dst.view(dst.shape[0], self.row_width),
            flat_cache,
            block_table,
            cu_seq_lens,
            seq_starts if seq_starts is not None else cu_seq_lens,
            scale if scale is not None else flat_cache,
        )


class PagedKVCacheGatherKernel(Kernel, PagedKVCacheGatherFwdInterface):
    """One gather and optional dequantization stage, tiled over packed output rows."""

    general = True

    @classmethod
    def entry_for(cls, call: PagedKVCacheGatherCall) -> Entry:
        args = dict(
            page_size=call.page_size,
            row_width=call.row_width,
            dtype=call.dtype,
            out_dtype=call.out_dtype,
            has_starts=call.has_starts,
            small_batch=call.batch if call.arch == 90 and 2 <= call.batch <= 8 else 0,
        )
        index = call.device_index
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self,
        page_size,
        row_width,
        dtype,
        out_dtype,
        has_starts=False,
        config=None,
        device_index=None,
        small_batch=0,
    ):
        super().__init__(device_index=device_index)
        self.page_size, self.row_width = page_size, row_width
        self.dtype, self.out_dtype, self.has_starts = dtype, out_dtype, has_starts
        self.small_batch = small_batch
        self.kernel = _paged_cache_gather_kernel(
            page_size,
            row_width,
            self.dtype_str,
            self.dtype_to_str(out_dtype),
            has_starts,
            small_batch,
        )
        self.init_config(config)

    @property
    def default_config(self):
        # Narrow column tiles sustain the 16-bit copies of two requests at width 1024 and above.
        if self.small_batch == 2 and self.row_width >= 1024:
            return {"block_rows": 4, "block_cols": 256, "threads": 128}
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
            # Tune once on 131072 packed rows; runtime batch/pool/table extents do not
            # change the compiled program or trigger a new tuning problem.
            batch, length = self.small_batch or 8, 16384
            table_width = (length + int(self.has_starts) + self.page_size - 1) // self.page_size
            num_pages = batch * table_width
            cache = torch.zeros(
                (num_pages * self.page_size, self.row_width), dtype=self.dtype, device=device
            )
            cu = torch.arange(batch + 1, device=device, dtype=torch.int32) * length
            table = torch.arange(num_pages, device=device, dtype=torch.int32).view(
                batch, table_width
            )
            return [
                torch.empty((batch * length, self.row_width), dtype=self.out_dtype, device=device),
                cache,
                table,
                cu,
                torch.ones(batch, dtype=torch.int32, device=device) if self.has_starts else cu,
                torch.ones(1, dtype=torch.float32, device=device)
                if self.dtype == torch.float8_e4m3fn
                else cache,
            ]

        return supply

    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None) -> None:
        flat_dst = dst.view(dst.shape[0], self.row_width)
        flat_cache = cache.view(cache.shape[0] * cache.shape[1], self.row_width)
        # Absent optional arguments reuse existing tensors as unread placeholders.
        self.program(self._compile)(
            flat_dst,
            flat_cache,
            block_table,
            cu_seq_lens,
            seq_starts if self.has_starts else cu_seq_lens,
            scale if scale is not None else flat_cache,
        )

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)
