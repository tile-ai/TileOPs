"""Scatter new K/V rows into the page pool their slots name.

One block owns a tile of tokens and a column tile of their rows. A token whose slot is
``-1`` is skipped; every other token reads its slot once and writes the same columns of
its key and value rows, so the guard and the slot load are paid once for the two stores.
An FP8 cache divides by the scale and uses a saturating cast that preserves NaN.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import (
    PagedKVCacheWriteCall,
    PagedKVCacheWriteFwdInterface,
)
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["PagedKVCacheWriteKernel"]


@functools.lru_cache(maxsize=32)
def _paged_cache_write_kernel(
    heads_kv: int,
    dim: int,
    physical_tokens: int,
    in_dtype: str,
    cache_dtype: str,
) -> Callable:
    row_width = heads_kv * dim
    quantized = cache_dtype == "float8_e4m3fn"

    @tilelang.jit(out_idx=[])
    def _func(block_rows: int, block_cols: int, threads: int) -> Callable:
        # The tokens written are a dynamic extent: a decode step writes one row per
        # running request, and a kernel built per token count would compile every step.
        tokens = T.dynamic("tokens")

        if quantized:

            @T.prim_func
            def _paged_cache_write_fp8(
                K: T.Tensor((tokens, row_width), in_dtype),  # type: ignore
                V: T.Tensor((tokens, row_width), in_dtype),  # type: ignore
                KPages: T.Tensor((physical_tokens, row_width), cache_dtype),  # type: ignore
                VPages: T.Tensor((physical_tokens, row_width), cache_dtype),  # type: ignore
                Slots: T.Tensor((tokens,), "int64"),  # type: ignore
                KScale: T.Tensor((1,), "float32"),  # type: ignore
                VScale: T.Tensor((1,), "float32"),  # type: ignore
            ) -> None:
                with T.Kernel(
                    T.ceildiv(tokens, block_rows),
                    T.ceildiv(row_width, block_cols),
                    threads=threads,
                ) as (row_tile, col_tile):
                    for i, c in T.Parallel(block_rows, block_cols):
                        token = row_tile * block_rows + i
                        col = col_tile * block_cols + c
                        if token < tokens and col < row_width:
                            slot = Slots[token]
                            if slot >= 0:
                                # TileLang's FP8 cast saturates finite overflow and
                                # infinities while preserving NaN.
                                KPages[slot, col] = T.Cast("float32", K[token, col]) / KScale[0]
                                VPages[slot, col] = T.Cast("float32", V[token, col]) / VScale[0]

            return _paged_cache_write_fp8

        @T.prim_func
        def _paged_cache_write_16bit(
            K: T.Tensor((tokens, row_width), in_dtype),  # type: ignore
            V: T.Tensor((tokens, row_width), in_dtype),  # type: ignore
            KPages: T.Tensor((physical_tokens, row_width), cache_dtype),  # type: ignore
            VPages: T.Tensor((physical_tokens, row_width), cache_dtype),  # type: ignore
            Slots: T.Tensor((tokens,), "int64"),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(tokens, block_rows),
                T.ceildiv(row_width, block_cols),
                threads=threads,
            ) as (row_tile, col_tile):
                for i, c in T.Parallel(block_rows, block_cols):
                    token = row_tile * block_rows + i
                    col = col_tile * block_cols + c
                    if token < tokens and col < row_width:
                        slot = Slots[token]
                        if slot >= 0:
                            KPages[slot, col] = K[token, col]
                            VPages[slot, col] = V[token, col]

        return _paged_cache_write_16bit

    return _func


class PagedKVCacheWriteKernel(Kernel, PagedKVCacheWriteFwdInterface):
    """One pass over the tokens, a row tile and a column tile per block."""

    @classmethod
    def entry_for(cls, call: PagedKVCacheWriteCall) -> Entry:
        args = dict(
            heads_kv=call.heads_kv,
            dim=call.dim,
            num_pages=call.num_pages,
            page_size=call.page_size,
            dtype=call.dtype,
            cache_dtype=call.cache_dtype,
        )
        index = call.device.index if call.device is not None else None
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self,
        heads_kv: int,
        dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        """Build the program for one page pool geometry.

        Args:
            heads_kv: Key/value heads one token's row holds.
            dim: Elements per head in a row.
            num_pages: Pages in the pool.
            page_size: Rows in a page.
            dtype: Torch dtype of the rows the call passes in.
            cache_dtype: Torch dtype of the page pool; float8_e4m3fn quantizes on write.
            config: Tile sizes and thread count, or None for ``default_config``.
            device_index: The device the program is built for, or None for the current one.
        """
        super().__init__(device_index=device_index)
        self.heads_kv, self.dim = heads_kv, dim
        self.num_pages, self.page_size = num_pages, page_size
        self.dtype, self.cache_dtype = dtype, cache_dtype
        self.row_width = heads_kv * dim
        self.physical_tokens = num_pages * page_size
        self.quantized = cache_dtype == torch.float8_e4m3fn
        self.kernel = _paged_cache_write_kernel(
            heads_kv,
            dim,
            self.physical_tokens,
            self.dtype_str,
            self.dtype_to_str(cache_dtype),
        )
        self.init_config(config)

    @property
    def default_config(self):
        # A modest row tile exposes parallelism on small pools; a 128-column tile
        # keeps contiguous stores while masking narrower rows.
        return {"block_rows": 16, "block_cols": 128, "threads": 128}

    @property
    def autotune_configs(self):
        return [
            dict(block_rows=r, block_cols=c, threads=t)
            for r, c, t in [
                (16, 128, 128),
                (32, 128, 128),
                (8, 256, 128),
                (4, 512, 128),
                (2, 1024, 128),
                (16, 256, 256),
                (8, 512, 256),
                (4, 1024, 256),
            ]
        ]

    @property
    def autotune_supply_prog(self):
        def supply(params):
            device = torch.device("cuda", self.device_index)
            # A negative slot skips its token, so the slots a candidate is timed on must
            # name rows: the tuner's own randint(-2, 3) would time a collapsed write.
            tokens = min(32, self.physical_tokens)
            rows = torch.zeros((tokens, self.row_width), dtype=self.dtype, device=device)
            pages = torch.zeros(
                (self.physical_tokens, self.row_width), dtype=self.cache_dtype, device=device
            )
            slots = torch.arange(tokens, dtype=torch.int64, device=device) % self.physical_tokens
            if not self.quantized:
                return [rows, rows.clone(), pages, pages.clone(), slots]
            scale = torch.ones(1, dtype=torch.float32, device=device)
            return [rows, rows.clone(), pages, pages.clone(), slots, scale, scale.clone()]

        return supply

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Write the rows the slots name into both pools; see the interface for the contract."""
        # A call of no tokens writes nothing. The page pools it is handed still hold
        # elements, so the generated empty-call rule does not skip the launch, and a
        # zero-extent grid is not a legal one; the launch is skipped here instead.
        if k.shape[0] == 0:
            return
        # A slot indexes the page pool as one flat run of rows, so the kernel reads the
        # four-dimensional tensors the manifest declares as rows of row_width elements.
        args = [
            k.view(k.shape[0], self.row_width),
            v.view(v.shape[0], self.row_width),
            k_pages.view(self.physical_tokens, self.row_width),
            v_pages.view(self.physical_tokens, self.row_width),
            slot_mapping,
        ]
        if self.quantized:
            args += [k_scale, v_scale]
        self.kernel(**self.config)(*args)
