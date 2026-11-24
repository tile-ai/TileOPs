"""Scatter new K/V rows into the page pool their slots name.

A thread owns one vector of a token's row: it loads the token's slot and the vector of the
key and of the value row together, then stores both where the slot names, or nothing for a
slot of ``-1``. An FP8 cache divides by the scale and uses a saturating cast that preserves
NaN.
"""

import math
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._kernels.attention.call_spec import (
    PagedKVCacheWriteCall,
    PagedKVCacheWriteFwdInterface,
)
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = ["PagedKVCacheWriteKernel"]


@target_cache(maxsize=32)
def _paged_cache_write_kernel(
    row_width: int,
    physical_tokens: int,
    in_dtype: str,
    cache_dtype: str,
) -> Callable:
    quantized = cache_dtype == "float8_e4m3fn"
    # A thread loads one 16-byte vector of an input row, or the widest smaller power of
    # two that divides the row, so no vector straddles two rows.
    in_bytes = DataType(in_dtype).bits // 8
    vec = math.gcd(row_width, VECTOR_ACCESS_BYTES // in_bytes)
    lanes = row_width // vec
    # A vector of 4, 8 or 16 bytes moves through PTX vector loads and predicated stores;
    # a narrower one keeps element loads and a branched store.
    loaded_bytes = vec * in_bytes
    stored_bytes = vec * DataType(cache_dtype).bits // 8

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            # Loads are bounded by the token clamp and stores by the slot predicate, which
            # tests both ends of the pool; the guard would split the loads into branches
            # that each recompute the address.
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        },
    )
    def _func(threads: int) -> Callable:
        # The tokens written are a dynamic extent: a decode step writes one row per
        # running request, and a kernel built per token count would compile every step.
        # It is 64-bit because the vector count, tokens * lanes, can pass 2**31.
        tokens = T.dynamic("tokens", "int64")

        @T.macro
        def write_rows(K, V, KPages, VPages, Slots, KScale, VScale):
            load = {16: T.ldg128, 8: T.ldg64, 4: T.ldg32}.get(loaded_bytes)
            store, word_dtype = {
                16: (T.stg128, "uint32x4"),
                8: (T.stg64, "uint32x2"),
                4: (T.stg32, "uint32"),
            }.get(stored_bytes, (None, None))
            with T.Kernel(T.ceildiv(tokens * lanes, threads), threads=threads) as bx:
                vector = T.cast(bx, "int64") * threads + T.get_thread_binding()
                # A thread past the last token loads the last token's row and stores
                # nothing, so every load is unconditional.
                token = T.min(vector // lanes, tokens - 1)
                col = vector % lanes * vec
                slot = Slots[token]
                # A negative slot skips the token; a slot past the pool writes nothing.
                named = vector // lanes < tokens and slot >= 0 and slot < physical_tokens
                rows = T.alloc_local((2, vec), in_dtype)
                stored = T.alloc_local((2, vec), cache_dtype)
                # Both rows are loaded before either store. ptxas sinks a non-coherent
                # (`.nc`) load whose only use is a predicated store below the slot test,
                # which serializes the slot load and the row loads; a coherent load may
                # not pass the other row's store, so the slot and both rows are in flight
                # together and a thread waits on memory once.
                for r, src in ((0, K), (1, V)):
                    if load is not None:
                        rows[r, T.Ramp(0, 1, vec)] = T.reinterpret(
                            load(src[token, col]), f"{in_dtype}x{vec}"
                        )
                    else:
                        for i in T.vectorized(vec):
                            rows[r, i] = src[token, col + i]
                for r, scale in ((0, KScale), (1, VScale)):
                    for i in T.vectorized(vec):
                        if quantized:
                            # TileLang's FP8 cast saturates finite overflow and
                            # infinities while preserving NaN.
                            stored[r, i] = T.Cast("float32", rows[r, i]) / scale[0]
                        else:
                            stored[r, i] = rows[r, i]
                for r, pages in ((0, KPages), (1, VPages)):
                    if store is not None:
                        held = T.BufferLoad(stored, [r, T.Ramp(0, 1, vec)])
                        store(pages[slot, col], T.reinterpret(held, word_dtype), pred=named)
                    elif named:
                        for i in T.vectorized(vec):
                            pages[slot, col + i] = stored[r, i]

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
                write_rows(K, V, KPages, VPages, Slots, KScale, VScale)

            return _paged_cache_write_fp8

        @T.prim_func
        def _paged_cache_write_16bit(
            K: T.Tensor((tokens, row_width), in_dtype),  # type: ignore
            V: T.Tensor((tokens, row_width), in_dtype),  # type: ignore
            KPages: T.Tensor((physical_tokens, row_width), cache_dtype),  # type: ignore
            VPages: T.Tensor((physical_tokens, row_width), cache_dtype),  # type: ignore
            Slots: T.Tensor((tokens,), "int64"),  # type: ignore
        ) -> None:
            write_rows(K, V, KPages, VPages, Slots, None, None)

        return _paged_cache_write_16bit

    return _func


class PagedKVCacheWriteKernel(Kernel, PagedKVCacheWriteFwdInterface):
    """One vector of a token's key and value rows per thread, in one pass over the tokens."""

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
        index = call.device_index
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self,
        heads_kv: int,
        dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype,
        config: dict | None = None,
        device_index: int | None = None,
    ):
        """Build the program for one page pool geometry.

        Args:
            heads_kv: Key/value heads one token's row holds.
            dim: Elements per head in a row.
            num_pages: Pages in the pool.
            page_size: Rows in a page.
            dtype: Torch dtype of the rows the call passes in.
            cache_dtype: Torch dtype of the page pool; float8_e4m3fn quantizes on write.
            config: Threads per block, or None for ``default_config``.
            device_index: The device the program is built for, or None for the current one.
        """
        super().__init__(device_index=device_index)
        self.dtype, self.cache_dtype = dtype, cache_dtype
        self.row_width = heads_kv * dim
        self.physical_tokens = num_pages * page_size
        self.quantized = cache_dtype == torch.float8_e4m3fn
        self.kernel = _paged_cache_write_kernel(
            self.row_width,
            self.physical_tokens,
            self.dtype_str,
            self.dtype_to_str(cache_dtype),
        )
        self.init_config(config)

    @property
    def default_config(self):
        # Four warps per block: a block covers 128 vectors, 2 KiB of 16-bit rows, so even a
        # short decode step spreads over many SMs.
        return {"threads": 128}

    @property
    def autotune_configs(self):
        return [dict(threads=t) for t in (64, 128, 256, 512)]

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
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
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
        self.program(self._compile)(*args)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)
