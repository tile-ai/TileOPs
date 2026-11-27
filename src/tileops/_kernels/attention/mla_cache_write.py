"""Write each token's MLA latent ``kv_c`` and rope key ``k_pe`` into the cache row its slot names.

An FP8 cache divides by the scale and uses a saturating cast that preserves NaN.
"""

import math
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import (
    MLAKVCacheWriteCall,
    MLAKVCacheWriteFwdInterface,
)
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import MAX_BLOCK_THREADS, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._utils import WARP_LANES

__all__ = ["MLAKVCacheWriteKernel"]


def _vector(columns: int, itemsize: int) -> int:
    """The elements of the widest access, at most 16 bytes, that tiles a row exactly."""
    return math.gcd(columns, VECTOR_ACCESS_BYTES // itemsize)


def _row_vectors(latent_dim: int, rope_dim: int, itemsize: int, rope: bool) -> tuple[int, int]:
    """The vectors of one row the copy moves: the latent's, and k_pe's unless it is rotated."""
    latent = latent_dim // _vector(latent_dim, itemsize)
    pe = 0 if rope else rope_dim // _vector(rope_dim, itemsize)
    return latent, pe


@target_cache(maxsize=32)
def _mla_cache_write_kernel(
    latent_dim: int,
    rope_dim: int,
    physical_tokens: int,
    in_dtype: str,
    cache_dtype: str,
    rope_layout: str | None,
    max_position: int,
    cos_sin_dtype: str,
) -> Callable:
    width = latent_dim + rope_dim
    half = rope_dim // 2
    quantized = cache_dtype == "float8_e4m3fn"
    rope = rope_layout is not None
    neox = rope_layout == "neox"
    # Elements of one rotation half a rope thread holds: four spread a 64-wide k_pe over
    # 16 threads with 8-byte accesses. An interleaved thread holds whole pairs, so its span
    # is even; a neox thread holds the same span of both halves.
    if neox:
        span = 4 if half % 4 == 0 else (2 if half % 2 == 0 else 1)
        rope_lanes = half // span
    else:
        span = 4 if rope_dim % 4 == 0 else 2
        rope_lanes = rope_dim // span
    itemsize = getattr(torch, in_dtype).itemsize
    latent_vector = _vector(latent_dim, itemsize)
    pe_vector = _vector(rope_dim, itemsize)
    latent_row, pe_row = _row_vectors(latent_dim, rope_dim, itemsize, rope)

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            # Two tokens write one row only if their slots are equal, and distinct slots
            # are the caller's obligation; the checker cannot see that and reports the
            # stores of a block's tokens as a race.
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        },
    )
    def _func(block_rows: int, threads: int) -> Callable:
        # The tokens written are a dynamic extent: a decode step writes one row per
        # running request, and a kernel built per token count would compile every step.
        tokens = T.dynamic("tokens")
        rope_items = block_rows * rope_lanes
        rope_rounds = (rope_items + threads - 1) // threads
        # k_pe's vectors are numbered after the latent's, so they start on the threads the
        # latent left idle.
        latent_items = block_rows * latent_row
        pe_items = block_rows * pe_row

        def _quantize(value, scale):
            # Division, not a multiply by the reciprocal: the two round differently before the
            # FP8 cast, and the stored code must be that of value / scale.
            return value / scale if quantized else value

        # TileLang moves a span as one vector where it proves the offset aligned in every
        # row, and element by element where it does not.
        @T.macro
        def _span_load(dst, src, row, start, count):
            for v in T.vectorized(count):
                dst[v] = src[row, start + v]

        @T.macro
        def _span_store(cache, slot, start, values, scale):
            for v in T.vectorized(span):
                cache[slot, start + v] = _quantize(values[v], scale)

        @T.macro
        def _rotate(KPe, Cache, Slots, Positions, CosSin, scale, bx, tx):
            x = T.alloc_local((span,), in_dtype)
            y = T.alloc_local((span,), in_dtype)
            cos = T.alloc_local((span,), cos_sin_dtype)
            sin = T.alloc_local((span,), cos_sin_dtype)
            out_x = T.alloc_local((span,), "float32")
            out_y = T.alloc_local((span,), "float32")
            for rnd in T.serial(rope_rounds):
                item = rnd * threads + tx
                token = bx * block_rows + item // rope_lanes
                lane = item % rope_lanes
                if item < rope_items and token < tokens:
                    pos = Positions[token]
                    if neox:
                        # Pair i is (k_pe[i], k_pe[i + half]): the thread holds a span of
                        # each half and the angles of that span.
                        _span_load(x, KPe, token, lane * span, span)
                        _span_load(y, KPe, token, half + lane * span, span)
                        _span_load(cos, CosSin, pos, lane * span, span)
                        _span_load(sin, CosSin, pos, half + lane * span, span)
                    else:
                        # Pair i is (k_pe[2i], k_pe[2i + 1]): the thread's span holds
                        # span / 2 whole pairs and needs that many angles.
                        _span_load(x, KPe, token, lane * span, span)
                        _span_load(cos, CosSin, pos, lane * (span // 2), span // 2)
                        _span_load(sin, CosSin, pos, half + lane * (span // 2), span // 2)
                    slot = Slots[token]
                    if slot >= 0:
                        # Each product and the sum round once in float32, as the
                        # reference evaluates them; a fused multiply-add would round once
                        # less and move some results by one step of the stored dtype.
                        if neox:
                            for v in T.unroll(span):
                                a = T.Cast("float32", x[v])
                                b = T.Cast("float32", y[v])
                                c = T.Cast("float32", cos[v])
                                s = T.Cast("float32", sin[v])
                                rot_a = T.ieee_sub(T.ieee_mul(a, c), T.ieee_mul(b, s))
                                rot_b = T.ieee_add(T.ieee_mul(b, c), T.ieee_mul(a, s))
                                out_x[v] = T.Cast("float32", T.Cast(in_dtype, rot_a))
                                out_y[v] = T.Cast("float32", T.Cast(in_dtype, rot_b))
                            _span_store(Cache, slot, latent_dim + lane * span, out_x, scale)
                            _span_store(Cache, slot, latent_dim + half + lane * span, out_y, scale)
                        else:
                            for p in T.unroll(span // 2):
                                a = T.Cast("float32", x[2 * p])
                                b = T.Cast("float32", x[2 * p + 1])
                                c = T.Cast("float32", cos[p])
                                s = T.Cast("float32", sin[p])
                                rot_a = T.ieee_sub(T.ieee_mul(a, c), T.ieee_mul(b, s))
                                rot_b = T.ieee_add(T.ieee_mul(b, c), T.ieee_mul(a, s))
                                out_x[2 * p] = T.Cast("float32", T.Cast(in_dtype, rot_a))
                                out_x[2 * p + 1] = T.Cast("float32", T.Cast(in_dtype, rot_b))
                            _span_store(Cache, slot, latent_dim + lane * span, out_x, scale)

        def _rounds(first, items):
            # The rounds of the thread loop whose items fall in [first, first + items).
            return first // threads, (first + items + threads - 1) // threads

        def _depth(first, items):
            # Vectors of the range a thread holds, one a round; an empty range allocates one.
            low, high = _rounds(first, items)
            return max(1, high - low)

        latent_depth = _depth(0, latent_items)
        pe_depth = _depth(latent_items, pe_items)
        # Each copied range: (columns, offset in the cache row, vector, first item, items).
        latent_range = (latent_dim, 0, latent_vector, 0, latent_items)
        pe_range = (rope_dim, latent_dim, pe_vector, latent_items, pe_items)

        def _place(bx, tx, rnd, columns, vector, first):
            # The range's item a thread takes in a round, the item's token and first column.
            item = rnd * threads + tx - first
            vectors = columns // vector
            return item, bx * block_rows + item // vectors, item % vectors * vector

        @T.macro
        def _gather(Src, held, bx, tx, rng):
            columns, _, vector, first, items = rng
            low, high = _rounds(first, items)
            for rnd in T.serial(low, high):
                item, token, column = _place(bx, tx, rnd, columns, vector, first)
                for v in T.vectorized(vector):
                    held[rnd - low, v] = T.if_then_else(
                        item >= 0 and item < items and token < tokens,
                        Src[token, column + v],
                        T.cast(0, in_dtype),
                    )

        @T.macro
        def _scatter(held, Cache, Slots, scale, bx, tx, rng):
            columns, offset, vector, first, items = rng
            low, high = _rounds(first, items)
            for rnd in T.serial(low, high):
                item, token, column = _place(bx, tx, rnd, columns, vector, first)
                if item >= 0 and item < items and token < tokens:
                    slot = Slots[token]
                    if slot >= 0:
                        for v in T.vectorized(vector):
                            if quantized:
                                # TileLang's FP8 cast saturates finite overflow and
                                # infinities while preserving NaN.
                                Cache[slot, offset + column + v] = _quantize(
                                    T.Cast("float32", held[rnd - low, v]), scale
                                )
                            else:
                                Cache[slot, offset + column + v] = held[rnd - low, v]

        @T.macro
        def _write(KvC, KPe, Cache, Slots, Scale, Positions, CosSin):
            with T.Kernel(T.ceildiv(tokens, block_rows), threads=threads) as bx:
                tx = T.get_thread_binding()
                scale = Scale[0] if quantized else None
                # Every vector a thread owns is loaded before any slot is read, so the
                # loads are in flight together and none waits on a slot.
                latent = T.alloc_local((latent_depth, latent_vector), in_dtype)
                pe = T.alloc_local((pe_depth, pe_vector), in_dtype)
                if latent_items > 0:
                    _gather(KvC, latent, bx, tx, latent_range)
                if pe_items > 0:
                    _gather(KPe, pe, bx, tx, pe_range)
                if latent_items > 0:
                    _scatter(latent, Cache, Slots, scale, bx, tx, latent_range)
                if pe_items > 0:
                    _scatter(pe, Cache, Slots, scale, bx, tx, pe_range)
                if rope and rope_dim > 0:
                    _rotate(KPe, Cache, Slots, Positions, CosSin, scale, bx, tx)

        if rope and quantized:

            @T.prim_func
            def _mla_cache_write_fp8_rope(
                KvC: T.Tensor((tokens, latent_dim), in_dtype),  # type: ignore
                KPe: T.Tensor((tokens, rope_dim), in_dtype),  # type: ignore
                Cache: T.Tensor((physical_tokens, width), cache_dtype),  # type: ignore
                Slots: T.Tensor((tokens,), "int64"),  # type: ignore
                Scale: T.Tensor((1,), "float32"),  # type: ignore
                Positions: T.Tensor((tokens,), "int64"),  # type: ignore
                CosSin: T.Tensor((max_position, rope_dim), cos_sin_dtype),  # type: ignore
            ) -> None:
                _write(KvC, KPe, Cache, Slots, Scale, Positions, CosSin)

            return _mla_cache_write_fp8_rope

        if rope:

            @T.prim_func
            def _mla_cache_write_rope(
                KvC: T.Tensor((tokens, latent_dim), in_dtype),  # type: ignore
                KPe: T.Tensor((tokens, rope_dim), in_dtype),  # type: ignore
                Cache: T.Tensor((physical_tokens, width), cache_dtype),  # type: ignore
                Slots: T.Tensor((tokens,), "int64"),  # type: ignore
                Positions: T.Tensor((tokens,), "int64"),  # type: ignore
                CosSin: T.Tensor((max_position, rope_dim), cos_sin_dtype),  # type: ignore
            ) -> None:
                _write(KvC, KPe, Cache, Slots, None, Positions, CosSin)

            return _mla_cache_write_rope

        if quantized:

            @T.prim_func
            def _mla_cache_write_fp8(
                KvC: T.Tensor((tokens, latent_dim), in_dtype),  # type: ignore
                KPe: T.Tensor((tokens, rope_dim), in_dtype),  # type: ignore
                Cache: T.Tensor((physical_tokens, width), cache_dtype),  # type: ignore
                Slots: T.Tensor((tokens,), "int64"),  # type: ignore
                Scale: T.Tensor((1,), "float32"),  # type: ignore
            ) -> None:
                _write(KvC, KPe, Cache, Slots, Scale, None, None)

            return _mla_cache_write_fp8

        @T.prim_func
        def _mla_cache_write_16bit(
            KvC: T.Tensor((tokens, latent_dim), in_dtype),  # type: ignore
            KPe: T.Tensor((tokens, rope_dim), in_dtype),  # type: ignore
            Cache: T.Tensor((physical_tokens, width), cache_dtype),  # type: ignore
            Slots: T.Tensor((tokens,), "int64"),  # type: ignore
        ) -> None:
            _write(KvC, KPe, Cache, Slots, None, None, None)

        return _mla_cache_write_16bit

    return _func


class MLAKVCacheWriteKernel(Kernel, MLAKVCacheWriteFwdInterface):
    """One pass over the tokens, a few whole rows per block."""

    @classmethod
    def entry_for(cls, call: MLAKVCacheWriteCall) -> Entry:
        args = dict(
            latent_dim=call.latent_dim,
            rope_dim=call.rope_dim,
            num_pages=call.num_pages,
            page_size=call.page_size,
            dtype=call.dtype,
            cache_dtype=call.cache_dtype,
            rope_layout=call.rope_layout if call.fuse_rope else None,
            cos_sin_dtype=call.cos_sin_dtype if call.fuse_rope else None,
            max_position=call.max_position if call.fuse_rope else 0,
        )
        index = call.device_index
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self,
        latent_dim: int,
        rope_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype,
        rope_layout: str | None = None,
        cos_sin_dtype: torch.dtype | None = None,
        max_position: int = 0,
        config: dict | None = None,
        device_index: int | None = None,
    ):
        """Build the program for one cache geometry and RoPE form.

        Args:
            latent_dim: Elements of ``kv_c`` a row holds.
            rope_dim: Elements of ``k_pe`` a row holds.
            num_pages: Pages in the cache.
            page_size: Rows in a page.
            dtype: Torch dtype of ``kv_c`` and ``k_pe``.
            cache_dtype: Torch dtype of the cache; float8_e4m3fn quantizes on write.
            rope_layout: ``"neox"`` or ``"interleaved"`` to rotate ``k_pe`` as it is
                written, or None to copy it.
            cos_sin_dtype: Torch dtype of ``cos_sin_cache`` when rotating.
            max_position: Rows of ``cos_sin_cache`` when rotating.
            config: Rows per block and thread count, or None for ``default_config``.
            device_index: The device the program is built for, or None for the current one.
        """
        super().__init__(device_index=device_index)
        self.latent_dim, self.rope_dim = latent_dim, rope_dim
        self.dtype, self.cache_dtype = dtype, cache_dtype
        self.rope_layout, self.cos_sin_dtype = rope_layout, cos_sin_dtype
        self.max_position = max_position
        self.row_width = latent_dim + rope_dim
        self.physical_tokens = num_pages * page_size
        self.quantized = cache_dtype == torch.float8_e4m3fn
        self.kernel = _mla_cache_write_kernel(
            latent_dim,
            rope_dim,
            self.physical_tokens,
            self.dtype_str,
            self.dtype_to_str(cache_dtype),
            rope_layout,
            max_position,
            self.dtype_to_str(cos_sin_dtype) if cos_sin_dtype is not None else "float32",
        )
        self.init_config(config)

    @property
    def default_config(self):
        # Two tokens a block, more of them leave a decode-sized call too few blocks. A
        # thread for each vector the block copies, in whole warps: a thread that owns two
        # stores the second after the first, and reads its slot only then.
        rows = 2
        vectors = sum(
            _row_vectors(
                self.latent_dim, self.rope_dim, self.dtype.itemsize, self.rope_layout is not None
            )
        )
        warps = -(-rows * vectors // WARP_LANES)
        return {"block_rows": rows, "threads": min(MAX_BLOCK_THREADS, max(1, warps) * WARP_LANES)}

    @property
    def autotune_configs(self):
        return [
            dict(block_rows=r, threads=t)
            for r, t in [(1, 64), (2, 128), (4, 256), (8, 512), (2, 64), (4, 128)]
        ]

    @property
    def autotune_supply_prog(self):
        def supply(params):
            device = torch.device("cuda", self.device_index)
            # A negative slot skips its token, so the slots a candidate is timed on must
            # name rows: the tuner's own randint(-2, 3) would time a collapsed write.
            tokens = min(32, self.physical_tokens)
            kv_c = torch.zeros((tokens, self.latent_dim), dtype=self.dtype, device=device)
            k_pe = torch.zeros((tokens, self.rope_dim), dtype=self.dtype, device=device)
            cache = torch.zeros(
                (self.physical_tokens, self.row_width), dtype=self.cache_dtype, device=device
            )
            args = [kv_c, k_pe, cache, torch.arange(tokens, dtype=torch.int64, device=device)]
            if self.quantized:
                args.append(torch.ones(1, dtype=torch.float32, device=device))
            if self.rope_layout is not None:
                positions = torch.arange(tokens, dtype=torch.int64, device=device)
                cos_sin = torch.zeros(
                    (self.max_position, self.rope_dim), dtype=self.cos_sin_dtype, device=device
                )
                args += [positions % self.max_position, cos_sin]
            return args

        return supply

    def forward(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        scale: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        cos_sin_cache: torch.Tensor | None = None,
    ) -> None:
        # A call of no tokens writes nothing. The cache it is handed still holds elements,
        # so the generated empty-call rule does not skip the launch, and a zero-extent grid
        # is not a legal one; the launch is skipped here instead.
        if kv_c.shape[0] == 0:
            return
        # A slot indexes the cache as one flat run of rows, so the kernel reads the
        # three-dimensional cache the manifest declares as rows of row_width elements.
        args = [kv_c, k_pe, kv_cache.view(self.physical_tokens, self.row_width), slot_mapping]
        if self.quantized:
            args.append(scale)
        if self.rope_layout is not None:
            args += [positions, cos_sin_cache]
        self.program(self._compile)(*args)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)
