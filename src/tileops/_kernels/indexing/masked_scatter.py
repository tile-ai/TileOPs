"""The masked scatter: a row-mask prefix sum, then a copy of whole rows."""

import math
from collections.abc import Callable
from typing import Any

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.indexing.call_spec import MaskedScatterCall, MaskedScatterFwdInterface
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = ["MaskedScatterKernel"]


def _rows(
    input_shape: tuple[int, ...], mask_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    """The broadcast shape of input and mask, the mask's shape over its row axes, and the
    output's row count and row width.

    The row axes are the leading axes up to the last one the mask has extent above 1 on.
    Along the axes after them the mask has extent 1, so each row of their product is all
    true or all false. The signature has checked that the shapes broadcast.
    """
    n = max(len(input_shape), len(mask_shape))
    a = (1,) * (n - len(input_shape)) + tuple(input_shape)
    m = (1,) * (n - len(mask_shape)) + tuple(mask_shape)
    lead = n
    while lead > 0 and m[lead - 1] == 1:
        lead -= 1
    out_shape = tuple(y if x == 1 else x for x, y in zip(a, m, strict=True))
    return out_shape, m[:lead], math.prod(out_shape[:lead]), math.prod(out_shape[lead:])


# The copy CTA's threads and the vectors each moves. `_tile` sizes the tile from both, the copy
# program launches the threads, and the tile-count program relies on a tile of at most their
# product rows; four vectors give each thread four loads in flight before its stores.
_COPY_THREADS, _VECTORS_PER_THREAD = 128, 4


def _tile(cols: int, itemsize: int) -> tuple[int, int, int]:
    """The vector width and the rows and columns of a copy CTA's tile, for rows of *cols*.

    A vector is the most elements up to 16 bytes that *cols* divides, so every row starts on
    its boundary. The tile holds a vector for each thread's every load: up to 64 vectors of a
    row, and as many rows as fill the rest.
    """
    vec = 1
    while vec * 2 * itemsize <= 16 and cols % (vec * 2) == 0:
        vec *= 2
    vecs = 1 << min(6, max(0, (cols // vec).bit_length() - 1))
    return vec, _COPY_THREADS * _VECTORS_PER_THREAD // vecs, vecs * vec


@T.macro
def _block_exclusive_scan(x, sums, warps):
    """Turn each thread's ``x[0]`` into the sum of ``x[0]`` over the threads before it.

    ``sums[warps - 1]`` holds the block's total afterwards.
    """
    tx = T.get_thread_binding()
    lane = tx % 32
    warp = tx // 32
    own = T.alloc_local((1,), "int32")
    own[0] = x[0]
    for d in T.unroll(5):
        y = T.shfl_up(x[0], 1 << d)
        if lane >= (1 << d):
            x[0] += y
    if lane == 31:
        sums[warp] = x[0]
    T.sync_threads()
    if warp == 0:
        s = T.alloc_local((1,), "int32")
        s[0] = T.if_then_else(lane < warps, sums[lane], 0)
        for d in T.unroll(5):
            y = T.shfl_up(s[0], 1 << d)
            if lane >= (1 << d):
                s[0] += y
        if lane < warps:
            sums[lane] = s[0]
    T.sync_threads()
    x[0] -= own[0]
    if warp > 0:
        x[0] += sums[warp - 1]


@target_cache(maxsize=4)
def _tile_counts_kernel() -> Callable:
    """The program writing each row tile's count of true rows."""

    @tilelang.jit(out_idx=[])
    def _func(block_rows):
        # Eight warps: a tile of at most _COPY_THREADS * _VECTORS_PER_THREAD = 512 rows spans
        # at most one warp's 16-byte loads, so a CTA of whole warps never splits a tile.
        threads = 256
        # A thread loads up to one 16-byte vector of mask bytes; the threads of one tile are
        # consecutive lanes of a warp.
        per = min(block_rows, 16)
        group = block_rows // per
        rows = T.dynamic("rows")
        tiles = T.dynamic("tiles")

        @T.prim_func
        def count_tiles(mask: T.Tensor((rows,), "uint8"), counts: T.Tensor((tiles,), "int32")):
            with T.Kernel(T.ceildiv(tiles * group, threads), threads=threads) as bx:
                part = bx * threads + T.get_thread_binding()
                first = part * per
                b = T.alloc_local((per,), "uint8")
                s = T.alloc_var("int32", init=0)
                if first + per <= rows:
                    for v in T.vectorized(per):
                        b[v] = mask[first + v]
                else:
                    for v in T.unroll(per):
                        b[v] = T.if_then_else(first + v < rows, mask[first + v], T.uint8(0))
                for v in T.unroll(per):
                    s += T.cast(b[v] != 0, "int32")
                for d in T.unroll(group.bit_length() - 1):
                    s += T.shfl_xor(s, 1 << d)
                if part % group == 0 and part // group < tiles:
                    counts[part // group] = s

        return count_tiles

    return _func


@target_cache(maxsize=4)
def _tile_offsets_kernel() -> Callable:
    """The one-CTA program writing the exclusive prefix sum of the tile counts."""

    @tilelang.jit(out_idx=[])
    def _func():
        # One CTA scans every count, so it takes a block's most threads; four counts a thread
        # cover 4096 tiles a pass, 32768 rows of 8-row tiles.
        threads, items = 1024, 4
        tiles = T.dynamic("tiles")
        warps = threads // 32
        chunk = threads * items

        @T.prim_func
        def scan_tiles(counts: T.Tensor((tiles,), "int32"), offsets: T.Tensor((tiles,), "int32")):
            with T.Kernel(1, threads=threads):
                tx = T.get_thread_binding()
                sums = T.alloc_shared((warps,), "int32")
                c = T.alloc_local((items,), "int32")
                x = T.alloc_local((1,), "int32")
                carry = T.alloc_var("int32", init=0)
                for k in T.serial(T.ceildiv(tiles, chunk)):
                    first = k * chunk + tx * items
                    x[0] = 0
                    for j in T.unroll(items):
                        c[j] = T.if_then_else(first + j < tiles, counts[first + j], 0)
                        x[0] += c[j]
                    _block_exclusive_scan(x, sums, warps)
                    run = T.alloc_var("int32")
                    run = carry + x[0]
                    for j in T.unroll(items):
                        if first + j < tiles:
                            offsets[first + j] = run
                        run += c[j]
                    carry += sums[warps - 1]
                    T.sync_threads()

        return scan_tiles

    return _func


@target_cache(maxsize=32)
def _row_copy_kernel(vec: int, dtype: str) -> Callable:
    """The program writing each output row from source or input, rows a CTA tile, for rows
    whose width *vec* divides."""

    @tilelang.jit(out_idx=[])
    def _func(block_rows, block_cols):
        threads = _COPY_THREADS
        rows = T.dynamic("rows")
        cols = T.dynamic("cols")
        tiles = T.dynamic("tiles")
        src_rows = T.dynamic("src_rows")
        warps = threads // 32
        vecs = block_cols // vec
        steps = block_rows * vecs // threads
        # Consecutive rows a thread takes in the tile's own scan.
        items = max(1, block_rows // threads)

        @T.prim_func
        def copy_rows(
            input: T.Tensor((rows, cols), dtype),
            mask: T.Tensor((rows,), "uint8"),
            offsets: T.Tensor((tiles,), "int32"),
            source: T.Tensor((src_rows, cols), dtype),
            out: T.Tensor((rows, cols), dtype),
        ):
            T.assume(cols % vec == 0)
            col_tiles = T.ceildiv(cols, block_cols)
            # CUDA holds at most 65535 CTAs on grid y; a CTA takes every 65535th column tile
            # past that.
            grid_y = T.min(col_tiles, 65535)
            with T.Kernel(T.ceildiv(rows, block_rows), grid_y, threads=threads) as (bx, by):
                tx = T.get_thread_binding()
                sums = T.alloc_shared((warps,), "int32")
                # The source row of each of the tile's rows, -1 for a row input keeps.
                src = T.alloc_shared((block_rows,), "int32")
                m = T.alloc_local((items,), "int32")
                x = T.alloc_local((1,), "int32")
                x[0] = 0
                for j in T.unroll(items):
                    r = tx * items + j
                    row = bx * block_rows + r
                    m[j] = T.if_then_else(
                        r < block_rows and row < rows, T.cast(mask[row] != 0, "int32"), 0
                    )
                    x[0] += m[j]
                _block_exclusive_scan(x, sums, warps)
                run = T.alloc_var("int32")
                run = offsets[bx] + x[0]
                for j in T.unroll(items):
                    r = tx * items + j
                    if r < block_rows:
                        # A source too short for a row is not read past its end.
                        src[r] = T.if_then_else(m[j] != 0 and run < src_rows, run, -1)
                    run += m[j]
                T.sync_threads()
                buf = T.alloc_local((steps * vec,), dtype)
                for k in T.serial(T.ceildiv(col_tiles - by, grid_y)):
                    first_col = (by + k * grid_y) * block_cols
                    for it in T.unroll(steps):
                        idx = it * threads + tx
                        r = idx // vecs
                        row = bx * block_rows + r
                        col = first_col + idx % vecs * vec
                        if row < rows and col < cols:
                            if src[r] >= 0:
                                for v in T.vectorized(vec):
                                    buf[it * vec + v] = source[src[r], col + v]
                            else:
                                for v in T.vectorized(vec):
                                    buf[it * vec + v] = input[row, col + v]
                    for it in T.unroll(steps):
                        idx = it * threads + tx
                        row = bx * block_rows + idx // vecs
                        col = first_col + idx % vecs * vec
                        if row < rows and col < cols:
                            for v in T.vectorized(vec):
                                out[row, col + v] = buf[it * vec + v]

        return copy_rows

    return _func


class MaskedScatterKernel(Kernel, MaskedScatterFwdInterface):
    """The output as ``rows x cols``, where the mask is constant along each row.

    ``cols`` is the product of the trailing axes along which the mask has extent 1. Three
    programs run: each row tile's true count, one CTA's exclusive scan of those counts, and
    a copy in which each CTA scans its own rows and moves whole vectors of up to 16 bytes from
    source or input.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: MaskedScatterCall) -> str | None:
        _, _, rows, cols = _rows(call.input_shape, call.mask_shape)
        # Row and column indices, tile offsets and source rows are int32, with room for a last
        # tile's padding past the row count or the row width.
        if max(rows, cols) > 2**31 - 2**16:
            return "serves at most 2**31 - 2**16 mask rows of at most as many elements"
        _, block_rows, block_cols = _tile(cols, call.dtype.itemsize)
        if -(-rows // block_rows) * -(-cols // block_cols) >= 2**31:
            return "launches at most 2**31 - 1 copy tiles"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: MaskedScatterCall) -> Entry:
        # The row width is dynamic: calls whose widths share a vector width and a tile share
        # one build.
        _, _, _, cols = _rows(call.input_shape, call.mask_shape)
        tile = _tile(cols, call.dtype.itemsize)
        index = None if call.device is None else call.device.index
        return (*tile, call.dtype, index), lambda: cls(*tile, call.dtype, device_index=index)

    def __init__(
        self,
        vec: int,
        block_rows: int,
        block_cols: int,
        dtype: torch.dtype,
        config=None,
        device_index=None,
    ):
        super().__init__(device_index=device_index)
        self.block_rows, self.block_cols = block_rows, block_cols
        self.dtype = dtype
        self.kernel = _row_copy_kernel(vec, self.dtype_str)
        # Sets the config only: the programs compile at the first launch.
        self.init_config(config)

    @property
    def default_config(self) -> dict[str, Any]:
        return {"block_rows": self.block_rows, "block_cols": self.block_cols}

    def _compile(self, cfg: dict) -> tuple:
        return (
            _tile_counts_kernel()(cfg["block_rows"]),
            _tile_offsets_kernel()(),
            self.kernel(cfg["block_rows"], cfg["block_cols"]),
        )

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        self._require_devices(input=input, mask=mask, source=source)
        out_shape, row_axes, rows, cols = _rows(input.shape, mask.shape)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        lead = len(row_axes)
        # The mask at each row: its trailing extent-1 axes dropped, its leading ones expanded.
        row_mask = mask.reshape(row_axes).expand(out_shape[:lead]).reshape(rows).view(torch.uint8)
        # A source row is a run of cols elements; a partial last row and the rows past the
        # row count are never read.
        flat = source.reshape(-1)
        source_rows = flat[: min(flat.numel() // cols, rows) * cols].view(-1, cols)
        block_rows = self.config["block_rows"]
        tiles = -(-rows // block_rows)
        counts = torch.empty(tiles, dtype=torch.int32, device=input.device)
        offsets = torch.empty_like(counts)
        count, scan, copy = self.program(self._compile)
        count(row_mask, counts)
        scan(counts, offsets)
        copy(
            input.expand(out_shape).reshape(rows, cols).contiguous(),
            row_mask,
            offsets,
            source_rows,
            out.view(rows, cols),
        )
        return out
