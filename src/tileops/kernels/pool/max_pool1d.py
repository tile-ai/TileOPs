import functools
from typing import Any, Callable, ClassVar, NamedTuple, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import dtype_itemsize, pool_output_dim, window_span

__all__ = ["MaxPool1dKernel", "MaxPool1dWithIndicesKernel"]

_NEG_INF = float("-inf")
_NAN = float("nan")
_JIT_FLAGS = ["-O3", "-DENABLE_BF16"]

# Taps one fragment of the row-reduce body may hold. Above this the fragment costs more
# registers than the occupancy it buys back.
_MAX_FRAGMENT_TAPS = 8192

# Elements between two staged rows, past the span itself. Off a whole number of banks,
# so two lanes reducing the same output of neighbouring rows do not serialize. Measured
# on an H200; re-measure elsewhere.
_STAGE_ROW_PAD = 8


class _Plan(NamedTuple):
    """Which body reads a shape, and the extents that body needs."""

    body: str
    # Outputs along the pooled axis.
    out_l: int
    # Whether every tap of every window lies inside the row.
    always_in_bounds: bool
    # Elements the staged tile holds in front of the row, and in all. Both zero unless
    # the body is "staged".
    head: int
    span: int
    # Full-width accesses one window takes. Zero unless the body is "rowreduce".
    window_vectors: int


def _plan(
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    with_indices: bool,
) -> _Plan:
    """The body that reads this shape, chosen from the window geometry alone.

    ``rowreduce`` takes a shape whose row yields one output. The window is then the
    row's own reduction, and taking the taps of several rows off one fragment is what
    keeps a block wide enough to fill the device -- one output per lane leaves a launch
    of one thread per row. It emits no position, so it is not offered with indices.

    ``staged`` takes a shape whose windows overlap, so a tap feeds more than one output.
    The row reaches shared memory once and every tap comes off it. A block takes as many
    rows as its lanes have outputs to reduce, so a narrow output row does not leave them
    idle.

    ``windowed`` takes the rest, where each element feeds one window and staging the row
    would add a copy of it for nothing.

    Args:
        l_in: Elements one row holds.
        kernel_w: Taps one window has.
        stride_w: Elements between two consecutive windows.
        pad_w: Elements the leftmost window reaches in front of the row.
        dilation_w: Elements between two taps of one window.
        ceil_mode: Whether the output extent rounds up.
        dtype: TileLang name of the element type.
        with_indices: Whether the kernel also emits each maximum's position.

    Returns:
        The plan the builder for the chosen body reads.
    """
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    reach = dilation_w * (kernel_w - 1) + 1
    always_in_bounds = pad_w == 0 and (out_l - 1) * stride_w + reach <= l_in
    itemsize = dtype_itemsize(dtype)
    window_bytes = kernel_w * itemsize

    if (
        not with_indices
        and out_l == 1
        and always_in_bounds
        and pad_w == 0
        and dilation_w == 1
        and window_bytes % VECTOR_ACCESS_BYTES == 0
        # A fragment whose trailing extent is not a power of two has no layout.
        and kernel_w & (kernel_w - 1) == 0
        and kernel_w <= _MAX_FRAGMENT_TAPS
    ):
        return _Plan(
            "rowreduce", out_l, always_in_bounds, 0, 0, window_bytes // VECTOR_ACCESS_BYTES
        )

    if reach > stride_w:
        # The whole row at once, so every block's origin is the row's own start and the
        # block step puts no constraint on the access width.
        _, head, span = window_span(out_l, 0, l_in, kernel_w, stride_w, pad_w, dilation_w, dtype)
        if span * itemsize <= STATIC_SHARED_BYTES:
            return _Plan("staged", out_l, always_in_bounds, head, span, 0)

    return _Plan("windowed", out_l, always_in_bounds, 0, 0, 0)


def _stage_rows(rows: int, out_l: int, span: int, itemsize: int, threads: int) -> Tuple[int, int]:
    """Rows one staged tile holds, and the elements between two of them.

    One row per lane's worth of outputs: below that a block would leave lanes with
    nothing to reduce, and above it the extra rows only cost residency. A tile holding
    more than one row pads the stride between them off a whole number of banks, so two
    lanes reducing the same output of neighbouring rows do not serialize.

    ``rows`` dividing the count is what leaves the last block whole, and the shared
    budget is the second bound.
    """
    wanted = max(threads // out_l, 1)
    count = 1
    while count * 2 <= wanted and rows % (count * 2) == 0:
        count *= 2
    while count > 1:
        stride = span + _STAGE_ROW_PAD
        if count * stride * itemsize <= STATIC_SHARED_BYTES:
            return count, stride
        count //= 2
    return 1, span


def _block_rows(rows: int, kernel_w: int, window_vectors: int, threads: int) -> int:
    """Rows the row-reduce body takes per block.

    One lane per full-width access of a window keeps the load coalesced across the
    block: a warp then covers whole windows rather than one element of each. The
    fragment holds every tap of every row the block takes, so its budget is the second
    bound, and ``rows`` dividing the count is what leaves the last block whole.
    """
    wanted = min(
        max(threads // window_vectors, 1),
        max(_MAX_FRAGMENT_TAPS // kernel_w, 1),
    )
    count = 1
    while count * 2 <= wanted and rows % (count * 2) == 0:
        count *= 2
    return count


class _Launch:
    """The launch shapes worth offering for one plan.

    These figures describe this kernel's access patterns, not the device, and are held
    here so that a later kernel does not read them as general truths.
    """

    # Outputs one block covers where the body reads `block_ol`, and the block sizes
    # each body is timed at. A body whose rate falls off either side of one thread
    # count is given that count and not a space: the rows this kernel is measured on
    # run in single-digit microseconds, where a candidate's own measurement is too
    # coarse to rank the rest.
    _BLOCK_OUTPUTS: ClassVar[int] = 256
    _FALLBACK_THREADS: ClassVar[int] = 128
    _CANDIDATES: ClassVar[dict] = {
        # A wider block covers more of the flattened output, which is what keeps the
        # ragged block at the end of a row from being most of a launch.
        "windowed": ((256, 128), (512, 256), (1024, 256)),
        # The staging copy is the whole cost, and it is issued fastest by a block that
        # does not outnumber the row's full-width accesses.
        "staged": ((256, 128),),
        # One lane per full-width access of a window, so the block follows the window.
        "rowreduce": ((256, 256), (256, 512)),
    }
    # Threads a launch needs before a candidate is offered; under this the blocks do not
    # fill the device.
    _MIN_LAUNCH_THREADS: ClassVar[int] = 1 << 16

    def __init__(self, rows: int, kernel_w: int, itemsize: int, plan: _Plan) -> None:
        self._rows = rows
        self._kernel_w = kernel_w
        self._itemsize = itemsize
        self._plan = plan

    def default(self) -> dict:
        return {"block_ol": self._BLOCK_OUTPUTS, "threads": self._FALLBACK_THREADS}

    def tuned(self) -> list[dict]:
        candidates = [
            {"block_ol": block_ol, "threads": threads}
            for block_ol, threads in self._CANDIDATES[self._plan.body]
        ]
        return [
            candidate
            for candidate in candidates
            if self._blocks(candidate) * candidate["threads"] >= self._MIN_LAUNCH_THREADS
        ] or [self.default()]

    def _blocks(self, candidate: dict) -> int:
        if self._plan.body == "windowed":
            total = self._rows * self._plan.out_l
            return (total + candidate["block_ol"] - 1) // candidate["block_ol"]
        if self._plan.body == "staged":
            stage_rows, _ = _stage_rows(
                self._rows,
                self._plan.out_l,
                self._plan.span,
                self._itemsize,
                candidate["threads"],
            )
            return self._rows // stage_rows
        return self._rows // _block_rows(
            self._rows, self._kernel_w, self._plan.window_vectors, candidate["threads"]
        )


def _windowed_scan(
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    dtype: str,
    accum_dtype: str,
    always_in_bounds: bool,
):
    """The windowed body's scan over one output, reading each tap where it lies.

    ``T.max`` drops NaN, so the two ways to propagate it as PyTorch does are a second
    accumulator saying one was seen, and a select per tap that lets NaN into the value
    and keeps it there -- a later value fails ``val > NaN``. Which is cheaper follows
    the bounds test: where every tap is in range the flag costs one boolean or per tap
    against the select's whole compare, and where a tap is tested anyway the select
    rides in the test's shadow and the flag's extra register does not.
    """

    @T.macro
    def _store(x, out, row, ol, idx):
        max_val = T.alloc_var(T.float32)
        max_val = T.cast(_NEG_INF, accum_dtype)
        if always_in_bounds:
            has_nan = T.alloc_var(T.bool)
            has_nan = False
            for kw in T.serial(kernel_w):
                val = T.cast(x[row, ol * stride_w - pad_w + kw * dilation_w], accum_dtype)
                has_nan = has_nan | T.isnan(val)
                max_val = T.max(max_val, val)
            out[idx] = T.cast(T.if_then_else(has_nan, T.cast(_NAN, accum_dtype), max_val), dtype)
        else:
            for kw in T.serial(kernel_w):
                iw = ol * stride_w - pad_w + kw * dilation_w
                if (iw >= 0) and (iw < l_in):
                    val = T.cast(x[row, iw], accum_dtype)
                    max_val = T.if_then_else(T.isnan(val) or (val > max_val), val, max_val)
            out[idx] = T.cast(max_val, dtype)

    return _store


def _windowed_indices_scan(
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    dtype: str,
    accum_dtype: str,
    always_in_bounds: bool,
):
    """The windowed body's scan over one output, with the tap that won it."""

    @T.macro
    def _store(x, out, indices, row, ol, idx):
        max_val = T.alloc_var(T.float32)
        max_idx = T.alloc_var(T.int32)
        # -1 until a NaN is seen, so it is also the flag saying one was.
        nan_idx = T.alloc_var(T.int32)
        max_val = T.cast(_NEG_INF, accum_dtype)
        nan_idx = -1
        if always_in_bounds:
            # An expression, not a variable: a variable is opaque to the range analysis,
            # and each tap would then load under a bounds check this window's extent
            # rules out.
            iw0 = ol * stride_w - pad_w
            max_idx = iw0
        else:
            # A variable, because each tap is bounds-tested anyway and this then stays
            # out of all of them.
            iw0 = T.alloc_var(T.int32)
            iw0 = ol * stride_w - pad_w
            # The first tap the row holds, on the dilation grid. Only padding starts a
            # window before position 0, and that tap is the position PyTorch reports
            # when every tap the row holds is -inf.
            max_idx = iw0 + dilation_w * T.ceildiv(T.max(-iw0, 0), dilation_w)
        for kw in T.serial(kernel_w):
            iw = iw0 + kw * dilation_w
            if always_in_bounds or ((iw >= 0) and (iw < l_in)):
                val = T.cast(x[row, iw], accum_dtype)
                # `max_val` is never NaN and NaN fails `>`, so the compare rejects NaN
                # without a separate test. Strict `>` also keeps the seed against a tap
                # equal to it, and the first maximum against a later equal one, which is
                # what PyTorch reports.
                take = val > max_val
                max_val = T.if_then_else(take, val, max_val)
                max_idx = T.if_then_else(take, iw, max_idx)
                nan_idx = T.if_then_else(T.isnan(val), iw, nan_idx)

        # PyTorch reports the last NaN a window visited.
        out[idx] = T.cast(T.if_then_else(nan_idx >= 0, T.cast(_NAN, accum_dtype), max_val), dtype)
        indices[idx] = T.cast(T.if_then_else(nan_idx >= 0, nan_idx, max_idx), "int64")

    return _store


def _windowed_builder(
    rows: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    dtype: str,
    accum_dtype: str,
    out_l: int,
    always_in_bounds: bool,
):
    """One output per lane, each tap read where it lies."""
    total_output = rows * out_l

    @tilelang.jit(out_idx=[1], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        # Every lane of every block owns an output, so none of them carries a test.
        tail_free = total_output % block_ol == 0
        store = _windowed_scan(
            l_in,
            kernel_w,
            stride_w,
            pad_w,
            dilation_w,
            dtype,
            accum_dtype,
            always_in_bounds,
        )

        @T.prim_func
        def _main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((total_output,), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_ol), threads=threads) as bx:
                for i in T.Parallel(block_ol):
                    idx = bx * block_ol + i
                    if tail_free:
                        store(x, out, idx // out_l, idx % out_l, idx)
                    else:
                        if idx < total_output:
                            store(x, out, idx // out_l, idx % out_l, idx)

        return _main

    return _build


def _windowed_indices_builder(
    rows: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    dtype: str,
    accum_dtype: str,
    out_l: int,
    always_in_bounds: bool,
):
    """The windowed body, also emitting each maximum's position."""
    total_output = rows * out_l

    @tilelang.jit(out_idx=[1, 2], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        tail_free = total_output % block_ol == 0
        store = _windowed_indices_scan(
            l_in,
            kernel_w,
            stride_w,
            pad_w,
            dilation_w,
            dtype,
            accum_dtype,
            always_in_bounds,
        )

        @T.prim_func
        def _main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((total_output,), dtype),  # type: ignore
            indices: T.Tensor((total_output,), "int64"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_ol), threads=threads) as bx:
                for i in T.Parallel(block_ol):
                    idx = bx * block_ol + i
                    if tail_free:
                        store(x, out, indices, idx // out_l, idx % out_l, idx)
                    else:
                        if idx < total_output:
                            store(x, out, indices, idx // out_l, idx % out_l, idx)

        return _main

    return _build


def _stage_macro(l_in: int, head: int, staged: int, tail: int, stage_rows: int, dtype: str):
    """The staging pass: the block's rows, with -inf outside each of them.

    The head and the staged extent both divide the access width, so the copy is
    full-width and the two fills touch only elements outside a row. A tap reading one of
    those elements reads -inf, which is the value PyTorch pads a max-pool window with
    and which no tap can win against.
    """

    @T.macro
    def _stage(tile, x, row0):
        if head:
            for r, i in T.Parallel(stage_rows, head):
                tile[r, i] = T.cast(_NEG_INF, dtype)
        if tail:
            for r, i in T.Parallel(stage_rows, tail):
                tile[r, head + staged + i] = T.cast(_NEG_INF, dtype)
        T.copy(x[row0 : row0 + stage_rows, 0:staged], tile[:, head : head + staged])

    return _stage


def _staged_scan(
    kernel_w: int, stride_w: int, dilation_w: int, base: int, dtype: str, accum_dtype: str
):
    """The staged body's scan over one output, every tap read off the tile.

    Resident taps make the scan arithmetic-bound, which wants the opposite trade to the
    windowed body: no tap carries a bounds test, so NaN rides in a second accumulator
    rather than costing a select per tap.
    """

    @T.macro
    def _store(tile, out, r, row, ol):
        max_val = T.alloc_var(T.float32)
        has_nan = T.alloc_var(T.bool)
        max_val = T.cast(_NEG_INF, accum_dtype)
        has_nan = False
        for kw in T.serial(kernel_w):
            val = T.cast(tile[r, base + ol * stride_w + kw * dilation_w], accum_dtype)
            has_nan = has_nan | T.isnan(val)
            max_val = T.max(max_val, val)

        out[row, ol] = T.cast(T.if_then_else(has_nan, T.cast(_NAN, accum_dtype), max_val), dtype)

    return _store


def _staged_indices_scan(
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    base: int,
    dtype: str,
    accum_dtype: str,
):
    """The staged body's scan over one output, with the tap that won it."""

    @T.macro
    def _store(tile, out, indices, r, row, ol):
        max_val = T.alloc_var(T.float32)
        max_idx = T.alloc_var(T.int32)
        nan_idx = T.alloc_var(T.int32)
        max_val = T.cast(_NEG_INF, accum_dtype)
        nan_idx = -1
        iw0 = ol * stride_w - pad_w
        # The first tap the row holds, which is the position PyTorch reports when every
        # tap the row holds is -inf.
        max_idx = iw0 + dilation_w * T.ceildiv(T.max(-iw0, 0), dilation_w)
        for kw in T.serial(kernel_w):
            val = T.cast(tile[r, base + ol * stride_w + kw * dilation_w], accum_dtype)
            # A padded tap is -inf in the tile, so it never wins and needs no test.
            take = val > max_val
            max_val = T.if_then_else(take, val, max_val)
            max_idx = T.if_then_else(take, iw0 + kw * dilation_w, max_idx)
            nan_idx = T.if_then_else(T.isnan(val), iw0 + kw * dilation_w, nan_idx)

        out[row, ol] = T.cast(
            T.if_then_else(nan_idx >= 0, T.cast(_NAN, accum_dtype), max_val), dtype
        )
        indices[row, ol] = T.cast(T.if_then_else(nan_idx >= 0, nan_idx, max_idx), "int64")

    return _store


def _staged_extents(l_in: int, head: int, span: int) -> Tuple[int, int]:
    """Elements of a row the tile holds, and elements of -inf past them.

    The span covers the last window's last tap and then the rest of that access width,
    which is short of the row's end wherever the windows overlap without dividing it.
    """
    staged = min(l_in, span - head)
    return staged, span - head - staged


def _staged_builder(
    rows: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    dtype: str,
    accum_dtype: str,
    out_l: int,
    head: int,
    span: int,
    itemsize: int,
):
    """A block's rows staged in shared memory once, every tap read off them."""
    # The tile carries the padding as -inf, so a tap needs no test and this offset.
    base = head - pad_w
    staged, tail = _staged_extents(l_in, head, span)

    @tilelang.jit(out_idx=[1], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        stage_rows, row_stride = _stage_rows(rows, out_l, span, itemsize, threads)
        stage = _stage_macro(l_in, head, staged, tail, stage_rows, dtype)
        store = _staged_scan(kernel_w, stride_w, dilation_w, base, dtype, accum_dtype)

        @T.prim_func
        def _main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_l), dtype),  # type: ignore
        ):
            with T.Kernel(rows // stage_rows, threads=threads) as bx:
                tile = T.alloc_shared((stage_rows, row_stride), dtype)
                row0 = bx * stage_rows
                stage(tile, x, row0)
                for r, ol in T.Parallel(stage_rows, out_l):
                    store(tile, out, r, row0 + r, ol)

        return _main

    return _build


def _staged_indices_builder(
    rows: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    dtype: str,
    accum_dtype: str,
    out_l: int,
    head: int,
    span: int,
    itemsize: int,
):
    """The staged body, also emitting each maximum's position."""
    base = head - pad_w
    staged, tail = _staged_extents(l_in, head, span)

    @tilelang.jit(out_idx=[1, 2], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        stage_rows, row_stride = _stage_rows(rows, out_l, span, itemsize, threads)
        stage = _stage_macro(l_in, head, staged, tail, stage_rows, dtype)
        store = _staged_indices_scan(
            kernel_w, stride_w, pad_w, dilation_w, base, dtype, accum_dtype
        )

        @T.prim_func
        def _main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_l), dtype),  # type: ignore
            indices: T.Tensor((rows, out_l), "int64"),  # type: ignore
        ):
            with T.Kernel(rows // stage_rows, threads=threads) as bx:
                tile = T.alloc_shared((stage_rows, row_stride), dtype)
                row0 = bx * stage_rows
                stage(tile, x, row0)
                for r, ol in T.Parallel(stage_rows, out_l):
                    store(tile, out, indices, r, row0 + r, ol)

        return _main

    return _build


def _rowreduce_builder(
    rows: int,
    l_in: int,
    kernel_w: int,
    dtype: str,
    accum_dtype: str,
    window_vectors: int,
):
    """One output per row, taken off a fragment holding the taps of several rows."""

    @tilelang.jit(out_idx=[1], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        block_rows = _block_rows(rows, kernel_w, window_vectors, threads)

        @T.prim_func
        def _main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((rows, 1), dtype),  # type: ignore
        ):
            with T.Kernel(rows // block_rows, threads=threads) as bx:
                taps = T.alloc_fragment((block_rows, kernel_w), accum_dtype)
                nans = T.alloc_fragment((block_rows, kernel_w), accum_dtype)
                best = T.alloc_fragment((block_rows,), accum_dtype)
                seen = T.alloc_fragment((block_rows,), accum_dtype)
                for r, kw in T.Parallel(block_rows, kernel_w):
                    taps[r, kw] = T.cast(x[bx * block_rows + r, kw], accum_dtype)
                # `T.reduce_max` drops NaN, so which taps were NaN is reduced alongside
                # the value and the two answers meet at the store.
                for r, kw in T.Parallel(block_rows, kernel_w):
                    nans[r, kw] = T.if_then_else(T.isnan(taps[r, kw]), 1.0, 0.0)
                T.reduce_max(taps, best, dim=1, clear=True)
                T.reduce_max(nans, seen, dim=1, clear=True)
                for r in T.Parallel(block_rows):
                    out[bx * block_rows + r, 0] = T.cast(
                        T.if_then_else(seen[r] > 0.0, T.cast(_NAN, accum_dtype), best[r]),
                        dtype,
                    )

        return _main

    return _build


def _build_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    with_indices: bool,
):
    """The jit builder for the body this shape's plan selects."""
    accum_dtype = "float"
    rows = n * c_in
    plan = _plan(l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype, with_indices)
    if plan.body == "rowreduce":
        return _rowreduce_builder(rows, l_in, kernel_w, dtype, accum_dtype, plan.window_vectors)
    if plan.body == "staged":
        build = _staged_indices_builder if with_indices else _staged_builder
        return build(
            rows,
            l_in,
            kernel_w,
            stride_w,
            pad_w,
            dilation_w,
            dtype,
            accum_dtype,
            plan.out_l,
            plan.head,
            plan.span,
            dtype_itemsize(dtype),
        )
    build = _windowed_indices_builder if with_indices else _windowed_builder
    return build(
        rows,
        l_in,
        kernel_w,
        stride_w,
        pad_w,
        dilation_w,
        dtype,
        accum_dtype,
        plan.out_l,
        plan.always_in_bounds,
    )


@functools.lru_cache(maxsize=32)
def _max_pool1d_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    return _build_kernel(
        n, c_in, l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype, False
    )


@functools.lru_cache(maxsize=32)
def _max_pool1d_with_indices_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    return _build_kernel(
        n, c_in, l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype, True
    )


def _launch_max_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_ol: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    kernel = _max_pool1d_kernel(
        n, c_in, l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype
    )(block_ol, threads)
    return kernel(x.contiguous().view(n * c_in, l_in)).view(n, c_in, out_l)


def _launch_max_pool1d_with_indices(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_ol: int,
    threads: int,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    kernel = _max_pool1d_with_indices_kernel(
        n, c_in, l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode, dtype
    )(block_ol, threads)
    out, indices = kernel(x.contiguous().view(n * c_in, l_in))
    return out.view(n, c_in, out_l), indices.view(n, c_in, out_l)


class _MaxPool1dKernelBase(Kernel):
    """Shape, launch planning and dispatch shared by the two 1d max-pool kernels.

    Concrete kernels supply ``_build``, ``_dispatch`` and ``_with_indices``; everything
    else -- parameter capture, output extents, config and launch -- is identical between
    the value-only and with-indices variants.
    """

    _build: ClassVar[Callable[..., Any]]
    _dispatch: ClassVar[Callable[..., Any]]
    _with_indices: ClassVar[bool]

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_w: int,
        stride_w: int,
        pad_w: int,
        dilation_w: int,
        ceil_mode: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"{type(self).__name__} supports float16, bfloat16, and float32, got {dtype}"
            )
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_w = kernel_w
        self.stride_w = stride_w
        self.pad_w = pad_w
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
        self.kernel = type(self)._build(
            n,
            c_in,
            l_in,
            kernel_w,
            stride_w,
            pad_w,
            dilation_w,
            ceil_mode,
            self.dtype_str,
        )
        self.init_config(config, tune)

    def _launch(self) -> _Launch:
        plan = _plan(
            self.l_in,
            self.kernel_w,
            self.stride_w,
            self.pad_w,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            type(self)._with_indices,
        )
        return _Launch(self.n * self.c_in, self.kernel_w, dtype_itemsize(self.dtype_str), plan)

    @property
    def default_config(self) -> dict:
        return self._launch().default()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._launch().tuned()

    def forward(self, x: torch.Tensor) -> Any:
        self._require_cuda(x=x)
        return type(self)._dispatch(
            self.n,
            self.c_in,
            self.l_in,
            self.kernel_w,
            self.stride_w,
            self.pad_w,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            self.config["block_ol"],
            self.config["threads"],
            x,
        )


class MaxPool1dKernel(_MaxPool1dKernelBase):
    """Max pooling forward kernel (return_indices=False)."""

    _build = staticmethod(_max_pool1d_kernel)
    _dispatch = staticmethod(_launch_max_pool1d)
    _with_indices = False


class MaxPool1dWithIndicesKernel(_MaxPool1dKernelBase):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool1d_with_indices_kernel)
    _dispatch = staticmethod(_launch_max_pool1d_with_indices)
    _with_indices = True
