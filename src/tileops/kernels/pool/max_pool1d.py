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
_ACCUM_DTYPE = "float"
_JIT_FLAGS = ["-O3", "-DENABLE_BF16"]

# Taps one row-reduce block may hold. It bounds the two f32 fragments together: 8192
# taps is 64 KB over a 256-thread block, or 64 registers a thread.
_MAX_FRAGMENT_TAPS = 8192


class _Shape(NamedTuple):
    """The geometry one kernel is built for, over rows flattened from ``(N, C)``."""

    rows: int
    l_in: int
    kernel_w: int
    stride_w: int
    pad_w: int
    dilation_w: int
    dtype: str

    @property
    def itemsize(self) -> int:
        return dtype_itemsize(self.dtype)


class _Plan(NamedTuple):
    """Which body reads a shape, and the extents that body needs."""

    body: str
    out_l: int
    # Whether every tap of every window lies inside the row.
    always_in_bounds: bool
    # The staged tile, in front of the row and in all. Zero outside the staged body.
    head: int
    span: int
    # Full-width accesses one window takes. Zero outside the row-reduce body.
    window_vectors: int


def _plan(shape: _Shape, ceil_mode: bool, with_indices: bool) -> _Plan:
    """The body that reads this shape, chosen from the window geometry alone.

    ``rowreduce`` takes a row yielding one output, and emits no position, so it is not
    offered with indices. ``staged`` takes overlapping windows, whose taps it reads off
    shared memory. ``windowed`` takes the rest and reads each tap where it lies.
    """
    rows, l_in, kernel_w, stride_w, pad_w, dilation_w, dtype = shape
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    reach = dilation_w * (kernel_w - 1) + 1
    always_in_bounds = pad_w == 0 and (out_l - 1) * stride_w + reach <= l_in
    window_bytes = kernel_w * shape.itemsize

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
        # Step 0: overlapping windows are staged a whole row at a time, so there is no
        # second block of a row whose origin the access width would have to divide.
        _, head, span = window_span(out_l, 0, l_in, kernel_w, stride_w, pad_w, dilation_w, dtype)
        if span * shape.itemsize <= STATIC_SHARED_BYTES:
            return _Plan("staged", out_l, always_in_bounds, head, span, 0)

    return _Plan("windowed", out_l, always_in_bounds, 0, 0, 0)


def _stage_rows(rows: int, out_l: int, span: int, itemsize: int, threads: int) -> Tuple[int, int]:
    """Rows one staged tile holds, and the elements between two of them.

    The count divides ``rows``, so no block is left ragged, and fits the shared budget.
    A tile of more than one row pads the stride between them by one full-width access,
    which moves two lanes reducing the same output off one bank and keeps the staging
    copy aligned.
    """
    pad = VECTOR_ACCESS_BYTES // itemsize
    wanted = max(threads // out_l, 1)
    count = 1
    while count * 2 <= wanted and rows % (count * 2) == 0:
        count *= 2
    while count > 1:
        if count * (span + pad) * itemsize <= STATIC_SHARED_BYTES:
            return count, span + pad
        count //= 2
    return 1, span


def _block_rows(rows: int, kernel_w: int, window_vectors: int, threads: int) -> int:
    """Rows the row-reduce body takes per block.

    One lane per full-width access of a window. The count divides ``rows`` and its
    fragments fit :data:`_MAX_FRAGMENT_TAPS`.
    """
    wanted = min(
        max(threads // window_vectors, 1),
        max(_MAX_FRAGMENT_TAPS // kernel_w, 1),
    )
    count = 1
    while count * 2 <= wanted and rows % (count * 2) == 0:
        count *= 2
    return count


def _windowed_scan(
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    dtype: str,
    always_in_bounds: bool,
):
    """The windowed body's scan over one output, reading each tap where it lies.

    ``T.max`` drops NaN, so PyTorch's propagation is spelled two ways: a flag where no
    tap is bounds-tested, a select per tap where one is. Both are kept because which is
    cheaper follows the bounds test.
    """

    @T.macro
    def _store(x, out, row, ol, idx):
        max_val = T.alloc_var(T.float32)
        max_val = T.cast(_NEG_INF, _ACCUM_DTYPE)
        if always_in_bounds:
            has_nan = T.alloc_var(T.bool)
            has_nan = False
            for kw in T.serial(kernel_w):
                val = T.cast(x[row, ol * stride_w - pad_w + kw * dilation_w], _ACCUM_DTYPE)
                has_nan = has_nan | T.isnan(val)
                max_val = T.max(max_val, val)
            out[idx] = T.cast(T.if_then_else(has_nan, T.cast(_NAN, _ACCUM_DTYPE), max_val), dtype)
        else:
            for kw in T.serial(kernel_w):
                iw = ol * stride_w - pad_w + kw * dilation_w
                if (iw >= 0) and (iw < l_in):
                    val = T.cast(x[row, iw], _ACCUM_DTYPE)
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
    always_in_bounds: bool,
):
    """The windowed body's scan over one output, with the tap that won it."""

    @T.macro
    def _store(x, out, indices, row, ol, idx):
        max_val = T.alloc_var(T.float32)
        max_idx = T.alloc_var(T.int32)
        # -1 until a NaN is seen, so it is also the flag saying one was.
        nan_idx = T.alloc_var(T.int32)
        max_val = T.cast(_NEG_INF, _ACCUM_DTYPE)
        nan_idx = -1
        if always_in_bounds:
            # Why an expression and not a variable: a variable is opaque to the range
            # analysis, and every tap would load under a bounds check it rules out.
            iw0 = ol * stride_w - pad_w
            max_idx = iw0
        else:
            iw0 = T.alloc_var(T.int32)
            iw0 = ol * stride_w - pad_w
            # The first tap the row holds, which is what PyTorch reports for a window
            # whose every in-row tap is -inf.
            max_idx = iw0 + dilation_w * T.ceildiv(T.max(-iw0, 0), dilation_w)
        for kw in T.serial(kernel_w):
            iw = iw0 + kw * dilation_w
            if always_in_bounds or ((iw >= 0) and (iw < l_in)):
                val = T.cast(x[row, iw], _ACCUM_DTYPE)
                # Strict `>` reports the first of equal maxima, as PyTorch does, and
                # rejects NaN without a separate test.
                take = val > max_val
                max_val = T.if_then_else(take, val, max_val)
                max_idx = T.if_then_else(take, iw, max_idx)
                nan_idx = T.if_then_else(T.isnan(val), iw, nan_idx)

        # PyTorch reports the last NaN a window visited.
        out[idx] = T.cast(T.if_then_else(nan_idx >= 0, T.cast(_NAN, _ACCUM_DTYPE), max_val), dtype)
        indices[idx] = T.cast(T.if_then_else(nan_idx >= 0, nan_idx, max_idx), "int64")

    return _store


def _windowed_builder(shape: _Shape, plan: _Plan):
    """One output per lane, each tap read where it lies."""
    rows, l_in, kernel_w, stride_w, pad_w, dilation_w, dtype = shape
    out_l, in_bounds = plan.out_l, plan.always_in_bounds
    total_output = rows * out_l

    @tilelang.jit(out_idx=[1], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        tail_free = total_output % block_ol == 0
        scan = _windowed_scan(l_in, kernel_w, stride_w, pad_w, dilation_w, dtype, in_bounds)

        @T.prim_func
        def _main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((total_output,), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_ol), threads=threads) as bx:
                for i in T.Parallel(block_ol):
                    idx = bx * block_ol + i
                    if tail_free:
                        scan(x, out, idx // out_l, idx % out_l, idx)
                    else:
                        if idx < total_output:
                            scan(x, out, idx // out_l, idx % out_l, idx)

        return _main

    return _build


def _windowed_indices_builder(shape: _Shape, plan: _Plan):
    """The windowed body, also emitting each maximum's position."""
    rows, l_in, kernel_w, stride_w, pad_w, dilation_w, dtype = shape
    out_l, in_bounds = plan.out_l, plan.always_in_bounds
    total_output = rows * out_l

    @tilelang.jit(out_idx=[1, 2], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        tail_free = total_output % block_ol == 0
        scan = _windowed_indices_scan(l_in, kernel_w, stride_w, pad_w, dilation_w, dtype, in_bounds)

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
                        scan(x, out, indices, idx // out_l, idx % out_l, idx)
                    else:
                        if idx < total_output:
                            scan(x, out, indices, idx // out_l, idx % out_l, idx)

        return _main

    return _build


def _staged_extents(l_in: int, head: int, span: int) -> Tuple[int, int]:
    """Elements of a row the tile holds, and elements of -inf past them.

    The span reaches no further than the last window's access width, which is short of
    the row's end wherever the windows overlap without dividing it.
    """
    staged = min(l_in, span - head)
    return staged, span - head - staged


def _stage_macro(head: int, staged: int, tail: int, stage_rows: int, dtype: str):
    """The staging pass: the block's rows, with -inf outside each of them.

    -inf is what PyTorch pads a max-pool window with, so a tap outside the row needs no
    test.
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


def _staged_scan(kernel_w: int, stride_w: int, dilation_w: int, base: int, dtype: str):
    """The staged body's scan over one output, every tap read off the tile."""

    @T.macro
    def _store(tile, out, r, row, ol):
        max_val = T.alloc_var(T.float32)
        has_nan = T.alloc_var(T.bool)
        max_val = T.cast(_NEG_INF, _ACCUM_DTYPE)
        has_nan = False
        for kw in T.serial(kernel_w):
            val = T.cast(tile[r, base + ol * stride_w + kw * dilation_w], _ACCUM_DTYPE)
            has_nan = has_nan | T.isnan(val)
            max_val = T.max(max_val, val)

        out[row, ol] = T.cast(T.if_then_else(has_nan, T.cast(_NAN, _ACCUM_DTYPE), max_val), dtype)

    return _store


def _staged_indices_scan(
    kernel_w: int, stride_w: int, pad_w: int, dilation_w: int, base: int, dtype: str
):
    """The staged body's scan over one output, with the tap that won it."""

    @T.macro
    def _store(tile, out, indices, r, row, ol):
        max_val = T.alloc_var(T.float32)
        max_idx = T.alloc_var(T.int32)
        nan_idx = T.alloc_var(T.int32)
        max_val = T.cast(_NEG_INF, _ACCUM_DTYPE)
        nan_idx = -1
        iw0 = ol * stride_w - pad_w
        # The first tap the row holds, which is what PyTorch reports for a window whose
        # every in-row tap is -inf.
        max_idx = iw0 + dilation_w * T.ceildiv(T.max(-iw0, 0), dilation_w)
        for kw in T.serial(kernel_w):
            val = T.cast(tile[r, base + ol * stride_w + kw * dilation_w], _ACCUM_DTYPE)
            take = val > max_val
            max_val = T.if_then_else(take, val, max_val)
            max_idx = T.if_then_else(take, iw0 + kw * dilation_w, max_idx)
            nan_idx = T.if_then_else(T.isnan(val), iw0 + kw * dilation_w, nan_idx)

        out[row, ol] = T.cast(
            T.if_then_else(nan_idx >= 0, T.cast(_NAN, _ACCUM_DTYPE), max_val), dtype
        )
        indices[row, ol] = T.cast(T.if_then_else(nan_idx >= 0, nan_idx, max_idx), "int64")

    return _store


def _staged_builder(shape: _Shape, plan: _Plan):
    """A block's rows staged in shared memory once, every tap read off them."""
    rows, l_in, kernel_w, stride_w, pad_w, dilation_w, dtype = shape
    itemsize = shape.itemsize
    out_l, head, span = plan.out_l, plan.head, plan.span
    # Tap ``kw`` of output ``ol`` sits at ``base + ol * stride + kw * dilation``.
    base = head - pad_w
    staged, tail = _staged_extents(l_in, head, span)

    @tilelang.jit(out_idx=[1], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        stage_rows, row_stride = _stage_rows(rows, out_l, span, itemsize, threads)
        stage = _stage_macro(head, staged, tail, stage_rows, dtype)
        scan = _staged_scan(kernel_w, stride_w, dilation_w, base, dtype)

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
                    scan(tile, out, r, row0 + r, ol)

        return _main

    return _build


def _staged_indices_builder(shape: _Shape, plan: _Plan):
    """The staged body, also emitting each maximum's position."""
    rows, l_in, kernel_w, stride_w, pad_w, dilation_w, dtype = shape
    itemsize = shape.itemsize
    out_l, head, span = plan.out_l, plan.head, plan.span
    base = head - pad_w
    staged, tail = _staged_extents(l_in, head, span)

    @tilelang.jit(out_idx=[1, 2], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        stage_rows, row_stride = _stage_rows(rows, out_l, span, itemsize, threads)
        stage = _stage_macro(head, staged, tail, stage_rows, dtype)
        scan = _staged_indices_scan(kernel_w, stride_w, pad_w, dilation_w, base, dtype)

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
                    scan(tile, out, indices, r, row0 + r, ol)

        return _main

    return _build


def _rowreduce_builder(shape: _Shape, plan: _Plan):
    """One output per row, taken off fragments holding the taps of several rows."""
    rows, l_in, kernel_w, dtype = shape.rows, shape.l_in, shape.kernel_w, shape.dtype
    window_vectors = plan.window_vectors

    @tilelang.jit(out_idx=[1], compile_flags=_JIT_FLAGS)
    def _build(block_ol: int, threads: int):
        block_rows = _block_rows(rows, kernel_w, window_vectors, threads)

        @T.prim_func
        def _main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((rows, 1), dtype),  # type: ignore
        ):
            with T.Kernel(rows // block_rows, threads=threads) as bx:
                taps = T.alloc_fragment((block_rows, kernel_w), _ACCUM_DTYPE)
                nans = T.alloc_fragment((block_rows, kernel_w), _ACCUM_DTYPE)
                best = T.alloc_fragment((block_rows,), _ACCUM_DTYPE)
                seen = T.alloc_fragment((block_rows,), _ACCUM_DTYPE)
                for r, kw in T.Parallel(block_rows, kernel_w):
                    taps[r, kw] = T.cast(x[bx * block_rows + r, kw], _ACCUM_DTYPE)
                # `T.reduce_max` drops NaN, so which taps were NaN is reduced too.
                for r, kw in T.Parallel(block_rows, kernel_w):
                    nans[r, kw] = T.if_then_else(T.isnan(taps[r, kw]), 1.0, 0.0)
                T.reduce_max(taps, best, dim=1, clear=True)
                T.reduce_max(nans, seen, dim=1, clear=True)
                for r in T.Parallel(block_rows):
                    out[bx * block_rows + r, 0] = T.cast(
                        T.if_then_else(seen[r] > 0.0, T.cast(_NAN, _ACCUM_DTYPE), best[r]),
                        dtype,
                    )

        return _main

    return _build


_BUILDERS = {
    ("rowreduce", False): _rowreduce_builder,
    ("staged", False): _staged_builder,
    ("staged", True): _staged_indices_builder,
    ("windowed", False): _windowed_builder,
    ("windowed", True): _windowed_indices_builder,
}


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
    shape = _Shape(n * c_in, l_in, kernel_w, stride_w, pad_w, dilation_w, dtype)
    plan = _plan(shape, ceil_mode, with_indices)
    return _BUILDERS[plan.body, with_indices](shape, plan)


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


class _MaxPool1dKernelBase(Kernel):
    """Shape, launch planning and dispatch shared by the two 1d max-pool kernels.

    Concrete kernels supply ``_build``, ``_shaped`` and ``_with_indices``; everything
    else -- parameter capture, output extents, config and launch -- is identical between
    the value-only and with-indices variants.
    """

    _build: ClassVar[Callable[..., Any]]
    _shaped: ClassVar[Callable[..., Any]]
    _with_indices: ClassVar[bool]

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]

    _BLOCK_OUTPUTS: ClassVar[int] = 256
    _FALLBACK_THREADS: ClassVar[int] = 128
    # ``(block_ol, threads)`` per body, measured on an H200. A body is given one thread
    # count where its rate falls off either side of it.
    _CANDIDATES: ClassVar[dict] = {
        "windowed": ((256, 128), (512, 256), (1024, 256)),
        "staged": ((256, 128),),
        "rowreduce": ((256, 256), (256, 512)),
    }
    # Threads a launch needs before a candidate is offered: below one resident block per
    # SM on a device of a few hundred SMs, a wider block is the better trade.
    _MIN_LAUNCH_THREADS: ClassVar[int] = 1 << 16

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

    @property
    def _shape(self) -> _Shape:
        return _Shape(
            self.n * self.c_in,
            self.l_in,
            self.kernel_w,
            self.stride_w,
            self.pad_w,
            self.dilation_w,
            self.dtype_str,
        )

    @property
    def _plan(self) -> _Plan:
        return _plan(self._shape, self.ceil_mode, type(self)._with_indices)

    def _blocks(self, candidate: dict) -> int:
        """Blocks the launch takes at *candidate*, which decides whether to offer it."""
        shape, plan = self._shape, self._plan
        if plan.body == "windowed":
            total = shape.rows * plan.out_l
            return (total + candidate["block_ol"] - 1) // candidate["block_ol"]
        if plan.body == "staged":
            stage_rows, _ = _stage_rows(
                shape.rows, plan.out_l, plan.span, shape.itemsize, candidate["threads"]
            )
            return shape.rows // stage_rows
        return shape.rows // _block_rows(
            shape.rows, shape.kernel_w, plan.window_vectors, candidate["threads"]
        )

    @property
    def default_config(self) -> dict:
        return {"block_ol": self._BLOCK_OUTPUTS, "threads": self._FALLBACK_THREADS}

    @property
    def autotune_configs(self) -> list[dict]:
        candidates = [
            {"block_ol": block_ol, "threads": threads}
            for block_ol, threads in self._CANDIDATES[self._plan.body]
        ]
        return [
            candidate
            for candidate in candidates
            if self._blocks(candidate) * candidate["threads"] >= self._MIN_LAUNCH_THREADS
        ] or [self.default_config]

    def forward(self, x: torch.Tensor) -> Any:
        self._require_cuda(x=x)
        kernel = self.kernel(self.config["block_ol"], self.config["threads"])
        rows = kernel(x.contiguous().view(self.n * self.c_in, self.l_in))
        return type(self)._shaped(rows, (self.n, self.c_in, self.out_l))


class MaxPool1dKernel(_MaxPool1dKernelBase):
    """Max pooling forward kernel (return_indices=False)."""

    _build = staticmethod(_max_pool1d_kernel)
    _with_indices = False

    @staticmethod
    def _shaped(result: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        return result.view(shape)


class MaxPool1dWithIndicesKernel(_MaxPool1dKernelBase):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool1d_with_indices_kernel)
    _with_indices = True

    @staticmethod
    def _shaped(
        result: Tuple[torch.Tensor, torch.Tensor], shape: Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values, indices = result
        return values.view(shape), indices.view(shape)
