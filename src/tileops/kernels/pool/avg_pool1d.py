import functools
from typing import NamedTuple, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["AvgPool1dKernel", "AvgPool1dSpatialKernel"]

# Tile widths a launch may take, widest first. The tail below 128 is what keeps a window
# too wide to stage at 128 outputs from having no width at all.
_BLOCK_OL_CHOICES = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
# Two warps per block measured fastest only on the widest workload, and is what
# the autotuner mispicks on a launch too short for it to tell candidates apart.
_THREAD_CHOICES = (128, 256)
# Measured best, or within one timer quantum of best, at all three manifest workloads.
_DEFAULT_BLOCK_OL = 512
_DEFAULT_THREADS = 128


def _itemsize(dtype: str) -> int:
    return 4 if dtype in ("float", "float32") else 2


def _round_up(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


class _Staging(NamedTuple):
    """Where one block's staged span sits in the row, and how wide its loads are."""

    # Elements one full-width access covers.
    vector_elems: int
    # Elements staged in front of the block's leftmost window.
    head: int
    # Elements the block stages.
    span: int

    @property
    def vectors(self) -> int:
        """Full-width loads the staging pass issues."""
        return self.span // self.vector_elems


def _staging(
    block_ol: int, l_in: int, kernel_l: int, stride_l: int, pad_l: int, dtype: str
) -> _Staging:
    """The staged span covering ``block_ol`` consecutive outputs.

    The staging load is vectorized and unguarded, so every span start has to sit on a
    multiple of the access width. Two things move a start: the block index, in steps of
    ``block_ol * stride_l``, and the slide back inside the row at the end of a row, which
    lands it on ``l_in - span``. The width is narrowed until it divides both. The head
    then absorbs the left padding, and a span wider than the row is the whole row.
    """
    vector_elems = VECTOR_ACCESS_BYTES // _itemsize(dtype)
    while vector_elems > 1 and (l_in % vector_elems or (block_ol * stride_l) % vector_elems):
        vector_elems //= 2
    head = _round_up(pad_l, vector_elems)
    reach = head + (block_ol - 1) * stride_l + kernel_l
    return _Staging(vector_elems, head, min(_round_up(reach, vector_elems), l_in))


def _block_ol_choices(l_in: int, kernel_l: int, stride_l: int, pad_l: int, dtype: str) -> list[int]:
    """The tile widths whose staged span fits the static shared budget, widest first.

    Raises:
        ValueError: When not even one output's window fits, which takes a kernel size
            in the tens of thousands.
    """
    budget = STATIC_SHARED_BYTES // _itemsize(dtype)
    fitting = [
        block_ol
        for block_ol in _BLOCK_OL_CHOICES
        if _staging(block_ol, l_in, kernel_l, stride_l, pad_l, dtype).span <= budget
    ]
    if not fitting:
        span = _staging(1, l_in, kernel_l, stride_l, pad_l, dtype).span
        raise ValueError(
            f"avg_pool1d stages the pooling window in shared memory: kernel_size="
            f"{kernel_l} needs {span * _itemsize(dtype)} B for a single output, over "
            f"the {STATIC_SHARED_BYTES} B a block gets"
        )
    return fitting


def _thread_configs(
    block_ol: int, l_in: int, kernel_l: int, stride_l: int, pad_l: int, dtype: str
) -> list[dict]:
    """Candidate configs at a fixed tile width: the block sizes worth timing on it.

    The width is not a candidate. Two widths a factor of two apart differ by a few
    percent here, which is under what the autotuner can resolve on a launch this short,
    so it is settled by :func:`_default_block_ol` instead. A block with more threads than
    the staging pass has full-width loads idles threads on the read this kernel is bound
    by, which leaves at most one candidate on a narrow tile.
    """
    staged = _staging(block_ol, l_in, kernel_l, stride_l, pad_l, dtype)
    return [
        {"block_ol": block_ol, "threads": threads}
        for threads in _THREAD_CHOICES
        if staged.vectors >= threads
    ] or [{"block_ol": block_ol, "threads": _DEFAULT_THREADS}]


def _default_block_ol(
    l_in: int, out_l: int, kernel_l: int, stride_l: int, pad_l: int, dtype: str
) -> int:
    """``_DEFAULT_BLOCK_OL``, narrowed twice.

    Narrowed to what the static shared budget holds, and to what the output row is
    wide enough to fill: a tile wider than the row only idles threads.
    """
    choices = _block_ol_choices(l_in, kernel_l, stride_l, pad_l, dtype)
    wanted = min(_DEFAULT_BLOCK_OL, max(out_l, choices[-1]))
    return next(block_ol for block_ol in choices if block_ol <= wanted)


@functools.lru_cache(maxsize=64)
def _avg_pool1d_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, ceil_mode)
    rows = n * c_in
    window_inside = pad_l == 0 and (out_l - 1) * stride_l + kernel_l <= l_in
    # Otherwise a window can overhang, and the divisor comes from its own extent.
    whole_window_divides = window_inside or (count_include_pad and not ceil_mode)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool1d_func(block_ol: int, threads: int):
        # Outputs `stride_l` apart share taps, so a warp taking one output each reads
        # one short stretch of the row `kernel_l` times over, a narrow load each time. A
        # block stages that stretch with full-width loads and takes every tap from it.
        staged = _staging(block_ol, l_in, kernel_l, stride_l, pad_l, dtype)
        vector_elems, head, span = staged
        blocks_per_row = (out_l + block_ol - 1) // block_ol
        tail_free = out_l % block_ol == 0
        # Only the first and the last block of a row hold an overhanging window, so the
        # blocks between them need neither the tap select nor a per-output divisor.
        interior_split = (
            not window_inside
            and blocks_per_row >= 3
            and block_ol * stride_l - pad_l >= 0
            and ((blocks_per_row - 1) * block_ol - 1) * stride_l - pad_l + kernel_l <= l_in
        )
        # Inside those two blocks the overhang is a handful of outputs at the row ends.
        clean_lo = -(-pad_l // stride_l)
        clean_hi = min((l_in + pad_l - kernel_l) // stride_l, out_l - 1)

        def _window_store(inside: bool):
            """A macro storing one output's mean, reading the taps from the tile.

            Args:
                inside: Whether every tap of this output is known to be in the row.
                    When it is not, the tap is read anyway and a select discards it,
                    so an overhanging window costs no branch.
            """

            @T.macro
            def _store(tile, offset, j, ol, out, out_row):
                total = T.alloc_var(T.float32)
                total = T.cast(0.0, accum_dtype)
                if inside:
                    # Written out rather than held in a var: bound analysis cannot
                    # place a var inside the tile, and guards the tap load in 64-bit
                    # addressing when it cannot.
                    for k in T.serial(kernel_l):
                        total += T.cast(tile[offset + j * stride_l + k], accum_dtype)
                else:
                    # A guard is emitted here whatever the index looks like, so the var
                    # keeps the address arithmetic from repeating per tap. The clamp
                    # keeps the discarded read inside the tile.
                    at = T.alloc_var(T.int32)
                    at = offset + j * stride_l
                    for k in T.serial(kernel_l):
                        tap = T.cast(tile[T.max(0, T.min(at + k, span - 1))], accum_dtype)
                        src = ol * stride_l - pad_l + k
                        total += T.if_then_else(
                            (src >= 0) and (src < l_in), tap, T.cast(0.0, accum_dtype)
                        )
                if inside or whole_window_divides:
                    out[out_row, ol] = T.cast(total * T.cast(1.0 / kernel_l, accum_dtype), dtype)
                else:
                    start = ol * stride_l - pad_l
                    if count_include_pad:
                        divisor = T.max(
                            T.min(start + kernel_l, l_in + pad_l) - T.max(start, -pad_l), 1
                        )
                    else:
                        divisor = T.max(T.min(start + kernel_l, l_in) - T.max(start, 0), 1)
                    out[out_row, ol] = T.cast(total / T.cast(divisor, accum_dtype), dtype)

            return _store

        _interior = _window_store(True)
        _boundary = _window_store(window_inside)

        @T.macro
        def _edge_split(tile, offset, j, ol, out, out_row):
            """One edge block, testing each output for the clean path."""
            if (ol >= clean_lo) and (ol <= clean_hi):
                _interior(tile, offset, j, ol, out, out_row)
            else:
                if ol < out_l:
                    _boundary(tile, offset, j, ol, out, out_row)

        @T.macro
        def _edge_plain(tile, offset, j, ol, out, out_row):
            """One edge block, on the overhang path throughout."""
            if ol < out_l:
                _boundary(tile, offset, j, ol, out, out_row)

        # Testing each output costs two comparisons and buys the constant divisor, which
        # pays only where a window's divisor is its own extent.
        _edge = _edge_plain if whole_window_divides else _edge_split

        @T.prim_func
        def _avg_pool1d_main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_l), dtype),  # type: ignore
        ):
            with T.Kernel(blocks_per_row, rows, threads=threads) as (bx, by):
                tile = T.alloc_shared((span,), dtype)
                # Sliding the span back inside the row costs a few already-staged
                # elements at the two ends and keeps every staging load unguarded.
                start = T.max(0, T.min(bx * (block_ol * stride_l) - head, l_in - span))
                for i in T.Parallel(staged.vectors):
                    for v in T.vectorized(vector_elems):
                        tile[i * vector_elems + v] = x[by, start + i * vector_elems + v]
                offset = bx * (block_ol * stride_l) - pad_l - start
                for j in T.Parallel(block_ol):
                    ol = bx * block_ol + j
                    if window_inside:
                        if tail_free:
                            _interior(tile, offset, j, ol, out, by)
                        else:
                            if ol < out_l:
                                _interior(tile, offset, j, ol, out, by)
                    else:
                        if interior_split:
                            if (bx > 0) and (bx < blocks_per_row - 1):
                                _interior(tile, offset, j, ol, out, by)
                            else:
                                _edge(tile, offset, j, ol, out, by)
                        else:
                            _edge(tile, offset, j, ol, out, by)

        return _avg_pool1d_main

    return _avg_pool1d_func


def _avg_pool1d_spatial_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dtype: str = "float16",
):
    """Zero-padded, floor-mode 1d average pooling.

    Every window then spans the full kernel once the padding is counted, which is
    what ``_avg_pool1d_kernel`` emits for these two flags.
    """
    return _avg_pool1d_kernel(n, c_in, l_in, kernel_l, stride_l, pad_l, False, True, dtype)


def _launch_avg_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: str,
    block_ol: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, ceil_mode)
    kernel = _avg_pool1d_kernel(
        n,
        c_in,
        l_in,
        kernel_l,
        stride_l,
        pad_l,
        ceil_mode,
        count_include_pad,
        dtype,
    )(block_ol, threads)

    return kernel(x.contiguous().view(n * c_in, l_in)).view(n, c_in, out_l)


def _launch_avg_pool1d_spatial(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dtype: str,
    block_ol: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _launch_avg_pool1d(
        n, c_in, l_in, kernel_l, stride_l, pad_l, False, True, dtype, block_ol, threads, x
    )


class AvgPool1dSpatialKernel(Kernel):
    """Fast path for common NCL avg_pool1d workloads.

    Raises:
        ValueError: When one pooling window does not fit the shared memory a block
            stages it in, which takes a ``kernel_size`` in the tens of thousands.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_l: int,
        stride_l: int,
        pad_l: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.dtype = dtype
        self.out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, False)

        self.kernel = _avg_pool1d_spatial_kernel(
            n,
            c_in,
            l_in,
            kernel_l,
            stride_l,
            pad_l,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_ol": _default_block_ol(
                self.l_in,
                self.out_l,
                self.kernel_l,
                self.stride_l,
                self.pad_l,
                self.dtype_str,
            ),
            "threads": _DEFAULT_THREADS,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return _thread_configs(
            self.default_config["block_ol"],
            self.l_in,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.dtype_str,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        return _launch_avg_pool1d_spatial(
            self.n,
            self.c_in,
            self.l_in,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.dtype_str,
            self.config["block_ol"],
            self.config["threads"],
            x,
        )


class AvgPool1dKernel(Kernel):
    """Average pooling over an NCL row, with every PyTorch flag combination.

    Raises:
        ValueError: When one pooling window does not fit the shared memory a block
            stages it in, which takes a ``kernel_size`` in the tens of thousands.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_l: int,
        stride_l: int,
        pad_l: int,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype
        self.out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, ceil_mode)

        self.kernel = _avg_pool1d_kernel(
            n,
            c_in,
            l_in,
            kernel_l,
            stride_l,
            pad_l,
            ceil_mode,
            count_include_pad,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_ol": _default_block_ol(
                self.l_in,
                self.out_l,
                self.kernel_l,
                self.stride_l,
                self.pad_l,
                self.dtype_str,
            ),
            "threads": _DEFAULT_THREADS,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return _thread_configs(
            self.default_config["block_ol"],
            self.l_in,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.dtype_str,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        return _launch_avg_pool1d(
            self.n,
            self.c_in,
            self.l_in,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.ceil_mode,
            self.count_include_pad,
            self.dtype_str,
            self.config["block_ol"],
            self.config["threads"],
            x,
        )
