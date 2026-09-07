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
# The best block size is not a function of the shape any rule here fits, so it is tuned.
_THREAD_CHOICES = (64, 128, 256)
# Threads a launch needs before a narrow block is worth offering: below this the blocks
# do not fill the device, and giving each thread more outputs only lengthens the tail.
_MIN_LAUNCH_THREADS = 1 << 16
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

    The tile is a window on the row's own coordinates, starting `head` elements before
    the block's leftmost window and holding zeros wherever it reaches outside the row.
    Every staging load is a whole group of `vector_elems`, so each group is either
    entirely inside the row or entirely outside it, and the width is narrowed until the
    three things that move a group boundary all divide it: the block step
    ``block_ol * stride_l``, the head, and the row end ``l_in``.
    """
    vector_elems = VECTOR_ACCESS_BYTES // _itemsize(dtype)
    while vector_elems > 1 and (l_in % vector_elems or (block_ol * stride_l) % vector_elems):
        vector_elems //= 2
    head = _round_up(pad_l, vector_elems)
    reach = head + (block_ol - 1) * stride_l + kernel_l
    return _Staging(vector_elems, head, _round_up(reach, vector_elems))


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
    block_ol: int,
    rows: int,
    l_in: int,
    out_l: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dtype: str,
) -> list[dict]:
    """Candidate configs at a fixed tile width: the block sizes worth timing on it.

    The width is not a candidate. Two widths a factor of two apart differ by a few
    percent here, which is under what the autotuner can resolve on a launch this short,
    so it is settled by :func:`_default_block_ol` instead. Two block sizes are dropped:
    one with more threads than the staging pass has full-width loads idles threads on the
    read this kernel is bound by, and one whose launch falls short of
    ``_MIN_LAUNCH_THREADS`` leaves the device unfilled.
    """
    staged = _staging(block_ol, l_in, kernel_l, stride_l, pad_l, dtype)
    blocks = rows * ((out_l + block_ol - 1) // block_ol)
    return [
        {"block_ol": block_ol, "threads": threads}
        for threads in _THREAD_CHOICES
        if staged.vectors >= threads and blocks * threads >= _MIN_LAUNCH_THREADS
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
    # The outputs whose window lies inside the row, and so divides by the kernel width.
    clean_lo = -(-pad_l // stride_l)
    clean_hi = min((l_in + pad_l - kernel_l) // stride_l, out_l - 1)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool1d_func(block_ol: int, threads: int):
        # Outputs `stride_l` apart share taps, so a warp taking one output each reads
        # one short stretch of the row `kernel_l` times over, a narrow load each time. A
        # block stages that stretch with full-width loads and takes every tap from it.
        staged = _staging(block_ol, l_in, kernel_l, stride_l, pad_l, dtype)
        vector_elems, head, span = staged
        # The tile holds zeros where it reaches outside the row, so a tap needs no test
        # of its own and its index into the tile is the same in every block.
        base = head - pad_l
        blocks_per_row = (out_l + block_ol - 1) // block_ol
        tail_free = out_l % block_ol == 0
        # Only the first and the last block of a row reach outside it.
        edge_free = pad_l == 0 and (blocks_per_row - 1) * block_ol * stride_l + span <= l_in

        @T.macro
        def _stage_inside(tile, x, origin, row):
            """Every group is in the row: one unguarded full-width load each."""
            for i in T.Parallel(staged.vectors):
                for v in T.vectorized(vector_elems):
                    tile[i * vector_elems + v] = x[row, origin + i * vector_elems + v]

        @T.macro
        def _stage_edge(tile, x, origin, row):
            """Zero the tile, then load the groups the row covers.

            The row covers whole groups only, so the second pass tests one per group and
            its load stays unguarded. Predicating the load itself instead, in one pass,
            measured slower than doing two.
            """
            for i in T.Parallel(staged.vectors):
                for v in T.vectorized(vector_elems):
                    tile[i * vector_elems + v] = T.cast(0.0, dtype)
            for i in T.Parallel(staged.vectors):
                if (origin + i * vector_elems >= 0) and (origin + (i + 1) * vector_elems <= l_in):
                    for v in T.vectorized(vector_elems):
                        tile[i * vector_elems + v] = x[row, origin + i * vector_elems + v]

        @T.macro
        def _store(tile, j, ol, out, out_row, whole: bool):
            """Store the mean of the window `tile` holds for output ``ol``."""
            total = T.alloc_var(T.float32)
            total = T.cast(0.0, accum_dtype)
            for k in T.serial(kernel_l):
                total += T.cast(tile[base + j * stride_l + k], accum_dtype)
            if whole:
                out[out_row, ol] = T.cast(total * T.cast(1.0 / kernel_l, accum_dtype), dtype)
            else:
                start = ol * stride_l - pad_l
                if count_include_pad:
                    divisor = T.max(T.min(start + kernel_l, l_in + pad_l) - T.max(start, -pad_l), 1)
                else:
                    divisor = T.max(T.min(start + kernel_l, l_in) - T.max(start, 0), 1)
                out[out_row, ol] = T.cast(total / T.cast(divisor, accum_dtype), dtype)

        @T.macro
        def _store_output(tile, j, ol, out, out_row):
            """Store output ``ol``, taking its divisor from the window where it needs to."""
            if whole_window_divides:
                _store(tile, j, ol, out, out_row, True)
            else:
                # A window's divisor is its own extent only where it overhangs the row.
                if (ol >= clean_lo) and (ol <= clean_hi):
                    _store(tile, j, ol, out, out_row, True)
                else:
                    _store(tile, j, ol, out, out_row, False)

        @T.macro
        def _reduce(tile, bx, out, out_row):
            """One output per lane, over the outputs this block owns."""
            for j in T.Parallel(block_ol):
                ol = bx * block_ol + j
                if tail_free:
                    _store_output(tile, j, ol, out, out_row)
                else:
                    if ol < out_l:
                        _store_output(tile, j, ol, out, out_row)

        @T.prim_func
        def _avg_pool1d_main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_l), dtype),  # type: ignore
        ):
            with T.Kernel(blocks_per_row, rows, threads=threads) as (bx, by):
                tile = T.alloc_shared((span,), dtype)
                origin = bx * (block_ol * stride_l) - head
                if edge_free:
                    _stage_inside(tile, x, origin, by)
                else:
                    if (bx > 0) and (bx < blocks_per_row - 1):
                        _stage_inside(tile, x, origin, by)
                    else:
                        _stage_edge(tile, x, origin, by)
                _reduce(tile, bx, out, by)

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
            self.n * self.c_in,
            self.l_in,
            self.out_l,
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
            self.n * self.c_in,
            self.l_in,
            self.out_l,
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
