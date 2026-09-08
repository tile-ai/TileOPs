import functools
from typing import ClassVar, NamedTuple, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import dtype_itemsize, pool_output_dim

__all__ = ["AvgPool1dKernel", "AvgPool1dSpatialKernel"]


class _Span(NamedTuple):
    """The stretch of a row one block stages, in the row's own coordinates."""

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


def _round_up(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


def _span(
    tile_outputs: int, l_in: int, kernel_l: int, stride_l: int, pad_l: int, dtype: str
) -> _Span:
    """The stretch of a row a block stages to cover ``tile_outputs`` outputs.

    Every staging load is a whole group of ``vector_elems``, and a group is loaded or
    zeroed as one, so each has to sit either wholly inside the row or wholly outside it.
    The width is narrowed until the three things that move a group boundary all divide
    it: the block step, the head, and the row end.
    """
    vector_elems = VECTOR_ACCESS_BYTES // dtype_itemsize(dtype)
    step = tile_outputs * stride_l
    while vector_elems > 1 and (l_in % vector_elems or step % vector_elems):
        vector_elems //= 2
    head = _round_up(pad_l, vector_elems)
    reach = head + (tile_outputs - 1) * stride_l + kernel_l
    return _Span(vector_elems, head, _round_up(reach, vector_elems))


class _WindowStaging:
    """What one block stages, and the launch shapes worth offering for it.

    These figures describe this kernel's access pattern, not the device, and are held
    here so that a later kernel does not read them as general truths.
    """

    # Tile widths a launch may take, widest first. The tail below 128 is what keeps a
    # window too wide to stage at 128 outputs from having no width at all.
    _TILE_CHOICES: ClassVar[Tuple[int, ...]] = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
    # Outputs a block covers before the shared budget or the output row narrows it.
    _TILE_OUTPUTS: ClassVar[int] = 512
    _THREAD_CHOICES: ClassVar[Tuple[int, ...]] = (64, 128, 256)
    # Taken when every block size is ruled out, which needs a tile of a few outputs.
    _FALLBACK_THREADS: ClassVar[int] = 128
    # Threads a launch needs before a narrow block is worth offering: under this the
    # blocks do not fill the device, and widening each thread's run only lengthens it.
    _MIN_LAUNCH_THREADS: ClassVar[int] = 1 << 16

    def __init__(
        self,
        rows: int,
        l_in: int,
        out_l: int,
        kernel_l: int,
        stride_l: int,
        pad_l: int,
        dtype: str,
    ) -> None:
        self._rows = rows
        self._l_in = l_in
        self._out_l = out_l
        self._kernel_l = kernel_l
        self._stride_l = stride_l
        self._pad_l = pad_l
        self._dtype = dtype

    def _span(self, tile_outputs: int) -> _Span:
        return _span(
            tile_outputs, self._l_in, self._kernel_l, self._stride_l, self._pad_l, self._dtype
        )

    def widths(self) -> list[int]:
        """The tile widths whose span fits the static shared budget, widest first.

        Raises:
            ValueError: When not even one output's window fits, which takes a kernel
                size in the tens of thousands.
        """
        budget = STATIC_SHARED_BYTES // dtype_itemsize(self._dtype)
        fitting = [width for width in self._TILE_CHOICES if self._span(width).span <= budget]
        if not fitting:
            need = self._span(1).span * dtype_itemsize(self._dtype)
            raise ValueError(
                f"avg_pool1d stages the pooling window in shared memory: kernel_size="
                f"{self._kernel_l} needs {need} B for a single output, over the "
                f"{STATIC_SHARED_BYTES} B a block gets"
            )
        return fitting

    def width(self) -> int:
        """``_TILE_OUTPUTS``, narrowed to the shared budget and to the output row.

        A tile wider than the row only idles threads.
        """
        fitting = self.widths()
        wanted = min(self._TILE_OUTPUTS, max(self._out_l, fitting[-1]))
        return next(width for width in fitting if width <= wanted)

    def default(self) -> dict:
        return {"block_ol": self.width(), "threads": self._FALLBACK_THREADS}

    def tuned(self) -> list[dict]:
        """Block sizes worth timing at the chosen width, which is not itself a candidate.

        Two widths a factor of two apart differ by less than the autotuner resolves on a
        launch this short. Two block sizes are dropped: one holding more threads than the
        staging pass has full-width loads idles them on the read this kernel is bound by,
        and one whose launch falls short of ``_MIN_LAUNCH_THREADS`` leaves the device
        unfilled.
        """
        width = self.width()
        vectors = self._span(width).vectors
        blocks = self._rows * ((self._out_l + width - 1) // width)
        return [
            {"block_ol": width, "threads": threads}
            for threads in self._THREAD_CHOICES
            if vectors >= threads and blocks * threads >= self._MIN_LAUNCH_THREADS
        ] or [{"block_ol": width, "threads": self._FALLBACK_THREADS}]


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
        # Outputs `stride_l` apart share taps, so a warp taking one output each reads one
        # short stretch of the row `kernel_l` times over, a narrow load each time. A block
        # stages that stretch with full-width loads and takes every tap from it.
        vector_elems, head, span = _span(block_ol, l_in, kernel_l, stride_l, pad_l, dtype)
        vectors = span // vector_elems
        # The tile holds zeros where it reaches outside the row, so no tap carries a test
        # and its index into the tile is this same constant in every block.
        base = head - pad_l
        blocks_per_row = (out_l + block_ol - 1) // block_ol
        tail_free = out_l % block_ol == 0
        # Only the first and the last block of a row reach outside it.
        edge_free = pad_l == 0 and (blocks_per_row - 1) * block_ol * stride_l + span <= l_in

        @T.macro
        def _stage_inside(tile, x, origin, row):
            """Every group is in the row: one unguarded full-width load each."""
            for i in T.Parallel(vectors):
                for v in T.vectorized(vector_elems):
                    tile[i * vector_elems + v] = x[row, origin + i * vector_elems + v]

        @T.macro
        def _stage_edge(tile, x, origin, row):
            """Zero the tile, then load the groups the row covers.

            The row covers whole groups, so the second pass tests one per group and its
            load stays unguarded.
            """
            for i in T.Parallel(vectors):
                for v in T.vectorized(vector_elems):
                    tile[i * vector_elems + v] = T.cast(0.0, dtype)
            for i in T.Parallel(vectors):
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


class _AvgPool1dKernelBase(Kernel):
    """Shape, launch planning and dispatch shared by the two avg_pool1d kernels.

    The two differ only in which PyTorch flags a caller may set; the staged span and the
    configs worth offering follow from the shape alone.

    Raises:
        ValueError: When one pooling window does not fit the shared memory a block
            stages it in, which takes a ``kernel_size`` in the tens of thousands.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]

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

    def _staging(self) -> _WindowStaging:
        return _WindowStaging(
            self.n * self.c_in,
            self.l_in,
            self.out_l,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.dtype_str,
        )

    @property
    def default_config(self) -> dict:
        return self._staging().default()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._staging().tuned()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        kernel = self.kernel(self.config["block_ol"], self.config["threads"])
        rows = kernel(x.contiguous().view(self.n * self.c_in, self.l_in))
        return rows.view(self.n, self.c_in, self.out_l)


class AvgPool1dSpatialKernel(_AvgPool1dKernelBase):
    """Fast path for common NCL avg_pool1d workloads.

    Zero-padded and floor-mode, so every window spans the full kernel once the padding
    is counted.
    """

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
        super().__init__(n, c_in, l_in, kernel_l, stride_l, pad_l, False, True, dtype, config, tune)


class AvgPool1dKernel(_AvgPool1dKernelBase):
    """Average pooling over an NCL row, with every PyTorch flag combination."""
