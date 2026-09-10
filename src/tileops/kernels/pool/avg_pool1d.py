import functools
from typing import ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import STATIC_SHARED_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import (
    ACCUM_DTYPE,
    AvgPoolWindow,
    WindowSpan,
    dtype_itemsize,
    window_span,
)

__all__ = ["AvgPool1dKernel", "AvgPool1dSpatialKernel"]


class _WindowStaging:
    """What one block stages, and the launch shapes worth offering for it.

    These figures describe this kernel's access pattern, not the device, and are held
    here so that a later kernel does not read them as general truths.
    """

    # Tile widths a launch may take, widest first. The tail below 128 is for a window
    # too wide to stage at 128 outputs.
    _TILE_CHOICES: ClassVar[Tuple[int, ...]] = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
    # Outputs a block covers before the shared budget or the output row narrows it.
    _TILE_OUTPUTS: ClassVar[int] = 512
    _THREAD_CHOICES: ClassVar[Tuple[int, ...]] = (64, 128, 256)
    # Taken when every block size is ruled out.
    _FALLBACK_THREADS: ClassVar[int] = 128
    # Threads a launch needs before a narrow block is offered; under this the blocks
    # do not fill the device.
    _MIN_LAUNCH_THREADS: ClassVar[int] = 1 << 16

    def __init__(self, window: AvgPoolWindow, dtype: str) -> None:
        self._window = window
        self._dtype = dtype
        self._out_l = window.out[0]
        self._kernel_l = window.kernel[0]

    def _span(self, tile_outputs: int) -> WindowSpan:
        window = self._window
        (l_in,), (kernel_l,), (stride_l,), (pad_l,) = (
            window.size,
            window.kernel,
            window.stride,
            window.pad,
        )
        return window_span(
            tile_outputs,
            tile_outputs * stride_l,
            l_in,
            kernel_l,
            stride_l,
            pad_l,
            1,
            self._dtype,
        )

    def widths(self) -> list[int]:
        """The tile widths whose span fits the static shared budget, widest first."""
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
        """``_TILE_OUTPUTS``, narrowed to the shared budget and to the output row."""
        fitting = self.widths()
        wanted = min(self._TILE_OUTPUTS, max(self._out_l, fitting[-1]))
        return next(width for width in fitting if width <= wanted)

    def default(self) -> dict:
        return {"block_ol": self.width(), "threads": self._FALLBACK_THREADS}

    def tuned(self) -> list[dict]:
        """Block sizes worth timing at the chosen width, which is not itself a candidate.

        A block with more threads than the staging pass has loads idles them on the read
        this kernel is bound by.
        """
        width = self.width()
        vectors = self._span(width).vectors
        blocks = self._window.rows * ((self._out_l + width - 1) // width)
        return [
            {"block_ol": width, "threads": threads}
            for threads in self._THREAD_CHOICES
            if vectors >= threads and blocks * threads >= self._MIN_LAUNCH_THREADS
        ] or [{"block_ol": width, "threads": self._FALLBACK_THREADS}]


@functools.lru_cache(maxsize=32)
def _avg_pool1d_kernel(window: AvgPoolWindow, dtype: str):
    """One block per row tile, the stretch it pools staged in shared memory."""
    (l_in,), (kernel_l,), (stride_l,), (pad_l,) = (
        window.size,
        window.kernel,
        window.stride,
        window.pad,
    )
    (out_l,) = window.out
    rows = window.rows
    count_include_pad = window.count_include_pad
    whole_window_divides = window.whole_window_divides
    # The outputs whose window lies inside the row, and so divides by the kernel width.
    clean_lo = -(-pad_l // stride_l)
    clean_hi = min((l_in + pad_l - kernel_l) // stride_l, out_l - 1)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool1d_func(block_ol: int, threads: int):
        # A warp taking one output each reads one short stretch of the row `kernel_l`
        # times over. A block stages that stretch with full-width loads instead.
        vector_elems, head, span = window_span(
            block_ol, block_ol * stride_l, l_in, kernel_l, stride_l, pad_l, 1, dtype
        )
        vectors = span // vector_elems
        # The tile holds zeros outside the row, so a tap needs no test and this offset.
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

            The row covers whole groups, so the test is per group and the load unguarded.
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
            total = T.cast(0.0, ACCUM_DTYPE)
            for k in T.serial(kernel_l):
                total += T.cast(tile[base + j * stride_l + k], ACCUM_DTYPE)
            if whole:
                out[out_row, ol] = T.cast(total / T.cast(kernel_l, ACCUM_DTYPE), dtype)
            else:
                start = ol * stride_l - pad_l
                if count_include_pad:
                    divisor = T.max(T.min(start + kernel_l, l_in + pad_l) - T.max(start, -pad_l), 1)
                else:
                    divisor = T.max(T.min(start + kernel_l, l_in) - T.max(start, 0), 1)
                out[out_row, ol] = T.cast(total / T.cast(divisor, ACCUM_DTYPE), dtype)

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
        self.dtype = dtype
        self.window = AvgPoolWindow(
            rows=n * c_in,
            size=(l_in,),
            kernel=(kernel_l,),
            stride=(stride_l,),
            pad=(pad_l,),
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=None,
        )
        self.kernel = _avg_pool1d_kernel(self.window, self.dtype_str)
        self.init_config(config, tune)

    def _staging(self) -> _WindowStaging:
        return _WindowStaging(self.window, self.dtype_str)

    @property
    def default_config(self) -> dict:
        return self._staging().default()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._staging().tuned()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        kernel = self.kernel(self.config["block_ol"], self.config["threads"])
        rows = kernel(x.contiguous().view(self.window.rows, *self.window.size))
        return rows.view(self.n, self.c_in, *self.window.out)


class AvgPool1dSpatialKernel(_AvgPool1dKernelBase):
    """Fast path for common NCL avg_pool1d workloads: zero-padded, floor-mode."""

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
