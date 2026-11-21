import itertools
from collections.abc import Callable
from typing import Any, ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.pool.call_spec import (
    MaxPool2dFwdInterface,
    MaxPool2dIndicesFwdInterface,
    MaxPoolCall,
)
from tileops._kernels.pool.common import (
    ACCUM_DTYPE,
    MaxPoolKernelBase,
    axis_inside,
    pool_output_dim,
)

__all__ = ["MaxPool2dKernel", "MaxPool2dWarpKernel", "MaxPool2dWithIndicesKernel"]


def _band(tile: int, stride: int, kernel: int, dilation: int) -> int:
    """The input extent *tile* outputs of one thread span on an axis."""
    return (tile - 1) * stride + (kernel - 1) * dilation + 1


@target_cache(maxsize=32)
def _max_pool2d_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
):
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    rows = n * c_in
    rows_inside = axis_inside(h_in, out_h, kernel_h, stride_h, pad_h, dilation_h)
    cols_inside = axis_inside(w_in, out_w, kernel_w, stride_w, pad_w, dilation_w)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool2d_func(block_m: int, threads: int, tile_h: int, tile_w: int):
        band_h = _band(tile_h, stride_h, kernel_h, dilation_h)
        band_w = _band(tile_w, stride_w, kernel_w, dilation_w)
        tiles_h = -(-out_h // tile_h)
        tiles_w = -(-out_w // tile_w)
        total = rows * tiles_h * tiles_w
        tile_full = total % block_m == 0
        tile_exact = out_h % tile_h == 0 and out_w % tile_w == 0
        window_inside = rows_inside and cols_inside
        # Every tile starts the same distance into its row, so one check settles
        # 16-byte alignment for all of them.
        vector_width = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
        vector_band = (
            window_inside and band_w % vector_width == 0 and (tile_w * stride_w) % vector_width == 0
        )

        def safe_h(ih):
            return ih if rows_inside else T.max(0, T.min(ih, h_in - 1))

        def safe_w(iw):
            return iw if cols_inside else T.max(0, T.min(iw, w_in - 1))

        @T.prim_func
        def _max_pool2d_main(
            x: T.Tensor((rows, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, block_m), threads=threads) as tile:
                for i in T.Parallel(block_m):
                    idx = tile * block_m + i
                    if tile_full or idx < total:
                        plane_tile = idx // tiles_w
                        tw = idx - plane_tile * tiles_w
                        row = plane_tile // tiles_h
                        th = plane_tile - row * tiles_h
                        top = th * tile_h * stride_h - pad_h
                        left = tw * tile_w * stride_w - pad_w
                        band = T.alloc_local((band_w,), dtype)
                        across = T.alloc_local((band_h * tile_w,), ACCUM_DTYPE)
                        for r in T.serial(band_h):
                            ih = top + r
                            if vector_band:
                                for run in T.serial(band_w // vector_width):
                                    for t in T.vectorized(vector_width):
                                        band[run * vector_width + t] = x[
                                            row, ih, left + run * vector_width + t
                                        ]
                            else:
                                for t in T.serial(band_w):
                                    iw = left + t
                                    if window_inside:
                                        band[t] = x[row, ih, iw]
                                    else:
                                        # A position outside the input is read
                                        # from a clamped address and replaced, so
                                        # every thread walks the same band.
                                        band[t] = T.if_then_else(
                                            (rows_inside or ((ih >= 0) and (ih < h_in)))
                                            and (cols_inside or ((iw >= 0) and (iw < w_in))),
                                            x[row, safe_h(ih), safe_w(iw)],
                                            -T.infinity(dtype),
                                        )
                            # Row maxima: the column pass below reads each one
                            # kernel_h times, so the band is read once.
                            for j in T.serial(tile_w):
                                run = T.alloc_var(T.float32)
                                run = T.cast(band[j * stride_w], ACCUM_DTYPE)
                                for kw in T.serial(kernel_w - 1):
                                    v = T.cast(
                                        band[j * stride_w + (kw + 1) * dilation_w],
                                        ACCUM_DTYPE,
                                    )
                                    # NaN enters `run` and never leaves, since a
                                    # later value fails `v > NaN`.
                                    run = T.if_then_else(T.isnan(v) or (v > run), v, run)
                                across[r * tile_w + j] = run
                        for a in T.serial(tile_h):
                            for j in T.serial(tile_w):
                                run = T.alloc_var(T.float32)
                                run = across[(a * stride_h) * tile_w + j]
                                for kh in T.serial(kernel_h - 1):
                                    v = across[(a * stride_h + (kh + 1) * dilation_h) * tile_w + j]
                                    run = T.if_then_else(T.isnan(v) or (v > run), v, run)
                                oh_ = th * tile_h + a
                                ow_ = tw * tile_w + j
                                if tile_exact or ((oh_ < out_h) and (ow_ < out_w)):
                                    out[row, oh_, ow_] = T.cast(run, dtype)

        return _max_pool2d_main

    return _max_pool2d_func


@target_cache(maxsize=32)
def _max_pool2d_with_indices_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
):
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    rows = n * c_in
    plane = out_h * out_w
    total = rows * plane
    rows_inside = axis_inside(h_in, out_h, kernel_h, stride_h, pad_h, dilation_h)
    cols_inside = axis_inside(w_in, out_w, kernel_w, stride_w, pad_w, dilation_w)

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool2d_with_indices_func(block_m: int, threads: int):
        tile_full = total % block_m == 0

        def safe_w(iw):
            """*iw* as an index that is always in range; see the value-only kernel."""
            return iw if cols_inside else T.max(0, T.min(iw, w_in - 1))

        @T.prim_func
        def _max_pool2d_with_indices_main(
            x: T.Tensor((rows, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_h, out_w), dtype),  # type: ignore
            indices: T.Tensor((rows, out_h, out_w), "int64"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, block_m), threads=threads) as tile:
                for i in T.Parallel(block_m):
                    idx = tile * block_m + i
                    if tile_full or idx < total:
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        row = plane_row // out_h
                        oh = plane_row - row * out_h
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        run = T.alloc_var(T.float32)
                        best = T.alloc_var(T.int32)
                        run = -T.infinity(ACCUM_DTYPE)
                        # Seeded at the window's first tap inside the input, so a
                        # window of nothing but -inf reports that tap.
                        best = (top + T.max(0, T.ceildiv(-top, dilation_h)) * dilation_h) * w_in + (
                            left + T.max(0, T.ceildiv(-left, dilation_w)) * dilation_w
                        )
                        for kh in T.serial(kernel_h):
                            ih = top + kh * dilation_h
                            if rows_inside or ((ih >= 0) and (ih < h_in)):
                                for kw in T.serial(kernel_w):
                                    iw = left + kw * dilation_w
                                    # Why: a branch on the column splits the warp
                                    # at the row edges; the test rides the update.
                                    live = cols_inside or ((iw >= 0) and (iw < w_in))
                                    v = T.cast(x[row, ih, safe_w(iw)], ACCUM_DTYPE)
                                    # Strict > keeps the first maximum; a NaN takes
                                    # the position and holds it, so the last NaN in
                                    # the window wins.
                                    take = live and (T.isnan(v) or (v > run))
                                    run = T.if_then_else(take, v, run)
                                    best = T.if_then_else(take, ih * w_in + iw, best)
                        out[row, oh, ow] = T.cast(run, dtype)
                        indices[row, oh, ow] = T.cast(best, "int64")

        return _max_pool2d_with_indices_main

    return _max_pool2d_with_indices_func


class _MaxPool2dKernelBase(MaxPoolKernelBase):
    """The constructor both 2d max-pool kernels share; a subclass names its ``_build``."""

    _build: ClassVar[Callable[..., Any]]

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        ceil_mode: bool,
        dtype: torch.dtype,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"{type(self).__name__} supports float16, bfloat16, and float32, got {dtype}"
            )
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.dtype = dtype
        self.out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
        self.out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
        self.kernel = type(self)._build(
            n,
            c_in,
            h_in,
            w_in,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            ceil_mode,
            self.dtype_str,
        )
        self.init_config(config)


@T.macro
def _nan_max(a, b):
    """*b* where it is NaN or larger than *a*, so a NaN tap wins as in the generic kernel."""
    return T.if_then_else(T.isnan(b) or b > a, b, a)


@target_cache(maxsize=32)
def _max_pool2d_warp_kernel(planes: int, h_in: int, w_in: int, dtype: str):
    out_h, out_w = (h_in - 1) // 2, (w_in - 1) // 2
    # Three warps split the output rows; a lane takes one of at most 31 output columns.
    rows_per_warp = (out_h + 2) // 3
    band_h = 2 * rows_per_warp + 1

    @tilelang.jit(out_idx=[1])
    def build():
        @T.prim_func
        def main(
            x: T.Tensor[(planes, h_in, w_in), dtype],
            y: T.Tensor[(planes, out_h, out_w), dtype],
        ):
            with T.Kernel(planes, threads=96) as plane:
                thread = T.get_thread_binding()
                column = thread % 32
                row = thread // 32 * rows_per_warp
                across = T.alloc_local((band_h,), "float32")
                # A lane reads two columns and gets the rightmost tap from its neighbour.
                # Reducing across each row first preserves the generic kernel's tie order.
                for i in T.unroll(band_h):
                    ih = T.min(row * 2 + i, h_in - 1)
                    left = T.cast(x[plane, ih, T.min(column * 2, w_in - 1)], "float32")
                    middle = T.cast(x[plane, ih, T.min(column * 2 + 1, w_in - 1)], "float32")
                    right = T.shfl_down(left, 1)
                    across[i] = _nan_max(_nan_max(left, middle), right)
                for j in T.unroll(rows_per_warp):
                    if row + j < out_h and column < out_w:
                        upper = _nan_max(across[j * 2], across[j * 2 + 1])
                        y[plane, row + j, column] = T.cast(
                            _nan_max(upper, across[j * 2 + 2]), dtype
                        )

        return main

    return build


@target_cache(maxsize=32)
def _max_pool2d_half_warp_kernel(planes: int):
    """Pool 33×33 maps, assigning a half-warp to two rows of sixteen outputs."""

    dtype = "bfloat16"

    @tilelang.jit(out_idx=[1])
    def build():
        @T.prim_func
        def main(
            x: T.Tensor[(planes, 33, 33), dtype],
            y: T.Tensor[(planes, 16, 16), dtype],
        ):
            with T.Kernel(planes, threads=128) as plane:
                thread = T.get_thread_binding()
                column = thread % 16
                band = thread // 16
                rows = T.alloc_local((5,), "float32")
                for r in T.unroll(5):
                    left = T.cast(x[plane, 4 * band + r, 2 * column], "float32")
                    middle = T.cast(x[plane, 4 * band + r, 2 * column + 1], "float32")
                    right = T.alloc_var("float32", init=T.shfl_down(left, 1, width=16))
                    if column == 15:
                        right = T.cast(x[plane, 4 * band + r, 32], "float32")
                    rows[r] = _nan_max(_nan_max(left, middle), right)
                for r in T.unroll(2):
                    y[plane, 2 * band + r, column] = _nan_max(
                        _nan_max(rows[2 * r], rows[2 * r + 1]), rows[2 * r + 2]
                    )

        return main

    return build


class MaxPool2dWarpKernel(Kernel, MaxPool2dFwdInterface):
    """Share adjacent, unpadded 3-by-3 FP16 or BF16 pooling windows within a warp."""

    supported_archs = [90]
    preferred_over = frozenset({"max_pool2d_kernel"})

    @classmethod
    def refusal(cls, call: MaxPoolCall) -> str | None:
        if call.window != (3, 3) or call.stride != (2, 2):
            return "requires a 3-by-3 window with stride two"
        if (
            call.pad != (0, 0)
            or call.dilation != (1, 1)
            or call.dtype not in (torch.float16, torch.bfloat16)
        ):
            return "requires FP16 or BF16, no padding, and unit dilation"
        h_in, w_in = call.size
        if ((h_in, w_in) != (33, 33) or call.dtype != torch.bfloat16) and not all(
            47 <= extent <= 63 and extent % 2 for extent in (h_in, w_in)
        ):
            return "requires a 33-by-33 input or odd spatial extents from 47 to 63"
        if call.n * call.c_in * h_in * w_in >= 2**31:
            return "requires fewer than 2**31 input elements for 32-bit indexing"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: MaxPoolCall) -> Entry:
        index = call.device.index if call.device is not None else None
        args = (call.n * call.c_in, *call.size, call.dtype)
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(self, planes, h_in, w_in, dtype, *, device_index=None):
        super().__init__(device_index=device_index)
        self.planes, self.h_in, self.w_in = planes, h_in, w_in
        self.dtype = dtype
        self.half_warp = (h_in, w_in) == (33, 33)
        self.kernel = (
            _max_pool2d_half_warp_kernel(planes)
            if self.half_warp
            else _max_pool2d_warp_kernel(planes, h_in, w_in, self.dtype_str)
        )
        self.init_config()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.program(lambda config: self.kernel(**config))(
            x.reshape(self.planes, self.h_in, self.w_in)
        )
        return y.view(x.shape[0], x.shape[1], (self.h_in - 1) // 2, (self.w_in - 1) // 2)


class MaxPool2dKernel(_MaxPool2dKernelBase, MaxPool2dFwdInterface):
    """Max pooling forward kernel (return_indices=False).

    One thread owns a ``tile_h`` by ``tile_w`` block of outputs. The input band
    those outputs share is read once into registers, and the window separates:
    each band row gives its row maxima, and the column pass takes the maximum
    down them. A tile of 1 by 1 is the plain one-output-per-thread schedule.
    """

    def _launch(self, program, x: torch.Tensor) -> torch.Tensor:
        planes = program(x.reshape(self.n * self.c_in, self.h_in, self.w_in))
        return planes.view(self.n, self.c_in, self.out_h, self.out_w)

    _build = staticmethod(_max_pool2d_kernel)

    def _tiles(self, axis: str) -> tuple[int, ...]:
        """Tile extents worth trying on one axis.

        A tile only pays where the band it reads is the band its outputs use.
        Once the stride outruns the kernel, or the dilation outruns the stride,
        the band holds positions no window reaches and the extra reads are lost.
        """
        if axis == "h":
            extent, stride, kernel, dilation = (
                self.out_h,
                self.stride_h,
                self.kernel_h,
                self.dilation_h,
            )
        else:
            extent, stride, kernel, dilation = (
                self.out_w,
                self.stride_w,
                self.kernel_w,
                self.dilation_w,
            )
        if not (dilation <= stride <= kernel):
            return (1,)
        return tuple(t for t in (1, 2, 4, 8) if t <= extent)

    @property
    def default_config(self) -> dict:
        if (
            self.build_target.arch == 90
            and self.dtype == torch.bfloat16
            and self.kernel_h == self.kernel_w == 3
            and self.stride_h == self.stride_w == 2
            and self.pad_h == self.pad_w == 0
            and self.dilation_h == self.dilation_w == 1
            and self.h_in == self.w_in == 65
        ):
            return {"block_m": 128, "threads": 128, "tile_h": 8, "tile_w": 1}
        tiles_h = self._tiles("h")
        tiles_w = self._tiles("w")
        return {
            "block_m": 256,
            "threads": 256,
            "tile_h": max(t for t in tiles_h if t <= 4),
            "tile_w": max(t for t in tiles_w if t <= 2),
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # Accumulators one thread holds for its output tile: the band row plus the row
        # maxima the column pass consumes.
        max_tile_registers = 64
        space = itertools.product(
            (64, 128, 256, 512), (64, 128, 256), self._tiles("h"), self._tiles("w")
        )
        configs = [
            {
                "block_m": block_m,
                "threads": threads,
                "tile_h": tile_h,
                "tile_w": tile_w,
            }
            for block_m, threads, tile_h, tile_w in space
            if threads <= block_m
            and _band(tile_h, self.stride_h, self.kernel_h, self.dilation_h) * tile_w
            + _band(tile_w, self.stride_w, self.kernel_w, self.dilation_w)
            <= max_tile_registers
        ]
        return configs or [self.default_config]


class MaxPool2dWithIndicesKernel(_MaxPool2dKernelBase, MaxPool2dIndicesFwdInterface):
    """Max pooling forward-with-indices kernel."""

    def _launch(self, program, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, positions = program(x.reshape(self.n * self.c_in, self.h_in, self.w_in))
        shape = (self.n, self.c_in, self.out_h, self.out_w)
        return values.view(shape), positions.view(shape)

    _build = staticmethod(_max_pool2d_with_indices_kernel)
