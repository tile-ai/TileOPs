import functools
import itertools
from typing import Any, Callable, ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["MaxPool2dKernel", "MaxPool2dWithIndicesKernel"]

# Accumulators one thread holds for its output tile: the band row plus the row
# maxima the column pass consumes.
_MAX_TILE_REGISTERS = 64


def _axis_inside(
    size_in: int, size_out: int, kernel: int, stride: int, pad: int, dilation: int
) -> bool:
    """Whether every window on this axis lies inside the input.

    True makes the axis's bounds test a compile-time truth, and it drops out.
    """
    return pad == 0 and (size_out - 1) * stride + (kernel - 1) * dilation < size_in


@functools.lru_cache(maxsize=32)
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
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    rows = n * c_in
    rows_inside = _axis_inside(h_in, out_h, kernel_h, stride_h, pad_h, dilation_h)
    cols_inside = _axis_inside(w_in, out_w, kernel_w, stride_w, pad_w, dilation_w)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool2d_func(block_m: int, threads: int, tile_h: int, tile_w: int):
        # Input extent one thread's outputs span on each axis.
        band_h = (tile_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1
        band_w = (tile_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1
        tiles_h = -(-out_h // tile_h)
        tiles_w = -(-out_w // tile_w)
        total = rows * tiles_h * tiles_w
        tile_full = total % block_m == 0
        tile_exact = out_h % tile_h == 0 and out_w % tile_w == 0
        window_inside = rows_inside and cols_inside
        # Every tile starts the same distance into its row, so one check settles
        # 16-byte alignment for all of them.
        vector_width = 16 // torch.empty((), dtype=getattr(torch, dtype)).element_size()
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
                        across = T.alloc_local((band_h * tile_w,), accum_dtype)
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
                                run = T.cast(band[j * stride_w], accum_dtype)
                                for kw in T.serial(kernel_w - 1):
                                    v = T.cast(
                                        band[j * stride_w + (kw + 1) * dilation_w],
                                        accum_dtype,
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


def _launch_max_pool2d(
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
    config: dict,
    x: torch.Tensor,
) -> torch.Tensor:
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    kernel = _max_pool2d_kernel(
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
        dtype,
    )(**config)
    return kernel(x.reshape(n * c_in, h_in, w_in)).view(n, c_in, out_h, out_w)


@functools.lru_cache(maxsize=32)
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
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    rows = n * c_in
    plane = out_h * out_w
    total = rows * plane
    rows_inside = _axis_inside(h_in, out_h, kernel_h, stride_h, pad_h, dilation_h)
    cols_inside = _axis_inside(w_in, out_w, kernel_w, stride_w, pad_w, dilation_w)

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
                        run = -T.infinity(accum_dtype)
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
                                    v = T.cast(x[row, ih, safe_w(iw)], accum_dtype)
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


def _launch_max_pool2d_with_indices(
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
    config: dict,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    values, positions = _max_pool2d_with_indices_kernel(
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
        dtype,
    )(**config)(x.reshape(n * c_in, h_in, w_in))
    return (
        values.view(n, c_in, out_h, out_w),
        positions.view(n, c_in, out_h, out_w),
    )


class _MaxPool2dKernelBase(Kernel):
    """Shared construction and dispatch for the 2d max-pool kernels.

    Concrete kernels supply ``_build`` and ``_dispatch``; everything else —
    parameter capture, output extents, config and launch — is identical between
    the value-only and with-indices variants.
    """

    _build: ClassVar[Callable[..., Any]]
    _dispatch: ClassVar[Callable[..., Any]]

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]

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
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
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
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 512, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([256, 512, 1024, 2048], [128, 256, 512])
            if threads <= block_m
        ]

    def forward(self, x: torch.Tensor) -> Any:
        self._require_cuda(x=x)
        return type(self)._dispatch(
            self.n,
            self.c_in,
            self.h_in,
            self.w_in,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dilation_h,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            dict(self.config),
            x,
        )


class MaxPool2dKernel(_MaxPool2dKernelBase):
    """Max pooling forward kernel (return_indices=False).

    One thread owns a ``tile_h`` by ``tile_w`` block of outputs. The input band
    those outputs share is read once into registers, and the window separates:
    each band row gives its row maxima, and the column pass takes the maximum
    down them. A tile of 1 by 1 is the plain one-output-per-thread schedule.
    """

    _build = staticmethod(_max_pool2d_kernel)
    _dispatch = staticmethod(_launch_max_pool2d)

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
            and self._band(tile_h, self.stride_h, self.kernel_h, self.dilation_h) * tile_w
            + self._band(tile_w, self.stride_w, self.kernel_w, self.dilation_w)
            <= _MAX_TILE_REGISTERS
        ]
        return configs or [self.default_config]

    @staticmethod
    def _band(tile: int, stride: int, kernel: int, dilation: int) -> int:
        return (tile - 1) * stride + (kernel - 1) * dilation + 1


class MaxPool2dWithIndicesKernel(_MaxPool2dKernelBase):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool2d_with_indices_kernel)
    _dispatch = staticmethod(_launch_max_pool2d_with_indices)
