import functools
import itertools
from typing import Any, Callable, ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["MaxPool3dKernel", "MaxPool3dWithIndicesKernel"]


def _axis_inside(
    size_in: int, size_out: int, kernel: int, stride: int, pad: int, dilation: int
) -> bool:
    """Whether every window on this axis lies inside the input.

    True makes the axis's bounds test a compile-time truth, and it drops out.
    """
    return pad == 0 and (size_out - 1) * stride + (kernel - 1) * dilation < size_in


@functools.lru_cache(maxsize=32)
def _max_pool3d_kernel(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, ceil_mode, dilation_d)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    rows = n * c_in
    total = rows * out_d * out_h * out_w
    window_inside = (
        _axis_inside(d_in, out_d, kernel_d, stride_d, pad_d, dilation_d)
        and _axis_inside(h_in, out_h, kernel_h, stride_h, pad_h, dilation_h)
        and _axis_inside(w_in, out_w, kernel_w, stride_w, pad_w, dilation_w)
    )

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool3d_func(block_m: int, threads: int):
        tile_full = total % block_m == 0

        @T.prim_func
        def _max_pool3d_main(
            x: T.Tensor((rows, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, block_m), threads=threads) as tile:
                for i in T.Parallel(block_m):
                    idx = tile * block_m + i
                    if tile_full or idx < total:
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        depth_row = plane_row // out_h
                        oh = plane_row - depth_row * out_h
                        row = depth_row // out_d
                        od = depth_row - row * out_d
                        front = od * stride_d - pad_d
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        run = T.alloc_var(T.float32)
                        run = -T.infinity(accum_dtype)
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = front + kd * dilation_d
                                    ih = top + kh * dilation_h
                                    iw = left + kw * dilation_w
                                    # Why: over this many taps, skipping the tap
                                    # beats keeping the warp together.
                                    if window_inside or (
                                        (id_ >= 0)
                                        and (id_ < d_in)
                                        and (ih >= 0)
                                        and (ih < h_in)
                                        and (iw >= 0)
                                        and (iw < w_in)
                                    ):
                                        v = T.cast(x[row, id_, ih, iw], accum_dtype)
                                        # NaN enters `run` and never leaves,
                                        # since a later value fails `v > NaN`.
                                        run = T.if_then_else(T.isnan(v) or (v > run), v, run)
                        out[row, od, oh, ow] = T.cast(run, dtype)

        return _max_pool3d_main

    return _max_pool3d_func


def _launch_max_pool3d(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    config: dict,
    x: torch.Tensor,
) -> torch.Tensor:
    out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, ceil_mode, dilation_d)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    kernel = _max_pool3d_kernel(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        ceil_mode,
        dtype,
    )(**config)
    return kernel(x.reshape(n * c_in, d_in, h_in, w_in)).view(n, c_in, out_d, out_h, out_w)


@functools.lru_cache(maxsize=32)
def _max_pool3d_with_indices_kernel(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, ceil_mode, dilation_d)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    rows = n * c_in
    total = rows * out_d * out_h * out_w
    window_inside = (
        _axis_inside(d_in, out_d, kernel_d, stride_d, pad_d, dilation_d)
        and _axis_inside(h_in, out_h, kernel_h, stride_h, pad_h, dilation_h)
        and _axis_inside(w_in, out_w, kernel_w, stride_w, pad_w, dilation_w)
    )

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool3d_with_indices_func(block_m: int, threads: int):
        tile_full = total % block_m == 0

        @T.prim_func
        def _max_pool3d_with_indices_main(
            x: T.Tensor((rows, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_d, out_h, out_w), dtype),  # type: ignore
            indices: T.Tensor((rows, out_d, out_h, out_w), "int64"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, block_m), threads=threads) as tile:
                for i in T.Parallel(block_m):
                    idx = tile * block_m + i
                    if tile_full or idx < total:
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        depth_row = plane_row // out_h
                        oh = plane_row - depth_row * out_h
                        row = depth_row // out_d
                        od = depth_row - row * out_d
                        front = od * stride_d - pad_d
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        run = T.alloc_var(T.float32)
                        best = T.alloc_var(T.int32)
                        run = -T.infinity(accum_dtype)
                        # Seeded at the window's first tap inside the input, so a
                        # window of nothing but -inf reports that tap.
                        best = (
                            (front + T.max(0, T.ceildiv(-front, dilation_d)) * dilation_d) * h_in
                            + (top + T.max(0, T.ceildiv(-top, dilation_h)) * dilation_h)
                        ) * w_in + (left + T.max(0, T.ceildiv(-left, dilation_w)) * dilation_w)
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = front + kd * dilation_d
                                    ih = top + kh * dilation_h
                                    iw = left + kw * dilation_w
                                    if window_inside or (
                                        (id_ >= 0)
                                        and (id_ < d_in)
                                        and (ih >= 0)
                                        and (ih < h_in)
                                        and (iw >= 0)
                                        and (iw < w_in)
                                    ):
                                        v = T.cast(x[row, id_, ih, iw], accum_dtype)
                                        # Strict > keeps the first maximum; a NaN
                                        # takes the position and holds it, so the
                                        # last NaN in the window wins.
                                        take = T.isnan(v) or (v > run)
                                        run = T.if_then_else(take, v, run)
                                        best = T.if_then_else(
                                            take, (id_ * h_in + ih) * w_in + iw, best
                                        )
                        out[row, od, oh, ow] = T.cast(run, dtype)
                        indices[row, od, oh, ow] = T.cast(best, "int64")

        return _max_pool3d_with_indices_main

    return _max_pool3d_with_indices_func


def _launch_max_pool3d_with_indices(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    config: dict,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, ceil_mode, dilation_d)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    values, positions = _max_pool3d_with_indices_kernel(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        ceil_mode,
        dtype,
    )(**config)(x.reshape(n * c_in, d_in, h_in, w_in))
    shape = (n, c_in, out_d, out_h, out_w)
    return values.view(shape), positions.view(shape)


class _MaxPool3dKernelBase(Kernel):
    """Shared construction and dispatch for the 3d max-pool kernels.

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
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dilation_d: int,
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
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_d = kernel_d
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_d = stride_d
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_d = dilation_d
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, ceil_mode, dilation_d)
        self.out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
        self.out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
        self.kernel = type(self)._build(
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            dilation_d,
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
            self.d_in,
            self.h_in,
            self.w_in,
            self.kernel_d,
            self.kernel_h,
            self.kernel_w,
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.pad_d,
            self.pad_h,
            self.pad_w,
            self.dilation_d,
            self.dilation_h,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            dict(self.config),
            x,
        )


class MaxPool3dKernel(_MaxPool3dKernelBase):
    """Max pooling forward kernel (return_indices=False).

    One thread owns one output position and folds its window into a register, so
    an output is written once and the window never leaves the thread.
    """

    _build = staticmethod(_max_pool3d_kernel)
    _dispatch = staticmethod(_launch_max_pool3d)


class MaxPool3dWithIndicesKernel(_MaxPool3dKernelBase):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool3d_with_indices_kernel)
    _dispatch = staticmethod(_launch_max_pool3d_with_indices)
