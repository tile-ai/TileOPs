import functools
import itertools
from typing import Any, Callable, ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["MaxPool2dKernel", "MaxPool2dWithIndicesKernel"]


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
    plane = out_h * out_w
    total = rows * plane
    rows_inside = _axis_inside(h_in, out_h, kernel_h, stride_h, pad_h, dilation_h)
    cols_inside = _axis_inside(w_in, out_w, kernel_w, stride_w, pad_w, dilation_w)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool2d_func(block_m: int, threads: int):
        tile_full = total % block_m == 0

        def safe_w(iw):
            """*iw* as an index that is always in range, clamping only when the
            window can reach past the row. The clamped value is discarded by the
            update below, so which column it names does not matter."""
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
                        # Two divisions, not three: the remainders come back as
                        # multiplies.
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        row = plane_row // out_h
                        oh = plane_row - row * out_h
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        run = T.alloc_var(T.float32)
                        run = -T.infinity(accum_dtype)
                        for kh in T.serial(kernel_h):
                            ih = top + kh * dilation_h
                            # One test per window row, and the load below needs no
                            # clamp on that axis because the test already proved it.
                            if rows_inside or ((ih >= 0) and (ih < h_in)):
                                for kw in T.serial(kernel_w):
                                    iw = left + kw * dilation_w
                                    # Neighbouring threads hold neighbouring columns,
                                    # so a branch here splits the warp at the row
                                    # edges. The column test rides the update instead.
                                    v = T.cast(x[row, ih, safe_w(iw)], accum_dtype)
                                    live = cols_inside or ((iw >= 0) and (iw < w_in))
                                    # NaN enters `run` and never leaves: a later
                                    # value fails `v > NaN`, which is how PyTorch
                                    # propagates it.
                                    run = T.if_then_else(live and (T.isnan(v) or (v > run)), v, run)
                        out[row, oh, ow] = T.cast(run, dtype)

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
    # The kernel addresses one plane per (batch, channel) pair, so the two leading
    # extents are folded away on the way in and restored on the way out.
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
    window_inside = rows_inside and cols_inside

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

                        max_val = T.alloc_var(T.float32)
                        has_nan = T.alloc_var(T.bool)
                        max_idx = T.alloc_var(T.int32)
                        nan_idx = T.alloc_var(T.int32)
                        first_valid = T.alloc_var(T.bool)
                        top = T.alloc_var(T.int32)
                        left = T.alloc_var(T.int32)
                        base_flat = T.alloc_var(T.int32)
                        max_val = -T.infinity(accum_dtype)
                        has_nan = False
                        first_valid = True
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        base_flat = top * w_in + left
                        if window_inside:
                            # Window element (0, 0) is in bounds here, so its flat
                            # index is the correct seed: an all--inf window reports the
                            # first position, matching PyTorch, and first_valid is
                            # unneeded.
                            max_idx = base_flat
                            nan_idx = base_flat
                        else:
                            max_idx = 0
                            nan_idx = 0
                        for kh in T.serial(kernel_h):
                            ih = top + kh * dilation_h
                            if rows_inside or ((ih >= 0) and (ih < h_in)):
                                for kw in T.serial(kernel_w):
                                    iw = left + kw * dilation_w
                                    # Neighbouring threads hold neighbouring columns,
                                    # so a branch here splits the warp at the row
                                    # edges. The column test rides the update instead.
                                    live = cols_inside or ((iw >= 0) and (iw < w_in))
                                    val = T.cast(x[row, ih, safe_w(iw)], accum_dtype)
                                    flat_idx = (
                                        base_flat + kh * (dilation_h * w_in) + (kw * dilation_w)
                                    )
                                    is_nan = live and T.isnan(val)
                                    # Branch-free update. Strict > keeps the first
                                    # maximum; NaN never touches max_val/max_idx and
                                    # records the last NaN visited, matching PyTorch.
                                    # A window wholly inside the input seeds
                                    # max_val from element (0, 0), so it has no
                                    # first-valid case to carry; the comparison
                                    # still decides.
                                    seed = False if window_inside else first_valid
                                    take = live and (not is_nan) and (seed or (val > max_val))
                                    max_val = T.if_then_else(take, val, max_val)
                                    max_idx = T.if_then_else(take, flat_idx, max_idx)
                                    first_valid = first_valid and ((not live) or is_nan)
                                    nan_idx = T.if_then_else(is_nan, flat_idx, nan_idx)
                                    has_nan = has_nan or is_nan

                        out[row, oh, ow] = T.cast(
                            T.if_then_else(has_nan, T.cast(float("nan"), accum_dtype), max_val),
                            dtype,
                        )
                        indices[row, oh, ow] = T.cast(
                            T.if_then_else(has_nan, nan_idx, max_idx), "int64"
                        )

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

    One thread owns one output position and folds its window into a register, so
    an output is written once and the window never leaves the thread.
    """

    _build = staticmethod(_max_pool2d_kernel)
    _dispatch = staticmethod(_launch_max_pool2d)


class MaxPool2dWithIndicesKernel(_MaxPool2dKernelBase):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool2d_with_indices_kernel)
    _dispatch = staticmethod(_launch_max_pool2d_with_indices)
