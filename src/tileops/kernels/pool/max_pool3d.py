import functools
import itertools
from typing import Any, Callable, ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["MaxPool3dKernel", "MaxPool3dWithIndicesKernel"]


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
    total_output = n * c_in * out_d * out_h * out_w

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool3d_func(block_m: int, threads: int):
        @T.prim_func
        def _max_pool3d_main(
            x: T.Tensor((n, c_in, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ow = out_idx % out_w
                        spatial_idx = out_idx // out_w
                        oh = spatial_idx % out_h
                        depth_idx = spatial_idx // out_h
                        od = depth_idx % out_d
                        channel_batch_idx = depth_idx // out_d
                        c_idx = channel_batch_idx % c_in
                        batch = channel_batch_idx // c_in

                        max_val = T.alloc_var(T.float32)
                        has_nan = T.alloc_var(T.bool)
                        max_val = T.cast(float("-inf"), accum_dtype)
                        has_nan = False
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = od * stride_d - pad_d + kd * dilation_d
                                    ih = oh * stride_h - pad_h + kh * dilation_h
                                    iw = ow * stride_w - pad_w + kw * dilation_w
                                    if (
                                        id_ >= 0
                                        and id_ < d_in
                                        and ih >= 0
                                        and ih < h_in
                                        and iw >= 0
                                        and iw < w_in
                                    ):
                                        val = T.cast(x[batch, c_idx, id_, ih, iw], accum_dtype)
                                        if T.isnan(val):
                                            has_nan = True
                                        max_val = T.max(max_val, val)

                        result = T.if_then_else(
                            has_nan,
                            T.cast(float("nan"), accum_dtype),
                            max_val,
                        )
                        out[batch, c_idx, od, oh, ow] = T.cast(result, dtype)

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
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _max_pool3d_kernel(
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
    )(block_m, threads)(x)


class _MaxPool3dKernelBase(Kernel):
    """Shared construction and dispatch for the 3d max-pool kernels.

    Concrete kernels supply ``_build`` and ``_dispatch``; everything else —
    parameter capture, output extents, config and launch — is identical
    between the value-only and with-indices variants.
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
        return {
            "block_m": 256,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([128, 256, 512], [128, 256, 512])
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
            self.config["block_m"],
            self.config["threads"],
            x,
        )


class MaxPool3dKernel(_MaxPool3dKernelBase):
    """Max pooling forward kernel (return_indices=False)."""

    _build = staticmethod(_max_pool3d_kernel)
    _dispatch = staticmethod(_launch_max_pool3d)


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
    spatial_output = out_d * out_h * out_w
    total_output = n * c_in * spatial_output
    # Static specialization: with zero padding and no ceil overshoot every window
    # lies fully inside the input, so the per-element bounds check can be dropped.
    always_in_bounds = (
        pad_d == 0
        and pad_h == 0
        and pad_w == 0
        and (out_d - 1) * stride_d + (kernel_d - 1) * dilation_d < d_in
        and (out_h - 1) * stride_h + (kernel_h - 1) * dilation_h < h_in
        and (out_w - 1) * stride_w + (kernel_w - 1) * dilation_w < w_in
    )

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool3d_with_indices_func(block_m: int, threads: int):
        @T.prim_func
        def _max_pool3d_with_indices_main(
            x: T.Tensor((n, c_in, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_d, out_h, out_w), dtype),  # type: ignore
            indices: T.Tensor((n, c_in, out_d, out_h, out_w), "int64"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ow = out_idx % out_w
                        spatial_idx = out_idx // out_w
                        oh = spatial_idx % out_h
                        depth_idx = spatial_idx // out_h
                        od = depth_idx % out_d
                        channel_batch_idx = depth_idx // out_d
                        c_idx = channel_batch_idx % c_in
                        batch = channel_batch_idx // c_in

                        max_val = T.alloc_var(T.float32)
                        has_nan = T.alloc_var(T.bool)
                        max_idx = T.alloc_var(T.int32)
                        nan_idx = T.alloc_var(T.int32)
                        first_valid = T.alloc_var(T.bool)
                        # Loop-invariant window corner and flat base, materialized so the
                        # decode is not re-inlined into every window element.
                        id0 = T.alloc_var(T.int32)
                        ih0 = T.alloc_var(T.int32)
                        iw0 = T.alloc_var(T.int32)
                        base_flat = T.alloc_var(T.int32)
                        max_val = T.cast(float("-inf"), accum_dtype)
                        has_nan = False
                        first_valid = True
                        id0 = od * stride_d - pad_d
                        ih0 = oh * stride_h - pad_h
                        iw0 = ow * stride_w - pad_w
                        base_flat = (id0 * h_in + ih0) * w_in + iw0
                        if always_in_bounds:
                            # Window element (0, 0, 0) is in bounds here, so its flat
                            # index is the correct seed: an all--inf window reports the
                            # first position, matching PyTorch, and first_valid is
                            # unneeded.
                            max_idx = base_flat
                            nan_idx = base_flat
                        else:
                            max_idx = 0
                            nan_idx = 0
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = id0 + kd * dilation_d
                                    ih = ih0 + kh * dilation_h
                                    iw = iw0 + kw * dilation_w
                                    if always_in_bounds:
                                        val = T.cast(x[batch, c_idx, id_, ih, iw], accum_dtype)
                                        flat_idx = base_flat + (
                                            kd * (dilation_d * h_in * w_in)
                                            + kh * (dilation_h * w_in)
                                            + kw * dilation_w
                                        )
                                        is_nan = T.isnan(val)
                                        # Branch-free update. Strict > keeps the first
                                        # maximum; NaN never touches max_val/max_idx and
                                        # records the last NaN visited, matching PyTorch.
                                        take = (not is_nan) and (val > max_val)
                                        max_val = T.if_then_else(take, val, max_val)
                                        max_idx = T.if_then_else(take, flat_idx, max_idx)
                                        nan_idx = T.if_then_else(is_nan, flat_idx, nan_idx)
                                        has_nan = has_nan or is_nan
                                    elif (
                                        id_ >= 0
                                        and id_ < d_in
                                        and ih >= 0
                                        and ih < h_in
                                        and iw >= 0
                                        and iw < w_in
                                    ):
                                        val = T.cast(x[batch, c_idx, id_, ih, iw], accum_dtype)
                                        flat_idx = base_flat + (
                                            kd * (dilation_d * h_in * w_in)
                                            + kh * (dilation_h * w_in)
                                            + kw * dilation_w
                                        )
                                        is_nan = T.isnan(val)
                                        take = (not is_nan) and (first_valid or (val > max_val))
                                        max_val = T.if_then_else(take, val, max_val)
                                        max_idx = T.if_then_else(take, flat_idx, max_idx)
                                        first_valid = first_valid and is_nan
                                        nan_idx = T.if_then_else(is_nan, flat_idx, nan_idx)
                                        has_nan = has_nan or is_nan

                        result = T.if_then_else(
                            has_nan,
                            T.cast(float("nan"), accum_dtype),
                            max_val,
                        )
                        out[batch, c_idx, od, oh, ow] = T.cast(result, dtype)
                        indices[batch, c_idx, od, oh, ow] = T.cast(
                            T.if_then_else(
                                has_nan,
                                nan_idx,
                                max_idx,
                            ),
                            "int64",
                        )

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
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _max_pool3d_with_indices_kernel(
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
    )(block_m, threads)(x)


class MaxPool3dWithIndicesKernel(_MaxPool3dKernelBase):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool3d_with_indices_kernel)
    _dispatch = staticmethod(_launch_max_pool3d_with_indices)
