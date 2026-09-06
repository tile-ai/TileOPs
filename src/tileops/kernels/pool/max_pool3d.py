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
                        # Three divisions, not five: the remainders come back as
                        # multiplies.
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        depth_row = plane_row // out_h
                        oh = plane_row - depth_row * out_h
                        row = depth_row // out_d
                        od = depth_row - row * out_d
                        front = od * stride_d - pad_d
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        # A local array, not a scalar: an accumulator written under
                        # interleaved loops and predicates does not carry across
                        # iterations as a scalar here.
                        run = T.alloc_local((1,), accum_dtype)
                        run[0] = -T.infinity(accum_dtype)
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = front + kd * dilation_d
                                    ih = top + kh * dilation_h
                                    iw = left + kw * dilation_w
                                    # One predicate for the whole window position.
                                    # Splitting it per axis, or trading it for a
                                    # select over a clamped address, both measured
                                    # slower on a 27-tap window: skipping the tap
                                    # is worth more than the branch costs.
                                    if window_inside or (
                                        (id_ >= 0)
                                        and (id_ < d_in)
                                        and (ih >= 0)
                                        and (ih < h_in)
                                        and (iw >= 0)
                                        and (iw < w_in)
                                    ):
                                        v = T.cast(x[row, id_, ih, iw], accum_dtype)
                                        # NaN enters `run` and never leaves: a later
                                        # value fails `v > NaN`, which is how PyTorch
                                        # propagates it.
                                        run[0] = T.if_then_else(
                                            T.isnan(v) or (v > run[0]), v, run[0]
                                        )
                        out[row, od, oh, ow] = T.cast(run[0], dtype)

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
    # The kernel addresses one volume per (batch, channel) pair, so the two
    # leading extents are folded away on the way in and restored on the way out.
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

                        # Local arrays, not scalars: a value written under
                        # interleaved loops and predicates does not carry across
                        # iterations as a scalar here.
                        max_val = T.alloc_local((1,), accum_dtype)
                        has_nan = T.alloc_local((1,), "bool")
                        max_idx = T.alloc_local((1,), "int32")
                        nan_idx = T.alloc_local((1,), "int32")
                        first_valid = T.alloc_local((1,), "bool")
                        front = od * stride_d - pad_d
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        base_flat = (front * h_in + top) * w_in + left
                        max_val[0] = -T.infinity(accum_dtype)
                        has_nan[0] = False
                        first_valid[0] = True
                        if window_inside:
                            # Window element (0, 0, 0) is in bounds here, so its flat
                            # index is the correct seed: an all--inf window reports
                            # the first position, matching PyTorch, and first_valid
                            # is unneeded.
                            max_idx[0] = base_flat
                            nan_idx[0] = base_flat
                        else:
                            max_idx[0] = 0
                            nan_idx[0] = 0
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = front + kd * dilation_d
                                    ih = top + kh * dilation_h
                                    iw = left + kw * dilation_w
                                    # One predicate for the whole window position;
                                    # see the value-only kernel for why it is not
                                    # split per axis.
                                    if window_inside or (
                                        (id_ >= 0)
                                        and (id_ < d_in)
                                        and (ih >= 0)
                                        and (ih < h_in)
                                        and (iw >= 0)
                                        and (iw < w_in)
                                    ):
                                        val = T.cast(x[row, id_, ih, iw], accum_dtype)
                                        flat_idx = base_flat + (
                                            kd * (dilation_d * h_in * w_in)
                                            + kh * (dilation_h * w_in)
                                            + kw * dilation_w
                                        )
                                        is_nan = T.isnan(val)
                                        # Branch-free update. Strict > keeps the
                                        # first maximum; NaN never touches
                                        # max_val/max_idx and records the last NaN
                                        # visited, matching PyTorch.
                                        #
                                        # A window wholly inside the input seeds
                                        # max_val from element (0, 0, 0), so it has
                                        # no first-valid case to carry; the
                                        # comparison still decides.
                                        seed = False if window_inside else first_valid[0]
                                        take = (not is_nan) and (seed or (val > max_val[0]))
                                        max_val[0] = T.if_then_else(take, val, max_val[0])
                                        max_idx[0] = T.if_then_else(take, flat_idx, max_idx[0])
                                        first_valid[0] = first_valid[0] and is_nan
                                        nan_idx[0] = T.if_then_else(is_nan, flat_idx, nan_idx[0])
                                        has_nan[0] = has_nan[0] or is_nan

                        out[row, od, oh, ow] = T.cast(
                            T.if_then_else(
                                has_nan[0], T.cast(float("nan"), accum_dtype), max_val[0]
                            ),
                            dtype,
                        )
                        indices[row, od, oh, ow] = T.cast(
                            T.if_then_else(has_nan[0], nan_idx[0], max_idx[0]), "int64"
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
