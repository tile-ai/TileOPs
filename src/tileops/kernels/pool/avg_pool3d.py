import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["AvgPool3dKernel", "AvgPool3dSpatialKernel"]


def _axis_inside(size_in: int, size_out: int, kernel: int, stride: int, pad: int) -> bool:
    """Whether every window on this axis lies inside the input.

    True makes the axis's bounds test a compile-time truth, and it drops out.
    """
    return pad == 0 and (size_out - 1) * stride + kernel <= size_in


# Wider than the 32 a single entry point needs: this cache now serves both, since
# _avg_pool3d_spatial_kernel resolves to an entry here rather than holding its own.
@functools.lru_cache(maxsize=64)
def _avg_pool3d_kernel(
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
    ceil_mode: bool,
    count_include_pad: bool,
    use_divisor_override: bool,
    divisor_override: int,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, ceil_mode)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode)
    rows = n * c_in
    total = rows * out_d * out_h * out_w
    window_inside = (
        _axis_inside(d_in, out_d, kernel_d, stride_d, pad_d)
        and _axis_inside(h_in, out_h, kernel_h, stride_h, pad_h)
        and _axis_inside(w_in, out_w, kernel_w, stride_w, pad_w)
    )
    # Without ceil mode a window reaches `size + pad` at the furthest, so counting
    # the padding gives every output the whole kernel. Ceil mode can overhang that,
    # and uncounted padding shortens the windows that do.
    whole_window_divides = window_inside or (count_include_pad and not ceil_mode)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool3d_func(block_m: int, threads: int):
        tile_full = total % block_m == 0

        @T.prim_func
        def _avg_pool3d_main(
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
                        total_val = T.alloc_var(T.float32)
                        total_val = T.cast(0.0, accum_dtype)
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = front + kd
                                    ih = top + kh
                                    iw = left + kw
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
                                        total_val += T.cast(x[row, id_, ih, iw], accum_dtype)
                        # An explicit divisor is used as given, negative included.
                        # Only a divisor read off the window takes a floor, or an
                        # empty window divides by zero.
                        if use_divisor_override:
                            divisor = T.cast(divisor_override, accum_dtype)
                        elif whole_window_divides:
                            divisor = T.cast(kernel_d * kernel_h * kernel_w, accum_dtype)
                        elif count_include_pad:
                            divisor = T.cast(
                                T.max(
                                    T.max(
                                        T.max(T.min(front + kernel_d, d_in + pad_d), 0)
                                        - T.max(front, -pad_d),
                                        0,
                                    )
                                    * T.max(
                                        T.max(T.min(top + kernel_h, h_in + pad_h), 0)
                                        - T.max(top, -pad_h),
                                        0,
                                    )
                                    * T.max(
                                        T.max(T.min(left + kernel_w, w_in + pad_w), 0)
                                        - T.max(left, -pad_w),
                                        0,
                                    ),
                                    1,
                                ),
                                accum_dtype,
                            )
                        else:
                            divisor = T.cast(
                                T.max(
                                    T.max(
                                        T.max(T.min(front + kernel_d, d_in), 0) - T.max(front, 0),
                                        0,
                                    )
                                    * T.max(
                                        T.max(T.min(top + kernel_h, h_in), 0) - T.max(top, 0),
                                        0,
                                    )
                                    * T.max(
                                        T.max(T.min(left + kernel_w, w_in), 0) - T.max(left, 0),
                                        0,
                                    ),
                                    1,
                                ),
                                accum_dtype,
                            )
                        out[row, od, oh, ow] = T.cast(
                            total_val / divisor,
                            dtype,
                        )

        return _avg_pool3d_main

    return _avg_pool3d_func


def _avg_pool3d_spatial_kernel(
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
    dtype: str = "float16",
):
    """Zero-padded, floor-mode 3d average pooling with no divisor override.

    Every window then spans the full kernel once the padding is counted, which is
    what ``_avg_pool3d_kernel`` emits for ``ceil_mode=False``,
    ``count_include_pad=True`` and no override.
    """
    return _avg_pool3d_kernel(
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
        False,
        True,
        False,
        0,
        dtype,
    )


def _launch_avg_pool3d_spatial(
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
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _launch_avg_pool3d(
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
        False,
        True,
        False,
        0,
        dtype,
        block_m,
        threads,
        x,
    )


def _launch_avg_pool3d(
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
    ceil_mode: bool,
    count_include_pad: bool,
    use_divisor_override: bool,
    divisor_override: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, ceil_mode)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode)
    kernel = _avg_pool3d_kernel(
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
        ceil_mode,
        count_include_pad,
        use_divisor_override,
        divisor_override,
        dtype,
    )(block_m, threads)
    return kernel(x.reshape(n * c_in, d_in, h_in, w_in)).view(n, c_in, out_d, out_h, out_w)


class AvgPool3dSpatialKernel(Kernel):
    """Fast path for common NCDHW avg_pool3d workloads."""

    supported_archs: list[int] = [80, 86, 89, 90]

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
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
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
        self.dtype = dtype
        self.out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, False)
        self.out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, False)
        self.out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, False)

        self.kernel = _avg_pool3d_spatial_kernel(
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
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([64, 128, 256], [128, 256])
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        return _launch_avg_pool3d_spatial(
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
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )


class AvgPool3dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

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
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
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
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.use_divisor_override = divisor_override is not None
        self.divisor_override = divisor_override or 0
        self.dtype = dtype
        self.out_d = pool_output_dim(d_in, kernel_d, stride_d, pad_d, ceil_mode)
        self.out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode)
        self.out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode)

        self.kernel = _avg_pool3d_kernel(
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
            ceil_mode,
            count_include_pad,
            self.use_divisor_override,
            self.divisor_override,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([64, 128, 256], [128, 256])
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        return _launch_avg_pool3d(
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
            self.ceil_mode,
            self.count_include_pad,
            self.use_divisor_override,
            self.divisor_override,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
