import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import ACCUM_DTYPE, AvgPoolWindow

__all__ = ["AvgPool2dKernel", "AvgPool2dSpatialKernel"]


@functools.lru_cache(maxsize=32)
def _avg_pool2d_kernel(window: AvgPoolWindow, dtype: str):
    """One output per thread, every tap read where it lies."""
    h_in, w_in = window.size
    kernel_h, kernel_w = window.kernel
    stride_h, stride_w = window.stride
    pad_h, pad_w = window.pad
    out_h, out_w = window.out
    (low_h, low_w), (limit_h, limit_w) = window.overlap
    rows, total = window.rows, window.outputs
    window_inside = window.window_inside
    one_divisor = window.whole_window_divides or window.divisor_override is not None
    divisor = window.divisor

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool2d_func(threads: int):
        # A compile-time truth when the launch divides, and the tail test drops out.
        block_full = total % threads == 0

        @T.macro
        def _store(total_val, oh, ow, out, row):
            """Store one output, taking its divisor from wherever this shape has it."""
            if one_divisor:
                out[row, oh, ow] = T.cast(total_val / T.cast(divisor, ACCUM_DTYPE), dtype)
            else:
                top = oh * stride_h - pad_h
                left = ow * stride_w - pad_w
                # An empty window would divide by zero, so the overlap takes a floor.
                extent = T.max(
                    T.max(T.min(top + kernel_h, limit_h) - T.max(top, low_h), 0)
                    * T.max(T.min(left + kernel_w, limit_w) - T.max(left, low_w), 0),
                    1,
                )
                out[row, oh, ow] = T.cast(total_val / T.cast(extent, ACCUM_DTYPE), dtype)

        @T.prim_func
        def _avg_pool2d_main(
            x: T.Tensor((rows, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, threads), threads=threads) as tile:
                for i in T.Parallel(threads):
                    idx = tile * threads + i
                    if block_full or idx < total:
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        row = plane_row // out_h
                        oh = plane_row - row * out_h
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        total_val = T.alloc_var(T.float32)
                        total_val = T.cast(0.0, ACCUM_DTYPE)
                        for kh in T.serial(kernel_h):
                            for kw in T.serial(kernel_w):
                                ih = top + kh
                                iw = left + kw
                                # Tested where it is read, so the test predicates the
                                # load; hoisting it per axis makes it a branch.
                                if window_inside or (
                                    (ih >= 0) and (ih < h_in) and (iw >= 0) and (iw < w_in)
                                ):
                                    total_val += T.cast(x[row, ih, iw], ACCUM_DTYPE)
                        _store(total_val, oh, ow, out, row)

        return _avg_pool2d_main

    return _avg_pool2d_func


class _AvgPool2dKernelBase(Kernel):
    """Shape, launch planning and dispatch shared by the two avg_pool2d kernels."""

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    # One output per thread, so the block size is the whole launch space: a parallel loop
    # wider than the block serializes it, and one narrower idles lanes.
    _THREAD_CHOICES: ClassVar[tuple[int, ...]] = (128, 256, 512)
    _DEFAULT_THREADS: ClassVar[int] = 256

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
        self.dtype = dtype
        self.window = AvgPoolWindow(
            rows=n * c_in,
            size=(h_in, w_in),
            kernel=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            pad=(pad_h, pad_w),
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        self.kernel = _avg_pool2d_kernel(self.window, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": self._DEFAULT_THREADS}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": threads} for threads in self._THREAD_CHOICES]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        kernel = self.kernel(self.config["threads"])
        planes = kernel(x.contiguous().view(self.window.rows, *self.window.size))
        return planes.view(self.n, self.c_in, *self.window.out)


class AvgPool2dSpatialKernel(_AvgPool2dKernelBase):
    """Fast path for common NCHW avg_pool2d workloads: floor-mode, one divisor."""

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
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__(
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
            False,
            True,
            None,
            dtype,
            config,
            tune,
        )


class AvgPool2dKernel(_AvgPool2dKernelBase):
    """Average pooling over an NCHW plane, with every PyTorch flag combination."""
