from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["Conv2dIm2ColKernel"]


def _conv2d_im2col_kernel(
    n: int,
    c: int,
    h: int,
    w: int,
    kernel_h: int,
    kernel_w: int,
    out_h: int,
    out_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dtype: str = "float16",
) -> Callable:
    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv2d_im2col_func(block_m: int, block_k: int, threads: int) -> Callable:
        out_rows = n * out_h * out_w
        out_cols = c * kernel_h * kernel_w

        @T.prim_func
        def _conv2d_im2col(
            x: T.Tensor((n, c, h, w), dtype),  # type: ignore
            cols: T.Tensor((out_rows, out_cols), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(out_rows, block_m),
                T.ceildiv(out_cols, block_k),
                threads=threads,
            ) as (bx, by):
                row_start = bx * block_m
                col_start = by * block_k

                for i, j in T.Parallel(block_m, block_k):
                    row = row_start + i
                    col = col_start + j

                    if row < out_rows and col < out_cols:
                        batch_idx = row // (out_h * out_w)
                        spatial_idx = row % (out_h * out_w)
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w

                        channel_idx = col // (kernel_h * kernel_w)
                        kernel_idx = col % (kernel_h * kernel_w)
                        kh = kernel_idx // kernel_w
                        kw = kernel_idx % kernel_w

                        ih = oh * stride_h + kh - pad_h
                        iw = ow * stride_w + kw - pad_w

                        cols[row, col] = T.if_then_else(
                            (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w),
                            x[batch_idx, channel_idx, ih, iw],
                            T.cast(0, dtype),
                        )

        return _conv2d_im2col

    return _conv2d_im2col_func


@torch.library.custom_op("top::conv2d_im2col_wrapped_kernel", mutates_args=())
def _conv2d_im2col_wrapped_kernel(
    n: int,
    c: int,
    h: int,
    w: int,
    kernel_h: int,
    kernel_w: int,
    out_h: int,
    out_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dtype: str,
    block_m: int,
    block_k: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _conv2d_im2col_kernel(
        n,
        c,
        h,
        w,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dtype,
    )(block_m, block_k, threads)(x)


@_conv2d_im2col_wrapped_kernel.register_fake
def _(
    n: int,
    c: int,
    h: int,
    w: int,
    kernel_h: int,
    kernel_w: int,
    out_h: int,
    out_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dtype: str,
    block_m: int,
    block_k: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (n * out_h * out_w, c * kernel_h * kernel_w),
        dtype=x.dtype,
        device=x.device,
    )


class Conv2dIm2ColKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        n: int,
        c: int,
        h: int,
        w: int,
        kernel_h: int,
        kernel_w: int,
        out_h: int,
        out_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c = c
        self.h = h
        self.w = w
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.out_h = out_h
        self.out_w = out_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dtype = dtype
        self.kernel = _conv2d_im2col_kernel(
            n,
            c,
            h,
            w,
            kernel_h,
            kernel_w,
            out_h,
            out_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_k": 128,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": 32, "block_k": 64, "threads": 128},
            {"block_m": 64, "block_k": 64, "threads": 128},
            {"block_m": 64, "block_k": 128, "threads": 128},
            {"block_m": 128, "block_k": 64, "threads": 256},
            {"block_m": 128, "block_k": 128, "threads": 256},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _conv2d_im2col_wrapped_kernel(
            self.n,
            self.c,
            self.h,
            self.w,
            self.kernel_h,
            self.kernel_w,
            self.out_h,
            self.out_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_k"],
            self.config["threads"],
            x,
        )
