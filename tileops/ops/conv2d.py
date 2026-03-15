from typing import Dict, Optional, Sequence, Tuple

import torch

from tileops.kernels.conv2d import Conv2dIm2ColKernel, PointwiseConvKernel
from tileops.kernels.gemm import GemmKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["Conv2dOp"]


def _pair(value: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected a pair, got {value}")
    return int(value[0]), int(value[1])


class Conv2dOp(Op):
    """Dense forward Conv2d with a dedicated 1x1 implicit-GEMM path."""

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_h, self.kernel_w = _pair(kernel_size)
        self.stride_h, self.stride_w = _pair(stride)
        self.pad_h, self.pad_w = _pair(padding)
        self.dtype = dtype

        if self.stride_h <= 0 or self.stride_w <= 0:
            raise ValueError("stride must be positive")
        if self.pad_h < 0 or self.pad_w < 0:
            raise ValueError("padding must be non-negative")

        self.out_h = (self.h + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        self.out_w = (self.w + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        if self.out_h <= 0 or self.out_w <= 0:
            raise ValueError(
                "Conv2d output spatial size must be positive; "
                f"got out_h={self.out_h}, out_w={self.out_w}"
            )

        self.m = self.n * self.out_h * self.out_w
        self.k = self.c_in * self.kernel_h * self.kernel_w
        self.is_pointwise = self.kernel_h == 1 and self.kernel_w == 1
        self.use_pointwise_gemm = (
            self.is_pointwise
            and self.stride_h == 1
            and self.stride_w == 1
            and self.pad_h == 0
            and self.pad_w == 0
        )
        self.dispatch_kernel(kernel_map)

        if self.is_pointwise:
            if self.use_pointwise_gemm:
                self.gemm_kernel = self.kernel_map["gemm_kernel"](
                    self.m,
                    self.c_out,
                    self.c_in,
                    self.dtype,
                    tune=tune,
                )
                self.kernel = self.gemm_kernel
            else:
                self.pointwise_kernel = self.kernel_map["pointwise_kernel"](
                    self.n,
                    self.c_in,
                    self.h,
                    self.w,
                    self.c_out,
                    self.out_h,
                    self.out_w,
                    self.stride_h,
                    self.stride_w,
                    self.pad_h,
                    self.pad_w,
                    self.dtype,
                    tune=tune,
                )
                self.kernel = self.pointwise_kernel
        else:
            self.im2col_kernel = self.kernel_map["im2col_kernel"](
                self.n,
                self.c_in,
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
                self.dtype,
                tune=tune,
            )
            self.gemm_kernel = self.kernel_map["gemm_kernel"](
                self.m,
                self.c_out,
                self.k,
                self.dtype,
                tune=tune,
            )
            self.kernel = self.im2col_kernel

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "im2col_kernel": Conv2dIm2ColKernel,
            "gemm_kernel": GemmKernel,
            "pointwise_kernel": PointwiseConvKernel,
        }

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if weight.dtype != self.dtype:
            raise ValueError(f"Expected weight.dtype {self.dtype}, got {weight.dtype}")
        if x.shape != (self.n, self.c_in, self.h, self.w):
            raise ValueError(
                f"Expected x shape {(self.n, self.c_in, self.h, self.w)}, got {tuple(x.shape)}"
            )
        if weight.shape != (self.c_out, self.c_in, self.kernel_h, self.kernel_w):
            raise ValueError(
                "Expected weight shape "
                f"{(self.c_out, self.c_in, self.kernel_h, self.kernel_w)}, "
                f"got {tuple(weight.shape)}"
            )
        if bias is not None:
            if not bias.is_cuda:
                raise ValueError("bias must be a CUDA tensor")
            if bias.dtype != self.dtype:
                raise ValueError(f"Expected bias.dtype {self.dtype}, got {bias.dtype}")
            if bias.shape != (self.c_out,):
                raise ValueError(f"Expected bias shape {(self.c_out,)}, got {tuple(bias.shape)}")

        if self.is_pointwise:
            if self.use_pointwise_gemm:
                x_2d = x.contiguous().permute(0, 2, 3, 1).contiguous().reshape(self.m, self.c_in)
                weight_2d = weight.contiguous().reshape(self.c_out, self.c_in).transpose(0, 1).contiguous()
                out_2d = self.gemm_kernel(x_2d, weight_2d)
                if bias is not None:
                    out_2d = out_2d + bias.reshape(1, self.c_out)
                return out_2d.reshape(self.n, self.out_h, self.out_w, self.c_out).permute(
                    0, 3, 1, 2
                ).contiguous()

            out = self.pointwise_kernel(
                x.contiguous(),
                weight.contiguous().reshape(self.c_out, self.c_in).transpose(0, 1).contiguous(),
            )
            if bias is not None:
                out = out + bias.reshape(1, self.c_out, 1, 1)
            return out.contiguous()

        cols = self.im2col_kernel(x.contiguous())
        weight_2d = weight.contiguous().reshape(self.c_out, self.k).transpose(0, 1).contiguous()
        out_2d = self.gemm_kernel(cols, weight_2d)
        if bias is not None:
            out_2d = out_2d + bias.reshape(1, self.c_out)

        return out_2d.reshape(self.n, self.out_h, self.out_w, self.c_out).permute(0, 3, 1, 2).contiguous()
