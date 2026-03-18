from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.conv2d import Conv2d1x1Kernel, Conv2dKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["Conv2dOp"]


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class Conv2dOp(Op):

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.has_bias = bias
        self.dtype = dtype

        if self.groups != 1:
            raise NotImplementedError("Conv2dOp currently supports groups=1 only")
        if self.dilation != (1, 1):
            raise NotImplementedError("Conv2dOp currently supports dilation=1 only")

        self.dispatch_kernel(kernel_map)
        kernel_name = "conv2d_kernel"
        kernel_kwargs = dict(
            n=n,
            c_in=c_in,
            h=h,
            w=w,
            c_out=c_out,
            stride_h=self.stride[0],
            stride_w=self.stride[1],
            pad_h=self.padding[0],
            pad_w=self.padding[1],
            dtype=dtype,
            has_bias=bias,
            tune=tune,
        )
        if self.kernel_size == (1, 1) and "conv2d_1x1_kernel" in self.kernel_map:
            kernel_name = "conv2d_1x1_kernel"
        else:
            kernel_kwargs["k_h"] = self.kernel_size[0]
            kernel_kwargs["k_w"] = self.kernel_size[1]
        self.kernel = self.kernel_map[kernel_name](**kernel_kwargs)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "conv2d_1x1_kernel": Conv2d1x1Kernel,
            "conv2d_kernel": Conv2dKernel,
        }

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel(x, weight, bias)
