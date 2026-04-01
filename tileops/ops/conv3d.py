from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.conv import Conv3dKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["Conv3dOp"]


def _triple(value: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value, value)


class Conv3dOp(Op):

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        c_out: int,
        kernel_size: int | Tuple[int, int, int],
        stride: int | Tuple[int, int, int] = 1,
        padding: int | Tuple[int, int, int] = 0,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.has_bias = bias
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        if "conv3d_kernel" not in self.kernel_map:
            raise NotImplementedError("Conv3dOp requires 'conv3d_kernel' in kernel_map")
        self.kernel = self.kernel_map["conv3d_kernel"](
            n=n,
            c_in=c_in,
            d_in=d_in,
            h_in=h_in,
            w_in=w_in,
            c_out=c_out,
            kernel_d=self.kernel_size[0],
            kernel_h=self.kernel_size[1],
            kernel_w=self.kernel_size[2],
            stride_d=self.stride[0],
            stride_h=self.stride[1],
            stride_w=self.stride[2],
            pad_d=self.padding[0],
            pad_h=self.padding[1],
            pad_w=self.padding[2],
            dtype=dtype,
            has_bias=bias,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"conv3d_kernel": Conv3dKernel}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel(x, weight, bias)
