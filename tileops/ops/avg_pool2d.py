from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pool import AvgPool2dKernel
from tileops.kernels.pool.common import normalize_2d

from .op import Op

__all__ = ["AvgPool2dOp"]


class AvgPool2dOp(Op):

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = normalize_2d(kernel_size)
        self.stride = self.kernel_size if stride is None else normalize_2d(stride)
        self.padding = normalize_2d(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        if "avg_pool2d_kernel" not in self.kernel_map:
            raise NotImplementedError("AvgPool2dOp requires 'avg_pool2d_kernel' in kernel_map")
        self.kernel = self.kernel_map["avg_pool2d_kernel"](
            n=n,
            c_in=c_in,
            h_in=h_in,
            w_in=w_in,
            kernel_h=self.kernel_size[0],
            kernel_w=self.kernel_size[1],
            stride_h=self.stride[0],
            stride_w=self.stride[1],
            pad_h=self.padding[0],
            pad_w=self.padding[1],
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"avg_pool2d_kernel": AvgPool2dKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x)
