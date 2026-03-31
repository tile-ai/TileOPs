from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pool import AvgPool1dKernel
from tileops.kernels.pool.common import normalize_1d

from .op import Op

__all__ = ["AvgPool1dOp"]


class AvgPool1dOp(Op):

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: int | tuple[int],
        stride: Optional[int | tuple[int]] = None,
        padding: int | tuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = normalize_1d(kernel_size)
        self.stride = self.kernel_size if stride is None else normalize_1d(stride)
        self.padding = normalize_1d(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        if "avg_pool1d_kernel" not in self.kernel_map:
            raise NotImplementedError("AvgPool1dOp requires 'avg_pool1d_kernel' in kernel_map")
        self.kernel = self.kernel_map["avg_pool1d_kernel"](
            n=n,
            c_in=c_in,
            l_in=l_in,
            kernel_l=self.kernel_size,
            stride_l=self.stride,
            pad_l=self.padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"avg_pool1d_kernel": AvgPool1dKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x)
