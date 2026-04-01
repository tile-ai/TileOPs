from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pool import AvgPool1dKernel
from tileops.kernels.pool.common import (
    _normalize_pool_dims,
    validate_channels_last_input,
    validate_pool_params,
)

from .op import Op

__all__ = ["AvgPool1dOp"]


class AvgPool1dOp(Op):
    """Average pooling over channels-last `NLC` inputs.

    This op intentionally uses the TileOPs channels-last contract rather than
    PyTorch's default `NCL` layout.
    """

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
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 1)[0]
        self.stride = (
            (self.kernel_size,)
            if stride is None
            else _normalize_pool_dims("stride", stride, 1)
        )[0]
        self.padding = _normalize_pool_dims("padding", padding, 1)[0]
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype
        validate_pool_params(
            ndim=1,
            kernel_size=(self.kernel_size,),
            stride=(self.stride,),
            padding=(self.padding,),
        )

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
        validate_channels_last_input(
            op_name=type(self).__name__,
            x_shape=tuple(x.shape),
            expected_shape=(self.n, self.l_in, self.c_in),
            layout="NLC",
        )
        return self.kernel(x)
