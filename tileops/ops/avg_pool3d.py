from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pool import AvgPool3dKernel
from tileops.kernels.pool.common import (
    _normalize_pool_dims,
    validate_channels_last_input,
    validate_pool_params,
)

from .op import Op

__all__ = ["AvgPool3dOp"]


class AvgPool3dOp(Op):
    """Average pooling over channels-last `NDHWC` inputs."""

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 3)
        self.stride = (
            self.kernel_size
            if stride is None
            else _normalize_pool_dims("stride", stride, 3)
        )
        self.padding = _normalize_pool_dims("padding", padding, 3)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype
        validate_pool_params(
            ndim=3,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=divisor_override,
        )

        self.dispatch_kernel(kernel_map)
        if "avg_pool3d_kernel" not in self.kernel_map:
            raise NotImplementedError("AvgPool3dOp requires 'avg_pool3d_kernel' in kernel_map")
        self.kernel = self.kernel_map["avg_pool3d_kernel"](
            n=n,
            c_in=c_in,
            d_in=d_in,
            h_in=h_in,
            w_in=w_in,
            kernel_d=self.kernel_size[0],
            kernel_h=self.kernel_size[1],
            kernel_w=self.kernel_size[2],
            stride_d=self.stride[0],
            stride_h=self.stride[1],
            stride_w=self.stride[2],
            pad_d=self.padding[0],
            pad_h=self.padding[1],
            pad_w=self.padding[2],
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"avg_pool3d_kernel": AvgPool3dKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_channels_last_input(
            op_name=type(self).__name__,
            x_shape=tuple(x.shape),
            expected_shape=(self.n, self.d_in, self.h_in, self.w_in, self.c_in),
            layout="NDHWC",
        )
        return self.kernel(x)
