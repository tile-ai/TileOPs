from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pool import MaxPool2dKernel
from tileops.kernels.pool.common import (
    _normalize_pool_dims,
    validate_channels_last_input,
    validate_pool_params,
)

from .op import Op

__all__ = ["MaxPool2dOp"]


class MaxPool2dOp(Op):
    """Max pooling over channels-last `NHWC` inputs."""

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 2)
        self.stride = (
            self.kernel_size
            if stride is None
            else _normalize_pool_dims("stride", stride, 2)
        )
        self.padding = _normalize_pool_dims("padding", padding, 2)
        self.dilation = _normalize_pool_dims("dilation", dilation, 2)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        validate_pool_params(
            ndim=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        self.dispatch_kernel(kernel_map)
        if "max_pool2d_kernel" not in self.kernel_map:
            raise NotImplementedError("MaxPool2dOp requires 'max_pool2d_kernel' in kernel_map")
        self.kernel = self.kernel_map["max_pool2d_kernel"](
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
            dilation_h=self.dilation[0],
            dilation_w=self.dilation[1],
            ceil_mode=ceil_mode,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"max_pool2d_kernel": MaxPool2dKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        validate_channels_last_input(
            op_name=type(self).__name__,
            x_shape=tuple(x.shape),
            expected_shape=(self.n, self.h_in, self.w_in, self.c_in),
            layout="NHWC",
            ambiguous_layout_shape=(self.n, self.c_in, self.h_in, self.w_in),
        )
        values, indices = self.kernel(x)
        if self.return_indices:
            return values, indices
        return values
