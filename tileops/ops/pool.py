from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool import AvgPool1dKernel, AvgPool2dKernel, AvgPool3dKernel
from tileops.kernels.pool.common import (
    _normalize_pool_dims,
    pool_output_dim,
    validate_channels_last_input,
    validate_pool_params,
)

from .op_base import Op

__all__ = ["AvgPool1dFwdOp", "AvgPool2dOp", "AvgPool3dOp"]


def _validate_positive_int(name: str, value: int, op_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{op_name} {name} must be an int")
    if value <= 0:
        raise ValueError(f"{op_name} {name} must be greater than zero")


class AvgPool1dFwdOp(Op):
    """Average pooling over PyTorch-compatible NCL inputs."""

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
        _validate_positive_int("n", n, "AvgPool1dFwdOp")
        _validate_positive_int("c_in", c_in, "AvgPool1dFwdOp")
        _validate_positive_int("l_in", l_in, "AvgPool1dFwdOp")
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
        self.tune = tune
        validate_pool_params(
            ndim=1,
            kernel_size=(self.kernel_size,),
            stride=(self.stride,),
            padding=(self.padding,),
        )
        self.dispatch_kernel(kernel_map)
        if "avg_pool1d_kernel" not in self.kernel_map:
            raise NotImplementedError("AvgPool1dFwdOp requires 'avg_pool1d_kernel' in kernel_map")
        self.out_l = pool_output_dim(
            self.l_in,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
        )
        self.dtype = dtype
        self.kernel = self.kernel_map["avg_pool1d_kernel"](
            n=n,
            c_in=c_in,
            l_in=l_in,
            kernel_l=self.kernel_size,
            stride_l=self.stride,
            pad_l=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"avg_pool1d_kernel": AvgPool1dKernel}

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 3:
            raise ValueError("AvgPool1dFwdOp expects input_shape to be 3D NCL")
        n, c_in, l_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0)}
        out_l = pool_output_dim(l_in, kernel_size, stride, padding, ceil_mode)
        return {"output": (n, c_in, out_l)}

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                "input.dtype must be float16, bfloat16, or float32, "
                f"got {input.dtype}"
            )
        if input.dtype != self.dtype:
            raise ValueError(f"input.dtype must match op dtype {self.dtype}, got {input.dtype}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate_dtypes(input)
        if input.ndim != 3:
            raise ValueError("AvgPool1dFwdOp expects input to be a 3D NCL tensor")
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")

        input = input.contiguous()
        expected_shape = (self.n, self.c_in, self.l_in)
        actual_shape = tuple(input.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"AvgPool1dFwdOp expects input shape {expected_shape}, "
                f"but got {actual_shape}"
            )
        return self.kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = torch.empty((), dtype=self.dtype).element_size()
        flops = self.n * self.c_in * self.out_l * self.kernel_size
        bytes_ = (self.n * self.c_in * self.l_in + self.n * self.c_in * self.out_l) * elem_bytes
        return flops, bytes_


class AvgPool2dOp(Op):
    """Average pooling over channels-last `NHWC` inputs.

    This op is API-compatible with PyTorch pooling parameters, but the tensor
    layout contract is channels-last rather than `NCHW`. Ambiguous shapes
    where `NHWC` and `NCHW` would be indistinguishable are rejected eagerly.
    """

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
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 2)
        self.stride = (
            self.kernel_size
            if stride is None
            else _normalize_pool_dims("stride", stride, 2)
        )
        self.padding = _normalize_pool_dims("padding", padding, 2)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype
        validate_pool_params(
            ndim=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=divisor_override,
        )

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
        validate_channels_last_input(
            op_name=type(self).__name__,
            x_shape=tuple(x.shape),
            expected_shape=(self.n, self.h_in, self.w_in, self.c_in),
            layout="NHWC",
            ambiguous_layout_shape=(self.n, self.c_in, self.h_in, self.w_in),
        )
        return self.kernel(x)


class AvgPool3dOp(Op):
    """Average pooling over channels-last `NDHWC` inputs.

    Ambiguous shapes where `NDHWC` and `NCDHW` would be indistinguishable are
    rejected eagerly.
    """

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
            ambiguous_layout_shape=(self.n, self.c_in, self.d_in, self.h_in, self.w_in),
        )
        return self.kernel(x)
