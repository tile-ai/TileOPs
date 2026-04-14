from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.convolution import Conv1dKernel, Conv2d1x1Kernel, Conv2dKernel, Conv3dKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["Conv1dBiasFwdOp", "Conv1dFwdOp", "Conv2dOp", "Conv3dOp"]


class Conv1dFwdOp(Op):

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        if "conv1d_kernel" not in self.kernel_map:
            raise NotImplementedError("Conv1dFwdOp requires 'conv1d_kernel' in kernel_map")
        self.kernel = self.kernel_map["conv1d_kernel"](
            n=n,
            c_in=c_in,
            l_in=l_in,
            c_out=c_out,
            kernel_l=kernel_size,
            stride_l=stride,
            pad_l=padding,
            dtype=dtype,
            has_bias=bias,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"conv1d_kernel": Conv1dKernel}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel(x, weight, bias)


class Conv1dBiasFwdOp(Conv1dFwdOp):
    """Conv1d forward with bias=True default.

    Identical to :class:`Conv1dFwdOp` but defaults ``bias=True`` so the
    manifest key ``Conv1dBiasFwdOp`` resolves to a distinct class name.
    """

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if not bias:
            raise ValueError(
                "Conv1dBiasFwdOp requires bias=True. "
                "Use Conv1dFwdOp for the no-bias variant."
            )
        super().__init__(
            n=n, c_in=c_in, l_in=l_in, c_out=c_out,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias, dtype=dtype, kernel_map=kernel_map, tune=tune,
        )


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
        self.has_bias = bias
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
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
        if (
            self.kernel_size == (1, 1)
            and self.stride == (1, 1)
            and self.padding == (0, 0)
            and "conv2d_1x1_kernel" in self.kernel_map
        ):
            self.kernel = self.kernel_map["conv2d_1x1_kernel"](**kernel_kwargs)
        elif "conv2d_kernel" in self.kernel_map:
            self.kernel = self.kernel_map["conv2d_kernel"](
                **kernel_kwargs,
                kernel_h=self.kernel_size[0],
                kernel_w=self.kernel_size[1],
            )
        else:
            raise NotImplementedError(
                "Conv2dOp requires 'conv2d_1x1_kernel' or 'conv2d_kernel' in kernel_map"
            )

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
