from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.convolution import Conv1dKernel, Conv2d1x1Kernel, Conv2dKernel, Conv3dKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["Conv1dBiasFwdOp", "Conv1dFwdOp", "Conv2dOp", "Conv3dOp"]


def _conv_tuple(
    value: int | Tuple[int, ...],
    dims: int,
    name: str,
    op_name: str,
) -> Tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError(f"{op_name} {name} must be an int or a {dims}-element tuple")
    if isinstance(value, int):
        return (value,) * dims
    if isinstance(value, tuple):
        if len(value) != dims:
            raise ValueError(f"{op_name} {name} must be an int or a {dims}-element tuple")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            raise TypeError(f"{op_name} {name} must contain only ints")
        return value
    raise TypeError(f"{op_name} {name} must be an int or a {dims}-element tuple")


def _conv_padding_to_tuple(
    padding: int | Tuple[int, ...] | str,
    stride: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    op_name: str,
    dilation: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, ...]:
    dims = len(kernel_size)
    if dilation is None:
        dilation = (1,) * dims
    if isinstance(padding, str):
        if padding == "valid":
            return (0,) * dims
        if padding == "same":
            if any(axis_stride != 1 for axis_stride in stride):
                raise ValueError(f"{op_name} padding='same' requires stride == 1")
            effective_kernel = tuple(
                axis_dilation * (axis_kernel - 1) + 1
                for axis_kernel, axis_dilation in zip(kernel_size, dilation, strict=True)
            )
            if any(axis_kernel % 2 == 0 for axis_kernel in effective_kernel):
                raise ValueError(
                    f"{op_name} padding='same' requires odd effective kernel_size values "
                    "with the current symmetric padding kernel"
                )
            return tuple(axis_kernel // 2 for axis_kernel in effective_kernel)
        raise ValueError(
            f"{op_name} padding must be an int, {dims}-element tuple, 'valid', or 'same'"
        )
    return _conv_tuple(padding, dims, "padding", op_name)


def _validate_positive_int(name: str, value: int, op_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{op_name} {name} must be an int")
    if value <= 0:
        raise ValueError(f"{op_name} {name} must be greater than zero")


def _validate_conv_groups(op_name: str, c_in: int, c_out: int, groups: int) -> None:
    _validate_positive_int("groups", groups, op_name)
    if c_in % groups != 0:
        raise ValueError(f"{op_name} c_in must be divisible by groups")
    if c_out % groups != 0:
        raise ValueError(f"{op_name} c_out must be divisible by groups")


def _validate_conv_params(
    *,
    op_name: str,
    input_size: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Optional[Tuple[int, ...]] = None,
) -> None:
    ndim = len(input_size)
    if dilation is None:
        dilation = (1,) * ndim
    if (
        len(kernel_size) != ndim
        or len(stride) != ndim
        or len(padding) != ndim
        or len(dilation) != ndim
    ):
        raise ValueError(
            f"{op_name} kernel_size, stride, padding, and dilation must match dimensionality"
        )

    for name, values in (
        ("input_size", input_size),
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("padding", padding),
        ("dilation", dilation),
    ):
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise TypeError(f"{op_name} {name} must contain only ints")

    if any(v <= 0 for v in input_size):
        raise ValueError(f"{op_name} input spatial dimensions must be greater than zero")
    if any(v <= 0 for v in kernel_size):
        raise ValueError(f"{op_name} kernel_size must be greater than zero")
    if any(v <= 0 for v in stride):
        raise ValueError(f"{op_name} stride must be greater than zero")
    if any(v < 0 for v in padding):
        raise ValueError(f"{op_name} padding must be non-negative")
    if any(v <= 0 for v in dilation):
        raise ValueError(f"{op_name} dilation must be greater than zero")

    output_size = tuple(
        (input_dim + 2 * pad - dilation_dim * (kernel_dim - 1) - 1) // stride_dim + 1
        for input_dim, kernel_dim, stride_dim, pad, dilation_dim in zip(
            input_size, kernel_size, stride, padding, dilation, strict=True
        )
    )
    if any(v <= 0 for v in output_size):
        raise ValueError(f"{op_name} output spatial dimensions must be greater than zero")


def _validate_tensor_shape(op_name: str, name: str, tensor: torch.Tensor, expected_shape: Tuple[int, ...]) -> None:
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise ValueError(f"{op_name} expects {name} shape {expected_shape}, but got {actual_shape}")


class Conv1dFwdOp(Op):

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int | Tuple[int],
        stride: int | Tuple[int] = 1,
        padding: int | Tuple[int] | str = 0,
        dilation: int | Tuple[int] = 1,
        groups: int = 1,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        _validate_positive_int("n", n, "Conv1d")
        _validate_positive_int("c_in", c_in, "Conv1d")
        _validate_positive_int("l_in", l_in, "Conv1d")
        _validate_positive_int("c_out", c_out, "Conv1d")
        _validate_conv_groups("Conv1d", c_in, c_out, groups)
        if groups != 1:
            raise NotImplementedError("Conv1d currently supports groups=1 only")
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        kernel_size_tuple = _conv_tuple(kernel_size, 1, "kernel_size", "Conv1d")
        stride_tuple = _conv_tuple(stride, 1, "stride", "Conv1d")
        dilation_tuple = _conv_tuple(dilation, 1, "dilation", "Conv1d")
        padding_tuple = _conv_padding_to_tuple(
            padding, stride_tuple, kernel_size_tuple, "Conv1d", dilation_tuple
        )
        _validate_conv_params(
            op_name="Conv1d",
            input_size=(l_in,),
            kernel_size=kernel_size_tuple,
            stride=stride_tuple,
            padding=padding_tuple,
            dilation=dilation_tuple,
        )
        self.kernel_size = kernel_size_tuple[0]
        self.stride = stride_tuple[0]
        self.padding = padding_tuple[0]
        self.dilation = dilation_tuple[0]
        self.groups = groups
        self.has_bias = False
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        if "conv1d_kernel" not in self.kernel_map:
            raise NotImplementedError("Conv1dFwdOp requires 'conv1d_kernel' in kernel_map")
        self.kernel = self.kernel_map["conv1d_kernel"](
            n=n,
            c_in=c_in,
            l_in=l_in,
            c_out=c_out,
            kernel_l=self.kernel_size,
            stride_l=self.stride,
            pad_l=self.padding,
            dilation_l=self.dilation,
            dtype=dtype,
            has_bias=False,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"conv1d_kernel": Conv1dKernel}

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _validate_tensor_shape("Conv1d", "input", input, (self.n, self.l_in, self.c_in))
        _validate_tensor_shape(
            "Conv1d",
            "weight",
            weight,
            (self.c_out, self.c_in, self.kernel_size),
        )
        return self.kernel(input, weight, None)


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
        kernel_size: int | Tuple[int],
        stride: int | Tuple[int] = 1,
        padding: int | Tuple[int] | str = 0,
        dilation: int | Tuple[int] = 1,
        groups: int = 1,
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
        _validate_positive_int("n", n, "Conv1d")
        _validate_positive_int("c_in", c_in, "Conv1d")
        _validate_positive_int("l_in", l_in, "Conv1d")
        _validate_positive_int("c_out", c_out, "Conv1d")
        _validate_conv_groups("Conv1d", c_in, c_out, groups)
        if groups != 1:
            raise NotImplementedError("Conv1d currently supports groups=1 only")
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        kernel_size_tuple = _conv_tuple(kernel_size, 1, "kernel_size", "Conv1d")
        stride_tuple = _conv_tuple(stride, 1, "stride", "Conv1d")
        dilation_tuple = _conv_tuple(dilation, 1, "dilation", "Conv1d")
        padding_tuple = _conv_padding_to_tuple(
            padding, stride_tuple, kernel_size_tuple, "Conv1d", dilation_tuple
        )
        _validate_conv_params(
            op_name="Conv1d",
            input_size=(l_in,),
            kernel_size=kernel_size_tuple,
            stride=stride_tuple,
            padding=padding_tuple,
            dilation=dilation_tuple,
        )
        self.kernel_size = kernel_size_tuple[0]
        self.stride = stride_tuple[0]
        self.padding = padding_tuple[0]
        self.dilation = dilation_tuple[0]
        self.groups = groups
        self.has_bias = True
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        if "conv1d_kernel" not in self.kernel_map:
            raise NotImplementedError("Conv1dBiasFwdOp requires 'conv1d_kernel' in kernel_map")
        self.kernel = self.kernel_map["conv1d_kernel"](
            n=n,
            c_in=c_in,
            l_in=l_in,
            c_out=c_out,
            kernel_l=self.kernel_size,
            stride_l=self.stride,
            pad_l=self.padding,
            dilation_l=self.dilation,
            dtype=dtype,
            has_bias=True,
            tune=tune,
        )

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        _validate_tensor_shape("Conv1d", "input", input, (self.n, self.l_in, self.c_in))
        _validate_tensor_shape(
            "Conv1d",
            "weight",
            weight,
            (self.c_out, self.c_in, self.kernel_size),
        )
        _validate_tensor_shape("Conv1d", "bias", bias, (self.c_out,))
        return self.kernel(input, weight, bias)


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    return _conv_tuple(value, 2, "value", "Conv2d")  # type: ignore[return-value]


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
        padding: int | Tuple[int, int] | str = 0,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        _validate_positive_int("n", n, "Conv2d")
        _validate_positive_int("c_in", c_in, "Conv2d")
        _validate_positive_int("h", h, "Conv2d")
        _validate_positive_int("w", w, "Conv2d")
        _validate_positive_int("c_out", c_out, "Conv2d")
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _conv_padding_to_tuple(padding, self.stride, self.kernel_size, "Conv2d")
        _validate_conv_params(
            op_name="Conv2d",
            input_size=(h, w),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
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
        _validate_tensor_shape("Conv2d", "x", x, (self.n, self.h, self.w, self.c_in))
        _validate_tensor_shape(
            "Conv2d",
            "weight",
            weight,
            (self.c_out, self.c_in, self.kernel_size[0], self.kernel_size[1]),
        )
        if bias is not None:
            _validate_tensor_shape("Conv2d", "bias", bias, (self.c_out,))
        return self.kernel(x, weight, bias)


def _triple(value: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    return _conv_tuple(value, 3, "value", "Conv3d")  # type: ignore[return-value]


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
        padding: int | Tuple[int, int, int] | str = 0,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        _validate_positive_int("n", n, "Conv3d")
        _validate_positive_int("c_in", c_in, "Conv3d")
        _validate_positive_int("d_in", d_in, "Conv3d")
        _validate_positive_int("h_in", h_in, "Conv3d")
        _validate_positive_int("w_in", w_in, "Conv3d")
        _validate_positive_int("c_out", c_out, "Conv3d")
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _conv_padding_to_tuple(padding, self.stride, self.kernel_size, "Conv3d")
        _validate_conv_params(
            op_name="Conv3d",
            input_size=(d_in, h_in, w_in),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
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
        _validate_tensor_shape(
            "Conv3d",
            "x",
            x,
            (self.n, self.d_in, self.h_in, self.w_in, self.c_in),
        )
        _validate_tensor_shape(
            "Conv3d",
            "weight",
            weight,
            (
                self.c_out,
                self.c_in,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            ),
        )
        if bias is not None:
            _validate_tensor_shape("Conv3d", "bias", bias, (self.c_out,))
        return self.kernel(x, weight, bias)
