from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.convolution import (
    Conv1dKernel,
    Conv1dPointwiseKernel,
    Conv2d1x1Kernel,
    Conv2dKernel,
    Conv3dKernel,
    GroupConv1dKernel,
    GroupConv2dKernel,
    GroupConv3dKernel,
)
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = [
    "Conv1dBiasFwdOp",
    "Conv1dFwdOp",
    "Conv2dBiasFwdOp",
    "Conv2dFwdOp",
    "Conv3dBiasFwdOp",
    "Conv3dFwdOp",
]


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
    padding: Tuple[int | Tuple[int, int], ...],
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
        ("dilation", dilation),
    ):
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise TypeError(f"{op_name} {name} must contain only ints")
    for pad in padding:
        if isinstance(pad, tuple):
            if len(pad) != 2:
                raise ValueError(f"{op_name} asymmetric padding entries must have length 2")
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in pad):
                raise TypeError(f"{op_name} padding must contain only ints")
        elif not isinstance(pad, int) or isinstance(pad, bool):
            raise TypeError(f"{op_name} padding must contain only ints or int pairs")

    if any(v <= 0 for v in input_size):
        raise ValueError(f"{op_name} input spatial dimensions must be greater than zero")
    if any(v <= 0 for v in kernel_size):
        raise ValueError(f"{op_name} kernel_size must be greater than zero")
    if any(v <= 0 for v in stride):
        raise ValueError(f"{op_name} stride must be greater than zero")
    if any(any(axis_pad < 0 for axis_pad in pad) if isinstance(pad, tuple) else pad < 0 for pad in padding):
        raise ValueError(f"{op_name} padding must be non-negative")
    if any(v <= 0 for v in dilation):
        raise ValueError(f"{op_name} dilation must be greater than zero")

    output_size = tuple(
        (
            input_dim
            + (sum(pad) if isinstance(pad, tuple) else 2 * pad)
            - dilation_dim * (kernel_dim - 1)
            - 1
        ) // stride_dim + 1
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


def _conv1d_padding_pair_and_l_out(
    l_in: int,
    kernel_size: int,
    stride: int | Tuple[int],
    padding: int | Tuple[int] | str,
    dilation: int | Tuple[int],
) -> tuple[int, int, int]:
    kernel_size_tuple = _conv_tuple(kernel_size, 1, "kernel_size", "Conv1d")
    stride_tuple = _conv_tuple(stride, 1, "stride", "Conv1d")
    dilation_tuple = _conv_tuple(dilation, 1, "dilation", "Conv1d")
    if padding == "same":
        if stride_tuple[0] != 1:
            raise ValueError("Conv1d padding='same' requires stride == 1")
        total_pad = dilation_tuple[0] * (kernel_size_tuple[0] - 1)
        pad_left = total_pad // 2
        return pad_left, total_pad - pad_left, l_in
    padding_tuple = _conv_padding_to_tuple(
        padding, stride_tuple, kernel_size_tuple, "Conv1d", dilation_tuple
    )
    l_out = (
        l_in
        + 2 * padding_tuple[0]
        - dilation_tuple[0] * (kernel_size_tuple[0] - 1)
        - 1
    ) // stride_tuple[0] + 1
    return padding_tuple[0], padding_tuple[0], l_out


def _conv1d_l_out(
    l_in: int,
    kernel_size: int,
    stride: int | Tuple[int],
    padding: int | Tuple[int] | str,
    dilation: int | Tuple[int],
) -> int:
    _, _, l_out = _conv1d_padding_pair_and_l_out(
        l_in,
        kernel_size,
        stride,
        padding,
        dilation,
    )
    return l_out


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
        _has_bias: bool = False,
    ) -> None:
        _validate_positive_int("n", n, "Conv1d")
        _validate_positive_int("c_in", c_in, "Conv1d")
        _validate_positive_int("l_in", l_in, "Conv1d")
        _validate_positive_int("c_out", c_out, "Conv1d")
        _validate_conv_groups("Conv1d", c_in, c_out, groups)
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.groups = groups
        self.c_in_g = c_in // groups
        self.c_out_g = c_out // groups
        kernel_size_tuple = _conv_tuple(kernel_size, 1, "kernel_size", "Conv1d")
        stride_tuple = _conv_tuple(stride, 1, "stride", "Conv1d")
        dilation_tuple = _conv_tuple(dilation, 1, "dilation", "Conv1d")
        pad_left, pad_right, out_l = _conv1d_padding_pair_and_l_out(
            l_in,
            kernel_size_tuple[0],
            stride_tuple,
            padding,
            dilation_tuple,
        )
        _validate_conv_params(
            op_name="Conv1d",
            input_size=(l_in,),
            kernel_size=kernel_size_tuple,
            stride=stride_tuple,
            padding=((pad_left, pad_right),),
            dilation=dilation_tuple,
        )
        self.kernel_size = kernel_size_tuple[0]
        self.stride = stride_tuple[0]
        self.padding = pad_left
        self.padding_right = pad_right
        self.padding_pair = (pad_left, pad_right)
        self._padding_arg = padding
        self.dilation = dilation_tuple[0]
        self.out_l = out_l
        self.has_bias = _has_bias
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        kernel_kwargs = dict(
            n=n,
            c_in=c_in,
            l_in=l_in,
            c_out=c_out,
            dtype=dtype,
            has_bias=_has_bias,
            tune=tune,
        )
        if (
            self.groups == 1
            and self.kernel_size == 1
            and self.stride == 1
            and self.padding == 0
            and self.dilation == 1
            and "conv1d_pointwise_kernel" in self.kernel_map
        ):
            self.kernel = self.kernel_map["conv1d_pointwise_kernel"](**kernel_kwargs)
        elif self.groups > 1 and "group_conv1d_kernel" in self.kernel_map:
            self.kernel = self.kernel_map["group_conv1d_kernel"](
                **kernel_kwargs,
                kernel_l=self.kernel_size,
                stride_l=self.stride,
                pad_l=self.padding_pair,
                dilation_l=self.dilation,
                groups=self.groups,
                c_in_g=self.c_in_g,
                c_out_g=self.c_out_g,
            )
        elif self.groups == 1 and "conv1d_kernel" in self.kernel_map:
            self.kernel = self.kernel_map["conv1d_kernel"](
                **kernel_kwargs,
                kernel_l=self.kernel_size,
                stride_l=self.stride,
                pad_l=self.padding_pair,
                dilation_l=self.dilation,
            )
        else:
            raise NotImplementedError(
                "Conv1dFwdOp requires 'conv1d_pointwise_kernel', 'conv1d_kernel', "
                "or 'group_conv1d_kernel' in kernel_map"
            )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "conv1d_pointwise_kernel": Conv1dPointwiseKernel,
            "conv1d_kernel": Conv1dKernel,
            "group_conv1d_kernel": GroupConv1dKernel,
        }

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(input, weight)
        _validate_tensor_shape("Conv1d", "input", input, (self.n, self.c_in, self.l_in))
        _validate_tensor_shape(
            "Conv1d",
            "weight",
            weight,
            (self.c_out, self.c_in // self.groups, self.kernel_size),
        )
        return self.kernel(input, weight, None)

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, int, int],
        weight_shape: tuple[int, int, int],
    ) -> Dict[str, tuple[int, int, int]]:
        n, _, l_in = input_shape
        c_out, _, kernel_size = weight_shape
        l_out = _conv1d_l_out(
            l_in,
            kernel_size,
            self.stride,
            getattr(self, "_padding_arg", self.padding),
            self.dilation,
        )
        return {"output": (n, c_out, l_out)}

    def _validate_dtypes(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(f"input.dtype must be float16 or bfloat16, got {input.dtype}")
        if weight.dtype != input.dtype:
            raise ValueError(
                f"weight.dtype must match input.dtype {input.dtype}, got {weight.dtype}"
            )
        if self.dtype is not None and input.dtype != self.dtype:
            raise ValueError(f"input.dtype must match op dtype {self.dtype}, got {input.dtype}")

    def eval_roofline(self) -> tuple[int, int]:
        l_out = self.out_l
        flops = 2 * self.n * self.c_out * l_out * self.c_in_g * self.kernel_size
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        bytes_ = (
            self.n * self.c_in * self.l_in
            + self.c_out * self.c_in_g * self.kernel_size
            + self.n * self.c_out * l_out
        ) * elem_bytes
        return int(flops), int(bytes_)


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
        super().__init__(
            n=n,
            c_in=c_in,
            l_in=l_in,
            c_out=c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
            kernel_map=kernel_map,
            tune=tune,
            _has_bias=True,
        )

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(input, weight, bias)
        _validate_tensor_shape("Conv1d", "input", input, (self.n, self.c_in, self.l_in))
        _validate_tensor_shape(
            "Conv1d",
            "weight",
            weight,
            (self.c_out, self.c_in // self.groups, self.kernel_size),
        )
        _validate_tensor_shape("Conv1d", "bias", bias, (self.c_out,))
        return self.kernel(input, weight, bias)

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, int, int],
        weight_shape: tuple[int, int, int],
        bias_shape: tuple[int],
    ) -> Dict[str, tuple[int, int, int]]:
        del bias_shape
        return super()._infer_output_shapes(input_shape, weight_shape)

    def _validate_dtypes(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super()._validate_dtypes(input, weight)
        if bias.dtype != input.dtype:
            raise ValueError(
                f"bias.dtype must match input.dtype {input.dtype}, got {bias.dtype}"
            )

    def eval_roofline(self) -> tuple[int, int]:
        l_out = self.out_l
        flops = (
            2 * self.n * self.c_out * l_out * self.c_in_g * self.kernel_size
            + self.n * self.c_out * l_out
        )
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        bytes_ = (
            self.n * self.c_in * self.l_in
            + self.c_out * self.c_in_g * self.kernel_size
            + self.c_out
            + self.n * self.c_out * l_out
        ) * elem_bytes
        return int(flops), int(bytes_)


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    return _conv_tuple(value, 2, "value", "Conv2d")  # type: ignore[return-value]


def _conv_out_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class Conv2dFwdOp(Op):

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
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        _has_bias: bool = False,
    ) -> None:
        _validate_positive_int("n", n, "Conv2d")
        _validate_positive_int("c_in", c_in, "Conv2d")
        _validate_positive_int("h", h, "Conv2d")
        _validate_positive_int("w", w, "Conv2d")
        _validate_positive_int("c_out", c_out, "Conv2d")
        _validate_conv_groups("Conv2d", c_in, c_out, groups)
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.groups = groups
        self.c_in_g = c_in // groups
        self.c_out_g = c_out // groups
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        dilation_tuple = _conv_tuple(dilation, 2, "dilation", "Conv2d")
        self.padding = _conv_padding_to_tuple(
            padding, self.stride, self.kernel_size, "Conv2d", dilation_tuple
        )
        _validate_conv_params(
            op_name="Conv2d",
            input_size=(h, w),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=dilation_tuple,
        )
        self.dilation = dilation_tuple
        self.out_h = _conv_out_dim(
            h, self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0]
        )
        self.out_w = _conv_out_dim(
            w, self.kernel_size[1], self.stride[1], self.padding[1], self.dilation[1]
        )
        self.has_bias = _has_bias
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
            has_bias=_has_bias,
            tune=tune,
        )
        if (
            self.groups == 1
            and self.kernel_size == (1, 1)
            and self.stride == (1, 1)
            and self.padding == (0, 0)
            and self.dilation == (1, 1)
            and "conv2d_1x1_kernel" in self.kernel_map
        ):
            self.kernel = self.kernel_map["conv2d_1x1_kernel"](**kernel_kwargs)
        elif self.groups == 1 and "conv2d_kernel" in self.kernel_map:
            self.kernel = self.kernel_map["conv2d_kernel"](
                **kernel_kwargs,
                kernel_h=self.kernel_size[0],
                kernel_w=self.kernel_size[1],
                dilation_h=self.dilation[0],
                dilation_w=self.dilation[1],
            )
        elif self.groups > 1 and "group_conv2d_kernel" in self.kernel_map:
            self.kernel = self.kernel_map["group_conv2d_kernel"](
                **kernel_kwargs,
                kernel_h=self.kernel_size[0],
                kernel_w=self.kernel_size[1],
                dilation_h=self.dilation[0],
                dilation_w=self.dilation[1],
                groups=self.groups,
                c_in_g=self.c_in_g,
                c_out_g=self.c_out_g,
            )
        else:
            raise NotImplementedError(
                "Conv2dFwdOp requires 'conv2d_1x1_kernel', 'conv2d_kernel', "
                "or 'group_conv2d_kernel' in kernel_map"
            )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "conv2d_1x1_kernel": Conv2d1x1Kernel,
            "conv2d_kernel": Conv2dKernel,
            "group_conv2d_kernel": GroupConv2dKernel,
        }

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(input, weight)
        _validate_tensor_shape("Conv2d", "input", input, (self.n, self.c_in, self.h, self.w))
        _validate_tensor_shape(
            "Conv2d",
            "weight",
            weight,
            (self.c_out, self.c_in_g, self.kernel_size[0], self.kernel_size[1]),
        )
        return self.kernel(input, weight, None)

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, int, int, int],
        weight_shape: tuple[int, int, int, int],
    ) -> Dict[str, tuple[int, int, int, int]]:
        n, _, h, w = input_shape
        c_out, _, kernel_h, kernel_w = weight_shape
        stride = _pair(self.stride)
        dilation = _conv_tuple(self.dilation, 2, "dilation", "Conv2d")
        padding = _conv_padding_to_tuple(
            self.padding, stride, (kernel_h, kernel_w), "Conv2d", dilation
        )
        out_h = _conv_out_dim(h, kernel_h, stride[0], padding[0], dilation[0])
        out_w = _conv_out_dim(w, kernel_w, stride[1], padding[1], dilation[1])
        return {"output": (n, c_out, out_h, out_w)}

    def _validate_dtypes(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        if input.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError(
                f"input.dtype must be float32, float16, or bfloat16, got {input.dtype}"
            )
        if weight.dtype != input.dtype:
            raise ValueError(
                f"weight.dtype must match input.dtype {input.dtype}, got {weight.dtype}"
            )
        if self.dtype is not None and input.dtype != self.dtype:
            raise ValueError(f"input.dtype must match op dtype {self.dtype}, got {input.dtype}")

    def eval_roofline(self) -> tuple[int, int]:
        flops = (
            2
            * self.n
            * self.c_out
            * self.out_h
            * self.out_w
            * self.c_in_g
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        bytes_ = (
            self.n * self.c_in * self.h * self.w
            + self.c_out * self.c_in_g * self.kernel_size[0] * self.kernel_size[1]
            + self.n * self.c_out * self.out_h * self.out_w
        ) * elem_bytes
        return int(flops), int(bytes_)


class Conv2dBiasFwdOp(Conv2dFwdOp):
    """Conv2d forward with bias=True default.

    Identical to :class:`Conv2dFwdOp` but defaults ``bias=True`` so the
    manifest key ``Conv2dBiasFwdOp`` resolves to a distinct class name.
    """

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
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if not bias:
            raise ValueError(
                "Conv2dBiasFwdOp requires bias=True. "
                "Use Conv2dFwdOp for the no-bias variant."
            )
        super().__init__(
            n=n,
            c_in=c_in,
            h=h,
            w=w,
            c_out=c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
            kernel_map=kernel_map,
            tune=tune,
            _has_bias=True,
        )

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(input, weight, bias)
        _validate_tensor_shape("Conv2d", "input", input, (self.n, self.c_in, self.h, self.w))
        _validate_tensor_shape(
            "Conv2d",
            "weight",
            weight,
            (self.c_out, self.c_in_g, self.kernel_size[0], self.kernel_size[1]),
        )
        _validate_tensor_shape("Conv2d", "bias", bias, (self.c_out,))
        return self.kernel(input, weight, bias)

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, int, int, int],
        weight_shape: tuple[int, int, int, int],
        bias_shape: tuple[int],
    ) -> Dict[str, tuple[int, int, int, int]]:
        del bias_shape
        return super()._infer_output_shapes(input_shape, weight_shape)

    def _validate_dtypes(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super()._validate_dtypes(input, weight)
        if bias.dtype != input.dtype:
            raise ValueError(
                f"bias.dtype must match input.dtype {input.dtype}, got {bias.dtype}"
            )

    def eval_roofline(self) -> tuple[int, int]:
        flops = (
            2
            * self.n
            * self.c_out
            * self.out_h
            * self.out_w
            * self.c_in_g
            * self.kernel_size[0]
            * self.kernel_size[1]
            + self.n * self.c_out * self.out_h * self.out_w
        )
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        bytes_ = (
            self.n * self.c_in * self.h * self.w
            + self.c_out * self.c_in_g * self.kernel_size[0] * self.kernel_size[1]
            + self.c_out
            + self.n * self.c_out * self.out_h * self.out_w
        ) * elem_bytes
        return int(flops), int(bytes_)


def _triple(value: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    return _conv_tuple(value, 3, "value", "Conv3d")  # type: ignore[return-value]


class Conv3dFwdOp(Op):

    def __init__(
        self,
        n: int,
        c_in: int,
        d: int,
        h: int,
        w: int,
        c_out: int,
        kernel_size: int | Tuple[int, int, int],
        stride: int | Tuple[int, int, int] = 1,
        padding: int | Tuple[int, int, int] | str = 0,
        dilation: int | Tuple[int, int, int] = 1,
        groups: int = 1,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        _has_bias: bool = False,
    ) -> None:
        _validate_positive_int("n", n, "Conv3d")
        _validate_positive_int("c_in", c_in, "Conv3d")
        _validate_positive_int("d", d, "Conv3d")
        _validate_positive_int("h", h, "Conv3d")
        _validate_positive_int("w", w, "Conv3d")
        _validate_positive_int("c_out", c_out, "Conv3d")
        _validate_conv_groups("Conv3d", c_in, c_out, groups)
        self.n = n
        self.c_in = c_in
        self.d = d
        self.h = h
        self.w = w
        self.c_out = c_out
        self.groups = groups
        self.c_in_g = c_in // groups
        self.c_out_g = c_out // groups
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        dilation_tuple = _conv_tuple(dilation, 3, "dilation", "Conv3d")
        self.padding = _conv_padding_to_tuple(
            padding, self.stride, self.kernel_size, "Conv3d", dilation_tuple
        )
        _validate_conv_params(
            op_name="Conv3d",
            input_size=(d, h, w),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=dilation_tuple,
        )
        self.dilation = dilation_tuple
        self.out_d = _conv_out_dim(
            d, self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0]
        )
        self.out_h = _conv_out_dim(
            h, self.kernel_size[1], self.stride[1], self.padding[1], self.dilation[1]
        )
        self.out_w = _conv_out_dim(
            w, self.kernel_size[2], self.stride[2], self.padding[2], self.dilation[2]
        )
        self.has_bias = _has_bias
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        kernel_kwargs = dict(
            n=n,
            c_in=c_in,
            d_in=d,
            h_in=h,
            w_in=w,
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
            dilation_d=self.dilation[0],
            dilation_h=self.dilation[1],
            dilation_w=self.dilation[2],
            dtype=dtype,
            has_bias=_has_bias,
            tune=tune,
        )
        if self.groups == 1 and "conv3d_kernel" in self.kernel_map:
            self.kernel = self.kernel_map["conv3d_kernel"](**kernel_kwargs)
        elif self.groups > 1 and "group_conv3d_kernel" in self.kernel_map:
            self.kernel = self.kernel_map["group_conv3d_kernel"](
                **kernel_kwargs,
                groups=self.groups,
                c_in_g=self.c_in_g,
                c_out_g=self.c_out_g,
            )
        else:
            raise NotImplementedError(
                "Conv3dFwdOp requires 'conv3d_kernel' or 'group_conv3d_kernel' in kernel_map"
            )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "conv3d_kernel": Conv3dKernel,
            "group_conv3d_kernel": GroupConv3dKernel,
        }

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(input, weight)
        _validate_tensor_shape(
            "Conv3d",
            "input",
            input,
            (self.n, self.c_in, self.d, self.h, self.w),
        )
        _validate_tensor_shape(
            "Conv3d",
            "weight",
            weight,
            (
                self.c_out,
                self.c_in_g,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            ),
        )
        return self.kernel(input, weight, None)

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, int, int, int, int],
        weight_shape: tuple[int, int, int, int, int],
    ) -> Dict[str, tuple[int, int, int, int, int]]:
        n, _, d, h, w = input_shape
        c_out, _, kernel_d, kernel_h, kernel_w = weight_shape
        stride = _triple(self.stride)
        dilation = _conv_tuple(self.dilation, 3, "dilation", "Conv3d")
        padding = _conv_padding_to_tuple(
            self.padding, stride, (kernel_d, kernel_h, kernel_w), "Conv3d", dilation
        )
        out_d = _conv_out_dim(d, kernel_d, stride[0], padding[0], dilation[0])
        out_h = _conv_out_dim(h, kernel_h, stride[1], padding[1], dilation[1])
        out_w = _conv_out_dim(w, kernel_w, stride[2], padding[2], dilation[2])
        return {"output": (n, c_out, out_d, out_h, out_w)}

    def _validate_dtypes(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        if input.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError(
                f"input.dtype must be float32, float16, or bfloat16, got {input.dtype}"
            )
        if weight.dtype != input.dtype:
            raise ValueError(
                f"weight.dtype must match input.dtype {input.dtype}, got {weight.dtype}"
            )
        if self.dtype is not None and input.dtype != self.dtype:
            raise ValueError(f"input.dtype must match op dtype {self.dtype}, got {input.dtype}")

    def eval_roofline(self) -> tuple[int, int]:
        flops = (
            2
            * self.n
            * self.c_out
            * self.out_d
            * self.out_h
            * self.out_w
            * self.c_in_g
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.kernel_size[2]
        )
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        bytes_ = (
            self.n * self.c_in * self.d * self.h * self.w
            + self.c_out
            * self.c_in_g
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.kernel_size[2]
            + self.n * self.c_out * self.out_d * self.out_h * self.out_w
        ) * elem_bytes
        return int(flops), int(bytes_)


class Conv3dBiasFwdOp(Conv3dFwdOp):
    """Conv3d forward with bias=True default.

    Identical to :class:`Conv3dFwdOp` but defaults ``bias=True`` so the
    manifest key ``Conv3dBiasFwdOp`` resolves to a distinct class name.
    """

    def __init__(
        self,
        n: int,
        c_in: int,
        d: int,
        h: int,
        w: int,
        c_out: int,
        kernel_size: int | Tuple[int, int, int],
        stride: int | Tuple[int, int, int] = 1,
        padding: int | Tuple[int, int, int] | str = 0,
        dilation: int | Tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if not bias:
            raise ValueError(
                "Conv3dBiasFwdOp requires bias=True. "
                "Use Conv3dFwdOp for the no-bias variant."
            )
        super().__init__(
            n=n,
            c_in=c_in,
            d=d,
            h=h,
            w=w,
            c_out=c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
            kernel_map=kernel_map,
            tune=tune,
            _has_bias=True,
        )

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(input, weight, bias)
        _validate_tensor_shape(
            "Conv3d",
            "input",
            input,
            (self.n, self.c_in, self.d, self.h, self.w),
        )
        _validate_tensor_shape(
            "Conv3d",
            "weight",
            weight,
            (
                self.c_out,
                self.c_in_g,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            ),
        )
        _validate_tensor_shape("Conv3d", "bias", bias, (self.c_out,))
        return self.kernel(input, weight, bias)

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, int, int, int, int],
        weight_shape: tuple[int, int, int, int, int],
        bias_shape: tuple[int],
    ) -> Dict[str, tuple[int, int, int, int, int]]:
        del bias_shape
        return super()._infer_output_shapes(input_shape, weight_shape)

    def _validate_dtypes(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super()._validate_dtypes(input, weight)
        if bias.dtype != input.dtype:
            raise ValueError(
                f"bias.dtype must match input.dtype {input.dtype}, got {bias.dtype}"
            )

    def eval_roofline(self) -> tuple[int, int]:
        flops = (
            2
            * self.n
            * self.c_out
            * self.out_d
            * self.out_h
            * self.out_w
            * self.c_in_g
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.kernel_size[2]
            + self.n * self.c_out * self.out_d * self.out_h * self.out_w
        )
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        bytes_ = (
            self.n * self.c_in * self.d * self.h * self.w
            + self.c_out
            * self.c_in_g
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.kernel_size[2]
            + self.c_out
            + self.n * self.c_out * self.out_d * self.out_h * self.out_w
        ) * elem_bytes
        return int(flops), int(bytes_)
