"""Convolution operators (host-side Op layer).

Provides:
  - Conv1dFwdOp: torch.nn.functional.conv1d
  - Conv2dFwdOp: torch.nn.functional.conv2d
  - Conv3dFwdOp: torch.nn.functional.conv3d
"""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.convolution import (
    Conv1dCall,
    Conv1dFwdInterface,
    Conv1dKernel,
    Conv1dPointwiseKernel,
    Conv1dUnitStrideKernel,
    Conv2d1x1Kernel,
    Conv2dCall,
    Conv2dFwdInterface,
    Conv2dKernel,
    Conv2dSymmetricKernel,
    Conv3dCall,
    Conv3dFwdInterface,
    Conv3dKernel,
    Conv3dNdhwcKernel,
    DepthwiseConv1dKernel,
    DepthwiseConv2dKernel,
    GroupConv1dKernel,
    GroupConv2dKernel,
    GroupConv3dKernel,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = [
    "Conv1dFwdOp",
    "Conv2dFwdOp",
    "Conv3dFwdOp",
]


def _axes(value: int | Tuple[int, ...], dims: int) -> Tuple[int, ...]:
    """A per-axis parameter as one value per spatial axis."""
    return (value,) * dims if isinstance(value, int) else tuple(value)


def _padding(
    padding: int | Tuple[int, ...] | str,
    kernel: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """The padding before and after every spatial axis.

    ``padding='same'`` with an odd total puts the extra element after, as torch does.
    """
    if padding == "valid":
        zeros = (0,) * len(kernel)
        return zeros, zeros
    if padding == "same":
        total = tuple(d * (k - 1) for k, d in zip(kernel, dilation, strict=True))
        before = tuple(t // 2 for t in total)
        return before, tuple(t - b for t, b in zip(total, before, strict=True))
    before = _axes(padding, len(kernel))
    return before, before


def _out_dim(size: int, kernel: int, stride: int, before: int, after: int, dilation: int) -> int:
    return (size + before + after - dilation * (kernel - 1) - 1) // stride + 1


class Conv1dFwdOp(Op):
    """1D convolution over an NCL input, as ``torch.nn.functional.conv1d``."""

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "conv1d_pointwise": Conv1dPointwiseKernel,
        "conv1d": Conv1dKernel,
        "conv1d_unit_stride": Conv1dUnitStrideKernel,
        "depthwise_conv1d": DepthwiseConv1dKernel,
        "group_conv1d": GroupConv1dKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"conv1d": Conv1dFwdInterface}

    def __init__(
        self,
        stride: int | Tuple[int] = 1,
        padding: int | Tuple[int] | str = 0,
        dilation: int | Tuple[int] = 1,
        groups: int = 1,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            stride: Stride, an int or a 1-tuple (default 1).
            padding: Padding each side, an int or a 1-tuple, or ``'valid'`` / ``'same'``
                (default 0). ``'same'`` puts the extra element of an odd total on the right.
            dilation: Dilation, an int or a 1-tuple (default 1).
            groups: Number of channel groups (default 1).
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the convolution.

        Args:
            input: Input, $[N \\times C_{in} \\times L]$.
            weight: Weight, $[C_{out} \\times C_{in}/\\text{groups} \\times k]$.
            bias: Per-output-channel bias, $[C_{out}]$, or ``None``.

        Returns:
            The convolution result, $[N \\times C_{out} \\times L_{out}]$.
        """
        return self._call_boundary(input, weight, bias)

    def _eager_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot
        follow.
        """
        n, c_in, l_in = input.shape
        c_out, c_in_g, kernel_l = weight.shape
        (stride,) = _axes(self.stride, 1)
        (dilation,) = _axes(self.dilation, 1)
        if self.padding == "same":
            total = dilation * (kernel_l - 1)
            pad_left, pad_right, out_l = total // 2, total - total // 2, l_in
        else:
            (pad,) = (0,) if self.padding == "valid" else _axes(self.padding, 1)
            pad_left = pad_right = pad
            out_l = _out_dim(l_in, kernel_l, stride, pad, pad, dilation)
        # A kernel is handed contiguous tensors, in the manifest's ``signature.inputs`` order;
        # a bias this call did not pass is ``None`` there.
        input = input.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        call = Conv1dCall(
            n=n,
            c_in=c_in,
            c_out=c_out,
            c_in_g=c_in_g,
            l_in=l_in,
            kernel_l=kernel_l,
            stride_l=stride,
            pad_left=pad_left,
            pad_right=pad_right,
            dilation_l=dilation,
            groups=self.groups,
            out_l=out_l,
            dtype=input.dtype,
            has_bias=bias is not None,
            device=input.device,
        )
        self.kernel = self.kernel_for("conv1d", call)
        return self.kernel(input, weight, bias)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])


class Conv2dFwdOp(Op):
    """2D convolution over an NCHW input, as ``torch.nn.functional.conv2d``.

    They multiply float32 operands in TF32, as cuDNN does by default.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "conv2d_1x1": Conv2d1x1Kernel,
        "conv2d_symmetric": Conv2dSymmetricKernel,
        "conv2d": Conv2dKernel,
        "depthwise_conv2d": DepthwiseConv2dKernel,
        "group_conv2d": GroupConv2dKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"conv2d": Conv2dFwdInterface}

    def __init__(
        self,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] | str = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            stride: Stride, an int or a 2-tuple (default 1).
            padding: Padding each side, an int or a 2-tuple, or ``'valid'`` / ``'same'``
                (default 0). ``'same'`` puts the extra element of an odd total after.
            dilation: Dilation, an int or a 2-tuple (default 1).
            groups: Number of channel groups (default 1).
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the convolution.

        Args:
            input: Input, $[N \\times C_{in} \\times H \\times W]$.
            weight: Weight, $[C_{out} \\times C_{in}/\\text{groups} \\times k_H \\times k_W]$.
            bias: Per-output-channel bias, $[C_{out}]$, or ``None``.

        Returns:
            The convolution result, $[N \\times C_{out} \\times H_{out} \\times W_{out}]$.
        """
        return self._call_boundary(input, weight, bias)

    def _eager_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot
        follow.
        """
        n, c_in, h, w = input.shape
        c_out, c_in_g, kernel_h, kernel_w = weight.shape
        stride = _axes(self.stride, 2)
        dilation = _axes(self.dilation, 2)
        padding, padding_end = _padding(self.padding, (kernel_h, kernel_w), dilation)
        input = input.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        call = Conv2dCall(
            n=n,
            c_in=c_in,
            c_out=c_out,
            c_in_g=c_in_g,
            h=h,
            w=w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride=stride,
            padding=padding,
            padding_end=None if padding_end == padding else padding_end,
            dilation=dilation,
            groups=self.groups,
            out_h=_out_dim(h, kernel_h, stride[0], padding[0], padding_end[0], dilation[0]),
            out_w=_out_dim(w, kernel_w, stride[1], padding[1], padding_end[1], dilation[1]),
            dtype=input.dtype,
            has_bias=bias is not None,
            device=input.device,
        )
        self.kernel = self.kernel_for("conv2d", call)
        return self.kernel(input, weight, bias)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])


class Conv3dFwdOp(Op):
    """3D convolution over an NCDHW input, as ``torch.nn.functional.conv3d``.

    They multiply float32 operands in TF32, as cuDNN does by default.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "conv3d": Conv3dKernel,
        "conv3d_ndhwc": Conv3dNdhwcKernel,
        "group_conv3d": GroupConv3dKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"conv3d": Conv3dFwdInterface}

    def __init__(
        self,
        stride: int | Tuple[int, int, int] = 1,
        padding: int | Tuple[int, int, int] | str = 0,
        dilation: int | Tuple[int, int, int] = 1,
        groups: int = 1,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            stride: Stride, an int or a 3-tuple (default 1).
            padding: Padding each side, an int or a 3-tuple, or ``'valid'`` / ``'same'``
                (default 0). ``'same'`` puts the extra element of an odd total after.
            dilation: Dilation, an int or a 3-tuple (default 1).
            groups: Number of channel groups (default 1).
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the convolution.

        Args:
            input: Input, $[N \\times C_{in} \\times D \\times H \\times W]$.
            weight: Weight, $[C_{out} \\times C_{in}/\\text{groups} \\times k_D \\times k_H
                \\times k_W]$.
            bias: Per-output-channel bias, $[C_{out}]$, or ``None``.

        Returns:
            The convolution result, $[N \\times C_{out} \\times D_{out} \\times H_{out}
            \\times W_{out}]$.
        """
        return self._call_boundary(input, weight, bias)

    def _eager_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot
        follow.
        """
        n, c_in, d, h, w = input.shape
        c_out, c_in_g, kernel_d, kernel_h, kernel_w = weight.shape
        kernel = (kernel_d, kernel_h, kernel_w)
        stride = _axes(self.stride, 3)
        dilation = _axes(self.dilation, 3)
        padding, padding_end = _padding(self.padding, kernel, dilation)
        out_d, out_h, out_w = (
            _out_dim(*axis)
            for axis in zip((d, h, w), kernel, stride, padding, padding_end, dilation, strict=True)
        )
        input = input.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        call = Conv3dCall(
            n=n,
            c_in=c_in,
            c_out=c_out,
            c_in_g=c_in_g,
            d=d,
            h=h,
            w=w,
            kernel_d=kernel_d,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride=stride,
            padding=padding,
            padding_end=None if padding_end == padding else padding_end,
            dilation=dilation,
            groups=self.groups,
            out_d=out_d,
            out_h=out_h,
            out_w=out_w,
            dtype=input.dtype,
            has_bias=bias is not None,
            device=input.device,
        )
        self.kernel = self.kernel_for("conv3d", call)
        return self.kernel(input, weight, bias)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])
