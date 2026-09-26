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
    Conv1dKernel,
    Conv1dPointwiseKernel,
    Conv2d1x1Kernel,
    Conv2dKernel,
    Conv2dSymmetricKernel,
    Conv3dKernel,
    Conv3dNdhwcKernel,
    GroupConv1dKernel,
    GroupConv2dKernel,
    GroupConv3dKernel,
)
from tileops.kernels.convolution.call_spec import (
    Conv1dCall,
    Conv2dCall,
    Conv3dCall,
    conv3d_ndhwc_region,
)
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from .op_base import Op

__all__ = [
    "Conv1dFwdOp",
    "Conv2dFwdOp",
    "Conv3dFwdOp",
]


def _axes(value: int | Tuple[int, ...], dims: int) -> Tuple[int, ...]:
    """A per-axis parameter as one value per spatial axis."""
    return (value,) * dims if isinstance(value, int) else tuple(value)


def _symmetric_padding(
    padding: int | Tuple[int, ...] | str,
    kernel: Tuple[int, ...],
    dilation: Tuple[int, ...],
    op_name: str,
) -> Tuple[int, ...]:
    """The padding each side of every spatial axis, which the 2D and 3D kernels take.

    Raises:
        ValueError: ``padding='same'`` with an even effective kernel, which pads one side more
            than the other; no in-tree kernel serves asymmetric padding.
    """
    if padding == "valid":
        return (0,) * len(kernel)
    if padding == "same":
        total = tuple(d * (k - 1) for k, d in zip(kernel, dilation, strict=True))
        if any(t % 2 for t in total):
            raise ValueError(
                f"{op_name} padding='same' with an even effective kernel pads asymmetrically, "
                "which no in-tree kernel serves"
            )
        return tuple(t // 2 for t in total)
    return _axes(padding, len(kernel))


def _out_dim(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class Conv1dFwdOp(Op):
    """1D convolution over an NCL input, as ``torch.nn.functional.conv1d``."""

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "conv1d_pointwise_kernel": Conv1dPointwiseKernel,
        "conv1d_kernel": Conv1dKernel,
        "group_conv1d_kernel": GroupConv1dKernel,
    }

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
            out_l = _out_dim(l_in, kernel_l, stride, pad, dilation)
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
            tune=self.tune,
            device=input.device,
        )
        self.kernel = self.kernel_for("conv1d", (input, weight, bias), call)
        return self.kernel(input, weight, bias)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])


class Conv2dFwdOp(Op):
    """2D convolution over an NCHW input, as ``torch.nn.functional.conv2d``.

    The in-tree kernels pad each axis symmetrically, so ``padding='same'`` with an even
    effective kernel is refused. They multiply float32 operands in TF32, as cuDNN does by
    default.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "conv2d_1x1_kernel": Conv2d1x1Kernel,
        "conv2d_symmetric_kernel": Conv2dSymmetricKernel,
        "conv2d_kernel": Conv2dKernel,
        "group_conv2d_kernel": GroupConv2dKernel,
    }

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
                (default 0).
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

        Raises:
            ValueError: ``padding='same'`` with an even effective kernel.
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
        padding = _symmetric_padding(self.padding, (kernel_h, kernel_w), dilation, "Conv2d")
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
            dilation=dilation,
            groups=self.groups,
            out_h=_out_dim(h, kernel_h, stride[0], padding[0], dilation[0]),
            out_w=_out_dim(w, kernel_w, stride[1], padding[1], dilation[1]),
            dtype=input.dtype,
            has_bias=bias is not None,
            tune=self.tune,
            device=input.device,
        )
        self.kernel = self.kernel_for("conv2d", (input, weight, bias), call)
        return self.kernel(input, weight, bias)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])


def _can_use_conv3d_ndhwc(
    *,
    groups: int,
    c_in: int,
    c_out: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    out_d: int,
    out_h: int,
    out_w: int,
    n: int,
    dtype: torch.dtype,
) -> bool:
    """Return whether the NDHWC Conv3d fast path should serve this call.

    This is a performance eligibility guard, not the full Conv3d validity check.
    The op layer has already validated the convolution shape and computes
    ``out_d``, ``out_h``, and ``out_w`` as::

        out_axis = floor((in_axis + 2 * pad_axis
                          - dilation_axis * (kernel_axis - 1) - 1)
                         / stride_axis) + 1

    The NDHWC fast path materializes input, weight, and output staging layouts
    so the activation gather reads channel-contiguous runs. Keep it limited to
    dense, 16-bit, non-pointwise calls large enough to amortize that fixed
    layout-transform cost.
    """
    return conv3d_ndhwc_region(
        Conv3dCall(
            arch=0,
            n=n,
            c_in=c_in,
            c_out=c_out,
            kernel_d=kernel_d,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            out_d=out_d,
            out_h=out_h,
            out_w=out_w,
            groups=groups,
            dtype=dtype,
        )
    )


class Conv3dFwdOp(Op):
    """3D convolution over an NCDHW input, as ``torch.nn.functional.conv3d``.

    The in-tree kernels pad each axis symmetrically, so ``padding='same'`` with an even
    effective kernel is refused. They multiply float32 operands in TF32, as cuDNN does by
    default.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "conv3d_kernel": Conv3dKernel,
        "conv3d_ndhwc_kernel": Conv3dNdhwcKernel,
        "group_conv3d_kernel": GroupConv3dKernel,
    }

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
                (default 0).
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

        Raises:
            ValueError: ``padding='same'`` with an even effective kernel.
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
        padding = _symmetric_padding(self.padding, kernel, dilation, "Conv3d")
        out_d, out_h, out_w = (
            _out_dim(size, k, s, p, dl)
            for size, k, s, p, dl in zip((d, h, w), kernel, stride, padding, dilation, strict=True)
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
            dilation=dilation,
            groups=self.groups,
            out_d=out_d,
            out_h=out_h,
            out_w=out_w,
            dtype=input.dtype,
            has_bias=bias is not None,
            tune=self.tune,
            device=input.device,
        )
        self.kernel = self.kernel_for("conv3d", (input, weight, bias), call)
        return self.kernel(input, weight, bias)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])
