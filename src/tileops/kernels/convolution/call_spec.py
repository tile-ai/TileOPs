"""The facts of one convolution call its in-tree kernels select and build on, and the kernel
interfaces their implementations inherit."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "Conv1dCall",
    "Conv1dFwdInterface",
    "Conv2dCall",
    "Conv2dFwdInterface",
    "Conv3dCall",
    "Conv3dFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class Conv1dCall(CallSpec):
    """One Conv1d call, as the op knows it after inferring the output length.

    The operator has already validated that the public input is NCL, the weight
    is OIL, and the output length is positive. The output length is::

        out_l = floor((l_in + pad_left + pad_right
                       - dilation_l * (kernel_l - 1) - 1)
                      / stride_l) + 1
    """

    n: int = 1
    c_in: int = 1
    c_out: int = 1
    c_in_g: int = 1
    l_in: int = 1
    kernel_l: int = 1
    stride_l: int = 1
    pad_left: int = 0
    pad_right: int = 0
    dilation_l: int = 1
    groups: int = 1
    out_l: int = 1
    dtype: torch.dtype = torch.float16
    has_bias: bool = False

    @property
    def c_out_g(self) -> int:
        """Output channels one group produces."""
        return self.c_out // self.groups


@dataclasses.dataclass(frozen=True)
class Conv2dCall(CallSpec):
    """One Conv2d call, as the op knows it after inferring the output extents.

    The operator has already validated that the public input is NCHW, the weight
    is OIHW, and the output extents are positive. For each spatial axis the
    output size is::

        out_axis = floor((in_axis + pad_axis + pad_end_axis
                          - dilation_axis * (kernel_axis - 1) - 1)
                         / stride_axis) + 1
    """

    n: int = 1
    c_in: int = 1
    c_out: int = 1
    c_in_g: int = 1
    h: int = 1
    w: int = 1
    kernel_h: int = 1
    kernel_w: int = 1
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    # The padding after each axis where it differs from ``padding``, the padding before;
    # ``padding='same'`` with an odd total puts the extra element here, as torch does.
    padding_end: Optional[tuple[int, int]] = None
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1
    out_h: int = 1
    out_w: int = 1
    dtype: torch.dtype = torch.float16
    has_bias: bool = False

    @property
    def c_out_g(self) -> int:
        """Output channels one group produces."""
        return self.c_out // self.groups

    @property
    def out_hw(self) -> int:
        """Output elements of one image, which an m tile is laid over."""
        return self.out_h * self.out_w


@dataclasses.dataclass(frozen=True)
class Conv3dCall(CallSpec):
    """One Conv3d call, as the op knows it after inferring the output extents.

    The operator has already validated that the public input is NCDHW, the
    weight is OIDHW, and the output extents are positive. For each spatial
    axis the output size is::

        out_axis = floor((in_axis + pad_axis + pad_end_axis
                          - dilation_axis * (kernel_axis - 1) - 1)
                         / stride_axis) + 1
    """

    n: int = 1
    c_in: int = 1
    c_out: int = 1
    c_in_g: int = 1
    d: int = 1
    h: int = 1
    w: int = 1
    kernel_d: int = 1
    kernel_h: int = 1
    kernel_w: int = 1
    stride: tuple[int, int, int] = (1, 1, 1)
    padding: tuple[int, int, int] = (0, 0, 0)
    # The padding after each axis where it differs from ``padding``, the padding before;
    # ``padding='same'`` with an odd total puts the extra element here, as torch does.
    padding_end: Optional[tuple[int, int, int]] = None
    dilation: tuple[int, int, int] = (1, 1, 1)
    groups: int = 1
    out_d: int = 1
    out_h: int = 1
    out_w: int = 1
    dtype: torch.dtype = torch.float16
    has_bias: bool = False

    @property
    def c_out_g(self) -> int:
        """Output channels one group produces."""
        return self.c_out // self.groups

    @property
    def kernel_volume(self) -> int:
        """Weight elements one input channel contributes to one output element."""
        return self.kernel_d * self.kernel_h * self.kernel_w

    @property
    def output_spatial(self) -> int:
        """Output elements per channel across the batch."""
        return self.n * self.out_d * self.out_h * self.out_w


class Conv1dFwdInterface(KernelInterface):
    """1D cross-correlation over an NCL input, as ``torch.nn.functional.conv1d``."""

    request = Conv1dCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convolve *x* with *weight*; nothing is written in place.

        Every tensor is contiguous on ``call.device`` and in ``call.dtype``, the bias
        included. The program is compiled for one side of ``call.has_bias``, so a call
        passes a bias exactly where the call spec said it would.

        Args:
            x: ``(call.n, call.c_in, call.l_in)``.
            weight: ``(call.c_out, call.c_in_g, call.kernel_l)``.
            bias: ``(call.c_out,)``, or ``None`` where ``call.has_bias`` is false.

        Returns:
            A new ``(call.n, call.c_out, call.out_l)`` tensor in ``call.dtype``.
            Out-of-bounds input positions read as zero; ``float32`` operands
            multiply in TF32, as cuDNN does by default.
        """


class Conv2dFwdInterface(KernelInterface):
    """2D cross-correlation over an NCHW input, as ``torch.nn.functional.conv2d``."""

    request = Conv2dCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convolve *x* with *weight*; nothing is written in place.

        Every tensor is contiguous on ``call.device`` and in ``call.dtype``, the bias
        included. The program is compiled for one side of ``call.has_bias``, so a call
        passes a bias exactly where the call spec said it would. An implementation that
        stages another layout allocates it itself.

        Args:
            x: ``(call.n, call.c_in, call.h, call.w)``.
            weight: ``(call.c_out, call.c_in_g, call.kernel_h, call.kernel_w)``.
            bias: ``(call.c_out,)``, or ``None`` where ``call.has_bias`` is false.

        Returns:
            A new ``(call.n, call.c_out, call.out_h, call.out_w)`` tensor in
            ``call.dtype``. Out-of-bounds input positions read as zero;
            ``float32`` operands multiply in TF32, as cuDNN does by default.
        """


class Conv3dFwdInterface(KernelInterface):
    """3D cross-correlation over an NCDHW input, as ``torch.nn.functional.conv3d``."""

    request = Conv3dCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convolve *x* with *weight*; nothing is written in place.

        Every tensor is contiguous on ``call.device`` and in ``call.dtype``, the bias
        included. The program is compiled for one side of ``call.has_bias``, so a call
        passes a bias exactly where the call spec said it would. An implementation that
        stages another layout allocates it itself.

        Args:
            x: ``(call.n, call.c_in, call.d, call.h, call.w)``.
            weight: ``(call.c_out, call.c_in_g, call.kernel_d, call.kernel_h,
                call.kernel_w)``.
            bias: ``(call.c_out,)``, or ``None`` where ``call.has_bias`` is false.

        Returns:
            A new ``(call.n, call.c_out, call.out_d, call.out_h, call.out_w)`` tensor in
            ``call.dtype``. Out-of-bounds input positions read as zero;
            ``float32`` operands multiply in TF32, as cuDNN does by default.
        """
