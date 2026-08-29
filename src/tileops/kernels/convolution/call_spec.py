"""Call records and implementation regions for convolution kernels."""

from __future__ import annotations

import dataclasses

import torch

from tileops.kernels.call_spec import CallSpec

__all__ = [
    "Conv1dCall",
    "Conv2dCall",
    "Conv3dCall",
    "conv1d_dense_region",
    "conv1d_group_region",
    "conv1d_pointwise_region",
    "conv2d_dense_region",
    "conv2d_group_region",
    "conv2d_pointwise_region",
    "conv2d_symmetric_region",
    "conv3d_dense_region",
    "conv3d_group_region",
    "conv3d_ndhwc_region",
]


@dataclasses.dataclass(frozen=True)
class Conv1dCall(CallSpec):
    """Semantic and shape facts used to select a Conv1d implementation.

    The operator has already validated that the public input is NCL, the weight
    is OIL, and the output length is positive. The output length is::

        out_l = floor((l_in + pad_left + pad_right
                       - dilation_l * (kernel_l - 1) - 1)
                      / stride_l) + 1

    Implementations state positive regions over these fields, and the general
    dense implementation serves only ``groups == 1``.
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


@dataclasses.dataclass(frozen=True)
class Conv2dCall(CallSpec):
    """Semantic and shape facts used to select a Conv2d implementation.

    The operator has already validated that the public input is NCHW, the weight
    is OIHW, and the output dimensions are positive. For each spatial axis the
    output size is::

        out_axis = floor((in_axis + 2 * pad_axis
                          - dilation_axis * (kernel_axis - 1) - 1)
                         / stride_axis) + 1

    Implementations state positive regions over these fields. Specialized
    regions are deliberately disjoint; for example the symmetric implicit-GEMM
    path excludes 1x1 pointwise calls.
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
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1
    out_h: int = 1
    out_w: int = 1
    dtype: torch.dtype = torch.float16
    has_bias: bool = False


@dataclasses.dataclass(frozen=True)
class Conv3dCall(CallSpec):
    """Semantic and shape facts used to select a Conv3d implementation.

    The operator has already validated that the public input is NCDHW, the
    weight is OIDHW, and the output dimensions are positive. For each spatial
    axis the output size is::

        out_axis = floor((in_axis + 2 * pad_axis
                          - dilation_axis * (kernel_axis - 1) - 1)
                         / stride_axis) + 1

    Implementations state positive regions over these fields. Architecture is
    inherited from :class:`tileops.kernels.call_spec.CallSpec` and checked by
    ``Kernel.refusal()`` before an implementation's region is evaluated.
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
    dilation: tuple[int, int, int] = (1, 1, 1)
    groups: int = 1
    out_d: int = 1
    out_h: int = 1
    out_w: int = 1
    dtype: torch.dtype = torch.float16
    has_bias: bool = False

    @property
    def kernel_volume(self) -> int:
        return self.kernel_d * self.kernel_h * self.kernel_w

    @property
    def output_spatial(self) -> int:
        return self.n * self.out_d * self.out_h * self.out_w


def conv1d_dense_region(call: Conv1dCall) -> bool:
    """The dense Conv1d fallback region."""

    return call.groups == 1


def conv1d_group_region(call: Conv1dCall) -> bool:
    """The grouped Conv1d region."""

    return call.groups > 1


def conv1d_pointwise_region(call: Conv1dCall) -> bool:
    """The dense 1x1 Conv1d region that lowers to a pointwise GEMM."""

    return (
        conv1d_dense_region(call)
        and call.kernel_l == 1
        and call.stride_l == 1
        and call.pad_left == 0
        and call.pad_right == 0
        and call.dilation_l == 1
    )


def conv2d_dense_region(call: Conv2dCall) -> bool:
    """The dense Conv2d fallback region."""

    return call.groups == 1


def conv2d_group_region(call: Conv2dCall) -> bool:
    """The grouped Conv2d region."""

    return call.groups > 1


def conv2d_pointwise_region(call: Conv2dCall) -> bool:
    """The dense 1x1 Conv2d region that lowers to a pointwise GEMM."""

    return (
        conv2d_dense_region(call)
        and call.kernel_h == 1
        and call.kernel_w == 1
        and call.stride == (1, 1)
        and call.padding == (0, 0)
        and call.dilation == (1, 1)
    )


def conv2d_symmetric_region(call: Conv2dCall) -> bool:
    """The dense symmetric Conv2d region staged through NHWC implicit GEMM."""

    return (
        conv2d_dense_region(call)
        and not conv2d_pointwise_region(call)
        and call.kernel_h == call.kernel_w
        and call.stride[0] == call.stride[1]
        and call.padding[0] == call.padding[1]
        and call.dilation[0] == call.dilation[1]
        and call.c_in % 32 == 0
    )


def conv3d_dense_region(call: Conv3dCall) -> bool:
    """The dense Conv3d fallback region."""

    return call.groups == 1


def conv3d_group_region(call: Conv3dCall) -> bool:
    """The grouped Conv3d region."""

    return call.groups > 1


def conv3d_ndhwc_region(call: Conv3dCall) -> bool:
    """The conservative NDHWC-staged dense Conv3d fast-path region.

    This region is a performance heuristic layered on top of dense Conv3d
    correctness. The implementation pays input, weight, and output layout
    transforms to make the activation gather channel-contiguous, so pointwise
    and small-output calls remain on the dense fallback.
    """

    return (
        conv3d_dense_region(call)
        and call.dtype in {torch.float16, torch.bfloat16}
        and call.c_in % 32 == 0
        and call.c_out >= 64
        and call.output_spatial >= 1024
        and call.kernel_volume > 1
    )
