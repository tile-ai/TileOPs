"""Workload definitions for the convolution op family."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.numerics import Exact, reference_tolerance
from workloads.workload_base import WorkloadBase


class _ConvWorkload(WorkloadBase):
    """A convolution's input, weight and bias, and its reference; a subclass names the torch
    function as ``_conv`` and the spatial extents of the input and the kernel."""

    def _extents(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        raise NotImplementedError

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        spatial, kernel = self._extents()
        device = run_device()
        x = torch.randn(self.n, self.c_in, *spatial, device=device, dtype=self.dtype)
        weight = torch.randn(
            self.c_out, self.c_in // self.groups, *kernel, device=device, dtype=self.dtype
        )
        bias = torch.zeros(self.c_out, device=device, dtype=self.dtype) if self.bias else None
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """The convolution on fp32 operands with cuDNN's TF32 off, rounded once to ``x.dtype``."""
        cudnn = torch.backends.cudnn
        saved = cudnn.allow_tf32
        cudnn.allow_tf32 = False
        try:
            out = self.torch_program(
                x.float(), weight.float(), None if bias is None else bias.float()
            )
        finally:
            cudnn.allow_tf32 = saved
        return out.to(x.dtype)

    def torch_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        out = self._conv(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out.contiguous()

    def verification(self, *inputs):
        return convolution_verification(
            inputs[0].dtype, padding=self.padding, kernel_shape=inputs[1].shape[2:], inputs=inputs
        )


@dataclass(eq=False)
class Conv1dWorkload(_ConvWorkload):
    _conv = staticmethod(F.conv1d)

    n: int
    c_in: int
    l_in: int
    c_out: int
    kernel_size: int
    stride: int
    padding: int | str
    dilation: int
    groups: int
    dtype: torch.dtype
    bias: bool = True

    @classmethod
    def from_call(cls, call: Any) -> "Conv1dWorkload":
        """The workload of one manifest call of ``Conv1dFwdOp``."""
        indices = call.indices
        return cls(
            indices["N"],
            indices["C_in"],
            indices["L_in"],
            indices["C_out"],
            indices["kW"],
            indices["stride"],
            indices["padding"],
            indices["dilation"],
            indices["groups"],
            getattr(torch, indices["T"]),
            bias=call.present("bias"),
        )

    def _extents(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self.l_in,), (self.kernel_size,)


@dataclass(eq=False)
class Conv2dWorkload(_ConvWorkload):
    _conv = staticmethod(F.conv2d)

    n: int
    c_in: int
    h: int
    w: int
    c_out: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    dtype: torch.dtype
    bias: bool = True

    @classmethod
    def from_call(cls, call: Any) -> "Conv2dWorkload":
        """The workload of one manifest call of ``Conv2dFwdOp``."""
        indices = call.indices
        return cls(
            indices["N"],
            indices["C_in"],
            indices["H"],
            indices["W"],
            indices["C_out"],
            (indices["kH"], indices["kW"]),
            indices["stride"],
            indices["padding"],
            indices["dilation"],
            indices["groups"],
            getattr(torch, indices["T"]),
            bias=call.present("bias"),
        )

    def _extents(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self.h, self.w), tuple(self.kernel_size)


@dataclass(eq=False)
class Conv3dWorkload(_ConvWorkload):
    _conv = staticmethod(F.conv3d)

    n: int
    c_in: int
    d: int
    h: int
    w: int
    c_out: int
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    padding: tuple[int, int, int]
    dilation: tuple[int, int, int]
    groups: int
    dtype: torch.dtype
    bias: bool = True

    @classmethod
    def from_call(cls, call: Any) -> "Conv3dWorkload":
        """The workload of one manifest call of ``Conv3dFwdOp``."""
        indices = call.indices
        return cls(
            indices["N"],
            indices["C_in"],
            indices["D"],
            indices["H"],
            indices["W"],
            indices["C_out"],
            (indices["kD"], indices["kH"], indices["kW"]),
            indices["stride"],
            indices["padding"],
            indices["dilation"],
            indices["groups"],
            getattr(torch, indices["T"]),
            bias=call.present("bias"),
        )

    def _extents(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self.d, self.h, self.w), tuple(self.kernel_size)


def convolution_verification(dtype, *, padding=0, kernel_shape=(), inputs=None):
    """One numerical policy for the 1D, 2D and 3D convolution families.

    Given the call's ``(x, weight, ...)``, the absolute tolerance is the relative
    one at the output's scale: an output near zero cancels terms of that scale
    and keeps their accumulation error. A normalized-error bound then rejects an
    output lost to that cancellation.
    """
    tolerance = reference_tolerance(dtype)
    # An even filter under "same" padding needs an asymmetric explicit pad. The
    # reference and explicit-pad kernels can choose different reduction orders.
    if dtype == torch.float16 and padding == "same" and any(k % 2 == 0 for k in kernel_shape):
        tolerance = {"atol": 2e-3 if len(kernel_shape) == 1 else 2e-2, "rtol": 3e-3}
    if dtype == torch.float32:
        tolerance = {"atol": 6e-2, "rtol": 1.6e-2}
    if inputs is not None:
        x, weight = inputs[0], inputs[1]
        scale = (
            weight[0].numel() ** 0.5
            * (
                x.float().nan_to_num(0, 0, 0).square().mean()
                * weight.float().nan_to_num(0, 0, 0).square().mean()
            )
            .sqrt()
            .item()
        )
        tolerance = tolerance | {
            "atol": max(tolerance["atol"], tolerance["rtol"] * scale),
            "normalized": 1e-3,
        }
    return Exact(**tolerance)
