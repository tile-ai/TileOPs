"""Workload definitions for the convolution op family."""

from typing import Any, Optional

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.workload_base import WorkloadBase


class Conv1dWorkload(WorkloadBase):
    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int | str,
        dilation: int,
        groups: int,
        dtype: torch.dtype,
        bias: bool = True,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype
        self.bias = bias

    @classmethod
    def from_call(cls, call: Any) -> "Conv1dWorkload":
        """The workload of one manifest call of ``Conv1dFwdOp``."""
        ix = call.ix
        return cls(
            ix["N"],
            ix["C_in"],
            ix["L_in"],
            ix["C_out"],
            ix["kW"],
            ix["stride"],
            ix["padding"],
            ix["dilation"],
            ix["groups"],
            getattr(torch, ix["T"]),
            bias=call.present("bias"),
        )

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(
            self.n, self.c_in, self.l_in, device=run_device(), dtype=self.dtype
        ).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            self.kernel_size,
            device=run_device(),
            dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device=run_device(), dtype=self.dtype) if self.bias else None
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = F.conv1d(
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


class Conv2dWorkload(WorkloadBase):
    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        dtype: torch.dtype,
        bias: bool = True,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype
        self.bias = bias

    @classmethod
    def from_call(cls, call: Any) -> "Conv2dWorkload":
        """The workload of one manifest call of ``Conv2dFwdOp``."""
        ix = call.ix
        return cls(
            ix["N"],
            ix["C_in"],
            ix["H"],
            ix["W"],
            ix["C_out"],
            (ix["kH"], ix["kW"]),
            ix["stride"],
            ix["padding"],
            ix["dilation"],
            ix["groups"],
            getattr(torch, ix["T"]),
            bias=call.present("bias"),
        )

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(
            self.n, self.c_in, self.h, self.w, device=run_device(), dtype=self.dtype
        ).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
            device=run_device(),
            dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device=run_device(), dtype=self.dtype) if self.bias else None
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = F.conv2d(
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


class Conv3dWorkload(WorkloadBase):
    def __init__(
        self,
        n: int,
        c_in: int,
        d: int,
        h: int,
        w: int,
        c_out: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
        groups: int,
        dtype: torch.dtype,
        bias: bool = True,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d = d
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype
        self.bias = bias

    @classmethod
    def from_call(cls, call: Any) -> "Conv3dWorkload":
        """The workload of one manifest call of ``Conv3dFwdOp``."""
        ix = call.ix
        return cls(
            ix["N"],
            ix["C_in"],
            ix["D"],
            ix["H"],
            ix["W"],
            ix["C_out"],
            (ix["kD"], ix["kH"], ix["kW"]),
            ix["stride"],
            ix["padding"],
            ix["dilation"],
            ix["groups"],
            getattr(torch, ix["T"]),
            bias=call.present("bias"),
        )

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.d,
            self.h,
            self.w,
            device=run_device(),
            dtype=self.dtype,
        ).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            device=run_device(),
            dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device=run_device(), dtype=self.dtype) if self.bias else None
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = F.conv3d(
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


def convolution_verification(dtype, *, padding=0, kernel_shape=(), inputs=None):
    """One numerical policy for the 1D, 2D and 3D convolution families.

    Given the call's ``(x, weight, ...)``, the absolute tolerance is the relative
    one at the output's scale: an output near zero cancels terms of that scale
    and keeps their accumulation error. A normalized-error bound then rejects an
    output lost to that cancellation.
    """
    from workloads.numerics import Exact, reference_tolerance

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
