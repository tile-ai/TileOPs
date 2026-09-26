"""Workload definitions for the convolution op family."""

from typing import Any, Optional

import torch
import torch.nn.functional as F

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
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            self.kernel_size,
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device="cuda", dtype=self.dtype) if self.bias else None
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
            self.n, self.c_in, self.h, self.w, device="cuda", dtype=self.dtype
        ).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device="cuda", dtype=self.dtype) if self.bias else None
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
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device="cuda", dtype=self.dtype) if self.bias else None
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
