"""Workload definitions for the pool op family."""

from typing import Optional

import torch

from workloads.workload_base import WorkloadBase  # noqa: F401


class AvgPool1dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: int,
        stride: Optional[int],
        padding: int,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

class AvgPool2dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int],
        stride: Optional[tuple[int, int]],
        padding: tuple[int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n, self.c_in, self.h_in, self.w_in, device="cuda", dtype=self.dtype
        ).contiguous()
        return (x,)

class AvgPool3dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int, int],
        stride: Optional[tuple[int, int, int]],
        padding: tuple[int, int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.d_in,
            self.h_in,
            self.w_in,
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        return (x,)

class MaxPool2dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int],
        stride: Optional[tuple[int, int]],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        ceil_mode: bool,
        dtype: torch.dtype,
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n, self.c_in, self.h_in, self.w_in, device="cuda", dtype=self.dtype
        ).contiguous()
        return (x,)

class MaxPool1dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: tuple[int],
        stride: Optional[tuple[int]],
        padding: tuple[int],
        dilation: tuple[int],
        ceil_mode: bool,
        dtype: torch.dtype,
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

class MaxPool3dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int, int],
        stride: Optional[tuple[int, int, int]],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
        ceil_mode: bool,
        dtype: torch.dtype,
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.d_in,
            self.h_in,
            self.w_in,
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        return (x,)

class AvgPoolWorkload(WorkloadBase):
    def __init__(
        self,
        ndim: int,
        kernel_size: int | tuple[int, ...],
        stride: Optional[int | tuple[int, ...]],
        padding: int | tuple[int, ...],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self, *shape: int) -> tuple[torch.Tensor]:
        x = torch.randn(*shape, device="cuda", dtype=self.dtype).contiguous()
        return (x,)


class MaxPoolWorkload(WorkloadBase):
    def __init__(
        self,
        ndim: int,
        kernel_size: tuple[int, ...],
        stride: Optional[tuple[int, ...]],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        ceil_mode: bool,
        dtype: torch.dtype,
        contiguous: bool = True,
        return_indices: bool = False,
    ) -> None:
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.contiguous = contiguous
        self.return_indices = return_indices

    def gen_inputs(self, *shape: int) -> tuple[torch.Tensor]:
        x = torch.randn(*shape, device="cuda", dtype=self.dtype)
        if self.contiguous:
            x = x.contiguous()
        else:
            # Non-contiguous view: transpose the last two dims twice so strides
            # differ but shape semantics stay N,C,<spatial dims>.
            x = x.transpose(-2, -1).contiguous().transpose(-2, -1)
            assert not x.is_contiguous()
        return (x,)
