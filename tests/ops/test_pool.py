from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel_base import Kernel
from tileops.ops import AvgPool1dOp, AvgPool2dOp, AvgPool3dOp


class _DummyKernel(Kernel):
    supported_archs = [80]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AvgPool1dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, tune", [
            pytest.param(
                2, 64, 512, 3, None, 1, False, True, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-k3-default-stride-fp16",
            ),
            pytest.param(
                2, 64, 512, 3, None, 1, False, True, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-k3-default-stride-bf16",
            ),
        ]),
    ]


class AvgPool1dTest(TestBase):

    def __init__(
        self,
        kernel_size: int,
        stride: int | None,
        padding: int,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: torch.dtype,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype

    def gen_inputs(self, n: int, c_in: int, l_in: int) -> tuple[torch.Tensor]:
        x = torch.randn(n, l_in, c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool1d(
            x.permute(0, 2, 1).contiguous(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
        )
        return out.permute(0, 2, 1).contiguous()


@AvgPool1dFixture
def test_avg_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: int,
    stride: int | None,
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool1dTest(kernel_size, stride, padding, ceil_mode, count_include_pad, dtype)
    op = AvgPool1dOp(
        n=n,
        c_in=c_in,
        l_in=l_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(n, c_in, l_in), atol=atol, rtol=rtol)

class AvgPool2dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune", [
            pytest.param(
                2, 64, 56, 56, (3, 3), None, (1, 1), False, True, None, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-3x3-default-stride-fp16",
            ),
            pytest.param(
                2, 64, 56, 56, (3, 3), None, (1, 1), False, True, None, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-3x3-default-stride-bf16",
            ),
        ]),
    ]


class AvgPool2dTest(TestBase):

    def __init__(
        self,
        kernel_size: tuple[int, int],
        stride: Optional[tuple[int, int]],
        padding: tuple[int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self, n: int, c_in: int, h_in: int, w_in: int) -> tuple[torch.Tensor]:
        x = torch.randn(n, h_in, w_in, c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool2d(
            x.permute(0, 3, 1, 2).contiguous(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )
        return out.permute(0, 2, 3, 1).contiguous()


@AvgPool2dFixture
def test_avg_pool2d(
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
    tune: bool,
) -> None:
    test = AvgPool2dTest(
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    op = AvgPool2dOp(
        n=n,
        c_in=c_in,
        h_in=h_in,
        w_in=w_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(n, c_in, h_in, w_in), atol=atol, rtol=rtol)

class AvgPool3dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune", [
            pytest.param(
                1, 32, 16, 28, 28, (2, 2, 2), None, (0, 0, 0), False, True, None, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-2x2x2-default-stride-fp16",
            ),
            pytest.param(
                1, 32, 16, 28, 28, (2, 2, 2), None, (0, 0, 0), False, True, None, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-2x2x2-default-stride-bf16",
            ),
        ]),
    ]


class AvgPool3dTest(TestBase):

    def __init__(
        self,
        kernel_size: tuple[int, int, int],
        stride: Optional[tuple[int, int, int]],
        padding: tuple[int, int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self, n: int, c_in: int, d_in: int, h_in: int, w_in: int) -> tuple[torch.Tensor]:
        x = torch.randn(n, d_in, h_in, w_in, c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool3d(
            x.permute(0, 4, 1, 2, 3).contiguous(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )
        return out.permute(0, 2, 3, 4, 1).contiguous()


@AvgPool3dFixture
def test_avg_pool3d(
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
    tune: bool,
) -> None:
    test = AvgPool3dTest(
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    op = AvgPool3dOp(
        n=n,
        c_in=c_in,
        d_in=d_in,
        h_in=h_in,
        w_in=w_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(n, c_in, d_in, h_in, w_in), atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
