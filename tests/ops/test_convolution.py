from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import Conv1dFwdOp, Conv2dOp, Conv3dOp

SKIPPED_CONV2D_CASES = {
    (2, 32, 32, 32, 64, (3, 3), (1, 1), (1, 1), torch.float16, False),
    (2, 32, 32, 32, 64, (3, 3), (1, 1), (1, 1), torch.bfloat16, False),
    (1, 32, 28, 28, 64, (5, 5), (1, 1), (2, 2), torch.float16, False),
}

SKIPPED_CONV3D_CASES = {
    (1, 16, 8, 32, 32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.float16, False),
    (1, 16, 8, 32, 32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.bfloat16, False),
    (1, 32, 32, 64, 64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.bfloat16, False),
}


class Conv1dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, l_in, c_out, kernel_size, stride, padding, dtype, tune", [
            pytest.param(
                2, 64, 512, 128, 3, 1, 1, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-tcn-k3-s1-fp16",
            ),
            pytest.param(
                2, 64, 512, 128, 3, 1, 1, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-tcn-k3-s1-bf16",
            ),
        ]),
    ]


class Conv1dTest(TestBase):

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(self.n, self.l_in, self.c_in, device="cuda", dtype=self.dtype).contiguous()
        weight = torch.randn(
            self.c_out, self.c_in, self.kernel_size,
            device="cuda", dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device="cuda", dtype=self.dtype).contiguous()
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = F.conv1d(
            x.permute(0, 2, 1).contiguous(),
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1,
        )
        return out.permute(0, 2, 1).contiguous()


@Conv1dFixture
def test_conv1d(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = Conv1dTest(n, c_in, l_in, c_out, kernel_size, stride, padding, dtype)
    op = Conv1dFwdOp(
        n=n,
        c_in=c_in,
        l_in=l_in,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class Conv2dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, h, w, c_out, kernel_size, stride, padding, dtype, tune", [
            pytest.param(
                2, 32, 32, 32, 64, (3, 3), (1, 1), (1, 1), torch.float16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp16-3x3",
            ),
            pytest.param(
                2, 32, 32, 32, 64, (3, 3), (1, 1), (1, 1), torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-bf16-3x3",
            ),
        ]),
    ]


class Conv2dTest(TestBase):

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
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(self.n, self.h, self.w, self.c_in, device="cuda", dtype=self.dtype).contiguous()
        weight = torch.randn(
            self.c_out, self.c_in, self.kernel_size[0], self.kernel_size[1],
            device="cuda", dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device="cuda", dtype=self.dtype).contiguous()
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = F.conv2d(
            x.permute(0, 3, 1, 2).contiguous(),
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1,
        )
        return out.permute(0, 2, 3, 1).contiguous()


@Conv2dFixture
def test_conv2d(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    if (n, c_in, h, w, c_out, kernel_size, stride, padding, dtype, tune) in SKIPPED_CONV2D_CASES:
        pytest.skip("Temporarily skipping known Conv2d failures under TileLang 5f70374c (#999).")
    test = Conv2dTest(n, c_in, h, w, c_out, kernel_size, stride, padding, dtype)
    op = Conv2dOp(
        n=n,
        c_in=c_in,
        h=h,
        w=w,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class Conv3dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dtype, tune", [
            pytest.param(
                1, 16, 8, 32, 32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.float16, False,
                marks=pytest.mark.smoke,
                id="smoke-3d-unet-k3-s1-fp16",
            ),
            pytest.param(
                1, 16, 8, 32, 32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-3d-unet-k3-s1-bf16",
            ),
        ]),
    ]


class Conv3dTest(TestBase):

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        c_out: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(
            self.n, self.d_in, self.h_in, self.w_in, self.c_in,
            device="cuda", dtype=self.dtype,
        ).contiguous()
        weight = torch.randn(
            self.c_out, self.c_in, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            device="cuda", dtype=self.dtype,
        ).contiguous()
        bias = torch.zeros(self.c_out, device="cuda", dtype=self.dtype).contiguous()
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = F.conv3d(
            x.permute(0, 4, 1, 2, 3).contiguous(),
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1,
        )
        return out.permute(0, 2, 3, 4, 1).contiguous()


@Conv3dFixture
def test_conv3d(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    if (n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dtype, tune) in SKIPPED_CONV3D_CASES:
        pytest.skip("Temporarily skipping known Conv3d failures under TileLang 5f70374c (#999).")
    test = Conv3dTest(n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dtype)
    op = Conv3dOp(
        n=n,
        c_in=c_in,
        d_in=d_in,
        h_in=h_in,
        w_in=w_in,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
