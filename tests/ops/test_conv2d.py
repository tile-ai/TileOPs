from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.convolution import Conv2d1x1Kernel, Conv2dKernel
from tileops.ops import Conv2dOp


class Conv2dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, h, w, c_out, kernel_size, stride, padding, dtype, tune", [
            pytest.param(
                2, 32, 32, 32, 64, (3, 3), (1, 1), (1, 1), torch.float16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp16-3x3",
            ),
            pytest.param(
                1, 3, 112, 112, 64, (3, 3), (2, 2), (1, 1), torch.float16, False,
                marks=pytest.mark.full,
                id="full-stem-3x3-s2-fp16",
            ),
            pytest.param(
                1, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), torch.float16, False,
                marks=pytest.mark.full,
                id="full-resblock-3x3-s1-fp16",
            ),
            pytest.param(
                1, 128, 56, 56, 256, (3, 3), (2, 2), (1, 1), torch.float16, False,
                marks=pytest.mark.full,
                id="full-stage-transition-3x3-s2-fp16",
            ),
            pytest.param(
                1, 32, 28, 28, 64, (5, 5), (1, 1), (2, 2), torch.float16, False,
                marks=pytest.mark.full,
                id="full-small-5x5-s1-fp16",
            ),
            pytest.param(
                1, 64, 28, 28, 128, (5, 5), (2, 2), (2, 2), torch.float16, False,
                marks=pytest.mark.full,
                id="full-small-5x5-s2-fp16",
            ),
            pytest.param(
                2, 32, 32, 32, 64, (1, 1), (1, 1), (0, 0), torch.float16, True,
                marks=pytest.mark.full,
                id="full-fp16-1x1-tuned",
            ),
            pytest.param(
                1, 64, 28, 28, 128, (3, 3), (2, 2), (1, 1), torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-stride2",
            ),
            pytest.param(
                2, 32, 32, 32, 64, (3, 3), (1, 1), (1, 1), torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-bf16-3x3",
            ),
            pytest.param(
                1, 64, 28, 28, 64, (1, 1), (1, 1), (0, 0), torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-bf16-1x1",
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
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
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

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
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


@pytest.mark.smoke
def test_conv2d_accepts_zero_bias() -> None:
    op = Conv2dOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=32,
        kernel_size=3,
        padding=1,
        bias=True,
    )
    x = torch.randn(1, 16, 16, 32, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(32, 32, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.zeros(32, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv2d(x.permute(0, 3, 1, 2).contiguous(), weight, bias=bias, padding=1)
    ref = ref.permute(0, 2, 3, 1).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv2d_dispatches_1x1_kernel() -> None:
    op = Conv2dOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=1,
        bias=True,
    )
    assert isinstance(op.kernel, Conv2d1x1Kernel)


@pytest.mark.smoke
def test_conv2d_does_not_dispatch_1x1_kernel_with_padding() -> None:
    op = Conv2dOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=1,
        padding=1,
        bias=True,
    )
    assert isinstance(op.kernel, Conv2dKernel)


@pytest.mark.smoke
def test_conv2d_dispatches_3x3_kernel() -> None:
    op = Conv2dOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=3,
        padding=1,
        bias=True,
    )
    assert isinstance(op.kernel, Conv2dKernel)


@pytest.mark.smoke
def test_conv2d_dispatches_5x5_kernel() -> None:
    op = Conv2dOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=5,
        padding=2,
        bias=True,
    )
    assert isinstance(op.kernel, Conv2dKernel)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
