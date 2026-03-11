from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import Conv2dOp


class Conv2dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, h, w, c_out, kernel_size, stride, padding, bias, dtype, tune", [
            pytest.param(
                1, 3, 56, 56, 64, 1, 2, 0, False, torch.float16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp16-3in-64out-1x1-s2-56-no-bias",
            ),
            pytest.param(
                1, 64, 112, 112, 128, 1, 1, 0, True, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-64in-128out-1x1-s1-112-bias",
            ),
            pytest.param(
                1, 128, 112, 112, 256, 3, 2, 1, False, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-bf16-128in-256out-3x3-s2-112-no-bias",
            ),
            pytest.param(
                1, 256, 56, 56, 512, 5, 1, 2, True, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-256in-512out-5x5-s1-56-bias",
            ),
            pytest.param(
                1, 512, 56, 56, 64, (1, 1), (1, 1), (0, 0), False, torch.bfloat16, True,
                marks=pytest.mark.full,
                id="full-bf16-512in-64out-1x1-s1-56-no-bias-tuned",
            ),
            pytest.param(
                1, 3, 224, 224, 64, (5, 5), (1, 1), (2, 2), True, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-3in-64out-5x5-s1-224-bias-tuple-args",
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
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        padding: int | Tuple[int, int],
        bias: bool,
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
        self.bias = bias
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        kernel_h, kernel_w = self.kernel_size if isinstance(self.kernel_size, tuple) else (
            self.kernel_size, self.kernel_size)
        weight_scale = 1.0 / (self.c_in * kernel_h * kernel_w)
        x = torch.randn(self.n, self.c_in, self.h, self.w, device="cuda", dtype=self.dtype)
        weight = weight_scale * torch.randn(
            self.c_out, self.c_in, kernel_h, kernel_w, device="cuda", dtype=self.dtype
        )
        bias = torch.randn(self.c_out, device="cuda", dtype=self.dtype) if self.bias else None
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


@Conv2dFixture
def test_conv2d(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_size: int | Tuple[int, int],
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
    bias: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = Conv2dTest(n, c_in, h, w, c_out, kernel_size, stride, padding, bias, dtype)
    op = Conv2dOp(
        n,
        c_in,
        h,
        w,
        c_out,
        kernel_size,
        stride=stride,
        padding=padding,
        dtype=dtype,
        tune=tune,
    )
    tolerances = {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float16 else {
        "atol": 1.6e-2,
        "rtol": 1.6e-2,
    }
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
