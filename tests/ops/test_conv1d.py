from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.conv1d import Conv1dKernel
from tileops.ops import Conv1dOp


class Conv1dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, l_in, c_out, kernel_size, stride, padding, dtype, tune", [
            pytest.param(
                2, 64, 512, 128, 3, 1, 1, torch.float16, False,
                marks=pytest.mark.smoke,
                id="smoke-tcn-k3-s1-fp16",
            ),
            pytest.param(
                4, 256, 32000, 512, 1, 1, 0, torch.float16, False,
                marks=pytest.mark.full,
                id="full-convtasnet-pointwise-k1-s1-fp16",
            ),
            pytest.param(
                4, 128, 4096, 256, 3, 1, 1, torch.float16, False,
                marks=pytest.mark.full,
                id="full-seanet-residual-k3-s1-fp16",
            ),
            pytest.param(
                4, 64, 16000, 128, 5, 2, 2, torch.float16, False,
                marks=pytest.mark.full,
                id="full-audio-downsample-k5-s2-fp16",
            ),
            pytest.param(
                1, 32, 256, 64, 7, 1, 3, torch.float16, False,
                marks=pytest.mark.full,
                id="full-small-seanet-stem-k7-s1-fp16",
            ),
            pytest.param(
                2, 128, 4096, 256, 3, 2, 1, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-sequence-downsample-k3-s2-bf16",
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
    op = Conv1dOp(
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


@pytest.mark.smoke
def test_conv1d_accepts_zero_bias() -> None:
    op = Conv1dOp(
        n=1,
        c_in=32,
        l_in=256,
        c_out=64,
        kernel_size=5,
        stride=2,
        padding=2,
        bias=True,
    )
    x = torch.randn(1, 256, 32, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.zeros(64, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv1d(x.permute(0, 2, 1).contiguous(), weight, bias=bias, stride=2, padding=2)
    ref = ref.permute(0, 2, 1).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv1d_dispatches_kernel() -> None:
    op = Conv1dOp(
        n=1,
        c_in=32,
        l_in=256,
        c_out=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    assert isinstance(op.kernel, Conv1dKernel)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
