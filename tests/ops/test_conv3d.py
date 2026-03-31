from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.conv import Conv3dKernel
from tileops.ops import Conv3dOp


class Conv3dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dtype, tune", [
            pytest.param(
                1, 16, 8, 32, 32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.float16, False,
                marks=pytest.mark.full,
                id="smoke-3d-unet-k3-s1-fp16",
            ),
            pytest.param(
                1, 3, 16, 112, 112, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.float16, False,
                marks=pytest.mark.full,
                id="full-r3d-stem-k3-s1-fp16",
            ),
            pytest.param(
                1, 64, 8, 56, 56, 128, (3, 3, 3), (2, 2, 2), (1, 1, 1), torch.float16, False,
                marks=pytest.mark.full,
                id="full-video-stage-downsample-k3-s2-fp16",
            ),
            pytest.param(
                1, 32, 32, 64, 64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-unet-encoder-k3-s1-bf16",
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
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
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
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
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


@pytest.mark.full
def test_conv3d_accepts_zero_bias() -> None:
    op = Conv3dOp(
        n=1,
        c_in=8,
        d_in=8,
        h_in=16,
        w_in=16,
        c_out=16,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
    )
    x = torch.randn(1, 8, 16, 16, 8, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.zeros(16, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv3d(
        x.permute(0, 4, 1, 2, 3).contiguous(),
        weight,
        bias=bias,
        stride=2,
        padding=1,
    )
    ref = ref.permute(0, 2, 3, 4, 1).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.full
def test_conv3d_dispatches_kernel() -> None:
    op = Conv3dOp(
        n=1,
        c_in=8,
        d_in=8,
        h_in=16,
        w_in=16,
        c_out=16,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    assert isinstance(op.kernel, Conv3dKernel)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
