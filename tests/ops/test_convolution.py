from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.convolution import (
    Conv1dKernel,
    Conv1dPointwiseKernel,
    Conv2d1x1Kernel,
    Conv2dKernel,
    Conv3dKernel,
    GroupConv1dKernel,
)
from tileops.ops import (
    Conv1dBiasFwdOp,
    Conv1dFwdOp,
    Conv2dBiasFwdOp,
    Conv2dFwdOp,
    Conv3dBiasFwdOp,
    Conv3dFwdOp,
)


class Conv1dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, l_in, c_out, kernel_size, stride, padding, dilation, groups, dtype, tune", [
            pytest.param(
                2, 64, 512, 128, 3, 1, 1, 1, 1, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-tcn-k3-s1-fp16",
            ),
            pytest.param(
                2, 64, 512, 128, 3, 1, 1, 1, 1, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-tcn-k3-s1-bf16",
            ),
            pytest.param(
                4, 256, 32000, 512, 1, 1, 0, 1, 1, torch.float16, False,
                marks=pytest.mark.full,
                id="full-convtasnet-pointwise-k1-s1-fp16",
            ),
            pytest.param(
                4, 128, 4096, 256, 3, 1, 1, 1, 1, torch.float16, False,
                marks=pytest.mark.full,
                id="full-seanet-residual-k3-s1-fp16",
            ),
            pytest.param(
                4, 64, 16000, 128, 5, 2, 2, 1, 1, torch.float16, False,
                marks=pytest.mark.full,
                id="full-audio-downsample-k5-s2-fp16",
            ),
            pytest.param(
                1, 32, 256, 64, 7, 1, 3, 1, 1, torch.float16, False,
                marks=pytest.mark.full,
                id="full-small-seanet-stem-k7-s1-fp16",
            ),
            pytest.param(
                2, 128, 4096, 256, 3, 2, 1, 1, 1, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-sequence-downsample-k3-s2-bf16",
            ),
            pytest.param(
                1, 32, 512, 64, 3, 1, 2, 2, 1, torch.float16, False,
                marks=pytest.mark.full,
                id="full-dilation-k3-d2-fp16",
            ),
            pytest.param(
                1, 32, 128, 64, 3, 1, "valid", 1, 1, torch.float16, False,
                marks=pytest.mark.full,
                id="full-padding-valid-fp16",
            ),
            pytest.param(
                1, 32, 128, 64, 3, 1, "same", 1, 1, torch.float16, False,
                marks=pytest.mark.full,
                id="full-padding-same-fp16",
            ),
            pytest.param(
                1, 32, 128, 64, 3, 1, 1, 1, 2, torch.float16, False,
                marks=pytest.mark.full,
                id="full-groups2-k3-fp16",
            ),
            pytest.param(
                1, 48, 128, 72, 3, 1, 1, 1, 3, torch.float16, False,
                marks=pytest.mark.full,
                id="full-groups3-coutg24-fp16",
            ),
            pytest.param(
                1, 64, 128, 64, 31, 1, 15, 1, 64, torch.float16, False,
                marks=pytest.mark.full,
                id="full-conformer-depthwise-k31-fp16",
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
        padding: int | str,
        dilation: int,
        groups: int,
        dtype: torch.dtype,
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

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
        weight = torch.randn(
            self.c_out, self.c_in // self.groups, self.kernel_size,
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
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out.contiguous()


@Conv1dFixture
def test_conv1d(
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
    tune: bool,
) -> None:
    test = Conv1dTest(n, c_in, l_in, c_out, kernel_size, stride, padding, dilation, groups, dtype)
    op = Conv1dBiasFwdOp(
        n=n,
        c_in=c_in,
        l_in=l_in,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=dtype,
        tune=tune,
    )
    if groups > 1:
        assert isinstance(op.kernel, GroupConv1dKernel)
        assert op.kernel.use_direct is (c_in // groups == 1 and c_out // groups == 1)
    atol, rtol = (1e-3, 1e-3)
    if dtype == torch.bfloat16:
        atol, rtol = (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_conv1d_no_bias_matches_torch() -> None:
    op = Conv1dFwdOp(
        n=1,
        c_in=32,
        l_in=256,
        c_out=64,
        kernel_size=5,
        stride=2,
        padding=2,
    )
    x = torch.randn(1, 32, 256, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv1d(x, weight, bias=None, stride=2, padding=2).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv1d_bias_requires_bias_tensor() -> None:
    op = Conv1dBiasFwdOp(
        n=1,
        c_in=32,
        l_in=256,
        c_out=64,
        kernel_size=5,
        stride=2,
        padding=2,
    )
    x = torch.randn(1, 32, 256, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.zeros(64, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv1d(x, weight, bias=bias, stride=2, padding=2).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "op_cls, dilation, use_bias",
    [
        pytest.param(Conv1dFwdOp, 2, False, marks=pytest.mark.smoke, id="no-bias"),
        pytest.param(Conv1dBiasFwdOp, 2, True, marks=pytest.mark.full, id="bias"),
    ],
)
def test_conv1d_dilation_matches_torch(op_cls, dilation, use_bias: bool) -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 32, 128, 64, 3
    stride, padding = 1, 2
    op_kwargs = {"bias": use_bias} if op_cls is Conv1dBiasFwdOp else {}
    op = op_cls(
        n=n,
        c_in=c_in,
        l_in=l_in,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        **op_kwargs,
    )
    x = torch.randn(n, c_in, l_in, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device="cuda", dtype=torch.float16).contiguous()
    bias = (
        torch.randn(c_out, device="cuda", dtype=torch.float16).contiguous()
        if use_bias else None
    )
    out = op(x, weight, bias) if use_bias else op(x, weight)
    ref = F.conv1d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=2,
    )
    ref = ref.contiguous()
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=3e-3)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, use_bias",
    [
        pytest.param(Conv1dFwdOp, False, id="no-bias"),
        pytest.param(Conv1dBiasFwdOp, True, id="bias"),
    ],
)
def test_conv1d_same_padding_even_kernel_matches_torch(op_cls, use_bias: bool) -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 16, 129, 32, 2
    op_kwargs = {"bias": use_bias} if op_cls is Conv1dBiasFwdOp else {}
    op = op_cls(
        n=n,
        c_in=c_in,
        l_in=l_in,
        c_out=c_out,
        kernel_size=kernel_size,
        padding="same",
        **op_kwargs,
    )
    x = torch.randn(n, c_in, l_in, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device="cuda", dtype=torch.float16).contiguous()
    bias = (
        torch.randn(c_out, device="cuda", dtype=torch.float16).contiguous()
        if use_bias else None
    )
    out = op(x, weight, bias) if use_bias else op(x, weight)
    ref = F.conv1d(x, weight, bias=bias, padding="same").contiguous()
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=3e-3)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, expected_kernel",
    [
        pytest.param(3, 1, 1, 1, Conv1dKernel, id="generic"),
        pytest.param(1, 1, 0, 1, Conv1dPointwiseKernel, id="pointwise"),
    ],
)
def test_conv1d_dispatches_kernel(
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    expected_kernel: type,
) -> None:
    op = Conv1dFwdOp(
        n=1,
        c_in=32,
        l_in=256,
        c_out=64,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    assert isinstance(op.kernel, expected_kernel)


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
                1, 64, 56, 56, 128, (3, 3), (2, 2), (1, 1), torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-bf16-3x3-s2",
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
        x = torch.randn(self.n, self.c_in, self.h, self.w, device="cuda", dtype=self.dtype).contiguous()
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
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1,
        )
        return out.contiguous()


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
    test = Conv2dTest(n, c_in, h, w, c_out, kernel_size, stride, padding, dtype)
    op = Conv2dBiasFwdOp(
        n=n,
        c_in=c_in,
        h=h,
        w=w,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_conv2d_no_bias_matches_torch() -> None:
    op = Conv2dFwdOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=5,
        stride=2,
        padding=2,
    )
    x = torch.randn(1, 32, 16, 16, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, 5, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv2d(
        x,
        weight,
        bias=None,
        stride=2,
        padding=2,
    )
    ref = ref.contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv2d_dispatches_1x1_kernel() -> None:
    op = Conv2dFwdOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=1,
    )
    assert isinstance(op.kernel, Conv2d1x1Kernel)


@pytest.mark.smoke
def test_conv2d_does_not_dispatch_1x1_kernel_with_padding() -> None:
    op = Conv2dFwdOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=1,
        padding=1,
    )
    assert isinstance(op.kernel, Conv2dKernel)


@pytest.mark.smoke
def test_conv2d_dispatches_3x3_kernel() -> None:
    op = Conv2dFwdOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=3,
        padding=1,
    )
    assert isinstance(op.kernel, Conv2dKernel)


@pytest.mark.smoke
def test_conv2d_dispatches_5x5_kernel() -> None:
    op = Conv2dFwdOp(
        n=1,
        c_in=32,
        h=16,
        w=16,
        c_out=64,
        kernel_size=5,
        padding=2,
    )
    assert isinstance(op.kernel, Conv2dKernel)


class Conv3dFixture(FixtureBase):
    PARAMS = [
        ("n, c_in, d, h, w, c_out, kernel_size, stride, padding, dtype, tune", [
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
        d: int,
        h: int,
        w: int,
        c_out: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dtype: torch.dtype,
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
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(
            self.n, self.c_in, self.d, self.h, self.w,
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
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1,
        )
        return out.contiguous()


@Conv3dFixture
def test_conv3d(
    n: int,
    c_in: int,
    d: int,
    h: int,
    w: int,
    c_out: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = Conv3dTest(n, c_in, d, h, w, c_out, kernel_size, stride, padding, dtype)
    op = Conv3dBiasFwdOp(
        n=n,
        c_in=c_in,
        d=d,
        h=h,
        w=w,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dtype=dtype,
        tune=tune,
    )
    atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2))
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_conv3d_no_bias_matches_torch() -> None:
    op = Conv3dFwdOp(
        n=1,
        c_in=8,
        d=4,
        h=16,
        w=16,
        c_out=16,
        kernel_size=3,
        stride=2,
        padding=1,
    )
    x = torch.randn(1, 8, 4, 16, 16, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv3d(
        x,
        weight,
        bias=None,
        stride=2,
        padding=1,
    )
    ref = ref.contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv3d_accepts_zero_bias() -> None:
    op = Conv3dBiasFwdOp(
        n=1,
        c_in=8,
        d=8,
        h=16,
        w=16,
        c_out=16,
        kernel_size=3,
        stride=2,
        padding=1,
    )
    x = torch.randn(1, 8, 8, 16, 16, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.zeros(16, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv3d(
        x,
        weight,
        bias=bias,
        stride=2,
        padding=1,
    )
    ref = ref.contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv3d_dispatches_kernel() -> None:
    op = Conv3dFwdOp(
        n=1,
        c_in=8,
        d=8,
        h=16,
        w=16,
        c_out=16,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    assert isinstance(op.kernel, Conv3dKernel)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
