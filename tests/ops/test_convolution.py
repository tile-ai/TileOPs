import pytest
import torch
import torch.nn.functional as F

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.test_base import FixtureBase, TestBase
from tileops.kernels.convolution import (
    Conv1dKernel,
    Conv1dPointwiseKernel,
    Conv2d1x1Kernel,
    Conv2dSymmetricKernel,
    Conv3dKernel,
    GroupConv1dKernel,
    GroupConv2dKernel,
    GroupConv3dKernel,
)
from tileops.ops import (
    Conv1dFwdOp,
    Conv2dFwdOp,
    Conv3dFwdOp,
)
from workloads.convolution import Conv1dWorkload, Conv2dWorkload, Conv3dWorkload

for _op_cls in (Conv1dFwdOp, Conv2dFwdOp, Conv3dFwdOp):
    register_compile_contract(_op_cls)


class Conv1dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, l_in, c_out, kernel_size, stride, padding, dilation, groups, dtype, tune",
            [
                pytest.param(
                    2,
                    64,
                    512,
                    128,
                    3,
                    1,
                    1,
                    1,
                    1,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-tcn-k3-s1-fp16",
                ),
                pytest.param(
                    2,
                    64,
                    512,
                    128,
                    3,
                    1,
                    1,
                    1,
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-tcn-k3-s1-bf16",
                ),
                pytest.param(
                    4,
                    256,
                    32000,
                    512,
                    1,
                    1,
                    0,
                    1,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-convtasnet-pointwise-k1-s1-fp16",
                ),
                pytest.param(
                    4,
                    128,
                    4096,
                    256,
                    3,
                    1,
                    1,
                    1,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-seanet-residual-k3-s1-fp16",
                ),
                pytest.param(
                    4,
                    64,
                    16000,
                    128,
                    5,
                    2,
                    2,
                    1,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-audio-downsample-k5-s2-fp16",
                ),
                pytest.param(
                    1,
                    32,
                    256,
                    64,
                    7,
                    1,
                    3,
                    1,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-small-seanet-stem-k7-s1-fp16",
                ),
                pytest.param(
                    2,
                    128,
                    4096,
                    256,
                    3,
                    2,
                    1,
                    1,
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-sequence-downsample-k3-s2-bf16",
                ),
                pytest.param(
                    1,
                    32,
                    512,
                    64,
                    3,
                    1,
                    2,
                    2,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-dilation-k3-d2-fp16",
                ),
                pytest.param(
                    1,
                    32,
                    128,
                    64,
                    3,
                    1,
                    "valid",
                    1,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-padding-valid-fp16",
                ),
                pytest.param(
                    1,
                    32,
                    128,
                    64,
                    3,
                    1,
                    "same",
                    1,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-padding-same-fp16",
                ),
                pytest.param(
                    1,
                    32,
                    128,
                    64,
                    3,
                    1,
                    1,
                    1,
                    2,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-groups2-k3-fp16",
                ),
                pytest.param(
                    1,
                    48,
                    128,
                    72,
                    3,
                    1,
                    1,
                    1,
                    3,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-groups3-coutg24-fp16",
                ),
                pytest.param(
                    1,
                    64,
                    128,
                    64,
                    31,
                    1,
                    15,
                    1,
                    64,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-conformer-depthwise-k31-fp16",
                ),
            ],
        ),
    ]


class Conv1dTest(Conv1dWorkload, TestBase):
    pass


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
    op = Conv1dFwdOp(
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        tune=tune,
    )
    atol, rtol = (1e-3, 1e-3)
    if dtype == torch.bfloat16:
        atol, rtol = (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)
    if groups > 1:
        assert isinstance(op.kernel, GroupConv1dKernel)
        assert op.kernel.use_direct is (c_in // groups == 1 and c_out // groups == 1)


@pytest.mark.smoke
def test_conv1d_no_bias_matches_torch() -> None:
    op = Conv1dFwdOp(stride=2, padding=2)
    x = torch.randn(1, 32, 256, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv1d(x, weight, bias=None, stride=2, padding=2).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv1d_bias_matches_torch() -> None:
    op = Conv1dFwdOp(stride=2, padding=2)
    x = torch.randn(1, 32, 256, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.zeros(64, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv1d(x, weight, bias=bias, stride=2, padding=2).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "dilation, use_bias",
    [
        pytest.param(2, False, marks=pytest.mark.smoke, id="no-bias"),
        pytest.param(2, True, marks=pytest.mark.full, id="bias"),
    ],
)
def test_conv1d_dilation_matches_torch(dilation, use_bias: bool) -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 32, 128, 64, 3
    stride, padding = 1, 2
    op = Conv1dFwdOp(
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    x = torch.randn(n, c_in, l_in, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.randn(c_out, device="cuda", dtype=torch.float16).contiguous() if use_bias else None
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
    "use_bias",
    [pytest.param(False, id="no-bias"), pytest.param(True, id="bias")],
)
def test_conv1d_same_padding_even_kernel_matches_torch(use_bias: bool) -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 16, 129, 32, 2
    op = Conv1dFwdOp(padding="same")
    x = torch.randn(n, c_in, l_in, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(c_out, c_in, kernel_size, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.randn(c_out, device="cuda", dtype=torch.float16).contiguous() if use_bias else None
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
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    x = torch.randn(1, 32, 256, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, kernel_size, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    assert isinstance(op.kernel, expected_kernel)
    ref = F.conv1d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation)
    torch.testing.assert_close(out, ref.contiguous(), atol=1e-3, rtol=1e-3)


class Conv2dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, h, w, c_out, kernel_size, stride, padding, dilation, groups, dtype, tune",
            [
                pytest.param(
                    2,
                    32,
                    32,
                    32,
                    64,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-fp16-3x3",
                ),
                pytest.param(
                    2,
                    32,
                    32,
                    32,
                    64,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bf16-3x3",
                ),
                # MobileNetV2 depthwise 3x3 block, reduced spatial size for smoke cost.
                pytest.param(
                    1,
                    16,
                    16,
                    16,
                    16,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    16,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-mobilenetv2-depthwise-small-fp16",
                ),
                pytest.param(
                    1,
                    3,
                    112,
                    112,
                    64,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-stem-3x3-s2-fp16",
                ),
                pytest.param(
                    1,
                    64,
                    56,
                    56,
                    64,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-resblock-3x3-s1-fp16",
                ),
                pytest.param(
                    1,
                    128,
                    56,
                    56,
                    256,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-stage-transition-3x3-s2-fp16",
                ),
                pytest.param(
                    1,
                    32,
                    28,
                    28,
                    64,
                    (5, 5),
                    (1, 1),
                    (2, 2),
                    (1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-small-5x5-s1-fp16",
                ),
                pytest.param(
                    1,
                    64,
                    28,
                    28,
                    128,
                    (5, 5),
                    (2, 2),
                    (2, 2),
                    (1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-small-5x5-s2-fp16",
                ),
                pytest.param(
                    2,
                    32,
                    32,
                    32,
                    64,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    torch.float16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fp16-1x1-tuned",
                ),
                pytest.param(
                    1,
                    64,
                    28,
                    28,
                    128,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fp16-stride2",
                ),
                pytest.param(
                    1,
                    64,
                    56,
                    56,
                    128,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-3x3-s2",
                ),
                pytest.param(
                    1,
                    64,
                    28,
                    28,
                    64,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-bf16-1x1",
                ),
                pytest.param(
                    1,
                    64,
                    32,
                    32,
                    128,
                    (3, 3),
                    (1, 1),
                    (2, 2),
                    (2, 2),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-deeplab-aspp-3x3-d2-fp16",
                ),
                # ResNeXt bottleneck grouped 3x3 convolution.
                pytest.param(
                    1,
                    128,
                    28,
                    28,
                    256,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    32,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-resnext-grouped-3x3-fp16",
                ),
            ],
        ),
    ]


class Conv2dTest(Conv2dWorkload, TestBase):
    pass


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
    dilation: tuple[int, int],
    groups: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = Conv2dTest(n, c_in, h, w, c_out, kernel_size, stride, padding, dilation, groups, dtype)
    op = Conv2dFwdOp(
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        tune=tune,
    )
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)
    if groups > 1:
        assert isinstance(op.kernel, GroupConv2dKernel)


@pytest.mark.smoke
def test_conv2d_no_bias_matches_torch() -> None:
    op = Conv2dFwdOp(
        stride=2,
        padding=4,
        dilation=2,
    )
    x = torch.randn(1, 32, 16, 16, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, 5, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv2d(
        x,
        weight,
        bias=None,
        stride=2,
        padding=4,
        dilation=2,
    )
    ref = ref.contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv2d_no_bias_grouped_matches_torch() -> None:
    groups = 8
    op = Conv2dFwdOp(
        padding=1,
        groups=groups,
    )
    x = torch.randn(1, 16, 16, 16, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(32, 2, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv2d(x, weight, bias=None, padding=1, groups=groups).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
@pytest.mark.parametrize("use_bias", [False, True], ids=["no-bias", "bias"])
def test_conv2d_depthwise_dispatches_the_direct_kernel(use_bias: bool) -> None:
    """One channel per group is a GEMM with M=1, so it gets a direct kernel instead."""
    channels = 32
    op = Conv2dFwdOp(padding=1, groups=channels)
    x = torch.randn(1, channels, 28, 28, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(channels, 1, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    bias = (
        torch.randn(channels, device="cuda", dtype=torch.float16).contiguous() if use_bias else None
    )

    out = op(x, weight, bias)

    assert isinstance(op.kernel, GroupConv2dKernel) and op.kernel.use_direct
    ref = F.conv2d(x, weight, bias=bias, padding=1, groups=channels).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv2d_dispatches_1x1_kernel() -> None:
    op = Conv2dFwdOp()
    x = torch.randn(1, 32, 32, 32, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 1, 1, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    assert isinstance(op.kernel, Conv2d1x1Kernel)
    ref = F.conv2d(x, weight, bias=None, padding=0)
    torch.testing.assert_close(out, ref.contiguous(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv2d_does_not_dispatch_1x1_kernel_with_padding() -> None:
    # Use c_in not divisible by 32 so the symmetric kernel is not selected and
    # the general kernel handles the padded 1x1 case without the im2col-TMA
    # constraints that affect the symmetric path.
    op = Conv2dFwdOp(padding=1)
    x = torch.randn(1, 16, 32, 32, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 16, 1, 1, device="cuda", dtype=torch.float16).contiguous()
    op(x, weight)
    assert not isinstance(op.kernel, Conv2d1x1Kernel)


@pytest.mark.smoke
def test_conv2d_dispatches_3x3_kernel() -> None:
    op = Conv2dFwdOp(padding=1)
    x = torch.randn(1, 32, 32, 32, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    assert isinstance(op.kernel, Conv2dSymmetricKernel)
    ref = F.conv2d(x, weight, bias=None, padding=1)
    torch.testing.assert_close(out, ref.contiguous(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv2d_dispatches_5x5_kernel() -> None:
    op = Conv2dFwdOp(padding=2)
    x = torch.randn(1, 32, 32, 32, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, 5, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    assert isinstance(op.kernel, Conv2dSymmetricKernel)
    ref = F.conv2d(x, weight, bias=None, padding=2)
    torch.testing.assert_close(out, ref.contiguous(), atol=1e-3, rtol=1e-3)


class Conv3dFixture(FixtureBase):
    PARAMS = [
        (
            "n, c_in, d, h, w, c_out, kernel_size, stride, padding, dilation, groups, dtype, tune",
            [
                pytest.param(
                    1,
                    16,
                    8,
                    32,
                    32,
                    32,
                    (3, 3, 3),
                    (1, 1, 1),
                    (1, 1, 1),
                    (1, 1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-3d-unet-k3-s1-fp16",
                ),
                pytest.param(
                    1,
                    16,
                    8,
                    32,
                    32,
                    32,
                    (3, 3, 3),
                    (1, 1, 1),
                    (1, 1, 1),
                    (1, 1, 1),
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-3d-unet-k3-s1-bf16",
                ),
                # Video depthwise 3D block, reduced size for smoke cost.
                pytest.param(
                    1,
                    8,
                    4,
                    12,
                    12,
                    8,
                    (3, 3, 3),
                    (1, 1, 1),
                    (1, 1, 1),
                    (1, 1, 1),
                    8,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-video-depthwise3d-small-fp16",
                ),
                pytest.param(
                    1,
                    3,
                    16,
                    112,
                    112,
                    64,
                    (3, 3, 3),
                    (1, 1, 1),
                    (1, 1, 1),
                    (1, 1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-r3d-stem-k3-s1-fp16",
                ),
                pytest.param(
                    1,
                    64,
                    8,
                    56,
                    56,
                    128,
                    (3, 3, 3),
                    (2, 2, 2),
                    (1, 1, 1),
                    (1, 1, 1),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-video-stage-downsample-k3-s2-fp16",
                ),
                pytest.param(
                    1,
                    32,
                    32,
                    64,
                    64,
                    64,
                    (3, 3, 3),
                    (1, 1, 1),
                    (1, 1, 1),
                    (1, 1, 1),
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                    id="full-unet-encoder-k3-s1-bf16",
                ),
                pytest.param(
                    1,
                    16,
                    8,
                    32,
                    32,
                    32,
                    (3, 3, 3),
                    (1, 1, 1),
                    (2, 2, 2),
                    (2, 2, 2),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-3d-aspp-3x3x3-d2-fp16",
                ),
                # 3D-ResNeXt/video backbone grouped 3x3x3 convolution.
                pytest.param(
                    1,
                    64,
                    8,
                    28,
                    28,
                    128,
                    (3, 3, 3),
                    (1, 1, 1),
                    (1, 1, 1),
                    (1, 1, 1),
                    32,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-3d-resnext-grouped-k3-fp16",
                ),
            ],
        ),
    ]


class Conv3dTest(Conv3dWorkload, TestBase):
    pass


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
    dilation: tuple[int, int, int],
    groups: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = Conv3dTest(
        n, c_in, d, h, w, c_out, kernel_size, stride, padding, dilation, groups, dtype
    )
    op = Conv3dFwdOp(
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        tune=tune,
    )
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)
    if groups > 1:
        assert isinstance(op.kernel, GroupConv3dKernel)


@pytest.mark.smoke
def test_conv3d_no_bias_matches_torch() -> None:
    op = Conv3dFwdOp(
        stride=2,
        padding=2,
        dilation=2,
    )
    x = torch.randn(1, 8, 8, 16, 16, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv3d(
        x,
        weight,
        bias=None,
        stride=2,
        padding=2,
        dilation=2,
    )
    ref = ref.contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv3d_no_bias_grouped_matches_torch() -> None:
    groups = 4
    op = Conv3dFwdOp(
        padding=1,
        groups=groups,
    )
    x = torch.randn(1, 8, 4, 12, 12, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(16, 2, 3, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv3d(x, weight, bias=None, padding=1, groups=groups).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_conv3d_accepts_zero_bias() -> None:
    op = Conv3dFwdOp(
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
    op = Conv3dFwdOp(stride=1, padding=1)
    x = torch.randn(1, 8, 8, 32, 32, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    op(x, weight)
    assert isinstance(op.kernel, Conv3dKernel)


@pytest.mark.smoke
def test_conv2d_dynamic_shape_kernel_cache_and_roofline() -> None:
    op = Conv2dFwdOp(stride=1, padding=1)
    x1 = torch.randn(1, 16, 32, 32, dtype=torch.float16, device="cuda")
    w1 = torch.randn(24, 16, 3, 3, dtype=torch.float16, device="cuda")
    x2 = torch.randn(2, 16, 32, 32, dtype=torch.float16, device="cuda")
    w2 = torch.randn(24, 16, 3, 3, dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="requires a prior forward"):
        op.eval_roofline()

    op(x1, w1)
    assert len(list(op.iter_kernels())) == 1
    flops, nbytes = op.eval_roofline()
    assert flops > 0
    assert nbytes > 0

    op(x1, w1)
    assert len(list(op.iter_kernels())) == 1

    op(x2, w2)
    assert len(list(op.iter_kernels())) == 2


@pytest.mark.smoke
def test_conv1d_depthwise_no_bias_matches_torch() -> None:
    """The depthwise-direct path is the one Conv1d variant no other no-bias case reaches."""
    groups = 32
    op = Conv1dFwdOp(padding=1, groups=groups)
    x = torch.randn(1, groups, 128, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(groups, 1, 3, device="cuda", dtype=torch.float16).contiguous()

    out = op(x, weight)

    assert isinstance(op.kernel, GroupConv1dKernel) and op.kernel.use_direct
    ref = F.conv1d(x, weight, bias=None, padding=1, groups=groups).contiguous()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
def test_a_kernel_built_without_a_bias_refuses_one() -> None:
    """Bias presence is compiled in, so the two sides are different programs.

    Built directly: through the op the two always agree, so this guard has no op-level
    entry point.
    """
    kernel = Conv2d1x1Kernel(
        n=1,
        c_in=32,
        h=8,
        w=8,
        c_out=32,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        dtype=torch.float16,
    )
    x = torch.randn(1, 32, 8, 8, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(32, 32, 1, 1, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.randn(32, device="cuda", dtype=torch.float16).contiguous()

    with pytest.raises(ValueError, match="built without a bias"):
        kernel(x, weight, bias)


@pytest.mark.smoke
def test_inputs_on_different_devices_are_rejected() -> None:
    """The kernel memo lets the first input's device speak for the rest, so they agree."""
    op = Conv2dFwdOp(padding=1)
    x = torch.randn(1, 8, 8, 8, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(4, 8, 3, 3, dtype=torch.float16).contiguous()

    with pytest.raises(ValueError, match="every input on cuda"):
        op(x, weight)


# --------------------------------------------------------------------------------------
# The compile boundary: the node in the graph is the op's, whichever target serves it
# --------------------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("use_bias", [False, True], ids=["no-bias", "bias"])
def test_conv2d_cold_traces_fullgraph_and_owns_its_graph_nodes(use_bias: bool) -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op = Conv2dFwdOp(padding=1)
    x = torch.randn(1, 32, 16, 16, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.randn(64, device="cuda", dtype=torch.float16).contiguous() if use_bias else None

    assert_op_owns_graph_nodes(op, x, weight, bias)
    torch.testing.assert_close(
        torch.compile(op, fullgraph=True)(x, weight, bias), op(x, weight, bias)
    )


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_conv1d_cold_traces_fullgraph_and_owns_its_graph_nodes() -> None:
    op = Conv1dFwdOp(padding=1)
    x = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, device="cuda", dtype=torch.float16).contiguous()

    assert_op_owns_graph_nodes(op, x, weight, None)
    torch.testing.assert_close(torch.compile(op, fullgraph=True)(x, weight), op(x, weight))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_conv3d_cold_traces_fullgraph_and_owns_its_graph_nodes() -> None:
    op = Conv3dFwdOp(padding=1)
    x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float16).contiguous()
    weight = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float16).contiguous()

    assert_op_owns_graph_nodes(op, x, weight, None)
    torch.testing.assert_close(torch.compile(op, fullgraph=True)(x, weight), op(x, weight))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_a_non_contiguous_input_compiles_to_the_shape_the_fake_promised() -> None:
    """The fake speaks before the body normalizes contiguity, so it promises contiguous."""
    op = Conv2dFwdOp(padding=1)
    x = torch.randn(1, 32, 16, 32, device="cuda", dtype=torch.float16)[:, :, :, ::2]
    weight = torch.randn(64, 32, 3, 3, device="cuda", dtype=torch.float16).contiguous()
    assert not x.is_contiguous()

    output = torch.compile(op, fullgraph=True)(x, weight)

    assert output.is_contiguous()
    torch.testing.assert_close(output, op(x, weight))
