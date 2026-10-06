import pytest
import torch
import torch.nn.functional as F

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import FixtureBase, TestBase
from tileops.kernels.convolution.call_spec import Conv1dCall, Conv2dCall, Conv3dCall
from tileops.ops import (
    Conv1dFwdOp,
    Conv2dFwdOp,
    Conv3dFwdOp,
)
from workloads.convolution import (
    Conv1dWorkload,
    Conv2dWorkload,
    Conv3dWorkload,
    convolution_verification,
)
from workloads.device import run_device
from workloads.numerics import compare_outputs

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
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="convolution")],
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
                    1,
                    17,
                    5,
                    7,
                    3,
                    1,
                    5,
                    2,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-padding-wider-than-row-fp16",
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
    op = Conv1dFwdOp(stride=stride, padding=padding, dilation=dilation, groups=groups, tune=tune)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_conv1d_no_bias_matches_torch() -> None:
    op = Conv1dFwdOp(stride=2, padding=2)
    x = torch.randn(1, 32, 256, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv1d(x, weight, bias=None, stride=2, padding=2).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv1d_bias_matches_torch() -> None:
    op = Conv1dFwdOp(stride=2, padding=2)
    x = torch.randn(1, 32, 256, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, device=run_device(), dtype=torch.float16).contiguous()
    bias = torch.zeros(64, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv1d(x, weight, bias=bias, stride=2, padding=2).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


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
    x = torch.randn(n, c_in, l_in, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(
        c_out, c_in, kernel_size, device=run_device(), dtype=torch.float16
    ).contiguous()
    bias = (
        torch.randn(c_out, device=run_device(), dtype=torch.float16).contiguous()
        if use_bias
        else None
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
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "use_bias",
    [pytest.param(False, id="no-bias"), pytest.param(True, id="bias")],
)
def test_conv1d_same_padding_even_kernel_matches_torch(use_bias: bool) -> None:
    n, c_in, l_in, c_out, kernel_size = 1, 16, 129, 32, 2
    op = Conv1dFwdOp(padding="same")
    x = torch.randn(n, c_in, l_in, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(
        c_out, c_in, kernel_size, device=run_device(), dtype=torch.float16
    ).contiguous()
    bias = (
        torch.randn(c_out, device=run_device(), dtype=torch.float16).contiguous()
        if use_bias
        else None
    )
    out = op(x, weight, bias) if use_bias else op(x, weight)
    ref = F.conv1d(x, weight, bias=bias, padding="same").contiguous()
    compare_outputs(
        out, ref, convolution_verification(out.dtype, padding="same", kernel_shape=weight.shape[2:])
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    [
        pytest.param(3, 1, 1, 1, id="unit-stride"),
        pytest.param(3, 2, 1, 1, id="generic"),
        pytest.param(1, 1, 0, 1, id="pointwise"),
    ],
)
def test_conv1d_dispatches_kernel(
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> None:
    op = Conv1dFwdOp(
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    x = torch.randn(1, 32, 256, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, kernel_size, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv1d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation)
    compare_outputs(out, ref.contiguous(), convolution_verification(out.dtype))


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_conv1d_unit_stride_under_tuning(tune: bool) -> None:
    """c_in = 130 puts each tap's weight columns off a 16-byte boundary and needs two channel
    blocks; the padding gives boundary CTAs, whichever taps per k tile tuning picks."""
    test = Conv1dTest(1, 130, 260, 67, 4, 1, 3, 2, 1, torch.float16)
    test.check(Conv1dFwdOp(padding=3, dilation=2, tune=tune), *test.gen_inputs())


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
                    torch.float32,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-fp32-3x3",
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
    op = Conv2dFwdOp(stride=stride, padding=padding, dilation=dilation, groups=groups, tune=tune)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_conv2d_no_bias_matches_torch() -> None:
    op = Conv2dFwdOp(
        stride=2,
        padding=4,
        dilation=2,
    )
    x = torch.randn(1, 32, 16, 16, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, 5, device=run_device(), dtype=torch.float16).contiguous()
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
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv2d_no_bias_grouped_matches_torch() -> None:
    groups = 8
    op = Conv2dFwdOp(
        padding=1,
        groups=groups,
    )
    x = torch.randn(1, 16, 16, 16, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(32, 2, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv2d(x, weight, bias=None, padding=1, groups=groups).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("use_bias", [False, True], ids=["no-bias", "bias"])
def test_conv2d_depthwise_dispatches_the_direct_kernel(use_bias: bool) -> None:
    """One channel per group is a GEMM with M=1, so it gets a direct kernel instead."""
    channels = 32
    op = Conv2dFwdOp(padding=1, groups=channels)
    x = torch.randn(1, channels, 28, 28, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(channels, 1, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
    bias = (
        torch.randn(channels, device=run_device(), dtype=torch.float16).contiguous()
        if use_bias
        else None
    )

    out = op(x, weight, bias)

    ref = F.conv2d(x, weight, bias=bias, padding=1, groups=channels).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv2d_dispatches_1x1_kernel() -> None:
    op = Conv2dFwdOp()
    x = torch.randn(1, 32, 32, 32, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 1, 1, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv2d(x, weight, bias=None, padding=0)
    compare_outputs(out, ref.contiguous(), convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv2d_dispatches_3x3_kernel() -> None:
    op = Conv2dFwdOp(padding=1)
    x = torch.randn(1, 32, 32, 32, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv2d(x, weight, bias=None, padding=1)
    compare_outputs(out, ref.contiguous(), convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv2d_dispatches_5x5_kernel() -> None:
    op = Conv2dFwdOp(padding=2)
    x = torch.randn(1, 32, 32, 32, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 5, 5, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv2d(x, weight, bias=None, padding=2)
    compare_outputs(out, ref.contiguous(), convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv2d_batch_with_partial_tile_leaves_the_symmetric_kernel() -> None:
    # out_h * out_w is 49 here, a multiple of no m tile the symmetric kernel builds,
    # so an m tile would span two images and its implicit GEMM would take the wrong
    # image for the tail. More than one image therefore goes elsewhere.
    op = Conv2dFwdOp()
    x = torch.randn(5, 96, 9, 9, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 96, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv2d(x, weight, bias=None)
    compare_outputs(out, ref.contiguous(), convolution_verification(out.dtype))


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
                    torch.float32,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-3d-unet-k3-s1-fp32",
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
                # Non-symmetric dilation/padding on the NDHWC fast path (c_in % 32 == 0).
                pytest.param(
                    1,
                    32,
                    8,
                    16,
                    16,
                    64,
                    (3, 3, 3),
                    (1, 1, 1),
                    (2, 1, 3),
                    (2, 1, 3),
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-3d-ndhwc-nonsymmetric-dilation-fp16",
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
    op = Conv3dFwdOp(stride=stride, padding=padding, dilation=dilation, groups=groups, tune=tune)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_conv3d_no_bias_matches_torch() -> None:
    op = Conv3dFwdOp(
        stride=2,
        padding=2,
        dilation=2,
    )
    x = torch.randn(1, 8, 8, 16, 16, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
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
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, conv, x_shape, w_shape",
    [
        pytest.param(Conv2dFwdOp, F.conv2d, (1, 16, 17, 20), (32, 16, 2, 3), id="conv2d"),
        # Wide and large enough for the NDHWC path.
        pytest.param(Conv3dFwdOp, F.conv3d, (1, 32, 8, 16, 16), (64, 32, 2, 3, 4), id="conv3d"),
    ],
)
def test_same_padding_with_an_odd_total_matches_torch(op_cls, conv, x_shape, w_shape) -> None:
    """torch puts the extra element of an odd ``'same'`` total after the axis."""
    x = torch.randn(*x_shape, device=run_device(), dtype=torch.float16)
    weight = torch.randn(*w_shape, device=run_device(), dtype=torch.float16)
    bias = torch.randn(w_shape[0], device=run_device(), dtype=torch.float16)
    out = op_cls(padding="same")(x, weight, bias)
    ref = conv(x, weight, bias=bias, padding="same")
    compare_outputs(
        out, ref, convolution_verification(out.dtype, padding="same", kernel_shape=weight.shape[2:])
    )


@pytest.mark.smoke
def test_conv3d_no_bias_grouped_matches_torch() -> None:
    groups = 4
    op = Conv3dFwdOp(
        padding=1,
        groups=groups,
    )
    x = torch.randn(1, 8, 4, 12, 12, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 2, 3, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight)
    ref = F.conv3d(x, weight, bias=None, padding=1, groups=groups).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv3d_accepts_zero_bias() -> None:
    op = Conv3dFwdOp(
        stride=2,
        padding=1,
    )
    x = torch.randn(1, 8, 8, 16, 16, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(16, 8, 3, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
    bias = torch.zeros(16, device=run_device(), dtype=torch.float16).contiguous()
    out = op(x, weight, bias)
    ref = F.conv3d(
        x,
        weight,
        bias=bias,
        stride=2,
        padding=1,
    )
    ref = ref.contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv3d_dispatches_ndhwc_kernel_no_bias() -> None:
    op = Conv3dFwdOp(stride=1, padding=1)
    x = torch.randn(1, 32, 8, 16, 16, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, 3, 3, device=run_device(), dtype=torch.float16).contiguous()

    out = op(x, weight)

    ref = F.conv3d(x, weight, bias=None, stride=1, padding=1).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv3d_ndhwc_tiles_straddle_batches() -> None:
    """Two batches of 1125 output positions: a tile of positions runs from one batch into
    the next, and each position lands in its own batch of the NCDHW result."""
    op = Conv3dFwdOp(stride=1, padding=1)
    x = torch.randn(2, 32, 5, 15, 15, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, 3, 3, device=run_device(), dtype=torch.float16).contiguous()

    out = op(x, weight)

    assert [type(k).__name__ for k in op.iter_kernels()] == ["Conv3dNdhwcKernel"]
    ref = F.conv3d(x, weight, bias=None, stride=1, padding=1).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize(
    ("op_type", "call", "served"),
    [
        pytest.param(
            Conv3dFwdOp,
            Conv3dCall(
                n=1,
                c_in=32,
                c_out=64,
                c_in_g=32,
                d=16,
                h=520,
                w=520,
                kernel_d=3,
                kernel_h=3,
                kernel_w=3,
                padding=(1, 1, 1),
                out_d=16,
                out_h=520,
                out_w=520,
            ),
            "conv3d",
            id="channels-last-m-tiles-past-grid-y",
        ),
        pytest.param(
            Conv1dFwdOp,
            Conv1dCall(
                n=1,
                c_in=16,
                c_out=64,
                c_in_g=16,
                l_in=33_600_000,
                kernel_l=3,
                pad_left=1,
                pad_right=1,
                out_l=33_600_000,
            ),
            "conv1d",
            id="unit-stride-output-past-int32",
        ),
        pytest.param(
            Conv2dFwdOp,
            Conv2dCall(
                n=128,
                c_in=1280,
                c_out=1280,
                c_in_g=2,
                h=8,
                w=8,
                kernel_h=3,
                kernel_w=3,
                padding=(1, 1),
                groups=640,
                out_h=8,
                out_w=8,
            ),
            None,
            id="grouped-planes-past-grid-z",
        ),
    ],
)
def test_conv_call_past_a_launch_limit_is_refused_during_selection(op_type, call, served) -> None:
    """A kernel that cannot launch or build a call leaves it to one that can, or selection
    raises with the limit rather than the launch failing."""
    (interface,) = op_type.interfaces
    if served is None:
        with pytest.raises(ValueError, match="blocks along grid z"):
            op_type().select_implementation(interface, call)
    else:
        assert op_type().select_implementation(interface, call) == served


@pytest.mark.smoke
def test_conv3d_roofline_ignores_the_serving_kernel_layout_traffic() -> None:
    """The channels-last kernel stages input and weight. Those buffers are
    intermediates of one implementation, and the roofline is the algorithm's minimum
    traffic, so the number does not move with them."""
    op = Conv3dFwdOp(stride=1, padding=1)
    x = torch.randn(1, 32, 8, 16, 16, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, 3, 3, device=run_device(), dtype=torch.float16).contiguous()

    op(x, weight)

    _, nbytes = op.eval_roofline()
    out_elems = 1 * 64 * 8 * 16 * 16
    input_elems = 1 * 32 * 8 * 16 * 16
    weight_elems = 64 * 32 * 3 * 3 * 3
    assert nbytes == (input_elems + weight_elems + out_elems) * x.element_size()


@pytest.mark.smoke
def test_conv3d_does_not_dispatch_ndhwc_for_pointwise() -> None:
    op = Conv3dFwdOp(stride=1, padding=0)
    x = torch.randn(1, 32, 8, 16, 16, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 1, 1, 1, device=run_device(), dtype=torch.float16).contiguous()

    out = op(x, weight)

    ref = F.conv3d(x, weight, bias=None, stride=1, padding=0).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv3d_does_not_dispatch_ndhwc_for_small_output() -> None:
    op = Conv3dFwdOp(stride=1, padding=1)
    x = torch.randn(1, 32, 2, 4, 4, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, 3, 3, device=run_device(), dtype=torch.float16).contiguous()

    out = op(x, weight)

    ref = F.conv3d(x, weight, bias=None, stride=1, padding=1).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


@pytest.mark.smoke
def test_conv1d_depthwise_no_bias_matches_torch() -> None:
    """The depthwise-direct path is the one Conv1d variant no other no-bias case reaches."""
    groups = 32
    op = Conv1dFwdOp(padding=1, groups=groups)
    x = torch.randn(1, groups, 128, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(groups, 1, 3, device=run_device(), dtype=torch.float16).contiguous()

    out = op(x, weight)

    ref = F.conv1d(x, weight, bias=None, padding=1, groups=groups).contiguous()
    compare_outputs(out, ref, convolution_verification(out.dtype))


# --------------------------------------------------------------------------------------
# The compile boundary: the node in the graph is the op's, whichever target serves it
# --------------------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("use_bias", [False, True], ids=["no-bias", "bias"])
def test_conv2d_cold_traces_fullgraph_and_owns_its_graph_nodes(use_bias: bool) -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op = Conv2dFwdOp(padding=1)
    x = torch.randn(1, 32, 16, 16, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
    bias = (
        torch.randn(64, device=run_device(), dtype=torch.float16).contiguous() if use_bias else None
    )

    assert_op_owns_graph_nodes(op, x, weight, bias)
    torch.testing.assert_close(
        torch.compile(op, fullgraph=True)(x, weight, bias), op(x, weight, bias)
    )


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_conv1d_cold_traces_fullgraph_and_owns_its_graph_nodes() -> None:
    op = Conv1dFwdOp(padding=1)
    x = torch.randn(1, 32, 128, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(64, 32, 3, device=run_device(), dtype=torch.float16).contiguous()

    assert_op_owns_graph_nodes(op, x, weight, None)
    torch.testing.assert_close(torch.compile(op, fullgraph=True)(x, weight), op(x, weight))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_conv3d_cold_traces_fullgraph_and_owns_its_graph_nodes() -> None:
    op = Conv3dFwdOp(padding=1)
    x = torch.randn(1, 16, 8, 8, 8, device=run_device(), dtype=torch.float16).contiguous()
    weight = torch.randn(32, 16, 3, 3, 3, device=run_device(), dtype=torch.float16).contiguous()

    assert_op_owns_graph_nodes(op, x, weight, None)
    torch.testing.assert_close(torch.compile(op, fullgraph=True)(x, weight), op(x, weight))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_a_non_contiguous_input_compiles_to_the_shape_the_fake_promised() -> None:
    """The fake speaks before the body normalizes contiguity, so it promises contiguous."""
    op = Conv2dFwdOp(padding=1)
    x = torch.randn(1, 32, 16, 32, device=run_device(), dtype=torch.float16)[:, :, :, ::2]
    weight = torch.randn(64, 32, 3, 3, device=run_device(), dtype=torch.float16).contiguous()
    assert not x.is_contiguous()

    output = torch.compile(op, fullgraph=True)(x, weight)

    assert output.is_contiguous()
    torch.testing.assert_close(output, op(x, weight))


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
def test_conv1d_graph_reads_updated_inference_weights() -> None:
    """Weight packing consumes live data, even without a Tensor version counter."""
    workload = Conv1dTest(2, 64, 512, 128, 3, 1, 1, 1, 1, torch.float16)
    op = Conv1dFwdOp(padding=1)
    with torch.inference_mode():
        inputs = workload.gen_inputs()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op(*inputs)
        inputs[1].mul_(0.5)
        graph.replay()
        workload.check(op, *inputs, runs=lambda *args: output)
