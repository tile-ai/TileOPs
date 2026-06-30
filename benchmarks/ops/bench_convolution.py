from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops import Conv1dBiasFwdOp, Conv2dBiasFwdOp, Conv3dBiasFwdOp


class Conv1dBenchCase:

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
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
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
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
        return F.conv1d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
        )


class Conv1dBenchmark(BenchmarkBase[Conv1dBenchCase]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        out_l = (t.l_in + 2 * t.padding - t.dilation * (t.kernel_size - 1) - 1) // t.stride + 1
        return 2.0 * t.n * t.c_out * out_l * t.c_in * t.kernel_size

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        out_l = (t.l_in + 2 * t.padding - t.dilation * (t.kernel_size - 1) - 1) // t.stride + 1
        bytes_ = (
            t.n * t.c_in * t.l_in
            + t.c_out * t.c_in * t.kernel_size
            + t.n * t.c_out * out_l
        ) * t.dtype.itemsize
        return bytes_


_CONV1D_BENCH_PARAMS = [
    pytest.param(4, 256, 32000, 512, 1, 1, 0, 1, torch.float16, True, id="convtasnet-pointwise-k1-s1-fp16"),
    pytest.param(4, 128, 4096, 256, 3, 1, 1, 1, torch.float16, True, id="seanet-k3-s1-fp16"),
    pytest.param(4, 64, 16000, 128, 5, 2, 2, 1, torch.float16, True, id="audio-downsample-k5-s2-fp16"),
    pytest.param(4, 128, 8192, 256, 7, 1, 3, 1, torch.float16, True, id="seanet-stem-k7-s1-fp16"),
    pytest.param(2, 128, 4096, 256, 3, 2, 1, 1, torch.bfloat16, True, id="sequence-downsample-k3-s2-bf16"),
    pytest.param(4, 128, 4096, 256, 3, 1, 2, 2, torch.float16, True, id="seanet-k3-s1-d2-fp16"),
]


@pytest.mark.parametrize(
    "n, c_in, l_in, c_out, kernel_size, stride, padding, dilation, dtype, tune",
    _CONV1D_BENCH_PARAMS,
)
def test_conv1d_bench(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = Conv1dBenchCase(n, c_in, l_in, c_out, kernel_size, stride, padding, dilation, dtype)
    bm = Conv1dBenchmark(test)
    inputs = test.gen_inputs()
    x, weight, bias = inputs

    op = Conv1dBiasFwdOp(
        n=n,
        c_in=c_in,
        l_in=l_in,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("conv1d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x, weight, bias)
    BenchmarkReport.record("conv1d", locals(), result_bl, tag="torch")


class Conv2dBenchCase:

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
        dilation: tuple[int, int],
        groups: int,
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
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(self.n, self.c_in, self.h, self.w, device="cuda", dtype=self.dtype).contiguous()
        weight = torch.randn(
            self.c_out, self.c_in // self.groups, self.kernel_size[0], self.kernel_size[1],
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
        return F.conv2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

class Conv2dBenchmark(BenchmarkBase[Conv2dBenchCase]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        out_h = (t.h + 2 * t.padding[0] - t.dilation[0] * (t.kernel_size[0] - 1) - 1) // t.stride[0] + 1
        out_w = (t.w + 2 * t.padding[1] - t.dilation[1] * (t.kernel_size[1] - 1) - 1) // t.stride[1] + 1
        c_in_g = t.c_in // t.groups
        return 2.0 * t.n * t.c_out * out_h * out_w * c_in_g * t.kernel_size[0] * t.kernel_size[1]

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        out_h = (t.h + 2 * t.padding[0] - t.dilation[0] * (t.kernel_size[0] - 1) - 1) // t.stride[0] + 1
        out_w = (t.w + 2 * t.padding[1] - t.dilation[1] * (t.kernel_size[1] - 1) - 1) // t.stride[1] + 1
        c_in_g = t.c_in // t.groups
        bytes_ = (
            t.n * t.c_in * t.h * t.w
            + t.c_out * c_in_g * t.kernel_size[0] * t.kernel_size[1]
            + t.n * t.c_out * out_h * out_w
        ) * t.dtype.itemsize
        return bytes_


_CONV2D_BENCH_PARAMS = [
    pytest.param(2, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, torch.float16, True, id="resnet-3x3-fp16"),
    pytest.param(1, 3, 112, 112, 64, (3, 3), (2, 2), (1, 1), (1, 1), 1, torch.float16, True, id="stem-3x3-s2-fp16"),
    pytest.param(1, 128, 56, 56, 256, (3, 3), (2, 2), (1, 1), (1, 1), 1, torch.float16, True, id="stage-transition-3x3-s2-fp16"),
    pytest.param(1, 256, 112, 112, 512, (3, 3), (1, 1), (1, 1), (1, 1), 1, torch.float16, True, id="highres-3x3-s1-fp16"),
    pytest.param(1, 64, 56, 56, 128, (5, 5), (1, 1), (2, 2), (1, 1), 1, torch.float16, True, id="midres-5x5-s1-fp16"),
    pytest.param(1, 128, 56, 56, 256, (5, 5), (2, 2), (2, 2), (1, 1), 1, torch.float16, True, id="stage-transition-5x5-s2-fp16"),
    pytest.param(1, 128, 28, 28, 128, (3, 3), (2, 2), (1, 1), (1, 1), 1, torch.bfloat16, True, id="stride2-bf16"),
    pytest.param(2, 64, 56, 56, 256, (1, 1), (1, 1), (0, 0), (1, 1), 1, torch.float16, True, id="resnet-1x1-fp16"),
    pytest.param(2, 128, 28, 28, 512, (1, 1), (1, 1), (0, 0), (1, 1), 1, torch.float16, True, id="bottleneck-expand-1x1-fp16"),
    pytest.param(2, 512, 28, 28, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, torch.float16, True, id="bottleneck-reduce-1x1-fp16"),
    pytest.param(1, 256, 14, 14, 1024, (1, 1), (1, 1), (0, 0), (1, 1), 1, torch.float16, True, id="late-stage-1x1-fp16"),
    pytest.param(1, 512, 7, 7, 2048, (1, 1), (1, 1), (0, 0), (1, 1), 1, torch.float16, True, id="classifier-1x1-fp16"),
    pytest.param(2, 64, 56, 56, 256, (1, 1), (1, 1), (0, 0), (1, 1), 1, torch.bfloat16, True, id="resnet-1x1-bf16"),
    # DeepLabV3/DeepLabV3+ ASPP branch: 3x3 atrous conv on stride-16 encoder features.
    pytest.param(1, 2048, 32, 32, 256, (3, 3), (1, 1), (12, 12), (12, 12), 1, torch.float16, True, id="deeplabv3-aspp-3x3-rate12-fp16"),
    # MobileNetV2 inverted residual depthwise 3x3 convolution.
    pytest.param(1, 32, 56, 56, 32, (3, 3), (1, 1), (1, 1), (1, 1), 32, torch.float16, True, id="mobilenetv2-depthwise-fp16"),
    # ResNeXt bottleneck grouped 3x3 convolution.
    pytest.param(1, 128, 28, 28, 256, (3, 3), (1, 1), (1, 1), (1, 1), 32, torch.float16, True, id="resnext-grouped-3x3-fp16"),
]


@pytest.mark.parametrize(
    "n, c_in, h, w, c_out, kernel_size, stride, padding, dilation, groups, dtype, tune",
    _CONV2D_BENCH_PARAMS,
)
def test_conv2d_bench(
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
    test = Conv2dBenchCase(n, c_in, h, w, c_out, kernel_size, stride, padding, dilation, groups, dtype)
    bm = Conv2dBenchmark(test)
    inputs = test.gen_inputs()
    x, weight, bias = inputs

    op = Conv2dBiasFwdOp(
        n=n,
        c_in=c_in,
        h=h,
        w=w,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("conv2d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x, weight, bias)
    BenchmarkReport.record("conv2d", locals(), result_bl, tag="torch")


class Conv3dBenchCase:

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
        dilation: tuple[int, int, int],
        groups: int,
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
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(
            self.n, self.c_in, self.d, self.h, self.w,
            device="cuda", dtype=self.dtype,
        ).contiguous()
        weight = torch.randn(
            self.c_out,
            self.c_in // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
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
        return F.conv3d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

class Conv3dBenchmark(BenchmarkBase[Conv3dBenchCase]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        out_d = (t.d + 2 * t.padding[0] - t.dilation[0] * (t.kernel_size[0] - 1) - 1) // t.stride[0] + 1
        out_h = (t.h + 2 * t.padding[1] - t.dilation[1] * (t.kernel_size[1] - 1) - 1) // t.stride[1] + 1
        out_w = (t.w + 2 * t.padding[2] - t.dilation[2] * (t.kernel_size[2] - 1) - 1) // t.stride[2] + 1
        c_in_g = t.c_in // t.groups
        return 2.0 * t.n * t.c_out * out_d * out_h * out_w * c_in_g * t.kernel_size[0] * t.kernel_size[1] * t.kernel_size[2]

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        out_d = (t.d + 2 * t.padding[0] - t.dilation[0] * (t.kernel_size[0] - 1) - 1) // t.stride[0] + 1
        out_h = (t.h + 2 * t.padding[1] - t.dilation[1] * (t.kernel_size[1] - 1) - 1) // t.stride[1] + 1
        out_w = (t.w + 2 * t.padding[2] - t.dilation[2] * (t.kernel_size[2] - 1) - 1) // t.stride[2] + 1
        c_in_g = t.c_in // t.groups
        bytes_ = (
            t.n * t.c_in * t.d * t.h * t.w
            + t.c_out * c_in_g * t.kernel_size[0] * t.kernel_size[1] * t.kernel_size[2]
            + t.n * t.c_out * out_d * out_h * out_w
        ) * t.dtype.itemsize
        return bytes_


_CONV3D_BENCH_PARAMS = [
    pytest.param(1, 3, 16, 112, 112, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1, torch.float16, True, id="r3d-stem-k3-s1-fp16"),
    pytest.param(1, 64, 8, 56, 56, 128, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), 1, torch.float16, True, id="video-stage-downsample-k3-s2-fp16"),
    pytest.param(1, 32, 32, 64, 64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1, torch.bfloat16, True, id="unet-encoder-k3-s1-bf16"),
    # 3D U-Net + 3D ASPP medical segmentation branch: 3x3x3 atrous conv on low-resolution volume features.
    pytest.param(1, 256, 8, 16, 16, 256, (3, 3, 3), (1, 1, 1), (6, 6, 6), (6, 6, 6), 1, torch.float16, True, id="3d-unet-aspp-3x3x3-rate6-fp16"),
    # 3D-ResNeXt/video backbone grouped 3x3x3 convolution.
    pytest.param(1, 64, 8, 28, 28, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 32, torch.float16, False, id="3d-resnext-grouped-k3-fp16"),
]


@pytest.mark.parametrize(
    "n, c_in, d, h, w, c_out, kernel_size, stride, padding, dilation, groups, dtype, tune",
    _CONV3D_BENCH_PARAMS,
)
def test_conv3d_bench(
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
    test = Conv3dBenchCase(n, c_in, d, h, w, c_out, kernel_size, stride, padding, dilation, groups, dtype)
    bm = Conv3dBenchmark(test)
    inputs = test.gen_inputs()
    x, weight, bias = inputs

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
        dilation=dilation,
        groups=groups,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("conv3d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x, weight, bias)
    BenchmarkReport.record("conv3d", locals(), result_bl, tag="torch")
