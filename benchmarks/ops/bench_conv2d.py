from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import Conv2dOp


class Conv2dBenchCase:

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


class Conv2dBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        out_h = (t.h + 2 * t.padding[0] - t.kernel_size[0]) // t.stride[0] + 1
        out_w = (t.w + 2 * t.padding[1] - t.kernel_size[1]) // t.stride[1] + 1
        return 2.0 * t.n * t.c_out * out_h * out_w * t.c_in * t.kernel_size[0] * t.kernel_size[1]

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        out_h = (t.h + 2 * t.padding[0] - t.kernel_size[0]) // t.stride[0] + 1
        out_w = (t.w + 2 * t.padding[1] - t.kernel_size[1]) // t.stride[1] + 1
        bytes_ = (
            t.n * t.c_in * t.h * t.w
            + t.c_out * t.c_in * t.kernel_size[0] * t.kernel_size[1]
            + t.n * t.c_out * out_h * out_w
        ) * t.dtype.itemsize
        return bytes_


_CONV2D_BENCH_PARAMS = [
    pytest.param(2, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), torch.float16, True, id="resnet-3x3-fp16"),
    pytest.param(1, 3, 112, 112, 64, (3, 3), (2, 2), (1, 1), torch.float16, True, id="stem-3x3-s2-fp16"),
    pytest.param(1, 128, 56, 56, 256, (3, 3), (2, 2), (1, 1), torch.float16, True, id="stage-transition-3x3-s2-fp16"),
    pytest.param(1, 256, 112, 112, 512, (3, 3), (1, 1), (1, 1), torch.float16, True, id="highres-3x3-s1-fp16"),
    pytest.param(1, 64, 56, 56, 128, (5, 5), (1, 1), (2, 2), torch.float16, True, id="midres-5x5-s1-fp16"),
    pytest.param(1, 128, 56, 56, 256, (5, 5), (2, 2), (2, 2), torch.float16, True, id="stage-transition-5x5-s2-fp16"),
    pytest.param(1, 128, 28, 28, 128, (3, 3), (2, 2), (1, 1), torch.bfloat16, True, id="stride2-bf16"),
    pytest.param(2, 64, 56, 56, 256, (1, 1), (1, 1), (0, 0), torch.float16, True, id="resnet-1x1-fp16"),
    pytest.param(2, 128, 28, 28, 512, (1, 1), (1, 1), (0, 0), torch.float16, True, id="bottleneck-expand-1x1-fp16"),
    pytest.param(2, 512, 28, 28, 128, (1, 1), (1, 1), (0, 0), torch.float16, True, id="bottleneck-reduce-1x1-fp16"),
    pytest.param(1, 256, 14, 14, 1024, (1, 1), (1, 1), (0, 0), torch.float16, True, id="late-stage-1x1-fp16"),
    pytest.param(1, 512, 7, 7, 2048, (1, 1), (1, 1), (0, 0), torch.float16, True, id="classifier-1x1-fp16"),
    pytest.param(2, 64, 56, 56, 256, (1, 1), (1, 1), (0, 0), torch.bfloat16, True, id="resnet-1x1-bf16"),
]


@pytest.mark.parametrize(
    "n, c_in, h, w, c_out, kernel_size, stride, padding, dtype, tune",
    _CONV2D_BENCH_PARAMS,
)
def test_conv2d_bench(
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
    test = Conv2dBenchCase(n, c_in, h, w, c_out, kernel_size, stride, padding, dtype)
    bm = Conv2dBenchmark(test)
    inputs = test.gen_inputs()

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
    result = bm.profile(op, *inputs, warmup=5, rep=10)
    BenchmarkReport.record("conv2d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs, warmup=5, rep=10)
    BenchmarkReport.record("conv2d", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
