from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_conv2d import Conv2dTest
from tileops.ops import Conv2dOp


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
    pytest.param(2, 64, 56, 56, 256, (1, 1), (1, 1), (0, 0), torch.float16, True, id="resnet-1x1-fp16"),
    pytest.param(1, 128, 28, 28, 128, (3, 3), (2, 2), (1, 1), torch.bfloat16, True, id="stride2-bf16"),
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
    test = Conv2dTest(n, c_in, h, w, c_out, kernel_size, stride, padding, dtype)
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
