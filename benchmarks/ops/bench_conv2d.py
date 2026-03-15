import os
from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_conv2d import Conv2dTest
from tileops.ops import Conv2dOp

ENABLE_TUNE = os.environ.get("CONV2D_BENCH_TUNE", "0") == "1"


class Conv2dBenchmarkFixture:
    PARAMS = [
        ("n, c_in, h, w, c_out, kernel_size, stride, padding, bias, dtype", [
            pytest.param(
                1, 32, 28, 28, 64, (1, 1), (2, 2), (1, 1), True, torch.float16,
                marks=pytest.mark.full,
                id="bench-fp16-32in-64out-1x1-s2-pad1-28-bias",
            ),
            pytest.param(
                1, 64, 56, 56, 256, 1, 1, 0, False, torch.float16,
                marks=pytest.mark.full,
                id="bench-fp16-64in-256out-1x1-s1-56",
            ),
            pytest.param(
                1, 64, 56, 56, 256, 1, 2, 0, False, torch.float16,
                marks=pytest.mark.full,
                id="bench-fp16-64in-256out-1x1-s2-56",
            ),
            pytest.param(
                1, 256, 56, 56, 512, 1, 1, 0, True, torch.float16,
                marks=pytest.mark.full,
                id="bench-fp16-256in-512out-1x1-s1-56-bias",
            ),
            pytest.param(
                1, 128, 112, 112, 512, 1, 1, 0, False, torch.bfloat16,
                marks=pytest.mark.full,
                id="bench-bf16-128in-512out-1x1-s1-112",
            ),
            pytest.param(
                1, 64, 56, 56, 64, 3, 1, 1, False, torch.float16,
                marks=pytest.mark.full,
                id="bench-fp16-64in-64out-3x3-s1-56",
            ),
            pytest.param(
                1, 512, 56, 56, 512, 3, 1, 1, False, torch.bfloat16,
                marks=pytest.mark.full,
                id="bench-bf16-512in-512out-3x3-s1-56",
            ),
            pytest.param(
                1, 128, 112, 112, 256, 3, 2, 1, False, torch.bfloat16,
                marks=pytest.mark.full,
                id="bench-bf16-128in-256out-3x3-s2-112",
            ),
            pytest.param(
                1, 64, 224, 224, 128, 5, 2, 2, False, torch.float16,
                marks=pytest.mark.full,
                id="bench-fp16-64in-128out-5x5-s2-224",
            ),
        ]),
    ]

    def __call__(self, fn):
        for names, values in reversed(self.PARAMS):
            fn = pytest.mark.parametrize(names, values)(fn)
        return fn


class Conv2dBenchmark(BenchmarkBase):
    def calculate_flops(self) -> Optional[float]:
        test = self.test
        kernel_h, kernel_w = test.kernel_size if isinstance(test.kernel_size, tuple) else (
            test.kernel_size, test.kernel_size)
        stride_h, stride_w = test.stride if isinstance(test.stride, tuple) else (
            test.stride, test.stride)
        pad_h, pad_w = test.padding if isinstance(test.padding, tuple) else (
            test.padding, test.padding)
        out_h = (test.h + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (test.w + 2 * pad_w - kernel_w) // stride_w + 1
        return 2.0 * test.n * test.c_out * out_h * out_w * test.c_in * kernel_h * kernel_w

    def calculate_memory(self) -> Optional[float]:
        test = self.test
        kernel_h, kernel_w = test.kernel_size if isinstance(test.kernel_size, tuple) else (
            test.kernel_size, test.kernel_size)
        stride_h, stride_w = test.stride if isinstance(test.stride, tuple) else (
            test.stride, test.stride)
        pad_h, pad_w = test.padding if isinstance(test.padding, tuple) else (
            test.padding, test.padding)
        out_h = (test.h + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (test.w + 2 * pad_w - kernel_w) // stride_w + 1
        bias_elems = test.c_out if test.bias else 0
        total_elems = (
            test.n * test.c_in * test.h * test.w
            + test.c_out * test.c_in * kernel_h * kernel_w
            + bias_elems
            + test.n * test.c_out * out_h * out_w
        )
        return total_elems * test.dtype.itemsize


@Conv2dBenchmarkFixture()
def test_conv2d_bench(
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
    test = Conv2dTest(n, c_in, h, w, c_out, kernel_size, stride, padding, bias, dtype)
    bm = Conv2dBenchmark(test)
    inputs = test.gen_inputs()

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
        tune=ENABLE_TUNE,
    )
    result = bm.profile(op, *inputs)
    kernel_size = str(kernel_size) if isinstance(kernel_size, tuple) else kernel_size
    stride = str(stride) if isinstance(stride, tuple) else stride
    padding = str(padding) if isinstance(padding, tuple) else padding
    BenchmarkReport.record("conv2d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("conv2d", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
