from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.kernels.pool.common import pool_output_dim
from tileops.ops import AvgPool2dOp


class AvgPool2dBenchCase:

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: Tuple[int, int],
        stride: Optional[Tuple[int, int]],
        padding: Tuple[int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.h_in, self.w_in, self.c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


class AvgPool2dBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        out_h = pool_output_dim(t.h_in, t.kernel_size[0], t.stride[0], t.padding[0], t.ceil_mode)
        out_w = pool_output_dim(t.w_in, t.kernel_size[1], t.stride[1], t.padding[1], t.ceil_mode)
        return t.n * t.c_in * out_h * out_w * t.kernel_size[0] * t.kernel_size[1]

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        out_h = pool_output_dim(t.h_in, t.kernel_size[0], t.stride[0], t.padding[0], t.ceil_mode)
        out_w = pool_output_dim(t.w_in, t.kernel_size[1], t.stride[1], t.padding[1], t.ceil_mode)
        return (t.n * t.c_in * t.h_in * t.w_in + t.n * t.c_in * out_h * out_w) * t.dtype.itemsize


_AVG_POOL2D_BENCH_PARAMS = [
    pytest.param(2, 64, 112, 112, (3, 3), (2, 2), (1, 1), False, True, None, torch.float16, True, id="vision-3x3-s2"),
    pytest.param(2, 128, 56, 56, (5, 5), (2, 2), (2, 2), False, True, None, torch.float16, True, id="vision-5x5-s2"),
    pytest.param(3, 96, 55, 57, (3, 5), (2, 2), (1, 2), True, False, 7, torch.bfloat16, True, id="ceil-divisor-bf16"),
]


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
    _AVG_POOL2D_BENCH_PARAMS,
)
def test_avg_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]],
    padding: Tuple[int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool2dBenchCase(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    bm = AvgPool2dBenchmark(test)
    inputs = test.gen_inputs()
    (x,) = inputs
    x_nchw = x.permute(0, 3, 1, 2).contiguous()

    op = AvgPool2dOp(
        n=n,
        c_in=c_in,
        h_in=h_in,
        w_in=w_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("avg_pool2d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x_nchw)
    BenchmarkReport.record("avg_pool2d", locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
