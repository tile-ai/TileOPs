from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.kernels.pool.common import pool_output_dim
from tileops.ops import AvgPool3dOp


class AvgPool3dBenchCase:

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_size: Tuple[int, int, int],
        stride: Optional[Tuple[int, int, int]],
        padding: Tuple[int, int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
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
        x = torch.randn(
            self.n, self.d_in, self.h_in, self.w_in, self.c_in, device="cuda", dtype=self.dtype
        ).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


class AvgPool3dBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        out_d = pool_output_dim(t.d_in, t.kernel_size[0], t.stride[0], t.padding[0], t.ceil_mode)
        out_h = pool_output_dim(t.h_in, t.kernel_size[1], t.stride[1], t.padding[1], t.ceil_mode)
        out_w = pool_output_dim(t.w_in, t.kernel_size[2], t.stride[2], t.padding[2], t.ceil_mode)
        return t.n * t.c_in * out_d * out_h * out_w * t.kernel_size[0] * t.kernel_size[1] * t.kernel_size[2]

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        out_d = pool_output_dim(t.d_in, t.kernel_size[0], t.stride[0], t.padding[0], t.ceil_mode)
        out_h = pool_output_dim(t.h_in, t.kernel_size[1], t.stride[1], t.padding[1], t.ceil_mode)
        out_w = pool_output_dim(t.w_in, t.kernel_size[2], t.stride[2], t.padding[2], t.ceil_mode)
        return (
            t.n * t.c_in * t.d_in * t.h_in * t.w_in + t.n * t.c_in * out_d * out_h * out_w
        ) * t.dtype.itemsize


_AVG_POOL3D_BENCH_PARAMS = [
    pytest.param(1, 32, 16, 56, 56, (2, 2, 2), (2, 2, 2), (0, 0, 0), False, True, None, torch.float16, True, id="video-2x2x2"),
    pytest.param(2, 64, 8, 28, 28, (2, 3, 3), (2, 2, 2), (1, 1, 1), True, False, None, torch.float16, True, id="ceil-video"),
    pytest.param(2, 24, 10, 20, 22, (2, 2, 3), (2, 2, 2), (0, 1, 1), False, True, 7, torch.bfloat16, True, id="divisor-bf16"),
]


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
    _AVG_POOL3D_BENCH_PARAMS,
)
def test_avg_pool3d_bench(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: Tuple[int, int, int],
    stride: Optional[Tuple[int, int, int]],
    padding: Tuple[int, int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool3dBenchCase(
        n,
        c_in,
        d_in,
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
    bm = AvgPool3dBenchmark(test)
    inputs = test.gen_inputs()
    (x,) = inputs
    x_ncdhw = x.permute(0, 4, 1, 2, 3).contiguous()

    op = AvgPool3dOp(
        n=n,
        c_in=c_in,
        d_in=d_in,
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
    BenchmarkReport.record("avg_pool3d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x_ncdhw)
    BenchmarkReport.record("avg_pool3d", locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
