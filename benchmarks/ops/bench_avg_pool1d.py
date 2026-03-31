from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import AvgPool1dOp


class AvgPool1dBenchCase:

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: int,
        stride: Optional[int],
        padding: int,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.l_in, self.c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
        )


class AvgPool1dBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        out_l = (t.l_in + 2 * t.padding - t.kernel_size + t.stride - 1) // t.stride + 1 if t.ceil_mode else (
            (t.l_in + 2 * t.padding - t.kernel_size) // t.stride + 1
        )
        return t.n * t.c_in * out_l * t.kernel_size

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        out_l = (t.l_in + 2 * t.padding - t.kernel_size + t.stride - 1) // t.stride + 1 if t.ceil_mode else (
            (t.l_in + 2 * t.padding - t.kernel_size) // t.stride + 1
        )
        return (t.n * t.c_in * t.l_in + t.n * t.c_in * out_l) * t.dtype.itemsize


_AVG_POOL1D_BENCH_PARAMS = [
    pytest.param(4, 128, 4096, 3, 2, 1, False, True, torch.float16, True, id="audio-downsample-fp16"),
    pytest.param(2, 256, 32000, 5, 4, 2, False, True, torch.float16, True, id="long-temporal-fp16"),
    pytest.param(2, 128, 2048, 4, 2, 1, True, False, torch.bfloat16, True, id="ceil-bf16"),
]


@pytest.mark.parametrize(
    "n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, tune",
    _AVG_POOL1D_BENCH_PARAMS,
)
def test_avg_pool1d_bench(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: int,
    stride: Optional[int],
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool1dBenchCase(n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype)
    bm = AvgPool1dBenchmark(test)
    inputs = test.gen_inputs()
    (x,) = inputs
    x_ncl = x.permute(0, 2, 1).contiguous()

    op = AvgPool1dOp(
        n=n,
        c_in=c_in,
        l_in=l_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("avg_pool1d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x_ncl)
    BenchmarkReport.record("avg_pool1d", locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
