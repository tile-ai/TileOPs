from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import Conv1dOp


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
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(self.n, self.l_in, self.c_in, device="cuda", dtype=self.dtype).contiguous()
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
            dilation=1,
            groups=1,
        )


class Conv1dBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        out_l = (t.l_in + 2 * t.padding - t.kernel_size) // t.stride + 1
        return 2.0 * t.n * t.c_out * out_l * t.c_in * t.kernel_size

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        out_l = (t.l_in + 2 * t.padding - t.kernel_size) // t.stride + 1
        bytes_ = (
            t.n * t.c_in * t.l_in
            + t.c_out * t.c_in * t.kernel_size
            + t.n * t.c_out * out_l
        ) * t.dtype.itemsize
        return bytes_


_CONV1D_BENCH_PARAMS = [
    pytest.param(4, 256, 32000, 512, 1, 1, 0, torch.float16, True, id="convtasnet-pointwise-k1-s1-fp16"),
    pytest.param(4, 128, 4096, 256, 3, 1, 1, torch.float16, True, id="seanet-k3-s1-fp16"),
    pytest.param(4, 64, 16000, 128, 5, 2, 2, torch.float16, True, id="audio-downsample-k5-s2-fp16"),
    pytest.param(4, 128, 8192, 256, 7, 1, 3, torch.float16, True, id="seanet-stem-k7-s1-fp16"),
    pytest.param(2, 128, 4096, 256, 3, 2, 1, torch.bfloat16, True, id="sequence-downsample-k3-s2-bf16"),
]


@pytest.mark.parametrize(
    "n, c_in, l_in, c_out, kernel_size, stride, padding, dtype, tune",
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
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = Conv1dBenchCase(n, c_in, l_in, c_out, kernel_size, stride, padding, dtype)
    bm = Conv1dBenchmark(test)
    inputs = test.gen_inputs()
    x, weight, bias = inputs
    x_ncl = x.permute(0, 2, 1).contiguous()

    op = Conv1dOp(
        n=n,
        c_in=c_in,
        l_in=l_in,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("conv1d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x_ncl, weight, bias)
    BenchmarkReport.record("conv1d", locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
