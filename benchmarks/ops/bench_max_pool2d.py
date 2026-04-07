from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.kernels.pool.common import pool_output_dim
from tileops.ops import MaxPool2dOp


class MaxPool2dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: Tuple[int, int],
        stride: Optional[Tuple[int, int]],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        ceil_mode: bool,
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.h_in, self.w_in, self.c_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )


class MaxPool2dBenchmark(BenchmarkBase):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        out_h = pool_output_dim(t.h_in, t.kernel_size[0], t.stride[0], t.padding[0], t.ceil_mode, t.dilation[0])
        out_w = pool_output_dim(t.w_in, t.kernel_size[1], t.stride[1], t.padding[1], t.ceil_mode, t.dilation[1])
        return t.n * t.c_in * out_h * out_w * t.kernel_size[0] * t.kernel_size[1]

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        out_h = pool_output_dim(t.h_in, t.kernel_size[0], t.stride[0], t.padding[0], t.ceil_mode, t.dilation[0])
        out_w = pool_output_dim(t.w_in, t.kernel_size[1], t.stride[1], t.padding[1], t.ceil_mode, t.dilation[1])
        return (t.n * t.c_in * t.h_in * t.w_in + t.n * t.c_in * out_h * out_w) * t.dtype.itemsize


_MAX_POOL2D_BASE_CASES = [
    (2, 64, 112, 112, (3, 3), (2, 2), (1, 1), (1, 1), False, "vision-3x3-s2"),
    (2, 128, 56, 56, (5, 5), (2, 2), (2, 2), (1, 1), False, "vision-5x5-s2"),
    (3, 96, 55, 57, (3, 3), (2, 2), (1, 1), (2, 1), True, "ceil-dilation-nonpow2"),
]

_MAX_POOL2D_BENCH_PARAMS = [
    pytest.param(*case[:-1], dtype, True, id=f"{case[-1]}-{str(dtype).split('.')[-1]}")
    for case in _MAX_POOL2D_BASE_CASES
    for dtype in (torch.float16, torch.bfloat16)
]


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _MAX_POOL2D_BENCH_PARAMS,
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool2dBenchCase(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
    )
    bm = MaxPool2dBenchmark(test)
    inputs = test.gen_inputs()
    (x,) = inputs
    x_nchw = x.permute(0, 3, 1, 2).contiguous()

    op = MaxPool2dOp(
        n=n,
        c_in=c_in,
        h_in=h_in,
        w_in=w_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=ceil_mode,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x_nchw)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
