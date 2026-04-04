from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import Conv3dOp


class Conv3dBenchCase:

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        c_out: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = torch.randn(
            self.n, self.d_in, self.h_in, self.w_in, self.c_in,
            device="cuda", dtype=self.dtype,
        ).contiguous()
        weight = torch.randn(
            self.c_out, self.c_in, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
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
            dilation=1,
            groups=1,
        )


class Conv3dBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        out_d = (t.d_in + 2 * t.padding[0] - t.kernel_size[0]) // t.stride[0] + 1
        out_h = (t.h_in + 2 * t.padding[1] - t.kernel_size[1]) // t.stride[1] + 1
        out_w = (t.w_in + 2 * t.padding[2] - t.kernel_size[2]) // t.stride[2] + 1
        return 2.0 * t.n * t.c_out * out_d * out_h * out_w * t.c_in * t.kernel_size[0] * t.kernel_size[1] * t.kernel_size[2]

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        out_d = (t.d_in + 2 * t.padding[0] - t.kernel_size[0]) // t.stride[0] + 1
        out_h = (t.h_in + 2 * t.padding[1] - t.kernel_size[1]) // t.stride[1] + 1
        out_w = (t.w_in + 2 * t.padding[2] - t.kernel_size[2]) // t.stride[2] + 1
        bytes_ = (
            t.n * t.c_in * t.d_in * t.h_in * t.w_in
            + t.c_out * t.c_in * t.kernel_size[0] * t.kernel_size[1] * t.kernel_size[2]
            + t.n * t.c_out * out_d * out_h * out_w
        ) * t.dtype.itemsize
        return bytes_


_CONV3D_BENCH_PARAMS = [
    pytest.param(1, 3, 16, 112, 112, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.float16, True, id="r3d-stem-k3-s1-fp16"),
    pytest.param(1, 64, 8, 56, 56, 128, (3, 3, 3), (2, 2, 2), (1, 1, 1), torch.float16, True, id="video-stage-downsample-k3-s2-fp16"),
    pytest.param(1, 32, 32, 64, 64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), torch.bfloat16, True, id="unet-encoder-k3-s1-bf16"),
]


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dtype, tune",
    _CONV3D_BENCH_PARAMS,
)
def test_conv3d_bench(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = Conv3dBenchCase(n, c_in, d_in, h_in, w_in, c_out, kernel_size, stride, padding, dtype)
    bm = Conv3dBenchmark(test)
    inputs = test.gen_inputs()
    x, weight, bias = inputs
    x_ncdhw = x.permute(0, 4, 1, 2, 3).contiguous()
    x_ndhwc = x_ncdhw.contiguous(memory_format=torch.channels_last_3d)

    op = Conv3dOp(
        n=n,
        c_in=c_in,
        d_in=d_in,
        h_in=h_in,
        w_in=w_in,
        c_out=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("conv3d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x_ncdhw, weight, bias)
    BenchmarkReport.record("conv3d", locals(), result_bl, tag="torch-ncdhw")

    result_bl = bm.profile(test.ref_program, x_ndhwc, weight, bias)
    BenchmarkReport.record("conv3d", locals(), result_bl, tag="torch-ndhwc")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
