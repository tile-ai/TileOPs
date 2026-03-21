from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_engram_decode import (
    EngramDecodeTest,
    _ref_engram_decode_step,
)
from tileops.ops.engram_decode import EngramDecodeOp


class EngramDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, d_mem, d, w = t.batch, t.d_mem, t.d, t.conv_kernel_size
        # GEMV: 2 * B * d_mem * d (k) + 2 * B * d_mem * d (v)
        # 2x RMSNorm(d): ~4d each -> 8*B*d
        # dot product: 2*B*d, sigmoid: ~10*B, gated mul: B*d
        # RMSNorm(v_hat): 4*B*d
        # dilated conv (w taps): w*2*B*d
        # SiLU + residual: ~10*B + B*d
        return (4 * B * d_mem * d
                + B * (8 * d + 2 * d + d + 4 * d + w * 2 * d + d)
                + 20 * B)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, d_mem, d, mcl, w = t.batch, t.d_mem, t.d, t.max_conv_len, t.conv_kernel_size
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Read: e_t (B*d_mem) + h_t (B*d) + conv_state (B*mcl*d) + W_K,W_V (2*d_mem*d)
        #        + weights (2*d + w*d)
        # Write: y_t (B*d) + new_conv_state (B*mcl*d)
        return (B * d_mem + B * d + 2 * B * mcl * d + 2 * d_mem * d
                + 2 * d + w * d + B * d) * elem


_ENGRAM_DECODE_BENCH_PARAMS = [
    pytest.param(1, 512, 256, 12, 4, 3, torch.float16, True, id="fp16-mainstream"),
    pytest.param(4, 1024, 512, 20, 4, 5, torch.float16, True, id="fp16-large"),
    pytest.param(8, 512, 256, 18, 4, 3, torch.bfloat16, True, id="bf16-batched"),
]


@pytest.mark.parametrize(
    "batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune",
    _ENGRAM_DECODE_BENCH_PARAMS,
)
def test_engram_decode_bench(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune):
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    bm = EngramDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = EngramDecodeOp(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(*args):
        return _ref_engram_decode_step(*args, max_conv_len=max_conv_len, dilation=dilation)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
