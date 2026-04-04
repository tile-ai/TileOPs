from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.engram_fwd import EngramGateConvFwdOp
from workloads.ops.engram_fwd import (
    EngramGateConvFwdTest,
    ref_engram_gate_conv_fwd,
)


class EngramGateConvFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        M, T, d = t.M, t.seq_len, t.d
        # 2x RMSNorm(d): ~4d each -> 8*M*T*d
        # dot product (d): 2*M*T*d
        # sigmoid: ~10*M*T
        # gated mul: M*T*d
        # RMSNorm(v_hat): 4*M*T*d
        # conv (kernel=4): 4*2*M*T*d
        # SiLU: ~10*M*T
        # residual add: M*T*d
        return M * T * (8 * d + 2 * d + d + 4 * d + 8 * d + d) + 20 * M * T

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        M, T, d = t.M, t.seq_len, t.d
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Read: H + k + v (3*M*T*d) + weights (2*d + 4*d)
        # Write: Y + vhat (2*M*T*d) + alpha + rrms*3 (4*M*T * 4bytes)
        return (5 * M * T * d) * elem + 4 * M * T * 4 + 6 * d * elem


_ENGRAM_GATE_CONV_FWD_BENCH_PARAMS = [
    pytest.param(1, 32, 256, torch.float16, True, id="fp16-small"),
    pytest.param(2, 64, 512, torch.float16, True, id="fp16-mainstream"),
    pytest.param(1, 128, 256, torch.bfloat16, True, id="bf16-long-seq"),
    pytest.param(2, 16, 256, torch.bfloat16, True, id="bf16-batched"),
]


@pytest.mark.parametrize("M, seq_len, d, dtype, tune", _ENGRAM_GATE_CONV_FWD_BENCH_PARAMS)
def test_engram_gate_conv_fwd_bench(M, seq_len, d, dtype, tune):
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    bm = EngramGateConvFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = EngramGateConvFwdOp(M, seq_len, d, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(*args):
        return ref_engram_gate_conv_fwd(*args)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
