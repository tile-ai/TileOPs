from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_engram_bwd import (
    EngramGateConvBwdTest,
)
from tileops.ops.engram_bwd import EngramGateConvBwdOp


class EngramGateConvBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        M, T, d = t.M, t.seq_len, t.d
        fwd_flops = M * T * (8 * d + 2 * d + d + 4 * d + 8 * d + d) + 20 * M * T
        return int(fwd_flops * 2.5)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        M, T, d = t.M, t.seq_len, t.d
        elem = torch.tensor([], dtype=t.dtype).element_size()
        read_bytes = 5 * M * T * d * elem + 6 * d * elem + 4 * M * T * 4
        write_bytes = 3 * M * T * d * elem + 10 * d * 4 + M * T * d * 4
        return read_bytes + write_bytes


_ENGRAM_GATE_CONV_BWD_BENCH_PARAMS = [
    pytest.param(1, 32, 256, torch.float16, True, id="fp16-small"),
    pytest.param(2, 64, 512, torch.float16, True, id="fp16-mainstream"),
    pytest.param(1, 128, 256, torch.bfloat16, True, id="bf16-long-seq"),
    pytest.param(2, 16, 256, torch.bfloat16, True, id="bf16-batched"),
]


@pytest.mark.parametrize("M, seq_len, d, dtype, tune", _ENGRAM_GATE_CONV_BWD_BENCH_PARAMS)
def test_engram_gate_conv_bwd_bench(M, seq_len, d, dtype, tune):
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    bm = EngramGateConvBwdBenchmark(test)
    inputs = test.gen_inputs()

    op = EngramGateConvBwdOp(M, seq_len, d, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("engram_gate_conv_bwd", locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
