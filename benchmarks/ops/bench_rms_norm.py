from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_rms_norm import RmsNormTest
from tileops.ops.norm.rms_norm import RmsNormOp


class RmsNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N squares + (N-1) adds for sum + 1 div + 1 add + 1 rsqrt + N muls (x*rrms) + N muls (weight)
        # Simplified: ~4N flops per row
        return 4 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only (not padded). Read x + read weight + write y."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read weight (N, broadcast) + write y (M*N)
        return (2 * t.m * t.n + t.n) * elem_bytes


_RMS_NORM_BENCH_PARAMS = [
    pytest.param(1024, 4096, torch.float16, True, id="mainstream-fp16"),
    pytest.param(4096, 4096, torch.bfloat16, True, id="throughput-bf16"),
    pytest.param(2048, 5120, torch.float16, True, id="non-power-of-two"),
    pytest.param(1025, 4096, torch.float16, True, id="tail-m"),
]


@pytest.mark.parametrize("m, n, dtype, tune", _RMS_NORM_BENCH_PARAMS)
def test_rms_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RmsNormTest(m, n, dtype)
    bm = RmsNormBenchmark(test)
    inputs = test.gen_inputs()

    op = RmsNormOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
