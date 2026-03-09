from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_fused_add_rmsnorm import FusedAddRmsNormFixture, FusedAddRmsNormTest
from tileops.ops.norm.fused_add_rmsnorm import FusedAddRmsNormOp


class FusedAddRmsNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N for add, N squares, (N-1) adds for sum, 1 div + 1 add + 1 rsqrt,
        # N muls (h*rrms), N muls (weight)
        # Simplified: ~5N flops per row
        return 5 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only (not padded).
        Read x + read residual + read weight + write y + write pre_norm.
        """
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read residual (M*N) + read weight (N)
        # + write y (M*N) + write pre_norm (M*N)
        return (4 * t.m * t.n + t.n) * elem_bytes


@FusedAddRmsNormFixture
def test_fused_add_rmsnorm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddRmsNormTest(m, n, dtype)
    bm = FusedAddRmsNormBenchmark(test)
    inputs = test.gen_inputs()

    op = FusedAddRmsNormOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("fused_add_rmsnorm", locals(), result, tag="tileops")

    # Baseline: PyTorch add + manual rmsnorm composite
    def baseline_fn(x, residual, weight):
        h = x + residual
        rms = torch.sqrt(h.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        y = ((h.float() / rms) * weight.float()).to(x.dtype)
        return y, h

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("fused_add_rmsnorm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
