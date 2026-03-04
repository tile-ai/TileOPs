from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_rms_norm import RmsNormFixture, RmsNormTest
from tileops.ops import RmsNormOp


class RmsNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # Per element: square(1) + add-to-sum(1) + mul-by-inv-N(1) + add-eps(1) + rsqrt(1) + mul-x(1) + mul-w(1)
        # Roughly ~5 flops per element for the reduction + 2 for the normalization
        return 7.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Read x (M*N) + read weight (N) + write y (M*N)
        # NOTE: For hidden sizes not aligned to 256, the Op pads x/weight at
        # runtime (extra copy). Memory here reflects useful bytes only;
        # real bandwidth for non-aligned shapes will be slightly higher.
        return (t.m * t.n + t.n + t.m * t.n) * t.dtype.itemsize


@RmsNormFixture
def test_rms_norm_bench(m: int, n: int, dtype: torch.dtype, eps: float) -> None:
    test = RmsNormTest(m, n, dtype, eps)
    bm = RmsNormBenchmark(test)
    inputs = test.gen_inputs()

    op = RmsNormOp(m, n, eps=eps, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("rms_norm", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("rms_norm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
