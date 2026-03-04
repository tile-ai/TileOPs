from typing import Optional

import pytest
import torch
from tests.ops.test_gemv import GemvFixture, GemvTest

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GemmOp


class GemvBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return 2.0 * self.test.n * self.test.k

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return (t.k + t.k * t.n + t.n) * t.dtype.itemsize


@GemvFixture
def test_gemv_bench(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    test = GemvTest(n, k, dtype)
    bm = GemvBenchmark(test)
    inputs = test.gen_inputs()

    op = GemmOp(1, n, k, trans_b=True, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gemv", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gemv", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
