from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_silu import SiluFixture, SiluTest
from tileops.ops import SiluOp


class SiluBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # Per element: neg(1) + exp(1) + add(1) + div(1) = ~4 flops
        return 4.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Read x (M*N) + write y (M*N)
        return (t.m * t.n + t.m * t.n) * t.dtype.itemsize


@SiluFixture
def test_silu_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = SiluTest(m, n, dtype)
    bm = SiluBenchmark(test)
    inputs = test.gen_inputs()

    op = SiluOp(m, n, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("silu", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("silu", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
