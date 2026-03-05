from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_silu_and_mul import SiluAndMulFixture, SiluAndMulTest
from tileops.ops import SiluAndMulOp


class SiluAndMulBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # Per element: neg(1) + exp(1) + add(1) + div(1) + mul_gate(1) = ~5 flops
        return 5.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Useful bytes only: read x (M*N) + read gate (M*N) + write y (M*N).
        # For non-aligned N, kernel operates on padded_n elements but
        # bandwidth_tbs is reported against useful bytes (actual data moved).
        return (t.m * t.n + t.m * t.n + t.m * t.n) * t.dtype.itemsize


@SiluAndMulFixture
def test_silu_and_mul_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = SiluAndMulTest(m, n, dtype)
    bm = SiluAndMulBenchmark(test)
    inputs = test.gen_inputs()

    op = SiluAndMulOp(m, n, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("silu_and_mul", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("silu_and_mul", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
