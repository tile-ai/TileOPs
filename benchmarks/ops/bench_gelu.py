from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gelu import GeluErfFixture, GeluTanhFixture, GeluTest
from tileops.ops import GeluOp


class GeluBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # tanh-approx: x³(2) + coeff*x³(1) + x+inner(1) + sqrt_coeff*inner(1)
        #   + tanh(~8) + 1+tanh(1) + 0.5*x*(1+tanh)(2) ≈ ~16 flops
        # exact (erf): x/sqrt2(1) + erf(~8) + 1+erf(1) + 0.5*x*(1+erf)(2) ≈ ~12
        # Use ~14 as average
        return 14.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Read x (M*N) + write y (M*N)
        return (t.m * t.n + t.m * t.n) * t.dtype.itemsize


@GeluTanhFixture
def test_gelu_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluTest(m, n, dtype, approximate='tanh')
    bm = GeluBenchmark(test)
    inputs = test.gen_inputs()

    op = GeluOp(m, n, dtype=dtype, approximate='tanh')
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gelu_tanh", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gelu_tanh", locals(), result_bl, tag="baseline")


@GeluErfFixture
def test_gelu_erf_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluTest(m, n, dtype, approximate='none')
    bm = GeluBenchmark(test)
    inputs = test.gen_inputs()

    op = GeluOp(m, n, dtype=dtype, approximate='none')
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gelu_erf", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gelu_erf", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
