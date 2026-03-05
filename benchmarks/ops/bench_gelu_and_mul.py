from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gelu_and_mul import (
    GeluAndMulErfFixture, GeluAndMulTanhFixture, GeluAndMulTest,
)
from tileops.ops import GeluAndMulOp


class GeluAndMulTanhBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # gelu_tanh per element: x^3(2 mul) + coeff_mul(1) + add(1) +
        # sqrt_mul(1) + tanh(~8) + add_1(1) + mul_half(1) + mul_x(1) +
        # mul_gate(1) = ~17 flops
        return 17.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Useful bytes only: read x (M*N) + read gate (M*N) + write y (M*N).
        # For non-aligned N, kernel operates on padded_n elements but
        # bandwidth_tbs is reported against useful bytes (actual data moved).
        return (t.m * t.n + t.m * t.n + t.m * t.n) * t.dtype.itemsize


class GeluAndMulErfBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # gelu_erf per element: div_sqrt2(1) + erf(~8) + add_1(1) +
        # mul_half(1) + mul_x(1) + mul_gate(1) = ~13 flops
        return 13.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Useful bytes only (see GeluAndMulTanhBenchmark comment).
        return (t.m * t.n + t.m * t.n + t.m * t.n) * t.dtype.itemsize


@GeluAndMulTanhFixture
def test_gelu_and_mul_tanh_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluAndMulTest(m, n, dtype, approximate='tanh')
    bm = GeluAndMulTanhBenchmark(test)
    inputs = test.gen_inputs()

    op = GeluAndMulOp(m, n, dtype=dtype, approximate='tanh')
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gelu_and_mul_tanh", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gelu_and_mul_tanh", locals(), result_bl, tag="baseline")


@GeluAndMulErfFixture
def test_gelu_and_mul_erf_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluAndMulTest(m, n, dtype, approximate='none')
    bm = GeluAndMulErfBenchmark(test)
    inputs = test.gen_inputs()

    op = GeluAndMulOp(m, n, dtype=dtype, approximate='none')
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gelu_and_mul_erf", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gelu_and_mul_erf", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
