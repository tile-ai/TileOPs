from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gelu_and_mul import (
    GeluAndMulErfFixture, GeluAndMulTanhFixture, GeluAndMulTest,
)
from tileops.ops import GeluAndMulOp


class GeluAndMulBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # gelu_tanh: ~16 flops + mul_gate(1) = ~17
        # gelu_erf: ~12 flops + mul_gate(1) = ~13
        # Use ~15 as average
        return 15.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Read x (M*N) + read gate (M*N) + write y (M*N)
        return (t.m * t.n + t.m * t.n + t.m * t.n) * t.dtype.itemsize


@GeluAndMulTanhFixture
def test_gelu_and_mul_tanh_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluAndMulTest(m, n, dtype, approximate='tanh')
    bm = GeluAndMulBenchmark(test)
    inputs = test.gen_inputs()

    op = GeluAndMulOp(m, n, dtype=dtype, approximate='tanh')
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gelu_and_mul_tanh", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gelu_and_mul_tanh", locals(), result_bl, tag="baseline")


@GeluAndMulErfFixture
def test_gelu_and_mul_erf_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluAndMulTest(m, n, dtype, approximate='none')
    bm = GeluAndMulBenchmark(test)
    inputs = test.gen_inputs()

    op = GeluAndMulOp(m, n, dtype=dtype, approximate='none')
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gelu_and_mul_erf", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gelu_and_mul_erf", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
