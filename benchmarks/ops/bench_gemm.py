from typing import Optional

import torch
import pytest

from tests.ops.test_gemm import GemmFixture, GemmTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GemmOp


class GemmBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return 2.0 * self.test.m * self.test.n * self.test.k

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return (t.m * t.k + t.k * t.n + t.m * t.n) * t.dtype.itemsize


@GemmFixture
def test_gemm_bench(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool,
                    tune: bool) -> None:
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    bm = GemmBenchmark(test)
    inputs = test.gen_inputs()

    op = GemmOp(m, n, k, trans_a=trans_a, trans_b=trans_b, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gemm", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gemm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
