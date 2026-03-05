from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gemm import GemmTest
from tests.test_base import FixtureBase
from tileops.ops import GemmOp


class GemvFixture(FixtureBase):
    PARAMS = [
        ("n, k, dtype, tune", [
            (1024, 1024, torch.float16, False),
            (7168, 16384, torch.float16, True),
            (18432, 7168, torch.float16, True),
            (1024, 1024, torch.bfloat16, False),
            (7168, 16384, torch.bfloat16, True),
            (18432, 7168, torch.bfloat16, True),
        ]),
    ]


class GemvBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return 2.0 * self.test.n * self.test.k

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return (t.k + t.k * t.n + t.n) * t.dtype.itemsize


@GemvFixture
def test_gemv_bench(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    test = GemmTest(1, n, k, dtype, trans_b=True)
    bm = GemvBenchmark(test)
    inputs = test.gen_inputs()

    op = GemmOp(1, n, k, trans_b=True, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gemv", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gemv", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
