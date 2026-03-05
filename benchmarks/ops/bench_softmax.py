from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_softmax import SoftmaxFixture, SoftmaxTest
from tileops.ops import SoftmaxOp


class SoftmaxBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # Per element: sub-max(1) + exp(1) + add-to-sum(1) + div(1) = ~4 flops
        # Plus the max reduction (~1 flop per element)
        return 5.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Read x (M*N) + write y (M*N)
        # NOTE: For hidden sizes not aligned to 256, the Op pads x at runtime
        # (extra copy). Memory here reflects useful bytes only.
        return (t.m * t.n + t.m * t.n) * t.dtype.itemsize


@SoftmaxFixture
def test_softmax_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = SoftmaxTest(m, n, dtype)
    bm = SoftmaxBenchmark(test)
    inputs = test.gen_inputs()

    op = SoftmaxOp(m, n, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("softmax", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("softmax", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
