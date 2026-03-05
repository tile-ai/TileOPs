from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_layer_norm import LayerNormFixture, LayerNormTest
from tileops.ops import LayerNormOp


class LayerNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # Per element: sub-mean(1) + square(1) + add-to-sum(1) + mul-inv-N(1) + add-eps(1)
        # + rsqrt(1) + sub-mean(1) + mul-rstd(1) + mul-weight(1) + add-bias(1)
        return 10.0 * self.test.m * self.test.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Read x (M*N) + read weight (N) + read bias (N) + write y (M*N)
        # NOTE: For hidden sizes not aligned to 256, the Op pads x/weight/bias
        # at runtime (extra copy). Memory here reflects useful bytes only;
        # real bandwidth for non-aligned shapes will be slightly higher.
        return (t.m * t.n + t.n + t.n + t.m * t.n) * t.dtype.itemsize


@LayerNormFixture
def test_layer_norm_bench(m: int, n: int, dtype: torch.dtype, eps: float) -> None:
    test = LayerNormTest(m, n, dtype, eps)
    bm = LayerNormBenchmark(test)
    inputs = test.gen_inputs()

    op = LayerNormOp(m, n, eps=eps, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("layer_norm", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("layer_norm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
