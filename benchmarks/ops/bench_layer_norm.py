from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_layer_norm import LayerNormFixture, LayerNormTest
from tileops.ops.layer_norm import LayerNormOp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class LayerNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N adds (sum) + 1 div (mean) + N subs (x-mean) + N squares + (N-1) adds
        # + 1 div + 1 add + 1 rsqrt + N muls ((x-mean)*inv_std) + N muls (weight) + N adds (bias)
        # Simplified: ~6N flops per row
        return 6 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only (not padded). Read x + read weight + read bias + write y."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read weight (N, broadcast) + read bias (N, broadcast) + write y (M*N)
        return (2 * t.m * t.n + 2 * t.n) * elem_bytes


@LayerNormFixture
def test_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LayerNormTest(m, n, dtype)
    bm = LayerNormBenchmark(test)
    inputs = test.gen_inputs()

    op = LayerNormOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("layer_norm", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("layer_norm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
