from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ada_layer_norm_zero import AdaLayerNormZeroFixture, AdaLayerNormZeroTest
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroOp


class AdaLayerNormZeroBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N mean + N variance + N normalize + N scale + N shift + N gate = ~6N
        return 6 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only (not padded).

        Read x + read scale + read shift + read gate + write y.
        """
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read scale (M*N) + read shift (M*N) + read gate (M*N) + write y (M*N)
        return 5 * t.m * t.n * elem_bytes


@AdaLayerNormZeroFixture
def test_ada_layer_norm_zero_bench(
    m: int, n: int, dtype: torch.dtype, tune: bool,
) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    bm = AdaLayerNormZeroBenchmark(test)
    inputs = test.gen_inputs()

    op = AdaLayerNormZeroOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("ada_layer_norm_zero", locals(), result, tag="tileops")

    # Baseline: PyTorch composite F.layer_norm + arithmetic
    def baseline_fn(x, scale, shift, gate):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=1e-5)
        return gate * (scale * normed + shift)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("ada_layer_norm_zero", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
