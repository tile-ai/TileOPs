from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ada_layer_norm import AdaLayerNormFixture, AdaLayerNormTest
from tileops.ops.norm.ada_layer_norm import AdaLayerNormOp


class AdaLayerNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N mean + N variance + N normalize + N scale + N shift = ~5N per row
        return 5 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only (not padded).

        Read x + read scale + read shift + write y.
        """
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read scale (M*N) + read shift (M*N) + write y (M*N)
        return 4 * t.m * t.n * elem_bytes


@AdaLayerNormFixture
def test_ada_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    bm = AdaLayerNormBenchmark(test)
    inputs = test.gen_inputs()

    op = AdaLayerNormOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("ada_layer_norm", locals(), result, tag="tileops")

    # Baseline: PyTorch composite F.layer_norm + arithmetic
    def baseline_fn(x, scale, shift):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=1e-5)
        return scale * normed + shift

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("ada_layer_norm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
