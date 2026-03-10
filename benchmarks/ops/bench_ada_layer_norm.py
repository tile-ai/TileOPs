from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ada_layer_norm import AdaLayerNormFixture, AdaLayerNormTest
from tests.ops.test_ada_layer_norm_zero import AdaLayerNormZeroFixture, AdaLayerNormZeroTest
from tileops.ops.norm.ada_layer_norm import AdaLayerNormOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroOp


class AdaLayerNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N for mean, N for variance, N for normalize, N for scale, N for shift
        # Simplified: ~5N flops per row
        return 5 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only. Read x + read scale + read shift + write y."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read scale (M*N) + read shift (M*N) + write y (M*N) = 4*M*N
        return 4 * t.m * t.n * elem_bytes


class AdaLayerNormZeroBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N for mean, N for variance, N for normalize, N for scale, N for shift, N for gate
        # Simplified: ~6N flops per row
        return 6 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only. Read x + read scale + read shift + read gate + write y."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read scale (M*N) + read shift (M*N) + read gate (M*N) + write y (M*N) = 5*M*N
        return 5 * t.m * t.n * elem_bytes


@AdaLayerNormFixture
def test_ada_layer_norm_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    bm = AdaLayerNormBenchmark(test)
    inputs = test.gen_inputs()

    op = AdaLayerNormOp(M=m, N=n, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("ada_layer_norm", locals(), result, tag="tileops")

    # Baseline: PyTorch composite F.layer_norm + arithmetic
    def baseline_fn(x, scale, shift):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=1e-5)
        return scale * normed + shift

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("ada_layer_norm", locals(), result_bl, tag="baseline")


@AdaLayerNormZeroFixture
def test_ada_layer_norm_zero_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    bm = AdaLayerNormZeroBenchmark(test)
    inputs = test.gen_inputs()

    op = AdaLayerNormZeroOp(M=m, N=n, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("ada_layer_norm_zero", locals(), result, tag="tileops")

    # Baseline: PyTorch composite F.layer_norm + arithmetic + gate
    def baseline_fn(x, scale, shift, gate):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=1e-5)
        return gate * (scale * normed + shift)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("ada_layer_norm_zero", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
