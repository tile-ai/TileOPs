from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_layer_norm import LayerNormTest
from tileops.ops.norm.layer_norm import LayerNormOp


class LayerNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N for mean, N for variance, N for normalize, N for scale, N for bias
        # Simplified: ~5N flops per row
        return 5 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only (not padded). Read x + read weight + read bias + write y."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read weight (N, broadcast) + read bias (N, broadcast) + write y (M*N)
        return (2 * t.m * t.n + 2 * t.n) * elem_bytes


_LAYER_NORM_BENCH_PARAMS = [
    pytest.param(1024, 4096, torch.float16, True, id="mainstream-fp16"),
    pytest.param(4096, 4096, torch.bfloat16, True, id="throughput-bf16"),
    pytest.param(2048, 5120, torch.float16, True, id="non-power-of-two"),
    pytest.param(1025, 4096, torch.float16, True, id="tail-m"),
]


@pytest.mark.parametrize("m, n, dtype, tune", _LAYER_NORM_BENCH_PARAMS)
def test_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LayerNormTest(m, n, dtype)
    bm = LayerNormBenchmark(test)
    inputs = test.gen_inputs()

    op = LayerNormOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("layer_norm", locals(), result, tag="tileops")

    # AC-10: baseline uses torch.nn.functional.layer_norm
    def baseline_fn(x, weight, bias):
        return F.layer_norm(x, (n,), weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("layer_norm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
