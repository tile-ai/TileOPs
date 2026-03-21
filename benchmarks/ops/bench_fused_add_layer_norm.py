from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_fused_add_layer_norm import FusedAddLayerNormTest
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormOp


class FusedAddLayerNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Per row: N adds (residual), N for mean, N for variance, N for normalize,
        # N for scale, N for bias = ~6N flops per row
        return 6 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only.  Read x + residual + weight + bias + write y + residual_out."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + read residual (M*N) + read weight (N) + read bias (N)
        # + write y (M*N) + write residual_out (M*N)
        return (4 * t.m * t.n + 2 * t.n) * elem_bytes


_FUSED_ADD_LAYER_NORM_BENCH_PARAMS = [
    pytest.param(1024, 4096, torch.float16, True, id="mainstream-fp16"),
    pytest.param(4096, 4096, torch.bfloat16, True, id="throughput-bf16"),
    pytest.param(1024, 3000, torch.float16, True, id="non-power-of-two"),
    pytest.param(1025, 4096, torch.float16, True, id="tail-m"),
]


@pytest.mark.parametrize("m, n, dtype, tune", _FUSED_ADD_LAYER_NORM_BENCH_PARAMS)
def test_fused_add_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormTest(m, n, dtype)
    bm = FusedAddLayerNormBenchmark(test)
    inputs = test.gen_inputs()

    op = FusedAddLayerNormOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: add + F.layer_norm (separate ops)
    def baseline_fn(x, residual, weight, bias):
        add_result = (x.float() + residual.float()).to(x.dtype)
        return F.layer_norm(add_result, (n,), weight=weight, bias=bias, eps=test.eps), add_result

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
