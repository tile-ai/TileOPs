from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from workloads.ada_layer_norm import AdaLayerNormTest
from workloads.ada_layer_norm_zero import AdaLayerNormZeroTest

_ADA_OP_NAME = "AdaLayerNormFwdOp"
_ADA_ZERO_OP_NAME = "AdaLayerNormZeroFwdOp"


class AdaLayerNormBenchmark(BenchmarkBase[AdaLayerNormTest]):

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test, op):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class AdaLayerNormZeroBenchmark(BenchmarkBase[AdaLayerNormZeroTest]):

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test, op):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _manifest_params(op_name):
    params = []
    for w in load_workloads(op_name):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype,
                                       id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype", _manifest_params(_ADA_OP_NAME))
def test_ada_layer_norm_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    inputs = test.gen_inputs()

    op = AdaLayerNormFwdOp(M=m, N=n, dtype=dtype)
    bm = AdaLayerNormBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: PyTorch composite F.layer_norm + arithmetic
    def baseline_fn(x, scale, shift):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=test.eps)
        return scale * normed + shift

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


@pytest.mark.parametrize("m, n, dtype", _manifest_params(_ADA_ZERO_OP_NAME))
def test_ada_layer_norm_zero_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    inputs = test.gen_inputs()

    op = AdaLayerNormZeroFwdOp(M=m, N=n, dtype=dtype)
    bm = AdaLayerNormZeroBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: PyTorch composite F.layer_norm + arithmetic + gate
    def baseline_fn(x, scale, shift, gate):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=test.eps)
        return gate * (scale * normed + shift)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
