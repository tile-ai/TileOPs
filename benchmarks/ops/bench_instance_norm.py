from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.norm.instance_norm import (
    InstanceNormFwdOp,
    InstanceNormFwdOpNoAffine,
)
from workloads.instance_norm import InstanceNormTest

_OP_NAME = "InstanceNormFwdOp"
_OP_NAME_NO_AFFINE = "InstanceNormFwdOpNoAffine"


class InstanceNormBenchmark(BenchmarkBase[InstanceNormTest]):

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


def _build_params(workloads):
    params = []
    for w in workloads:
        shape = w["x_shape"]
        n, c, spatial = shape[0], shape[1], tuple(shape[2:])
        label = w.get("label", f"{n}x{c}x{'x'.join(map(str, spatial))}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(n, c, spatial, dtype, True,
                                       id=f"{label}-{dtype_str}"))
    return params


_AFFINE_PARAMS = _build_params(load_workloads(_OP_NAME))
_NO_AFFINE_PARAMS = _build_params(load_workloads(_OP_NAME_NO_AFFINE))


@pytest.mark.parametrize("n, c, spatial, dtype, tune", _AFFINE_PARAMS)
def test_instance_norm_bench(n: int, c: int, spatial: tuple,
                             dtype: torch.dtype, tune: bool) -> None:
    test = InstanceNormTest(n, c, spatial, dtype)
    x, weight, bias = test.gen_inputs()

    op = InstanceNormFwdOp(tune=tune)
    bm = InstanceNormBenchmark(test, op)
    result = bm.profile(op, x, weight, bias)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: torch.nn.functional.instance_norm
    def baseline_fn(x, weight, bias):
        return F.instance_norm(x, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, x, weight, bias)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


@pytest.mark.parametrize("n, c, spatial, dtype, tune", _NO_AFFINE_PARAMS)
def test_instance_norm_no_affine_bench(n: int, c: int, spatial: tuple,
                                       dtype: torch.dtype, tune: bool) -> None:
    test = InstanceNormTest(n, c, spatial, dtype)
    x, _, _ = test.gen_inputs()

    op = InstanceNormFwdOpNoAffine(tune=tune)
    bm = InstanceNormBenchmark(test, op)
    # Running stats are required positional args (R16) but ignored on the
    # use_input_stats=True path; pass placeholders.
    rm = torch.zeros(c, dtype=torch.float32, device="cuda")
    rv = torch.ones(c, dtype=torch.float32, device="cuda")
    result = bm.profile(op, x, rm, rv)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_no_affine(x):
        return F.instance_norm(x, weight=None, bias=None, eps=1e-5)

    result_bl = bm.profile(baseline_no_affine, x)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
