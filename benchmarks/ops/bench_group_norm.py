from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.norm.group_norm import (
    GroupNormFwdOp,
    GroupNormFwdOpNoAffine,
)
from workloads.group_norm import GroupNormTest

_OP_NAME = "GroupNormFwdOp"
_OP_NAME_NO_AFFINE = "GroupNormFwdOpNoAffine"


class GroupNormBenchmark(BenchmarkBase[GroupNormTest]):

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
        num_groups = w.get("num_groups")
        if num_groups is None:
            raise KeyError(
                "Workload manifest must contain 'num_groups'"
            )
        label = w.get("label", f"{n}x{c}x{'x'.join(map(str, spatial))}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(n, c, spatial, num_groups, dtype, False,
                                       id=f"{label}-{dtype_str}"))
    return params


_AFFINE_PARAMS = _build_params(load_workloads(_OP_NAME))
_NO_AFFINE_PARAMS = _build_params(load_workloads(_OP_NAME_NO_AFFINE))


@pytest.mark.parametrize("n, c, spatial, num_groups, dtype, tune",
                         _AFFINE_PARAMS)
def test_group_norm_bench(n: int, c: int, spatial: tuple, num_groups: int,
                          dtype: torch.dtype, tune: bool) -> None:
    test = GroupNormTest(n, c, spatial, num_groups, dtype)
    x, weight, bias = test.gen_inputs()

    op = GroupNormFwdOp(
        N=n, C=c, spatial=spatial, num_groups=num_groups,
        dtype=dtype, tune=tune,
    )
    bm = GroupNormBenchmark(test, op)
    result = bm.profile(op, x, weight, bias)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: torch.nn.functional.group_norm
    def baseline_fn(x, weight, bias):
        return F.group_norm(x, num_groups, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, x, weight, bias)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


@pytest.mark.parametrize("n, c, spatial, num_groups, dtype, tune",
                         _NO_AFFINE_PARAMS)
def test_group_norm_no_affine_bench(n: int, c: int, spatial: tuple,
                                    num_groups: int, dtype: torch.dtype,
                                    tune: bool) -> None:
    test = GroupNormTest(n, c, spatial, num_groups, dtype)
    x, _, _ = test.gen_inputs()

    op = GroupNormFwdOpNoAffine(
        N=n, C=c, spatial=spatial, num_groups=num_groups,
        dtype=dtype, tune=tune,
    )
    bm = GroupNormBenchmark(test, op)
    result = bm.profile(op, x)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_no_affine(x):
        return F.group_norm(x, num_groups, weight=None, bias=None, eps=1e-5)

    result_bl = bm.profile(baseline_no_affine, x)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
