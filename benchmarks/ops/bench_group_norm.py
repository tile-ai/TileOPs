from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.norm.group_norm import GroupNormFwdOp
from workloads.group_norm import GroupNormTest

_OP_NAME = "GroupNormFwdOp"


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


def _manifest_params():
    params = []
    for w in load_workloads(_OP_NAME):
        shape = w["x_shape"]
        n, c, spatial = shape[0], shape[1], tuple(shape[2:])
        g = w.get("num_groups", w.get("groups"))
        if g is None:
            raise KeyError(
                f"Workload manifest for {_OP_NAME} must contain 'num_groups' or 'groups'"
            )
        label = w.get("label", f"{n}x{c}x{'x'.join(map(str, spatial))}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(n, c, spatial, g, dtype, True,
                                       id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("n, c, spatial, g, dtype, tune", _manifest_params())
def test_group_norm_bench(n: int, c: int, spatial: tuple, g: int,
                          dtype: torch.dtype, tune: bool) -> None:
    test = GroupNormTest(n, c, spatial, g, dtype)
    inputs = test.gen_inputs()

    op = GroupNormFwdOp(N=n, C=c, spatial=spatial, G=g, dtype=dtype, tune=tune)
    bm = GroupNormBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: torch.nn.functional.group_norm
    def baseline_fn(x, weight, bias):
        return F.group_norm(x, g, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
