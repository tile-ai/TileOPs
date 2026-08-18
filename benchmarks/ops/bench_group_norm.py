import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.group_norm import GroupNormFwdOp
from workloads.normalization import GroupNormWorkload

_OP_NAME = "GroupNormFwdOp"


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


_WORKLOADS = load_workloads(_OP_NAME)
_AFFINE_PARAMS = _build_params(
    [w for w in _WORKLOADS if "weight_shape" in w]
)
_NO_AFFINE_PARAMS = _build_params(
    [w for w in _WORKLOADS if "weight_shape" not in w]
)


@pytest.mark.parametrize("n, c, spatial, num_groups, dtype, tune",
                         _AFFINE_PARAMS)
def test_group_norm_bench(n: int, c: int, spatial: tuple, num_groups: int,
                          dtype: torch.dtype, tune: bool) -> None:
    test = GroupNormWorkload(n, c, spatial, num_groups, dtype)
    x, weight, bias = test.gen_inputs()

    op = GroupNormFwdOp(num_groups=num_groups, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    # Baseline: torch.nn.functional.group_norm
    def baseline_fn(x, weight, bias):
        return F.group_norm(x, num_groups, weight=weight, bias=bias, eps=1e-5)

    bm.compare({"tileops": op, "torch": baseline_fn}, x, weight, bias, record_as=op, params=locals())


@pytest.mark.parametrize("n, c, spatial, num_groups, dtype, tune",
                         _NO_AFFINE_PARAMS)
def test_group_norm_no_affine_bench(n: int, c: int, spatial: tuple,
                                    num_groups: int, dtype: torch.dtype,
                                    tune: bool) -> None:
    test = GroupNormWorkload(n, c, spatial, num_groups, dtype)
    x, _, _ = test.gen_inputs()

    op = GroupNormFwdOp(num_groups=num_groups, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    def baseline_no_affine(x):
        return F.group_norm(x, num_groups, weight=None, bias=None, eps=1e-5)

    bm.compare({"tileops": op, "torch": baseline_no_affine}, x, record_as=op, params=locals())
