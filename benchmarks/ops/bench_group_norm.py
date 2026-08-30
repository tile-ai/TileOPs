"""Benchmarks for GroupNormFwdOp, affine and not, against flag_gems and torch."""

import math

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_group_norm,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops.norm.group_norm import GroupNormFwdOp
from workloads.normalization import GroupNormWorkload


def _group_norm_args(w: dict, dtype: torch.dtype) -> tuple:
    n, c, *spatial = w["x_shape"]
    if "num_groups" not in w:
        raise KeyError("Workload manifest must contain 'num_groups'")
    return (n, c, tuple(spatial), w["num_groups"], dtype, False)


_WORKLOADS = load_workloads(GroupNormFwdOp)
_AFFINE_PARAMS = workload_params([w for w in _WORKLOADS if "weight_shape" in w], _group_norm_args)
_NO_AFFINE_PARAMS = workload_params(
    [w for w in _WORKLOADS if "weight_shape" not in w], _group_norm_args
)


@pytest.mark.parametrize("n, c, spatial, num_groups, dtype, tune", _AFFINE_PARAMS)
def test_group_norm_bench(
    n: int, c: int, spatial: tuple, num_groups: int, dtype: torch.dtype, tune: bool
) -> None:
    test = GroupNormWorkload(n, c, spatial, num_groups, dtype)
    x, weight, bias = test.gen_inputs()

    op = GroupNormFwdOp(num_groups=num_groups, tune=tune)
    bm = ManifestBenchmark(op, test)

    # Baseline: torch.nn.functional.group_norm
    def baseline_fn(x, weight, bias):
        return F.group_norm(x, num_groups, weight=weight, bias=bias, eps=1e-5)

    flaggems_fn = flaggems_group_norm(n, c, math.prod(spatial), num_groups, 1e-5)
    assert_matches_reference(
        flaggems_fn, baseline_fn, x, weight, bias, **reference_tolerance(dtype)
    )

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_fn,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
        weight,
        bias,
    )


@pytest.mark.parametrize("n, c, spatial, num_groups, dtype, tune", _NO_AFFINE_PARAMS)
def test_group_norm_no_affine_bench(
    n: int, c: int, spatial: tuple, num_groups: int, dtype: torch.dtype, tune: bool
) -> None:
    test = GroupNormWorkload(n, c, spatial, num_groups, dtype)
    x, _, _ = test.gen_inputs()

    op = GroupNormFwdOp(num_groups=num_groups, tune=tune)
    bm = ManifestBenchmark(op, test)

    def baseline_no_affine(x):
        return F.group_norm(x, num_groups, weight=None, bias=None, eps=1e-5)

    flaggems_fn = flaggems_group_norm(n, c, math.prod(spatial), num_groups, 1e-5)
    assert_matches_reference(flaggems_fn, baseline_no_affine, x, **reference_tolerance(dtype))

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_fn,
            "torch": baseline_no_affine,
            TORCH_COMPILE_TAG: compiled_reference(baseline_no_affine),
        },
        x,
    )
