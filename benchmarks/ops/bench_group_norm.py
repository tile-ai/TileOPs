"""Benchmarks for GroupNormFwdOp, affine and not, against flag_gems and torch."""

import math

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flaggems_group_norm,
)
from tileops.ops.norm.group_norm import GroupNormFwdOp

_CASES = bench.cases(GroupNormFwdOp)


def _affine(case: bench.Case) -> bool:
    return "weight" in case.params


def _bench(case: bench.Case) -> None:
    x, weight, bias = case.inputs
    op = GroupNormFwdOp(**case.arguments)
    groups, eps = (case.params["num_groups"], case.params["eps"])
    baseline_fn = case.reference
    n, c, *spatial = x.shape
    flaggems_fn = flaggems_group_norm(n, c, math.prod(spatial), groups, eps)
    functors = {
        FLAGGEMS_TAG: flaggems_fn,
        "torch": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }
    bench.Runner(op, case).compare(functors)


@pytest.mark.parametrize("case", [c for c in _CASES if _affine(c)], ids=lambda case: case.id)
def test_group_norm_bench(case) -> None:
    _bench(case)


@pytest.mark.parametrize("case", [c for c in _CASES if not _affine(c)], ids=lambda case: case.id)
def test_group_norm_no_affine_bench(case) -> None:
    _bench(case)
