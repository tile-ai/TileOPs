"""Benchmarks for GroupNormFwdOp, affine and not, against flag_gems and torch."""

import math

import pytest
import torch.nn.functional as F

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_group_norm,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.norm.group_norm import GroupNormFwdOp
from workloads.normalization import NormCall

_CALLS = manifest_calls(GroupNormFwdOp)


def _affine(param) -> bool:
    return param.values[0].specs["weight"] is not None


def _bench(call) -> None:
    workload = NormCall(call)
    x, weight, bias = inputs = workload.gen_inputs()
    op = GroupNormFwdOp(**workload.arguments())
    groups, eps = call.params["num_groups"], call.params["eps"]

    def baseline_fn(x, weight, bias):
        return F.group_norm(x, groups, weight=weight, bias=bias, eps=eps)

    n, c, *spatial = x.shape
    flaggems_fn = flaggems_group_norm(n, c, math.prod(spatial), groups, eps)
    tolerance = reference_tolerance(x.dtype)
    assert_matches_reference(op, baseline_fn, *inputs, **tolerance)
    assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **tolerance)
    ManifestBenchmark(op, workload).compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_fn,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", [p for p in _CALLS if _affine(p)])
def test_group_norm_bench(call) -> None:
    _bench(call)


@pytest.mark.parametrize("call", [p for p in _CALLS if not _affine(p)])
def test_group_norm_no_affine_bench(call) -> None:
    _bench(call)
