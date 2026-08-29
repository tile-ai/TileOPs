"""Benchmarks for InstanceNormFwdOp.

flag_gems ships no instance_norm, so the tag goes to its ``group_norm`` with one
group per channel, which computes the same thing. torch eager and inductor complete
the row.
"""

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
from tileops.ops.norm.instance_norm import InstanceNormFwdOp
from workloads.normalization import InstanceNormWorkload

_OP_NAME = "InstanceNormFwdOp"


def _instance_norm_args(w: dict, dtype: torch.dtype) -> tuple:
    """``(n, c, spatial, dtype, tune, affine)``; a row is affine exactly when it
    declares ``weight_shape``."""
    n, c, *spatial = w["x_shape"]
    return (n, c, tuple(spatial), dtype, True, "weight_shape" in w)


@pytest.mark.parametrize(
    "n, c, spatial, dtype, tune, affine",
    workload_params(load_workloads(_OP_NAME), _instance_norm_args),
)
def test_instance_norm_bench(
    n: int, c: int, spatial: tuple, dtype: torch.dtype, tune: bool, affine: bool
) -> None:
    test = InstanceNormWorkload(n, c, spatial, dtype)
    x, _, _, weight, bias = test.gen_inputs()
    if not affine:
        weight = bias = None

    op = InstanceNormFwdOp(tune=tune)
    bm = ManifestBenchmark(op, test)

    # Baseline: torch.nn.functional.instance_norm
    def baseline_fn(x, running_mean, running_var, weight, bias):
        return F.instance_norm(x, weight=weight, bias=bias, eps=1e-5)

    # One group per channel is instance norm.
    group_norm_fn = flaggems_group_norm(n, c, math.prod(spatial), c, 1e-5)

    def flaggems_fn(x, running_mean, running_var, weight, bias):
        return group_norm_fn(x, weight, bias)

    assert_matches_reference(
        flaggems_fn, baseline_fn, x, None, None, weight, bias, **reference_tolerance(dtype)
    )

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_fn,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
        None,
        None,
        weight,
        bias,
        record_as=op,
        params=locals(),
    )
