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
from benchmarks.benchmark_base import ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.instance_norm import InstanceNormFwdOp
from workloads.normalization import InstanceNormWorkload

_OP_NAME = "InstanceNormFwdOp"


def _build_params(workloads):
    """One param per workload row and dtype.

    A row applies the affine exactly when it declares ``weight_shape`` (R18.1).
    """
    params = []
    for w in workloads:
        shape = w["x_shape"]
        n, c, spatial = shape[0], shape[1], tuple(shape[2:])
        label = w.get("label", f"{n}x{c}x{'x'.join(map(str, spatial))}")
        affine = "weight_shape" in w
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(n, c, spatial, dtype, True, affine, id=f"{label}-{dtype_str}")
            )
    return params


@pytest.mark.parametrize(
    "n, c, spatial, dtype, tune, affine", _build_params(load_workloads(_OP_NAME))
)
def test_instance_norm_bench(
    n: int, c: int, spatial: tuple, dtype: torch.dtype, tune: bool, affine: bool
) -> None:
    test = InstanceNormWorkload(n, c, spatial, dtype)
    x, _, _, weight, bias = test.gen_inputs()
    if not affine:
        weight = bias = None

    op = InstanceNormFwdOp(tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)

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
