"""Benchmarks for InstanceNormFwdOp.

flag_gems ships no instance_norm, so the tag goes to its ``group_norm`` with one
group per channel, which computes the same thing. torch eager and inductor complete
the row, except where running statistics are passed.
"""

import math

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flaggems_group_norm,
)
from tileops.ops.norm.instance_norm import InstanceNormFwdOp


@pytest.mark.parametrize("case", bench.cases(InstanceNormFwdOp), ids=lambda case: case.id)
def test_instance_norm_bench(case) -> None:
    inputs = case.inputs
    x = inputs[0]
    op = InstanceNormFwdOp(**case.arguments, tune=True)
    use_input_stats, momentum, eps = (
        case.params[k] for k in ("use_input_stats", "momentum", "eps")
    )
    baseline_fn = case.reference
    implementations = {}
    if use_input_stats and inputs[1] is None:
        n, c, *spatial = x.shape
        group_norm_fn = flaggems_group_norm(n, c, math.prod(spatial), c, eps)

        def flaggems_fn(x, running_mean, running_var, weight, bias):
            return group_norm_fn(x, weight, bias)

        implementations[FLAGGEMS_TAG] = flaggems_fn
    implementations["torch"] = baseline_fn
    implementations[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    bench.Runner(op, case).compare(implementations)
