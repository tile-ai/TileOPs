"""Benchmarks for InstanceNormFwdOp.

flag_gems ships no instance_norm, so the tag goes to its ``group_norm`` with one
group per channel, which computes the same thing. torch eager and inductor complete
the row, except where running statistics are passed.
"""

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
from tileops.ops.norm.instance_norm import InstanceNormFwdOp
from workloads.normalization import RunningStatsCall


@pytest.mark.parametrize("call", manifest_calls(InstanceNormFwdOp))
def test_instance_norm_bench(call) -> None:
    workload = RunningStatsCall(call)
    inputs = workload.gen_inputs()
    x = inputs[0]
    op = InstanceNormFwdOp(**workload.arguments(), tune=True)
    use_input_stats, momentum, eps = (
        call.params[k] for k in ("use_input_stats", "momentum", "eps")
    )

    def baseline_fn(x, running_mean, running_var, weight, bias):
        if running_mean is not None:
            running_mean, running_var = running_mean.clone(), running_var.clone()
        return F.instance_norm(
            x, running_mean, running_var, weight, bias, use_input_stats, momentum, eps
        )

    tolerance = reference_tolerance(x.dtype)
    reference_inputs = tuple(t if t is None else t.clone() for t in inputs)
    assert_matches_reference(op, baseline_fn, *reference_inputs, **tolerance)
    functors = {"tileops": op}
    # One group per channel is instance norm by the input's statistics; flag_gems' group
    # norm neither reads nor writes running statistics, so only such a row carries it.
    if use_input_stats and inputs[1] is None:
        n, c, *spatial = x.shape
        group_norm_fn = flaggems_group_norm(n, c, math.prod(spatial), c, eps)

        def flaggems_fn(x, running_mean, running_var, weight, bias):
            return group_norm_fn(x, weight, bias)

        assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **tolerance)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)
