"""Benchmarks for InstanceNormFwdOp.

flag_gems ships no instance_norm, so the tag goes to its ``group_norm`` with one
group per channel, which computes the same thing. torch eager and inductor complete
the row, except where running statistics are passed.
"""

import math

import pytest

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flaggems_group_norm,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.norm.instance_norm import InstanceNormFwdOp
from workloads.norm import RunningStatsCall


@pytest.mark.parametrize("call", manifest_calls(InstanceNormFwdOp))
def test_instance_norm_bench(call) -> None:
    workload = RunningStatsCall(call)
    inputs = workload.gen_inputs()
    x = inputs[0]
    op = InstanceNormFwdOp(**workload.arguments(), tune=True)
    use_input_stats, momentum, eps = (
        call.params[k] for k in ("use_input_stats", "momentum", "eps")
    )
    baseline_fn = workload.ref_program
    functors = {"tileops": op}
    if use_input_stats and inputs[1] is None:
        n, c, *spatial = x.shape
        group_norm_fn = flaggems_group_norm(n, c, math.prod(spatial), c, eps)

        def flaggems_fn(x, running_mean, running_var, weight, bias):
            return group_norm_fn(x, weight, bias)

        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)
