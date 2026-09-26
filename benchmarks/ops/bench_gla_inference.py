"""Benchmark the GLA inference path, one case per manifest call, against FLA on identical inputs."""

import pytest

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import GLAInferenceFwdOp
from workloads.linear_attention import GLAInferenceCall


@pytest.mark.parametrize("call", manifest_calls(GLAInferenceFwdOp))
def test_gla_inference_dense_prefill_bench(call) -> None:
    workload = GLAInferenceCall(call)
    inputs = workload.gen_inputs()
    op = GLAInferenceFwdOp(**workload.arguments())
    tolerance = reference_tolerance(inputs[0].dtype)
    assert_matches_reference(op, workload.ref_program, *inputs, **tolerance)
    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": workload.ref_program}, *inputs)
