"""Benchmark Gated DeltaNet inference, one case per manifest call, against FLA."""

import pytest

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import GatedDeltaNetFwdOp
from workloads.linear_attention import GatedDeltaNetFwdCall


@pytest.mark.parametrize("call", manifest_calls(GatedDeltaNetFwdOp))
def test_gated_deltanet_fwd_bench(call) -> None:
    workload = GatedDeltaNetFwdCall(call)
    inputs = workload.gen_inputs()
    op = GatedDeltaNetFwdOp(**workload.arguments())
    assert_matches_reference(
        op, workload.ref_program, *inputs, **reference_tolerance(inputs[0].dtype)
    )
    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": workload.ref_program}, *inputs)
