"""Benchmark Gated DeltaNet inference, one case per manifest call, against FLA."""

import pytest
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import GatedDeltaNetFwdOp
from workloads.linear_attention import GatedDeltaNetFwdCall


@pytest.mark.parametrize("call", manifest_calls(GatedDeltaNetFwdOp))
def test_gated_deltanet_fwd_bench(call) -> None:
    workload = GatedDeltaNetFwdCall(call)
    inputs = workload.gen_inputs()
    q, k, v, g, beta, initial_state = inputs[:6]
    op = GatedDeltaNetFwdOp(**workload.arguments())
    # A single-token call runs FLA's recurrent kernel, a longer one its chunked kernel.
    fla_kernel = fused_recurrent_gated_delta_rule if q.shape[1] == 1 else chunk_gated_delta_rule

    def fla(*_args):
        return fla_kernel(
            q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=True
        )

    assert_matches_reference(op, fla, *inputs, **reference_tolerance(q.dtype))
    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": (fla, ())}, *inputs)
