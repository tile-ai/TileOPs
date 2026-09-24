"""Benchmark migrated equal-length Gated DeltaNet inference paths."""

from typing import Any

import pytest
import torch
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.benchmark_base import ManifestBenchmark, then_dtype, workload_params
from tileops.manifest import load_workloads
from tileops.ops import GatedDeltaNetFwdOp
from workloads.linear_attention import GatedDeltaNetFwdWorkload


def _dense_args(workload: dict[str, Any]) -> tuple[int, int, int, int, bool]:
    batch, seq_len, heads, dim = workload["q_shape"]
    return batch, seq_len, heads, dim, "initial_state_shape" in workload


_BENCH_PARAMS = workload_params(
    load_workloads(GatedDeltaNetFwdOp),
    then_dtype(_dense_args),
    smoke_first=True,
)


@pytest.mark.parametrize("batch, seq_len, heads, dim, has_initial_state, dtype", _BENCH_PARAMS)
def test_gated_deltanet_fwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    has_initial_state: bool,
    dtype: torch.dtype,
) -> None:
    workload = GatedDeltaNetFwdWorkload(
        batch,
        seq_len,
        heads,
        dim,
        dtype,
        has_initial_state=has_initial_state,
    )
    inputs = workload.gen_inputs()
    op = GatedDeltaNetFwdOp()

    if seq_len == 1:
        q, k, v, g, beta, initial_state = inputs

        def fla():
            return fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
            )

        baseline = (fla, ())

        def reference(*_args):
            return fla()
    else:

        def fla(*args):
            return chunk_gated_delta_rule(*args, output_final_state=True)

        baseline = fla
        reference = fla

    assert_matches_reference(op, reference, *inputs, **reference_tolerance(dtype))
    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": baseline}, *inputs)
