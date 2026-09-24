"""Benchmark the new GLA inference path against FLA on identical inputs."""

import pytest
import torch

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.benchmark_base import ManifestBenchmark, then_dtype, workload_params
from tileops.manifest import load_workloads
from tileops.ops import GLAInferenceFwdOp
from workloads.linear_attention import GLAInferenceWorkload


def _gla_inference_args(workload: dict) -> tuple[int, int, int, int, int, bool]:
    batch, seq_len, heads, dim_k = workload["q_shape"]
    return (
        batch,
        seq_len,
        heads,
        dim_k,
        workload["v_shape"][-1],
        "initial_state_shape" in workload,
    )


@pytest.mark.parametrize(
    "batch, seq_len, heads, dim_k, dim_v, has_initial_state, dtype, tune",
    workload_params(load_workloads(GLAInferenceFwdOp), then_dtype(_gla_inference_args, tune=False)),
)
def test_gla_inference_dense_prefill_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    has_initial_state: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    del tune
    workload = GLAInferenceWorkload(batch, seq_len, heads, dim_k, dim_v, dtype, has_initial_state)
    inputs = workload.gen_inputs()
    op = GLAInferenceFwdOp()
    assert_matches_reference(op, workload.ref_program, *inputs, **reference_tolerance(dtype))
    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": workload.ref_program}, *inputs)
