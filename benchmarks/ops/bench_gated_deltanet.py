"""Benchmark the migrated equal-length Gated DeltaNet prefill region."""

from typing import Any

import pytest
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.benchmark_base import ManifestBenchmark, then_dtype, workload_params
from tileops.manifest import load_workloads
from tileops.ops import GatedDeltaNetFwdOp
from workloads.linear_attention import GatedDeltaNetFwdWorkload


def _dense_prefill_args(workload: dict[str, Any]) -> tuple[int, int, int, int]:
    batch, seq_len, heads, dim = workload["q_shape"]
    return batch, seq_len, heads, dim


_BENCH_PARAMS = workload_params(
    load_workloads(GatedDeltaNetFwdOp),
    then_dtype(_dense_prefill_args),
    smoke_first=True,
)


@pytest.mark.parametrize("batch, seq_len, heads, dim, dtype", _BENCH_PARAMS)
def test_gated_deltanet_dense_prefill_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
) -> None:
    workload = GatedDeltaNetFwdWorkload(batch, seq_len, heads, dim, dtype)
    inputs = workload.gen_inputs()
    op = GatedDeltaNetFwdOp()

    def fla(*args):
        return chunk_gated_delta_rule(*args, output_final_state=True)

    assert_matches_reference(op, fla, *inputs, **reference_tolerance(dtype))
    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": fla}, *inputs)
