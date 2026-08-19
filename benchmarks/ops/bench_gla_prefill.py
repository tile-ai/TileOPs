"""Benchmark TileOps zero-state GLA prefill against FLA ``chunk_gla``."""

from typing import Any

import pytest
import torch
from fla.ops.gla import chunk_gla

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params
from tileops.manifest import load_workloads
from tileops.ops import GLAPrefillFwdOp
from workloads.linear_attention import GLAPrefillFwdWorkload

_OP_NAME = "GLAPrefillFwdOp"


def _gla_prefill_args(
    workload: dict[str, Any],
) -> tuple[int, int, int, int, int, int, float]:
    batch, seq_len, heads, dim_k = workload["q_shape"]
    _, v_seq_len, v_heads, dim_v = workload["v_shape"]
    if v_seq_len != seq_len or v_heads != heads:
        raise ValueError("GLA prefill q_shape and v_shape must share seq_len and heads")
    return (
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        workload.get("chunk_size", 64),
        workload.get("scale", dim_k**-0.5),
    )


_BENCH_PARAMS = manifest_params(load_workloads(_OP_NAME), _gla_prefill_args, tune=False)


@pytest.mark.parametrize(
    "batch, seq_len, heads, dim_k, dim_v, chunk_size, scale, dtype, tune",
    _BENCH_PARAMS,
)
def test_gla_prefill_fwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GLAPrefillFwdWorkload(batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype)
    inputs = test.gen_inputs()
    op = GLAPrefillFwdOp(chunk_size=chunk_size, scale=scale, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    def fla_prefill(q, k, v, g):
        return chunk_gla(
            q,
            k,
            v,
            g,
            scale=scale,
            output_final_state=True,
        )

    results = bm.compare({"tileops": op, "fla": fla_prefill}, *inputs)
    results["tileops"]["speedup_vs_fla"] = (
        results["fla"]["latency_ms"] / results["tileops"]["latency_ms"]
    )
    BenchmarkReport.record(op, locals(), results["tileops"], tag="tileops")
    BenchmarkReport.record(op, locals(), results["fla"], tag="fla")
