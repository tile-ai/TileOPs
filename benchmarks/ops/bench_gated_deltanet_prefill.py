"""Benchmark: TileOPs Gated DeltaNet inference prefill.

FLA is required, not optional: this file exists to compare against
chunk_gated_delta_rule, and the reference recurrence is not a comparison worth
recording -- it spends minutes or OOMs on the long-context Qwen rows.

The benchmark measures the serving-oriented BTHD layout because that is the
production fast path used by FLA/Qwen-style inference prefill.
"""

import inspect
from typing import Any, Sequence

import pytest
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params
from tileops.manifest import load_workloads
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.linear_attention import GatedDeltaNetPrefillFwdWorkload

_OP_NAME = "GatedDeltaNetPrefillFwdOp"
def _fla_prefill_fwd():
    """Return the FLA prefill baseline callable."""
    signature = inspect.signature(chunk_gated_delta_rule)
    supports_output_final_state = "output_final_state" in signature.parameters

    def baseline_fn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ):
        kwargs: dict[str, Any] = {"scale": 1.0}
        if supports_output_final_state:
            kwargs["output_final_state"] = True
        return chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)

    return baseline_fn


def _normalize_gdn_prefill_layout(layout: str) -> str:
    layout = layout.lower()
    if layout in ("bhtd", "bthd"):
        return layout
    raise ValueError(f"Unsupported layout: {layout}")


def convert_gdn_prefill_layout(
    tensors: Sequence[torch.Tensor],
    src_layout: str,
    dst_layout: str,
) -> tuple[torch.Tensor, ...]:
    src_layout = _normalize_gdn_prefill_layout(src_layout)
    dst_layout = _normalize_gdn_prefill_layout(dst_layout)
    if src_layout == dst_layout:
        return tuple(tensors)
    if {src_layout, dst_layout} != {"bhtd", "bthd"}:
        raise ValueError(f"Unsupported layout conversion: {src_layout} -> {dst_layout}")

    converted = []
    for tensor in tensors:
        if tensor.ndim == 4:
            converted.append(tensor.permute(0, 2, 1, 3).contiguous())
        elif tensor.ndim == 3:
            converted.append(tensor.permute(0, 2, 1).contiguous())
        else:
            raise ValueError(
                "GDN prefill layout conversion expects 3D gate tensors or "
                f"4D sequence tensors, got {tensor.ndim}D"
            )
    return tuple(converted)


def _gdn_prefill_args(
    workload: dict[str, Any],
) -> tuple[int, int, int, int, int, int, str]:
    layout = workload.get("layout", "bthd").lower()
    if layout == "bthd":
        batch, seq_len, heads, dim_k = workload["q_shape"]
        _, v_seq_len, v_heads, dim_v = workload["v_shape"]
    elif layout == "bhtd":
        batch, heads, seq_len, dim_k = workload["q_shape"]
        _, v_heads, v_seq_len, dim_v = workload["v_shape"]
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    if v_seq_len != seq_len or v_heads != heads:
        raise ValueError("GDN prefill q_shape and v_shape must share seq_len and heads")
    return (
        batch,
        heads,
        seq_len,
        dim_k,
        dim_v,
        workload.get("chunk_size", 64),
        layout,
    )


_BENCH_PARAMS = manifest_params(load_workloads(_OP_NAME), _gdn_prefill_args, tune=False)


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, layout, dtype, tune",
    _BENCH_PARAMS,
)
def test_gated_deltanet_prefill_fwd_bench(
    batch: int,
    heads: int,
    seq_len: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    layout: str,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    fla_fn = _fla_prefill_fwd()
    test = GatedDeltaNetPrefillFwdWorkload(
        batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, layout=layout
    )
    inputs = test.gen_inputs()

    op = GatedDeltaNetPrefillFwdOp(chunk_size=chunk_size, tune=tune, layout=layout)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    fla_inputs = convert_gdn_prefill_layout(inputs, layout, "bthd")
    functors = {"tileops": op, "fla": (fla_fn, fla_inputs)}

    # Recorded by hand: the tileops row carries a derived speedup field.
    results = bm.compare(functors, *inputs)
    results["tileops"]["speedup_vs_fla"] = (
        results["fla"]["latency_ms"] / results["tileops"]["latency_ms"]
    )
    BenchmarkReport.record(op, locals(), results["tileops"], tag="tileops")
    BenchmarkReport.record(op, locals(), results["fla"], tag="fla")
