"""Benchmark: TileOPs Gated DeltaNet inference prefill.

FLA is required, not optional: this file exists to compare against
chunk_gated_delta_rule, and the reference recurrence is not a comparison worth
recording -- it spends minutes or OOMs on the long-context Qwen rows.

The benchmark measures the serving-oriented BTHD layout because that is the
production fast path used by FLA/Qwen-style inference prefill.
"""

import functools
import inspect
from typing import Any, Sequence

import pytest
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from benchmarks.benchmark_base import (
    ManifestBenchmark,
    then_dtype,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import GatedDeltaNetPrefillBHTDFwdOp, GatedDeltaNetPrefillBTHDFwdOp
from workloads.linear_attention import GatedDeltaNetPrefillFwdWorkload


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
    layout: str = "bthd",
) -> tuple[int, int, int, int, int, int, str]:
    """Constructor arguments for one workload row, read in *layout*'s order."""
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


_BENCH_PARAMS = workload_params(
    load_workloads(GatedDeltaNetPrefillBTHDFwdOp), then_dtype(_gdn_prefill_args, tune=False)
)
_BHTD_BENCH_PARAMS = workload_params(
    load_workloads(GatedDeltaNetPrefillBHTDFwdOp),
    then_dtype(
        functools.partial(_gdn_prefill_args, layout="bhtd"),
        tune=False,
    ),
)


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, layout, dtype, tune",
    _BHTD_BENCH_PARAMS,
)
def test_gated_deltanet_prefill_bhtd_bench(
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
    """Head-major prefill against FLA in its own token-major layout.

    FLA reads token-major only, so its inputs are converted outside the timed region:
    each row compares the two kernels in the layout each was written for. A head-major
    caller reaching for FLA also pays that conversion, which this row does not report.

    Neither tag is asserted, for the reason this module's docstring gives.
    """
    test = GatedDeltaNetPrefillFwdWorkload(
        batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, layout=layout
    )
    inputs = test.gen_inputs()

    op = GatedDeltaNetPrefillBHTDFwdOp(chunk_size=chunk_size, tune=tune)
    bm = ManifestBenchmark(op, test)
    fla_inputs = convert_gdn_prefill_layout(inputs, layout, "bthd")
    bm.compare({"tileops": op, "fla": (_fla_prefill_fwd(), fla_inputs)}, *inputs)


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

    op = GatedDeltaNetPrefillBTHDFwdOp(chunk_size=chunk_size, tune=tune)
    bm = ManifestBenchmark(op, test)
    fla_inputs = convert_gdn_prefill_layout(inputs, layout, "bthd")
    functors = {"tileops": op, "fla": (fla_fn, fla_inputs)}

    bm.compare(functors, *inputs)
