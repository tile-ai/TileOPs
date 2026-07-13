"""Benchmark: TileOPs Gated DeltaNet inference prefill.

When FLA is installed, record it as the independent baseline. A pure-torch
reference is kept only for small manifest fallback rows; the long-context Qwen
rows require FLA so no-FLA runs do not spend minutes or OOM inside the
reference recurrence.

The benchmark measures the serving-oriented BTHD layout because that is the
production fast path used by FLA/Qwen-style inference prefill.
"""

import inspect
from typing import Any, Sequence

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params
from benchmarks.ops.bench_gated_deltanet import (
    compute_w_u_torch,
    kernel2_gated_deltanet_torch,
    prepare_wy_repr_gated_torch,
)
from tileops.manifest import load_workloads
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.gated_deltanet import GatedDeltaNetPrefillFwdTest

_OP_NAME = "GatedDeltaNetPrefillFwdOp"
_TORCH_FALLBACK_MAX_SEQ_LEN = 4096


try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None


class _GatedDeltaNetPrefillFwdTestBaseline(GatedDeltaNetPrefillFwdTest):
    """Adds a pure-torch fallback baseline for benchmark profiling."""

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, g, beta = convert_gdn_prefill_layout(
            (q, k, v, g, beta), self.layout, "bhtd"
        )
        batch, heads, seq_len, dim_k = k.shape
        dim_v = v.shape[-1]
        chunk_size = self.chunk_size
        num_chunks = seq_len // chunk_size
        g_cum = (
            g.float()
            .reshape(batch, heads, num_chunks, chunk_size)
            .cumsum(-1)
            .reshape(batch, heads, seq_len)
            .to(g.dtype)
        )
        aw, au = prepare_wy_repr_gated_torch(k, g_cum, beta, chunk_size)
        w, u = compute_w_u_torch(aw, au, k, v, beta, chunk_size)
        initial_state = torch.zeros(
            batch, heads, dim_k, dim_v, dtype=torch.float32, device=q.device
        )
        final_state, o = kernel2_gated_deltanet_torch(
            q, k, g_cum, w, u, initial_state, chunk_size
        )
        (o,) = convert_gdn_prefill_layout((o.to(self.dtype),), "bhtd", self.layout)
        return o, final_state.to(self.dtype)


def _fla_prefill_fwd():
    """Return the FLA prefill baseline callable, or None if unavailable."""
    if chunk_gated_delta_rule is None:
        return None

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
    if layout == "bhsd":
        return "bhtd"
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
    elif layout in ("bhtd", "bhsd"):
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


def _can_use_torch_fallback(seq_len: int) -> bool:
    return seq_len <= _TORCH_FALLBACK_MAX_SEQ_LEN


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
    if fla_fn is None and not _can_use_torch_fallback(seq_len):
        pytest.skip(
            "FLA is required for long-context GDN prefill benchmark rows; "
            "the pure-torch fallback is capped at "
            f"S <= {_TORCH_FALLBACK_MAX_SEQ_LEN}"
        )

    test = _GatedDeltaNetPrefillFwdTestBaseline(
        batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, layout=layout
    )
    inputs = test.gen_inputs()

    op = GatedDeltaNetPrefillFwdOp(
        batch,
        heads,
        seq_len,
        dim_k,
        dim_v,
        chunk_size,
        dtype,
        tune=tune,
        layout=layout,
    )
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)

    if fla_fn is not None:
        fla_inputs = convert_gdn_prefill_layout(inputs, layout, "bthd")
        result_fla = bm.profile(fla_fn, *fla_inputs)
        result["speedup_vs_fla"] = result_fla["latency_ms"] / result["latency_ms"]
        BenchmarkReport.record(op, locals(), result, tag="tileops")
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        BenchmarkReport.record(op, locals(), result, tag="tileops")
        result_ref = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_ref, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
