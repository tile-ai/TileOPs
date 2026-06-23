"""Benchmark: TileOPs Gated DeltaNet inference prefill.

When FLA is installed, record it as the independent baseline. Otherwise fall
back to a pure-torch reference so every benchmark row has a non-TileOps entry.

The benchmark measures the serving-oriented BTHD layout because that is the
production fast path used by FLA/Qwen-style inference prefill.
"""

import inspect
from typing import Any

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
        q, k, v, g, beta = _to_bhtd_layout(q, k, v, g, beta, self.layout)
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
        return _from_bhtd_layout(o.to(self.dtype), final_state.to(self.dtype), self.layout)


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


def _to_bhtd_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = layout.lower()
    if layout == "bthd":
        return (
            q.permute(0, 2, 1, 3).contiguous(),
            k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous(),
            g.permute(0, 2, 1).contiguous(),
            beta.permute(0, 2, 1).contiguous(),
        )
    if layout in ("bhtd", "bhsd"):
        return q, k, v, g, beta
    raise ValueError(f"Unsupported layout: {layout}")


def _from_bhtd_layout(
    o: torch.Tensor,
    final_state: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout.lower() == "bthd":
        return o.permute(0, 2, 1, 3).contiguous(), final_state
    return o, final_state


def _to_fla_public_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = layout.lower()
    if layout == "bthd":
        return q, k, v, g, beta
    if layout in ("bhtd", "bhsd"):
        return (
            q.permute(0, 2, 1, 3).contiguous(),
            k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous(),
            g.permute(0, 2, 1).contiguous(),
            beta.permute(0, 2, 1).contiguous(),
        )
    raise ValueError(f"Unsupported layout: {layout}")


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

    fla_fn = _fla_prefill_fwd()
    if fla_fn is not None:
        fla_inputs = _to_fla_public_layout(*inputs, layout)
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
