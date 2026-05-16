"""Benchmark for FusedMoeFwdOp / FusedMoeFwdCbFwdOp (routed MoE FFN).

Workload shapes come from each op's manifest ``workloads`` (via
``load_workloads``); the benchmark reports TileOPs latency (nopad + padded
layouts) alongside the manifest-derived roofline (``op.eval_roofline()``)
and a vLLM / torch-ref baseline.

Coverage:

  Op                       Models (manifest workloads)
  FusedMoeFwdOp            Qwen3-235B-A22B (softmax), DeepSeek-V3 (sigmoid)
  FusedMoeFwdCbFwdOp       Kimi K2 (sigmoid + correction_bias)

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_fused_moe.py -vvs
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk as _vllm_fused_topk,
    )
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.moe import FusedMoeFwdCbFwdOp, FusedMoeFwdOp, FusedTopKOp
from workloads.workload_base import WorkloadBase

_OP_NAME = "FusedMoeFwdOp"
_OP_NAME_CB = "FusedMoeFwdCbFwdOp"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class FusedMoeBenchTest(WorkloadBase):
    """Inputs for a single FusedMoe benchmark configuration."""

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str,
        renormalize: bool,
        with_correction_bias: bool,
        routed_scaling_factor: float,
        dtype: torch.dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev,
        )
        gating = torch.randn(
            self.num_tokens, self.num_experts, dtype=torch.float32, device=dev,
        )
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev) * 0.1
            if self.with_correction_bias else None
        )
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        return hidden, gating, correction_bias, w_gate_up, w_down

    def ref_program(self, *args):
        return None


class FusedMoeBenchmark(BenchmarkBase[FusedMoeBenchTest]):
    """Benchmark wrapper sourcing flops/bytes from the bound op's roofline."""

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test, op):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        cache = self._roofline_cache
        if cache is None:
            cache = self._op.eval_roofline()
            self._roofline_cache = cache
        return cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _routed_scaling_factor(w: dict) -> float:
    return float(w.get("routed_scaling_factor", 1.0))


def _renormalize(w: dict) -> bool:
    return bool(w.get("renormalize", False))


def _to_params(workloads, *, with_correction_bias: bool):
    """Convert a manifest entry's workloads to pytest params."""
    params = []
    for w in workloads:
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["num_tokens"], w["num_experts"], w["top_k"],
                w["hidden_size"], w["ffn_size"],
                w["scoring_func"], _renormalize(w),
                with_correction_bias,
                _routed_scaling_factor(w),
                dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


_FWD_PARAMS = _to_params(load_workloads(_OP_NAME), with_correction_bias=False)
_FWD_CB_PARAMS = _to_params(load_workloads(_OP_NAME_CB), with_correction_bias=True)


def _run_bench(
    op_cls,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    ffn_size: int,
    scoring_func: str,
    renormalize: bool,
    with_correction_bias: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
) -> None:
    test = FusedMoeBenchTest(
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        scoring_func, renormalize, with_correction_bias,
        routed_scaling_factor, dtype,
    )
    hidden, gating, correction_bias, w_gate_up, w_down = test.gen_inputs()

    common_kwargs = dict(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func=scoring_func, renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor, dtype=dtype,
    )

    if with_correction_bias:
        forward_args_tileops = (hidden, gating, correction_bias, w_gate_up, w_down)
    else:
        forward_args_tileops = (hidden, gating, w_gate_up, w_down)

    # -- TileOPs nopad -----------------------------------------------------
    op = op_cls(layout="nopad", **common_kwargs)
    bm = FusedMoeBenchmark(test, op)
    op(*forward_args_tileops)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, *forward_args_tileops)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # -- TileOPs padded ----------------------------------------------------
    op_pad = op_cls(layout="padded", **common_kwargs)
    op_pad(*forward_args_tileops)  # warmup
    torch.cuda.synchronize()

    result_pad = bm.profile(op_pad, *forward_args_tileops)
    BenchmarkReport.record(op_pad, locals(), result_pad, tag="tileops-padded")

    # -- vLLM baseline (optional) ------------------------------------------
    if _VLLM_AVAILABLE:
        gating_f32 = gating.float()

        def _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down):
            tw, tids, _ = _vllm_fused_topk(
                hidden_states=hidden,
                gating_output=gating_f32,
                topk=top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
            )
            out = _vllm_fused_experts(hidden, w_gate_up, w_down, tw, tids)
            if routed_scaling_factor != 1.0:
                out = out * routed_scaling_factor
            return out

        _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down)
        torch.cuda.synchronize()

        result_vllm = bm.profile(
            _vllm_fn, hidden, gating, correction_bias, w_gate_up, w_down,
        )
        BenchmarkReport.record(op, locals(), result_vllm, tag="vllm")
    else:
        # torch-ref baseline: memory-efficient per-expert GEMM loop.
        fk = FusedTopKOp(
            num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
            scoring_func=scoring_func, renormalize=renormalize,
            with_correction_bias=with_correction_bias,
        )
        topk_weights, topk_ids = fk(gating, correction_bias)
        output_buf = torch.zeros(
            num_tokens, hidden_size, dtype=torch.float32, device=hidden.device,
        )
        ids_i64 = topk_ids.to(torch.int64)

        def _ref_fn(hidden, gating, correction_bias, w_gate_up, w_down):
            E = w_gate_up.shape[0]
            ffn_dim = w_gate_up.shape[1] // 2
            output_buf.zero_()
            for e in range(E):
                mask = (ids_i64 == e)
                if not mask.any():
                    continue
                t_idx, k_idx = mask.nonzero(as_tuple=True)
                h = hidden[t_idx].float()
                gate_up = h @ w_gate_up[e].float().t()
                act = F.silu(gate_up[:, :ffn_dim]) * gate_up[:, ffn_dim:]
                down = act @ w_down[e].float().t()
                weights = topk_weights[t_idx, k_idx].float().unsqueeze(-1)
                output_buf.index_add_(0, t_idx, down * weights)
            return (output_buf * routed_scaling_factor).to(hidden.dtype)

        _ref_fn(hidden, gating, correction_bias, w_gate_up, w_down)
        torch.cuda.synchronize()

        result_ref = bm.profile(
            _ref_fn, hidden, gating, correction_bias, w_gate_up, w_down,
        )
        BenchmarkReport.record(op, locals(), result_ref, tag="torch-ref")


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
    " scoring_func, renormalize, with_correction_bias,"
    " routed_scaling_factor, dtype_str",
    _FWD_PARAMS,
)
def test_fused_moe_fwd_bench(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    scoring_func, renormalize, with_correction_bias,
    routed_scaling_factor, dtype_str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    _run_bench(
        FusedMoeFwdOp,
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        scoring_func, renormalize, with_correction_bias,
        routed_scaling_factor, dtype,
    )


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
    " scoring_func, renormalize, with_correction_bias,"
    " routed_scaling_factor, dtype_str",
    _FWD_CB_PARAMS,
)
def test_fused_moe_fwd_cb_bench(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    scoring_func, renormalize, with_correction_bias,
    routed_scaling_factor, dtype_str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    _run_bench(
        FusedMoeFwdCbFwdOp,
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        scoring_func, renormalize, with_correction_bias,
        routed_scaling_factor, dtype,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
