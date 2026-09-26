"""Benchmark for FusedMoeSharedExpertFwdOp — FusedMoE with shared expert support.

Workload shapes come from the op's manifest ``workloads`` (via
``manifest_calls``); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``) and a vLLM baseline.

Coverage: Kimi K2, DeepSeek-V3 and GLM-4.5, each at a small-route, a decode and
a prefill token count — all three route with sigmoid and a correction bias.

Baselines:
  - vllm: fused_topk_bias + fused_experts + F.linear shared MLP. Absent without
    vLLM installed -- no row is recorded rather than a slower stand-in.
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias as _vllm_fused_topk_bias,
    )

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.moe import FusedMoeSharedExpertFwdOp
from workloads.moe import FusedMoeSharedExpertWorkload


@pytest.mark.parametrize("call", manifest_calls(FusedMoeSharedExpertFwdOp))
def test_fused_moe_shared_expert_bench(call) -> None:
    test = FusedMoeSharedExpertWorkload(call)
    inputs = test.gen_inputs()
    hidden, gating, w_gate_up, w_down, correction_bias, shared_w_gate_up, shared_w_down = inputs
    op = FusedMoeSharedExpertFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, test)
    for actual, expected in zip(op(*inputs), test.ref_program(*inputs), strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)

    functors = {"tileops": op}

    # vLLM shared expert: separate gate/up weights [Fs, H].
    if _VLLM_AVAILABLE and shared_w_gate_up is not None and correction_bias is not None:
        ffn = shared_w_down.shape[1]
        sw_gate, sw_up = shared_w_gate_up[:ffn], shared_w_gate_up[ffn:]
        top_k, renormalize = op.top_k, op.renormalize
        scoring_func, scale = op.scoring_func, op.routed_scaling_factor

        def _vllm_fn(
            hidden, gating, w_gate_up, w_down, correction_bias, shared_w_gate_up, shared_w_down
        ):
            tw, tids = _vllm_fused_topk_bias(
                hidden_states=hidden,
                gating_output=gating,
                scoring_func=scoring_func,
                e_score_correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                routed_scaling_factor=scale,
            )
            routed_out = _vllm_fused_experts(hidden, w_gate_up, w_down, tw, tids)
            act = F.silu(F.linear(hidden, sw_gate)) * F.linear(hidden, sw_up)
            return F.linear(act, shared_w_down), routed_out

        functors["vllm"] = _vllm_fn
    else:
        # No baseline rather than a misleading one: the per-expert Python loop is a
        # correctness reference, so timing against it measures neither implementation.
        warnings.warn(
            "No baseline recorded for FusedMoeSharedExpertFwdOp: vLLM is not installed, or the "
            "row is routed-only and the vLLM path here always builds a shared expert.",
            stacklevel=2,
        )

    bm.compare(functors, *inputs)
