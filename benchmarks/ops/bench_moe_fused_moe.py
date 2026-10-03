"""Benchmarks for the routed MoE FFN with and without a shared expert, and for the shared
expert alone, one case per manifest call, against vLLM and torch.
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import VLLM_TAG
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.moe import FusedMoEFwdOp, FusedMoESharedExpertFwdOp, SharedExpertMLPFwdOp
from workloads.moe import (
    FusedMoeSharedExpertWorkload,
    FusedMoeWorkload,
    SharedExpertMLPWorkload,
)

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )

    _VLLM_FUSED_EXPERTS_AVAILABLE = True
except ImportError:
    _VLLM_FUSED_EXPERTS_AVAILABLE = False

try:
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk as _vllm_fused_topk,
    )

    _VLLM_FUSED_MOE_AVAILABLE = _VLLM_FUSED_EXPERTS_AVAILABLE
except ImportError:
    _VLLM_FUSED_MOE_AVAILABLE = False

try:
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias as _vllm_fused_topk_bias,
    )

    _VLLM_SHARED_EXPERT_AVAILABLE = _VLLM_FUSED_EXPERTS_AVAILABLE
except ImportError:
    _VLLM_SHARED_EXPERT_AVAILABLE = False


# FusedMoe: Qwen3-235B-A22B (softmax), DeepSeek-V3 (sigmoid) and Kimi K2 (sigmoid with a correction
# bias, passed when a row lists it in `some`).


@pytest.mark.parametrize("call", manifest_calls(FusedMoEFwdOp))
def test_fused_moe_fwd_bench(call) -> None:
    workload = FusedMoeWorkload(call)
    inputs = workload.gen_inputs()
    hidden, gating, w_gate_up, w_down, correction_bias = inputs
    op = FusedMoEFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    torch.testing.assert_close(
        op(*inputs).float(), workload.ref_program(*inputs).float(), rtol=3e-2, atol=3e-2
    )

    functors = {"tileops": op}

    # vLLM's ``fused_topk`` has no correction_bias parameter, so routing would diverge
    # from TileOPs on a row that passes one; those rows time the reference instead.
    if _VLLM_FUSED_MOE_AVAILABLE and correction_bias is None:
        top_k, renormalize, scale = op.top_k, op.renormalize, op.routed_scaling_factor

        def _vllm_fn(hidden, gating, w_gate_up, w_down, correction_bias):
            tw, tids, _ = _vllm_fused_topk(
                hidden_states=hidden,
                gating_output=gating,
                topk=top_k,
                renormalize=renormalize,
                scoring_func=op.scoring_func,
            )
            out = _vllm_fused_experts(hidden, w_gate_up, w_down, tw, tids)
            return out * scale if scale != 1.0 else out

        functors[VLLM_TAG] = _vllm_fn
    else:
        functors["torch-ref"] = workload.ref_program

    bm.compare(functors, *inputs)


# FusedMoeSharedExpert: Kimi K2, DeepSeek-V3 and GLM-4.5 at small-route, decode and prefill token
# counts. The vllm row is fused_topk_bias + fused_experts + a shared F.linear MLP; without vLLM no
# row is recorded.


@pytest.mark.parametrize("call", manifest_calls(FusedMoESharedExpertFwdOp))
def test_fused_moe_shared_expert_bench(call) -> None:
    workload = FusedMoeSharedExpertWorkload(call)
    inputs = workload.gen_inputs()
    hidden, gating, w_gate_up, w_down, correction_bias, shared_w_gate_up, shared_w_down = inputs
    op = FusedMoESharedExpertFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    for actual, expected in zip(op(*inputs), workload.ref_program(*inputs), strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)

    functors = {"tileops": op}

    # vLLM shared expert: separate gate/up weights [Fs, H].
    if (
        _VLLM_SHARED_EXPERT_AVAILABLE
        and shared_w_gate_up is not None
        and correction_bias is not None
    ):
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

        functors[VLLM_TAG] = _vllm_fn
    else:
        # No baseline rather than a misleading one: the per-expert Python loop is a
        # correctness reference, so timing against it measures neither implementation.
        warnings.warn(
            "No baseline recorded for FusedMoESharedExpertFwdOp: vLLM is not installed, or the "
            "row is routed-only and the vLLM path here always builds a shared expert.",
            stacklevel=2,
        )

    bm.compare(functors, *inputs)


# SharedExpertMLP: the shared expert alone, against the two F.linear projections and the
# gated activation torch runs for it.


@pytest.mark.parametrize("call", manifest_calls(SharedExpertMLPFwdOp))
def test_shared_expert_mlp_bench(call) -> None:
    workload = SharedExpertMLPWorkload(call)
    inputs = workload.gen_inputs()
    op = SharedExpertMLPFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    torch.testing.assert_close(
        op(*inputs).float(), workload.ref_program(*inputs).float(), rtol=3e-2, atol=3e-2
    )

    def _torch_fn(hidden, w_gate_up, w_down):
        gate, up = F.linear(hidden, w_gate_up).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, w_down)

    bm.compare({"tileops": op, "torch-cublas": _torch_fn}, *inputs)
