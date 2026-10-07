"""Routed and shared expert MLPs against vLLM, FlashInfer CUTLASS and QuACK."""

import pytest
import torch.nn.functional as F

from benchmarks import api as bench
from benchmarks.baselines import QUACK_TAG, VLLM_TAG, quack_op, vllm_op
from benchmarks.moe_baselines import flashinfer_experts
from tileops.ops.moe import FusedMoEFwdOp, FusedMoESharedExpertFwdOp, SharedExpertMLPFwdOp


def _routed_moe(op, experts):
    topk = vllm_op("fused_topk", "model_executor.layers.fused_moe.router.fused_topk_router")
    topk_bias = vllm_op(
        "fused_topk_bias", "model_executor.layers.fused_moe.router.fused_topk_bias_router"
    )

    def run(hidden, gating, w1, w2, correction_bias):
        if correction_bias is None:
            weights, ids, _ = topk(
                hidden_states=hidden,
                gating_output=gating,
                topk=op.top_k,
                renormalize=op.renormalize,
                scoring_func=op.scoring_func,
            )
        else:
            weights, ids = topk_bias(
                hidden_states=hidden,
                gating_output=gating,
                topk=op.top_k,
                renormalize=op.renormalize,
                scoring_func=op.scoring_func,
                e_score_correction_bias=correction_bias,
                routed_scaling_factor=1.0,
            )
        output = experts(hidden, w1, w2, weights, ids)
        return output * op.routed_scaling_factor if op.routed_scaling_factor != 1.0 else output

    return run


def _expert_backends(hidden, w1, w2, top_k):
    return {
        VLLM_TAG: vllm_op("fused_experts", "model_executor.layers.fused_moe.fused_moe"),
        "flashinfer-cutlass": flashinfer_experts(hidden, w1, w2, top_k),
    }


@pytest.mark.parametrize("case", bench.cases(FusedMoEFwdOp), ids=lambda case: case.id)
def test_fused_moe_fwd_bench(case) -> None:
    op = FusedMoEFwdOp(**case.arguments)
    hidden, _, w1, w2, _ = case.inputs
    bench.Runner(op, case).compare(
        {tag: _routed_moe(op, fn) for tag, fn in _expert_backends(hidden, w1, w2, op.top_k).items()}
    )


def _with_shared_expert(routed):
    def run(hidden, gating, w1, w2, bias, shared_w1, shared_w2):
        output = routed(hidden, gating, w1, w2, bias)
        shared = None
        if shared_w1 is not None:
            gate, up = F.linear(hidden, shared_w1).chunk(2, -1)
            shared = F.linear(F.silu(gate) * up, shared_w2)
        return shared, output

    return run


@pytest.mark.parametrize("case", bench.cases(FusedMoESharedExpertFwdOp), ids=lambda case: case.id)
def test_fused_moe_shared_expert_bench(case) -> None:
    op = FusedMoESharedExpertFwdOp(**case.arguments)
    hidden, _, w1, w2, *_ = case.inputs
    bench.Runner(op, case).compare(
        {
            tag: _with_shared_expert(_routed_moe(op, fn))
            for tag, fn in _expert_backends(hidden, w1, w2, op.top_k).items()
        }
    )


@pytest.mark.parametrize("case", bench.cases(SharedExpertMLPFwdOp), ids=lambda case: case.id)
def test_shared_expert_mlp_bench(case) -> None:
    op = SharedExpertMLPFwdOp(**case.arguments)
    gemm_act = quack_op("gemm_act", "quack.gemm_interface")
    gemm = quack_op("gemm", "quack.gemm_interface")

    def quack_fn(hidden, w1, w2):
        _, activated = gemm_act(
            hidden, w1.T, activation="swiglu", store_preact=False, concat_layout=("B",)
        )
        return gemm(activated, w2.T)

    def torch_fn(hidden, w1, w2):
        gate, up = F.linear(hidden, w1).chunk(2, -1)
        return F.linear(F.silu(gate) * up, w2)

    bench.Runner(op, case).compare({"torch-cublas": torch_fn, QUACK_TAG: quack_fn})
