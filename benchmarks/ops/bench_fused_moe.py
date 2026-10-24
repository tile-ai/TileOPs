"""Routed and shared expert MLPs against vLLM, FlashInfer CUTLASS and QuACK.

Every routed baseline routes with vLLM: `fused_topk`, `fused_topk_bias` with a correction
bias, and the fused CUDA `grouped_topk` for DeepSeek-V3 grouped routing
(`num_expert_group > 1`), which requires a bias, so a grouped baseline is sigmoid with a
bias. Each routes unscaled and scales the expert output once, as the op does.
"""

import pytest
import torch.nn.functional as F

from benchmarks import api as bench
from benchmarks.baselines import (
    QUACK_TAG,
    VLLM_GROUPED_TOPK_SIGMOID,
    VLLM_TAG,
    quack_op,
    vllm_op,
)
from benchmarks.moe_baselines import (
    flashinfer_experts,
    flashinfer_experts_skip_reason,
    vllm_experts,
)
from tileops.ops.moe import FusedMoEFwdOp, FusedMoESharedExpertFwdOp, SharedExpertMLPFwdOp


def _routed_moe(op, experts):
    topk = vllm_op("fused_topk", "model_executor.layers.fused_moe.router.fused_topk_router")
    topk_bias = vllm_op(
        "fused_topk_bias", "model_executor.layers.fused_moe.router.fused_topk_bias_router"
    )
    grouped_topk = vllm_op("grouped_topk")

    def run(hidden, gating, w1, w2, correction_bias, b_gate_up=None, b_down=None):
        if op.num_expert_group > 1:
            weights, ids = grouped_topk(
                gating,
                op.num_expert_group,
                op.topk_group,
                op.top_k,
                op.renormalize,
                1.0,
                correction_bias,
                VLLM_GROUPED_TOPK_SIGMOID,
            )
        elif correction_bias is None:
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
        biases = (
            {}
            if b_gate_up is None and b_down is None
            else {"b_gate_up": b_gate_up, "b_down": b_down}
        )
        output = experts(hidden, w1, w2, weights, ids, **biases)
        return output * op.routed_scaling_factor if op.routed_scaling_factor != 1.0 else output

    return run


def _expert_backends(op, hidden, w1, w2, correction_bias, b_gate_up=None, b_down=None):
    """The routed baselines, or none for a grouped row vLLM cannot route.

    vLLM's fused ``grouped_topk`` takes only a biased sigmoid, and both baselines route
    with it. FlashInfer CUTLASS is dropped where its binding refuses the experts.
    """
    if op.num_expert_group > 1 and (correction_bias is None or op.scoring_func != "sigmoid"):
        print(
            f"  [skip] {VLLM_TAG}, flashinfer-cutlass: grouped routing needs sigmoid and a correction bias"
        )
        return {}
    backends = {VLLM_TAG: vllm_experts(op.activation)}
    reason = flashinfer_experts_skip_reason(hidden, w2, b_gate_up, b_down)
    if reason is None:
        backends["flashinfer-cutlass"] = flashinfer_experts(
            hidden, w1, w2, op.top_k, op.activation, b_gate_up
        )
    else:
        print(f"  [skip] flashinfer-cutlass: {reason}")
    return backends


@pytest.mark.parametrize("case", bench.cases(FusedMoEFwdOp), ids=lambda case: case.id)
def test_fused_moe_fwd_bench(case) -> None:
    op = FusedMoEFwdOp(**case.arguments)
    hidden, _, w1, w2, correction_bias, b_gate_up, b_down = case.inputs
    backends = _expert_backends(op, hidden, w1, w2, correction_bias, b_gate_up, b_down)
    bench.Runner(op, case).compare({tag: _routed_moe(op, fn) for tag, fn in backends.items()})


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
    hidden, _, w1, w2, correction_bias, *_ = case.inputs
    implementations = {}
    # Under tensor parallelism the op returns this rank's partial shared output; the
    # baselines compute the whole shared expert.
    if op.tp_size == 1:
        backends = _expert_backends(op, hidden, w1, w2, correction_bias)
        implementations = {
            tag: _with_shared_expert(_routed_moe(op, fn)) for tag, fn in backends.items()
        }
    else:
        print(
            f"  [skip] {VLLM_TAG}, flashinfer-cutlass: a partial shared output, tp_size={op.tp_size}"
        )
    bench.Runner(op, case).compare(implementations)


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
