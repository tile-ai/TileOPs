"""Routed and shared expert MLPs against vLLM, FlashInfer CUTLASS and QuACK."""

import pytest
import torch.nn.functional as F

from benchmarks.baselines import QUACK_TAG, VLLM_TAG, quack_op, vllm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.moe_baselines import flashinfer_experts
from tileops.ops.moe import FusedMoEFwdOp, FusedMoESharedExpertFwdOp, SharedExpertMLPFwdOp
from workloads.moe import FusedMoeSharedExpertWorkload, FusedMoeWorkload, SharedExpertMLPWorkload


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


@pytest.mark.parametrize("call", manifest_calls(FusedMoEFwdOp))
def test_fused_moe_fwd_bench(call) -> None:
    workload = FusedMoeWorkload(call)
    inputs = workload.gen_inputs()
    op = FusedMoEFwdOp(**call.arguments({}))
    hidden, _, w1, w2, _ = inputs
    functors = {
        "tileops": op,
        **{
            tag: _routed_moe(op, fn)
            for tag, fn in _expert_backends(hidden, w1, w2, op.top_k).items()
        },
    }
    ManifestBenchmark(op, workload).compare(functors, *inputs, count_copies=True)


def _with_shared_expert(routed):
    def run(hidden, gating, w1, w2, bias, shared_w1, shared_w2):
        output = routed(hidden, gating, w1, w2, bias)
        shared = None
        if shared_w1 is not None:
            gate, up = F.linear(hidden, shared_w1).chunk(2, -1)
            shared = F.linear(F.silu(gate) * up, shared_w2)
        return shared, output

    return run


@pytest.mark.parametrize("call", manifest_calls(FusedMoESharedExpertFwdOp))
def test_fused_moe_shared_expert_bench(call) -> None:
    workload = FusedMoeSharedExpertWorkload(call)
    inputs = workload.gen_inputs()
    op = FusedMoESharedExpertFwdOp(**call.arguments({}))
    hidden, _, w1, w2, *_ = inputs
    functors = {
        "tileops": op,
        **{
            tag: _with_shared_expert(_routed_moe(op, fn))
            for tag, fn in _expert_backends(hidden, w1, w2, op.top_k).items()
        },
    }
    ManifestBenchmark(op, workload).compare(functors, *inputs, count_copies=True)


@pytest.mark.parametrize("call", manifest_calls(SharedExpertMLPFwdOp))
def test_shared_expert_mlp_bench(call) -> None:
    workload = SharedExpertMLPWorkload(call)
    inputs = workload.gen_inputs()
    op = SharedExpertMLPFwdOp(**call.arguments({}))
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

    functors = {"tileops": op, "torch-cublas": torch_fn, QUACK_TAG: quack_fn}
    ManifestBenchmark(op, workload).compare(functors, *inputs, count_copies=True)
