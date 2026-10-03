"""Expert MLPs against vLLM Triton, FlashInfer CUTLASS and the staged pipeline."""

import pytest
import torch

from benchmarks.baselines import vllm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.moe_baselines import flashinfer_experts
from benchmarks.verification import Exact
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    FusedMoEExpertsFwdOp,
    IndexedExpertMLPFwdOp,
    MoEExpertMLPFwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
    RoutingEpilogueSpec,
)
from workloads.moe import IndexedExpertMLPWorkload, MoeExpertsWorkload


@pytest.mark.parametrize("call", manifest_calls(FusedMoEExpertsFwdOp))
def test_moe_experts_bench(call) -> None:
    workload = MoeExpertsWorkload(call)
    inputs = workload.gen_inputs()
    output, hidden, w1, w2, topk_weights, topk_ids = inputs
    experts = FusedMoEExpertsFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(experts, workload)

    def _experts_fn(hidden, w1, w2, topk_weights, topk_ids):
        experts(output, hidden, w1, w2, topk_weights, topk_ids)
        return output

    functors = {
        "tileops": _experts_fn,
        "vllm-triton": vllm_op("fused_experts", "model_executor.layers.fused_moe.fused_moe"),
        "flashinfer-cutlass": flashinfer_experts(hidden, w1, w2, topk_ids.shape[-1]),
    }

    bm.compare(
        {tag: (fn, inputs[1:]) for tag, fn in functors.items()},
        *inputs,
        evidence=dict.fromkeys(functors, Exact(rtol=3e-2, atol=3e-2)),
    )


@pytest.mark.parametrize("call", manifest_calls(IndexedExpertMLPFwdOp))
def test_indexed_expert_mlp_bench(call) -> None:
    workload = IndexedExpertMLPWorkload(call)
    inputs = workload.gen_inputs()
    output, hidden, w1, w2, topk_weights, topk_ids = inputs
    indexed = IndexedExpertMLPFwdOp(**call.arguments({}))

    def _indexed_fn(hidden, w1, w2, topk_weights, topk_ids):
        indexed(output, hidden, w1, w2, topk_weights, topk_ids)
        return output

    # The staged pipeline is what the composite runs on every other shape, so it is the
    # comparator the indexed path has to beat.
    layout = ContiguousLayoutSpec.tight_physical_psum()
    pre = MoEPrePermuteFwdOp(layout, num_local_experts=w1.shape[0])
    mlp = MoEExpertMLPFwdOp(layout)
    epilogue = RoutingEpilogueSpec(routed_scaling_factor=indexed.routed_scaling_factor)
    post = MoEPostPermuteFwdOp(layout, epilogue)
    staged_output = torch.empty_like(output)

    def _staged_fn(hidden, w1, w2, topk_weights, topk_ids):
        expert_input, physical_ends, inverse = pre(hidden, topk_ids)
        expert_output = mlp(expert_input, w1, w2, physical_ends)
        post(expert_output, topk_weights, inverse, out=staged_output)
        return staged_output

    fused_experts = vllm_op("fused_experts", "model_executor.layers.fused_moe.fused_moe")

    def vllm_fn(hidden, w1, w2, topk_weights, topk_ids):
        out = fused_experts(hidden, w1, w2, topk_weights, topk_ids)
        return out * indexed.routed_scaling_factor

    cutlass = flashinfer_experts(hidden, w1, w2, topk_ids.shape[-1])

    def cutlass_fn(hidden, w1, w2, topk_weights, topk_ids):
        return cutlass(hidden, w1, w2, topk_weights, topk_ids) * indexed.routed_scaling_factor

    functors = {
        "tileops": _indexed_fn,
        "staged": _staged_fn,
        "vllm-triton": vllm_fn,
        "flashinfer-cutlass": cutlass_fn,
    }

    ManifestBenchmark(indexed, workload).compare(
        {tag: (fn, inputs[1:]) for tag, fn in functors.items()},
        *inputs,
        evidence=dict.fromkeys(functors, Exact(rtol=3e-2, atol=3e-2)),
    )
