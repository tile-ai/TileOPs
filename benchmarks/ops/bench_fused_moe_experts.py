"""Expert MLPs against vLLM Triton, FlashInfer CUTLASS and the staged pipeline."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import vllm_op
from benchmarks.moe_baselines import flashinfer_experts
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    FusedMoEExpertsFwdOp,
    IndexedExpertMLPFwdOp,
    MoEExpertMLPFwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
    RoutingEpilogueSpec,
)


def _routed(inputs: tuple, functors: dict) -> dict:
    """Each functor on the routed inputs, without the output buffer the op writes."""
    return {tag: bench.Implementation(run=fn, args=inputs[1:]) for tag, fn in functors.items()}


@pytest.mark.parametrize("case", bench.cases(FusedMoEExpertsFwdOp), ids=lambda case: case.id)
def test_moe_experts_bench(case) -> None:
    _, hidden, w1, w2, _, topk_ids = case.inputs
    experts = FusedMoEExpertsFwdOp(**case.arguments)
    functors = {
        "vllm-triton": vllm_op("fused_experts", "model_executor.layers.fused_moe.fused_moe"),
        "flashinfer-cutlass": flashinfer_experts(hidden, w1, w2, topk_ids.shape[-1]),
    }
    bench.Runner(experts, case).compare(_routed(case.inputs, functors))


@pytest.mark.parametrize("case", bench.cases(IndexedExpertMLPFwdOp), ids=lambda case: case.id)
def test_indexed_expert_mlp_bench(case) -> None:
    output, hidden, w1, w2, _, topk_ids = case.inputs
    indexed = IndexedExpertMLPFwdOp(**case.arguments)

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
        "staged": _staged_fn,
        "vllm-triton": vllm_fn,
        "flashinfer-cutlass": cutlass_fn,
    }
    bench.Runner(indexed, case).compare(_routed(case.inputs, functors))
