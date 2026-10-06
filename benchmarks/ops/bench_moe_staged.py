"""Benchmarks for the staged rank-grouped MoE preparation boundaries."""

import pytest
import torch
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)

from benchmarks import api as bench
from benchmarks.baselines import VLLM_TAG, flashinfer_op
from tileops.ops.moe import (
    MoEExpertMLPFwdOp,
    MoEGroupedGemmFwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
)
from workloads.moe import gated_activation


@pytest.mark.parametrize("case", bench.cases(MoEPrePermuteFwdOp), ids=lambda case: case.id)
def test_moe_pre_permute_bench(case) -> None:
    op = MoEPrePermuteFwdOp(**case.arguments)

    def _vllm_reference(hidden: torch.Tensor, expert_ids: torch.Tensor):
        rows, _, offsets, inverse, _ = moe_permute(hidden, None, expert_ids, op.num_local_experts)
        return (rows, offsets[1:].int(), inverse)

    bench.Runner(op, case).compare({VLLM_TAG: _vllm_reference, "torch-ref": case.reference})


@pytest.mark.parametrize("case", bench.cases(MoEPostPermuteFwdOp), ids=lambda case: case.id)
def test_moe_post_permute_bench(case) -> None:
    expert_output, weights, _ = case.inputs
    op = MoEPostPermuteFwdOp(**case.arguments)
    tokens, hidden = weights.shape[0], expert_output.shape[-1]
    out_vllm = torch.empty(tokens, hidden, dtype=expert_output.dtype, device=expert_output.device)

    def _vllm_reference(
        output: torch.Tensor,
        routing_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        moe_unpermute(out_vllm, output, routing_weights, inverse_indices)
        return out_vllm

    bench.Runner(op, case).compare({VLLM_TAG: _vllm_reference, "torch-ref": case.reference})


def _flashinfer_segment_gemm(ends: torch.Tensor):
    wrapper = flashinfer_op("gemm.SegmentGEMMWrapper")(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=ends.device)
    )
    indptr = torch.cat((ends.new_zeros(1), ends)).to(torch.int64)

    def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return wrapper.run(
            x,
            weight,
            batch_size=weight.shape[0],
            weight_column_major=True,
            seg_indptr=indptr,
        )

    return run


def _tight_psum(op) -> bool:
    """The layout the torch and flashinfer segment baselines express: tight rows, psum ends."""
    layout = op.layout
    return layout.kind == "contiguous" and layout.packing.value == "tight"


@pytest.mark.parametrize("case", bench.cases(MoEGroupedGemmFwdOp), ids=lambda case: case.id)
def test_moe_grouped_gemm_bench(case) -> None:
    _, b, _ = case.inputs
    op = MoEGroupedGemmFwdOp(**case.arguments)
    implementations = {}
    if _tight_psum(op) and op.out_dtype is None:
        b_kn = b.transpose(1, 2).contiguous()

        def _torch_grouped_mm(a_, _b, ends):
            output = torch._grouped_mm(a_, b_kn, offs=ends)
            return output if op.activation is None else gated_activation(output, op.activation)

        implementations["torch-grouped-mm"] = _torch_grouped_mm
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(MoEExpertMLPFwdOp), ids=lambda case: case.id)
def test_moe_expert_mlp_bench(case) -> None:
    _, w_gate_up, w_down, metadata = case.inputs
    op = MoEExpertMLPFwdOp(**case.arguments)
    gate_up_kn = w_gate_up.transpose(1, 2).contiguous()
    down_kn = w_down.transpose(1, 2).contiguous()

    def _torch_grouped_mlp(x_, _w_gate_up, _w_down, ends):
        gate_up = torch._grouped_mm(x_, gate_up_kn, offs=ends)
        activated = gated_activation(gate_up, op.activation)
        return torch._grouped_mm(activated, down_kn, offs=ends)

    segment_gemm = _flashinfer_segment_gemm(metadata)
    silu_and_mul = flashinfer_op("activation.silu_and_mul")

    def _flashinfer_mlp(x_, w_gate_up_, w_down_, _ends):
        return segment_gemm(silu_and_mul(segment_gemm(x_, w_gate_up_)), w_down_)

    bench.Runner(op, case).compare(
        {
            "torch-grouped-mm": _torch_grouped_mlp,
            "flashinfer-segment-mlp": _flashinfer_mlp,
        }
    )
