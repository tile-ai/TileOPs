"""Benchmarks for the staged rank-grouped MoE preparation boundaries."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import VLLM_TAG, flashinfer_op, vllm_op
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
    moe_permute = vllm_op("moe_permute", "model_executor.layers.fused_moe.moe_permute_unpermute")

    def _vllm_reference(hidden: torch.Tensor, expert_ids: torch.Tensor):
        rows, _, offsets, inverse, _ = moe_permute(hidden, None, expert_ids, op.num_local_experts)
        return (rows, offsets[1:].int(), inverse)

    implementations = {"torch-ref": case.reference}
    if _tight_psum(op):
        implementations[VLLM_TAG] = _vllm_reference
    else:
        print(f"  [skip] {VLLM_TAG}: moe_permute writes tight rows and their expert offsets")
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(MoEPostPermuteFwdOp), ids=lambda case: case.id)
def test_moe_post_permute_bench(case) -> None:
    expert_output, weights, _ = case.inputs
    op = MoEPostPermuteFwdOp(**case.arguments)
    tokens, hidden = weights.shape[0], expert_output.shape[-1]
    out_vllm = torch.empty(tokens, hidden, dtype=expert_output.dtype, device=expert_output.device)
    moe_unpermute = vllm_op(
        "moe_unpermute", "model_executor.layers.fused_moe.moe_permute_unpermute"
    )

    # moe_unpermute takes no routing scale; the epilogue's is applied to its result.
    scale = 1.0 if op.epilogue is None else op.epilogue.routed_scaling_factor

    def _vllm_reference(
        output: torch.Tensor,
        routing_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        # The inverse indices address the flat rows of every layout's expert output.
        moe_unpermute(out_vllm, output.reshape(-1, hidden), routing_weights, inverse_indices)
        return out_vllm if scale == 1.0 else out_vllm * scale

    implementations = {"torch-ref": case.reference}
    if op.out_dtype in (None, expert_output.dtype):
        implementations[VLLM_TAG] = _vllm_reference
    else:
        print(
            f"  [skip] {VLLM_TAG}: moe_unpermute writes {expert_output.dtype}, not {op.out_dtype}"
        )
    bench.Runner(op, case).compare(implementations)


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
    """The layout the vLLM, torch and flashinfer segment baselines express: tight rows, psum
    ends."""
    layout = op.layout
    return layout.kind == "contiguous" and layout.selection_key == "tight_physical_psum"


def _row_bias(bias, ends: torch.Tensor):
    """Each tight row's expert bias in fp32, or ``None``; built outside the timed callable."""
    if bias is None:
        return None
    counts = torch.diff(ends.long(), prepend=ends.new_zeros(1, dtype=torch.long))
    return bias.float().repeat_interleave(counts, dim=0)


def _bias_after_rounding(*biases) -> "str | None":
    """Why a torch._grouped_mm row has no ratio when the call passes a bias: the op adds a bias to
    the fp32 product, and torch._grouped_mm writes the product in the operand dtype."""
    if all(bias is None for bias in biases):
        return None
    return "adds the bias to the GEMM product after rounding it to the operand dtype"


def _biased(output: torch.Tensor, row_bias) -> torch.Tensor:
    """*output* plus its rows' bias, added in fp32 as the op adds it."""
    return output if row_bias is None else (output.float() + row_bias).to(output.dtype)


@pytest.mark.parametrize("case", bench.cases(MoEGroupedGemmFwdOp), ids=lambda case: case.id)
def test_moe_grouped_gemm_bench(case) -> None:
    # The optional per-expert bias follows the metadata.
    _, b, metadata, bias = case.inputs
    op = MoEGroupedGemmFwdOp(**case.arguments)
    implementations = {}
    # torch._grouped_mm writes only the operand dtype, so an fp32 out_dtype has no torch row.
    if _tight_psum(op) and op.out_dtype is None:
        b_kn = b.transpose(1, 2).contiguous()
        row_bias = _row_bias(bias, metadata)

        def _torch_grouped_mm(a_, _b, ends, _bias):
            output = _biased(torch._grouped_mm(a_, b_kn, offs=ends), row_bias)
            return output if op.activation is None else gated_activation(output, op.activation)

        implementations["torch-grouped-mm"] = bench.Implementation(
            run=_torch_grouped_mm, noncomparable_reason=_bias_after_rounding(bias)
        )
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(MoEExpertMLPFwdOp), ids=lambda case: case.id)
def test_moe_expert_mlp_bench(case) -> None:
    # The optional biases b_gate_up, b_down follow the metadata.
    _, w_gate_up, w_down, metadata, b_gate_up, b_down = case.inputs
    op = MoEExpertMLPFwdOp(**case.arguments)
    implementations = {}
    if _tight_psum(op):
        gate_up_kn = w_gate_up.transpose(1, 2).contiguous()
        down_kn = w_down.transpose(1, 2).contiguous()
        gate_up_bias = _row_bias(b_gate_up, metadata)
        down_bias = _row_bias(b_down, metadata)

        def _torch_grouped_mlp(x_, _w_gate_up, _w_down, ends, _b_gate_up, _b_down):
            gate_up = _biased(torch._grouped_mm(x_, gate_up_kn, offs=ends), gate_up_bias)
            activated = gated_activation(gate_up, op.activation)
            return _biased(torch._grouped_mm(activated, down_kn, offs=ends), down_bias)

        implementations["torch-grouped-mm"] = bench.Implementation(
            run=_torch_grouped_mlp, noncomparable_reason=_bias_after_rounding(b_gate_up, b_down)
        )
    else:
        print("  [skip] torch-grouped-mm, flashinfer-segment-mlp: they take tight rows' offsets")
    # FlashInfer's segment GEMM takes no bias and its activation here is silu_and_mul.
    if _tight_psum(op) and op.activation == "silu_and_mul" and b_gate_up is None and b_down is None:
        segment_gemm = _flashinfer_segment_gemm(metadata)
        silu_and_mul = flashinfer_op("activation.silu_and_mul")

        def _flashinfer_mlp(x_, w_gate_up_, w_down_, _ends, _b_gate_up, _b_down):
            return segment_gemm(silu_and_mul(segment_gemm(x_, w_gate_up_)), w_down_)

        implementations["flashinfer-segment-mlp"] = _flashinfer_mlp
    bench.Runner(op, case).compare(implementations)
