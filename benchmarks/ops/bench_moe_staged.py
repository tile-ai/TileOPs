"""Benchmarks for the staged rank-grouped MoE preparation boundaries."""

import pytest
import torch
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)

from benchmarks.baselines import flashinfer_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.moe import (
    MoeExpertMLPFwdOp,
    MoeGroupedGemmFwdOp,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
)
from workloads.moe import (
    MoeExpertMLPWorkload,
    MoeGroupedGemmWorkload,
    MoePostPermuteWorkload,
    MoePrePermuteWorkload,
    gated_activation,
    valid_rows,
)


@pytest.mark.parametrize("call", manifest_calls(MoePrePermuteFwdOp))
def test_moe_pre_permute_bench(call) -> None:
    workload = MoePrePermuteWorkload(call)
    hidden_states, local_ids = workload.gen_inputs()
    op = MoePrePermuteFwdOp(**call.arguments({}))
    benchmark = ManifestBenchmark(op, workload)
    expert_input, ends, inverse = op(hidden_states, local_ids)
    ref_input, ref_ends, ref_inverse = workload.ref_program(hidden_states, local_ids)
    # Row order inside an expert's segment is not specified; each route's row and the ends are.
    torch.testing.assert_close(ends, ref_ends, rtol=0, atol=0)
    torch.testing.assert_close(
        expert_input[inverse.long()], ref_input[ref_inverse.long()], rtol=0, atol=0
    )

    def _vllm_reference(hidden: torch.Tensor, expert_ids: torch.Tensor):
        return moe_permute(hidden, None, expert_ids, op.num_local_experts)

    benchmark.compare(
        {"tileops": op, "vllm": _vllm_reference, "torch-ref": workload.ref_program},
        hidden_states,
        local_ids,
    )


@pytest.mark.parametrize("call", manifest_calls(MoePostPermuteFwdOp))
def test_moe_post_permute_bench(call) -> None:
    workload = MoePostPermuteWorkload(call)
    expert_output, weights, inverse = workload.gen_inputs()
    op = MoePostPermuteFwdOp(**call.arguments({}))
    benchmark = ManifestBenchmark(op, workload)
    torch.testing.assert_close(
        op(expert_output, weights, inverse),
        workload.ref_program(expert_output, weights, inverse),
        rtol=2e-2,
        atol=2e-2,
    )

    tokens, hidden = weights.shape[0], expert_output.shape[-1]
    numel = inverse.numel()
    inverse_permuted_idx = torch.empty(numel, dtype=torch.int32, device=inverse.device)
    inverse_permuted_idx[inverse.long()] = torch.arange(
        numel, dtype=torch.int32, device=inverse.device
    )
    out_vllm = torch.empty(tokens, hidden, dtype=expert_output.dtype, device=expert_output.device)

    def _vllm_reference(
        output: torch.Tensor,
        routing_weights: torch.Tensor,
        _inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        moe_unpermute(out_vllm, output, routing_weights, inverse_permuted_idx)
        return out_vllm

    benchmark.compare(
        {"tileops": op, "vllm": _vllm_reference, "torch-ref": workload.ref_program},
        expert_output,
        weights,
        inverse,
    )


def _assert_valid_rows_match(out: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor) -> None:
    flat_out = out.reshape(-1, out.shape[-1])[valid].float()
    flat_ref = ref.reshape(-1, ref.shape[-1])[valid].float()
    torch.testing.assert_close(flat_out, flat_ref, rtol=2e-2, atol=1e-1)


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


@pytest.mark.parametrize("call", manifest_calls(MoeGroupedGemmFwdOp))
def test_moe_grouped_gemm_bench(call) -> None:
    workload = MoeGroupedGemmWorkload(call)
    a, b, metadata = workload.gen_inputs()
    op = MoeGroupedGemmFwdOp(**call.arguments({}))
    benchmark = ManifestBenchmark(op, workload)
    valid = valid_rows(op.layout, metadata, a.numel() // a.shape[-1], b.shape[0])
    ref = workload.ref_program(a, b, metadata)
    _assert_valid_rows_match(op(a, b, metadata), ref, valid)
    functors = {"tileops": op}

    # torch._grouped_mm takes tight segments and writes the operand dtype.
    if _tight_psum(op) and op.out_dtype is None:
        b_kn = b.transpose(1, 2).contiguous()

        def _torch_grouped_mm(a_, _b, ends):
            output = torch._grouped_mm(a_, b_kn, offs=ends)
            return output if op.activation is None else gated_activation(output, op.activation)

        _assert_valid_rows_match(_torch_grouped_mm(a, b, metadata), ref, valid)
        functors["torch-grouped-mm"] = _torch_grouped_mm
    benchmark.compare(functors, a, b, metadata)


@pytest.mark.parametrize("call", manifest_calls(MoeExpertMLPFwdOp))
def test_moe_expert_mlp_bench(call) -> None:
    workload = MoeExpertMLPWorkload(call)
    x, w_gate_up, w_down, metadata = workload.gen_inputs()
    op = MoeExpertMLPFwdOp(**call.arguments({}))
    benchmark = ManifestBenchmark(op, workload)
    valid = valid_rows(op.layout, metadata, x.numel() // x.shape[-1], w_down.shape[0])
    ref = workload.ref_program(x, w_gate_up, w_down, metadata)
    _assert_valid_rows_match(op(x, w_gate_up, w_down, metadata), ref, valid)

    gate_up_kn = w_gate_up.transpose(1, 2).contiguous()
    down_kn = w_down.transpose(1, 2).contiguous()

    def _torch_grouped_mlp(x_, _w_gate_up, _w_down, ends):
        gate_up = torch._grouped_mm(x_, gate_up_kn, offs=ends)
        activated = gated_activation(gate_up, op.activation)
        return torch._grouped_mm(activated, down_kn, offs=ends)

    _assert_valid_rows_match(_torch_grouped_mlp(x, w_gate_up, w_down, metadata), ref, valid)
    segment_gemm = _flashinfer_segment_gemm(metadata)
    silu_and_mul = flashinfer_op("activation.silu_and_mul")

    def _flashinfer_mlp(x_, w_gate_up_, w_down_, _ends):
        return segment_gemm(silu_and_mul(segment_gemm(x_, w_gate_up_)), w_down_)

    _assert_valid_rows_match(_flashinfer_mlp(x, w_gate_up, w_down, metadata), ref, valid)
    benchmark.compare(
        {
            "tileops": op,
            "torch-grouped-mm": _torch_grouped_mlp,
            "flashinfer-segment-mlp": _flashinfer_mlp,
        },
        x,
        w_gate_up,
        w_down,
        metadata,
    )
