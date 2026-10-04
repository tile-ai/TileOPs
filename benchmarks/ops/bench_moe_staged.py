"""Benchmarks for the staged rank-grouped MoE preparation boundaries."""

import pytest
import torch
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)

from benchmarks.baselines import VLLM_TAG, flashinfer_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.moe import (
    MoEExpertMLPFwdOp,
    MoEGroupedGemmFwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
)
from workloads.moe import (
    MoeExpertMLPWorkload,
    MoeGroupedGemmWorkload,
    MoePostPermuteWorkload,
    MoePrePermuteWorkload,
    gated_activation,
    valid_rows,
)
from workloads.numerics import Custom, Exact


@pytest.mark.parametrize("call", manifest_calls(MoEPrePermuteFwdOp))
def test_moe_pre_permute_bench(call) -> None:
    workload = MoePrePermuteWorkload(call)
    hidden_states, local_ids = workload.gen_inputs()
    op = MoEPrePermuteFwdOp(**call.arguments({}))
    benchmark = ManifestBenchmark(op, workload)

    def _vllm_reference(hidden: torch.Tensor, expert_ids: torch.Tensor):
        rows, _, offsets, inverse, _ = moe_permute(hidden, None, expert_ids, op.num_local_experts)
        return rows, offsets[1:].int(), inverse

    def validate(got, expected):
        rows, ends, inverse = got
        ref_rows, ref_ends, ref_inverse = expected
        assert rows.shape == ref_rows.shape and rows.dtype == ref_rows.dtype
        assert inverse.shape == ref_inverse.shape and inverse.dtype == ref_inverse.dtype
        torch.testing.assert_close(ends, ref_ends, rtol=0, atol=0)
        torch.testing.assert_close(
            inverse.long().sort().values,
            torch.arange(inverse.numel(), device=inverse.device),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rows[inverse.long()], ref_rows[ref_inverse.long()], rtol=0, atol=0
        )
        owners = torch.searchsorted(ends, inverse, right=True)
        torch.testing.assert_close(owners, local_ids.flatten().to(owners.dtype), rtol=0, atol=0)

    benchmark.compare(
        {"tileops": op, VLLM_TAG: _vllm_reference, "torch-ref": workload.ref_program},
        hidden_states,
        local_ids,
        evidence=dict.fromkeys(
            ("tileops", VLLM_TAG),
            Custom(validate, "route permutation and expert segment ownership"),
        ),
    )


@pytest.mark.parametrize("call", manifest_calls(MoEPostPermuteFwdOp))
def test_moe_post_permute_bench(call) -> None:
    workload = MoePostPermuteWorkload(call)
    expert_output, weights, inverse = workload.gen_inputs()
    op = MoEPostPermuteFwdOp(**call.arguments({}))
    benchmark = ManifestBenchmark(op, workload)
    tokens, hidden = weights.shape[0], expert_output.shape[-1]
    out_vllm = torch.empty(tokens, hidden, dtype=expert_output.dtype, device=expert_output.device)

    def _vllm_reference(
        output: torch.Tensor,
        routing_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        moe_unpermute(out_vllm, output, routing_weights, inverse_indices)
        return out_vllm

    benchmark.compare(
        {"tileops": op, VLLM_TAG: _vllm_reference, "torch-ref": workload.ref_program},
        expert_output,
        weights,
        inverse,
        evidence=dict.fromkeys(("tileops", VLLM_TAG), Exact(rtol=2e-2, atol=2e-2)),
    )


def _assert_valid_rows_match(out: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype
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


@pytest.mark.parametrize("call", manifest_calls(MoEGroupedGemmFwdOp))
def test_moe_grouped_gemm_bench(call) -> None:
    workload = MoeGroupedGemmWorkload(call)
    a, b, metadata = workload.gen_inputs()
    op = MoEGroupedGemmFwdOp(**call.arguments({}))
    benchmark = ManifestBenchmark(op, workload)
    valid = valid_rows(op.layout, metadata, a.numel() // a.shape[-1], b.shape[0])
    functors = {"tileops": op}

    # torch._grouped_mm takes tight segments and writes the operand dtype.
    if _tight_psum(op) and op.out_dtype is None:
        b_kn = b.transpose(1, 2).contiguous()

        def _torch_grouped_mm(a_, _b, ends):
            output = torch._grouped_mm(a_, b_kn, offs=ends)
            return output if op.activation is None else gated_activation(output, op.activation)

        functors["torch-grouped-mm"] = _torch_grouped_mm
    benchmark.compare(
        functors,
        a,
        b,
        metadata,
        evidence=dict.fromkeys(
            functors,
            Custom(
                lambda got, ref: _assert_valid_rows_match(got, ref, valid),
                "defined rows of grouped layout",
            ),
        ),
    )


@pytest.mark.parametrize("call", manifest_calls(MoEExpertMLPFwdOp))
def test_moe_expert_mlp_bench(call) -> None:
    workload = MoeExpertMLPWorkload(call)
    x, w_gate_up, w_down, metadata = workload.gen_inputs()
    op = MoEExpertMLPFwdOp(**call.arguments({}))
    benchmark = ManifestBenchmark(op, workload)
    valid = valid_rows(op.layout, metadata, x.numel() // x.shape[-1], w_down.shape[0])

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
        evidence=dict.fromkeys(
            ("tileops", "torch-grouped-mm", "flashinfer-segment-mlp"),
            Custom(
                lambda got, ref: _assert_valid_rows_match(got, ref, valid),
                "defined rows of expert layout",
            ),
        ),
    )
