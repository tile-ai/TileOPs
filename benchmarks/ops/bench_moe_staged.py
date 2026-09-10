"""Benchmarks for the staged rank-grouped MoE preparation boundaries."""

import pytest
import torch
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)

from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MoeExpertMLPFwdOp,
    MoeGroupedGemmFwdOp,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
)
from tileops.ops.moe.contracts import layout_from_preset
from workloads.moe import (
    MoeExpertMLPStagedWorkload,
    MoeGroupedGemmStagedWorkload,
    MoePermuteWorkload,
    MoeUnpermuteWorkload,
)


def _pre_args(workload: dict, dtype: torch.dtype) -> tuple[int, int, int, int, torch.dtype]:
    tokens, hidden = workload["hidden_states_shape"]
    _, top_k = workload["local_expert_ids_shape"]
    return tokens, top_k, workload["num_local_experts"], hidden, dtype


@pytest.mark.parametrize(
    "tokens,top_k,num_local_experts,hidden,dtype",
    workload_params(load_workloads(MoePrePermuteFwdOp), _pre_args),
)
def test_moe_pre_permute_bench(
    tokens: int, top_k: int, num_local_experts: int, hidden: int, dtype: torch.dtype
) -> None:
    layout = ContiguousLayoutSpec.tight_physical_psum()
    op = MoePrePermuteFwdOp(layout, num_local_experts)
    workload = MoePermuteWorkload(tokens, top_k, num_local_experts, hidden, dtype)
    hidden_states, local_ids = workload.gen_inputs()
    benchmark = ManifestBenchmark(op, workload)
    token_rows = torch.arange(tokens, device=hidden_states.device).repeat_interleave(top_k)
    tight_slots = torch.arange(tokens * top_k, dtype=torch.int32, device=hidden_states.device)

    def _torch_reference(hidden: torch.Tensor, expert_ids: torch.Tensor):
        flat_ids = expert_ids.flatten().to(torch.int64)
        sorted_flat_indices = torch.argsort(flat_ids, stable=True)
        expert_input = hidden[token_rows[sorted_flat_indices]]
        counts = torch.bincount(flat_ids, minlength=num_local_experts)
        physical_ends = torch.cumsum(counts, dim=0).to(torch.int32)
        inverse = torch.empty_like(sorted_flat_indices, dtype=torch.int32)
        inverse[sorted_flat_indices] = tight_slots
        return expert_input, physical_ends, inverse

    def _vllm_reference(hidden: torch.Tensor, expert_ids: torch.Tensor):
        return moe_permute(hidden, None, expert_ids, num_local_experts)

    benchmark.compare(
        {"tileops": op, "vllm": _vllm_reference, "torch-ref": _torch_reference},
        hidden_states,
        local_ids,
    )


def _post_args(workload: dict, dtype: torch.dtype) -> tuple[int, int, int, torch.dtype]:
    rows, hidden = workload["expert_output_shape"]
    tokens, top_k = workload["topk_weights_shape"]
    assert rows == tokens * top_k
    return tokens, top_k, hidden, dtype


@pytest.mark.parametrize(
    "tokens,top_k,hidden,dtype",
    workload_params(load_workloads(MoePostPermuteFwdOp), _post_args),
)
def test_moe_post_permute_bench(tokens: int, top_k: int, hidden: int, dtype: torch.dtype) -> None:
    layout = ContiguousLayoutSpec.tight_physical_psum()
    op = MoePostPermuteFwdOp(layout)
    workload = MoeUnpermuteWorkload(tokens, top_k, hidden, dtype)
    expert_output, inverse, weights = workload.gen_inputs()
    benchmark = ManifestBenchmark(op, workload)

    def _torch_reference(
        output: torch.Tensor,
        routing_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        inverse_long = inverse_indices.to(torch.int64)
        gathered = output[inverse_long].float()
        weighted = gathered.view(tokens, top_k, hidden) * routing_weights.unsqueeze(-1)
        return weighted.sum(dim=1).to(output.dtype)

    numel = tokens * top_k
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
        {"tileops": op, "vllm": _vllm_reference, "torch-ref": _torch_reference},
        expert_output,
        weights,
        inverse,
    )


def _layout_args(workload: dict) -> dict:
    extra = {k: workload[k] for k in ("alignment", "max_m") if k in workload}
    return {"layout": workload["layout"], **extra}


def _layout_spec(layout_args: dict):
    extra = {k: v for k, v in layout_args.items() if k != "layout"}
    return layout_from_preset(layout_args["layout"], **extra)


def _assert_valid_rows_match(out: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor) -> None:
    flat_out = out.reshape(-1, out.shape[-1])[valid].float()
    flat_ref = ref.reshape(-1, ref.shape[-1])[valid].float()
    torch.testing.assert_close(flat_out, flat_ref, rtol=2e-2, atol=1e-1)


def _gemm_args(workload: dict, dtype: torch.dtype) -> tuple:
    return (
        tuple(workload["a_shape"]),
        tuple(workload["b_shape"]),
        _layout_args(workload),
        workload.get("activation"),
        dtype,
    )


@pytest.mark.parametrize(
    "a_shape,b_shape,layout_args,activation,dtype",
    workload_params(load_workloads(MoeGroupedGemmFwdOp), _gemm_args),
)
def test_moe_grouped_gemm_bench(a_shape, b_shape, layout_args, activation, dtype) -> None:
    op = MoeGroupedGemmFwdOp(_layout_spec(layout_args), activation=activation)
    workload = MoeGroupedGemmStagedWorkload(
        a_shape, b_shape, dtype=dtype, activation=activation, **layout_args
    )
    a, b, metadata = workload.gen_inputs()
    benchmark = ManifestBenchmark(op, workload)
    torch._assert_async(op.layout_guard(a, b, metadata))
    ref = workload.ref_program(a, b, metadata)
    _assert_valid_rows_match(op(a, b, metadata), ref, workload.valid_rows)

    # FIXME(staged-rollout): compare through the replaced op's benchmark after migration.
    benchmark.compare({"tileops": op}, a, b, metadata)


def _mlp_args(workload: dict, dtype: torch.dtype) -> tuple:
    return (
        tuple(workload["expert_input_shape"]),
        tuple(workload["w_gate_up_shape"]),
        tuple(workload["w_down_shape"]),
        _layout_args(workload),
        dtype,
    )


@pytest.mark.parametrize(
    "x_shape,w_gate_up_shape,w_down_shape,layout_args,dtype",
    workload_params(load_workloads(MoeExpertMLPFwdOp), _mlp_args),
)
def test_moe_expert_mlp_bench(x_shape, w_gate_up_shape, w_down_shape, layout_args, dtype) -> None:
    op = MoeExpertMLPFwdOp(_layout_spec(layout_args))
    workload = MoeExpertMLPStagedWorkload(
        x_shape, w_gate_up_shape, w_down_shape, dtype=dtype, **layout_args
    )
    x, w_gate_up, w_down, metadata = workload.gen_inputs()
    benchmark = ManifestBenchmark(op, workload)
    ref = workload.ref_program(x, w_gate_up, w_down, metadata)
    _assert_valid_rows_match(op(x, w_gate_up, w_down, metadata), ref, workload.valid_rows)

    # FIXME(staged-rollout): compare through the replaced op's benchmark after migration.
    benchmark.compare({"tileops": op}, x, w_gate_up, w_down, metadata)
