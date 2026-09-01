"""Benchmarks for the staged rank-grouped MoE preparation boundaries."""

import pytest
import torch
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)

from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops.moe import ContiguousLayoutSpec, MoePostPermuteFwdOp, MoePrePermuteFwdOp
from workloads.moe import MoePermuteWorkload, MoeUnpermuteWorkload


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
