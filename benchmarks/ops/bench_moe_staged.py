"""Benchmarks for the staged rank-grouped MoE preparation boundaries."""

import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
)
from workloads.moe import MoePermuteWorkload, MoeUnpermuteWorkload


def _pre_args(workload: dict, _dtype: torch.dtype) -> tuple[int, int, int, int]:
    tokens, hidden = workload["hidden_states_shape"]
    _, top_k = workload["local_expert_ids_shape"]
    return tokens, top_k, workload["num_local_experts"], hidden


@pytest.mark.parametrize(
    "tokens,top_k,num_local_experts,hidden",
    workload_params(load_workloads(MoePrePermuteFwdOp), _pre_args),
)
def test_moe_pre_permute_bench(
    tokens: int, top_k: int, num_local_experts: int, hidden: int
) -> None:
    layout = ContiguousLayoutSpec.tight_physical_psum()
    op = MoePrePermuteFwdOp(layout, num_local_experts)
    workload = MoePermuteWorkload(tokens, top_k, num_local_experts, hidden, torch.bfloat16)
    x, local_ids = workload.gen_inputs()
    benchmark = ManifestBenchmark(op, workload)
    token_rows = torch.arange(tokens, device=x.device).repeat_interleave(top_k)
    tight_slots = torch.arange(tokens * top_k, dtype=torch.int32, device=x.device)

    def _torch_reference(hidden_states: torch.Tensor, expert_ids: torch.Tensor):
        flat_ids = expert_ids.flatten().to(torch.int64)
        sorted_flat_indices = torch.argsort(flat_ids, stable=True)
        expert_input = hidden_states[token_rows[sorted_flat_indices]]
        counts = torch.bincount(flat_ids, minlength=num_local_experts)
        physical_ends = torch.cumsum(counts, dim=0).to(torch.int32)
        inverse = torch.empty_like(sorted_flat_indices, dtype=torch.int32)
        inverse[sorted_flat_indices] = tight_slots
        return expert_input, physical_ends, inverse

    benchmark.compare({"tileops": op, "torch-ref": _torch_reference}, x, local_ids)


def _post_args(workload: dict, _dtype: torch.dtype) -> tuple[int, int, int]:
    rows, hidden = workload["expert_output_shape"]
    tokens, top_k = workload["topk_weights_shape"]
    assert rows == tokens * top_k
    return tokens, top_k, hidden


@pytest.mark.parametrize(
    "tokens,top_k,hidden",
    workload_params(load_workloads(MoePostPermuteFwdOp), _post_args),
)
def test_moe_post_permute_bench(tokens: int, top_k: int, hidden: int) -> None:
    layout = ContiguousLayoutSpec.tight_physical_psum()
    op = MoePostPermuteFwdOp(layout=layout)
    workload = MoeUnpermuteWorkload(tokens, top_k, hidden, torch.bfloat16)
    expert_output, inverse, weights = workload.gen_inputs()
    benchmark = ManifestBenchmark(op, workload)
    inverse_long = inverse.to(torch.int64)

    def _torch_reference(
        output: torch.Tensor,
        routing_weights: torch.Tensor,
        _inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        gathered = output[inverse_long].float()
        weighted = gathered.view(tokens, top_k, hidden) * routing_weights.unsqueeze(-1)
        return weighted.sum(dim=1).to(output.dtype)

    benchmark.compare(
        {"tileops": op, "torch-ref": _torch_reference},
        expert_output,
        weights,
        inverse,
    )
