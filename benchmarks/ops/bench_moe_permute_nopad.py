"""Benchmark for MoePermuteNopadFwdOp (tight layout, no padding).

Baselines:
  - vLLM moe_permute (optional): vLLM's CUDA kernel for tight permute.
  - PyTorch reference: vectorized gather with counting sort.

Real model configurations:
  Model              H     E    K
  Kimi K2          7168  384   8
  DeepSeek-V3      7168  256   8
  Qwen3-235B-A22B  7168  128   8
  Qwen3-30B-A3B    3072  128   8
"""

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.moe import MoePermuteNopadFwdOp
from workloads.moe import MoePermuteWorkload

_OP_NAME = "MoePermuteNopadFwdOp"

# Benchmark class


# Manifest-driven parametrize


def _manifest_params():
    """Convert manifest workloads to pytest params."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        total_tokens, hidden_size = w["hidden_states_shape"]
        topk_tokens, top_k = w["topk_ids_shape"]
        assert topk_tokens == total_tokens
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                total_tokens, top_k, w["num_experts"], w["num_experts_local"],
                hidden_size,
                id=f"{label}-{dtype_str}",
            ))
    return params


# Benchmark test


@pytest.mark.parametrize(
    "total_tokens, top_k, num_experts, num_experts_local, hidden_size",
    _manifest_params(),
)
def test_moe_permute_nopad_bench(
    total_tokens: int, top_k: int, num_experts: int, num_experts_local: int,
    hidden_size: int,
) -> None:
    dtype = torch.bfloat16
    workload = MoePermuteWorkload(total_tokens, top_k, num_experts, hidden_size, dtype)
    torch.manual_seed(42)
    hidden_states, topk_ids = workload.gen_inputs()

    # Under expert parallelism this rank owns the first num_experts_local ids;
    # the rest belong elsewhere.
    expert_map = None
    if num_experts_local < num_experts:
        expert_map = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=hidden_states.device)
        expert_map[:num_experts_local] = torch.arange(
            num_experts_local, dtype=torch.int32, device=hidden_states.device)

    # TileOPs
    op = MoePermuteNopadFwdOp(
        num_experts=num_experts, num_experts_local=num_experts_local)
    bm = ManifestBenchmark(_OP_NAME, op, workload)
    op(hidden_states, topk_ids, expert_map)  # warmup / JIT compile
    torch.cuda.synchronize()

    functors = {"tileops": op}

    if expert_map is not None:
        # No vLLM or torch column: their permute takes the whole expert table, so
        # the two would not measure the same work.
        bm.compare(
            functors, hidden_states, topk_ids, expert_map,
            record_as=op, params=locals(),
        )
        return

    # vLLM baseline (optional)
    if _VLLM_AVAILABLE:
        def _vllm_fn(hidden_states, topk_ids):
            return moe_permute(hidden_states, None, topk_ids, num_experts)

        _vllm_fn(hidden_states, topk_ids)  # warmup
        torch.cuda.synchronize()

        functors["vllm"] = _vllm_fn
    else:
        # PyTorch vectorized baseline: counting sort + gather
        numel = total_tokens * top_k
        perm_h_buf = torch.empty(numel, hidden_size, dtype=dtype, device=hidden_states.device)
        token_indices = torch.arange(total_tokens, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).flatten()
        scatter_indices = torch.empty(numel, dtype=torch.int64, device=hidden_states.device)

        def _torch_fn(hidden_states, topk_ids):
            gathered = hidden_states[token_indices]  # [T*K, H]
            flat_ids = topk_ids.flatten().to(torch.int64)

            # Vectorized counting and offsets
            counts = torch.bincount(flat_ids, minlength=num_experts)
            true_offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=flat_ids.device),
                                       counts.cumsum(0)[:-1]])

            # Sort by expert, compute within-expert rank, then invert
            sorted_idx = torch.argsort(flat_ids, stable=True)
            sorted_experts = flat_ids[sorted_idx]
            expert_first = torch.cat([torch.zeros(1, dtype=torch.int64, device=flat_ids.device),
                                       counts.cumsum(0)[:-1]])
            within_rank = torch.arange(numel, device=flat_ids.device) - expert_first[sorted_experts]
            scatter_for_sorted = true_offsets[sorted_experts] + within_rank
            scatter_indices[sorted_idx] = scatter_for_sorted

            perm_h_buf[scatter_indices] = gathered
            return perm_h_buf, true_offsets.to(torch.int32), counts.to(torch.int32)

        _torch_fn(hidden_states, topk_ids)  # warmup
        torch.cuda.synchronize()

        functors["torch-ref"] = _torch_fn

    bm.compare(functors, hidden_states, topk_ids, record_as=op, params=locals())
