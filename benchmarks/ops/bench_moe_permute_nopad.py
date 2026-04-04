"""Benchmark for MoePermuteNopadOp (tight layout, no padding).

Baselines:
  - vLLM moe_permute (optional): vLLM's CUDA kernel for tight permute.
  - PyTorch reference: vectorized gather with counting sort.

Real model configurations:
  Model              H     E    K
  Kimi K2          7168  384   8
  DeepSeek-V3      7168  256   8
  Qwen3-235B-A22B  7168  128   8
  Qwen3-30B-A3B    3072  128   8

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_permute_nopad.py -vvs
    conda run -n tileops python benchmarks/ops/bench_moe_permute_nopad.py
"""

from typing import Optional

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.moe import MoePermuteNopadOp
from workloads.base import WorkloadBase

_OP_NAME = "moe_permute_nopad"

# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class MoePermuteNopadTest(WorkloadBase):
    def __init__(self, total_tokens, top_k, num_experts, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        hidden_states = torch.randn(self.total_tokens, self.hidden_size, dtype=self.dtype, device=dev)
        topk_ids = torch.randint(0, self.num_experts, (self.total_tokens, self.top_k), dtype=torch.int32, device=dev)
        return hidden_states, topk_ids

    def ref_program(self, *args):
        return None


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MoePermuteNopadBenchmark(BenchmarkBase):

    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            t = self.workload
            elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
            self._roofline_cache = eval_roofline(
                _OP_NAME,
                total_tokens=t.total_tokens,
                top_k=t.top_k,
                num_experts=t.num_experts,
                hidden_size=t.hidden_size,
                elem_bytes=elem_bytes,
            )
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


# ---------------------------------------------------------------------------
# Manifest-driven parametrize
# ---------------------------------------------------------------------------


def _manifest_params():
    """Convert manifest workloads to pytest params."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["total_tokens"], w["top_k"], w["num_experts"], w["hidden_size"],
                id=f"{label}-{dtype_str}",
            ))
    return params


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total_tokens, top_k, num_experts, hidden_size",
    _manifest_params(),
)
def test_moe_permute_nopad_bench(
    total_tokens: int, top_k: int, num_experts: int, hidden_size: int
) -> None:
    dtype = torch.bfloat16
    test = MoePermuteNopadTest(total_tokens, top_k, num_experts, hidden_size, dtype)
    bm = MoePermuteNopadBenchmark(test)
    hidden_states, topk_ids = test.gen_inputs()

    # TileOPs
    op = MoePermuteNopadOp(total_tokens, top_k, num_experts, hidden_size, dtype)
    op(hidden_states, topk_ids)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, hidden_states, topk_ids)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # vLLM baseline (optional)
    if _VLLM_AVAILABLE:
        def _vllm_fn(hidden_states, topk_ids):
            return moe_permute(hidden_states, None, topk_ids, num_experts)

        _vllm_fn(hidden_states, topk_ids)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(_vllm_fn, hidden_states, topk_ids)
        BenchmarkReport.record(op, locals(), result_vllm, tag="vllm")
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

        result_torch = bm.profile(_torch_fn, hidden_states, topk_ids)
        BenchmarkReport.record(op, locals(), result_torch, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
