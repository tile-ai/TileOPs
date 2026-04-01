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
from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoePermuteNopadOp

# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class MoePermuteNopadTest(TestBase):
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
# Benchmark fixture
# ---------------------------------------------------------------------------


class MoePermuteNopadBenchFixture(FixtureBase):
    """Production-scale configs for throughput benchmarking.

    Columns: total_tokens, top_k, num_experts, hidden_size
    """
    PARAMS = [
        ("total_tokens, top_k, num_experts, hidden_size", [
            # ── Kimi K2: H=7168, E=384, K=8 ─────────────────────────────
            (1,    8, 384, 7168),
            (32,   8, 384, 7168),
            (512,  8, 384, 7168),
            (4096, 8, 384, 7168),
            # ── DeepSeek-V3: H=7168, E=256, K=8 ─────────────────────────
            (1,    8, 256, 7168),
            (32,   8, 256, 7168),
            (512,  8, 256, 7168),
            (4096, 8, 256, 7168),
            # ── Qwen3-235B-A22B: H=7168, E=128, K=8 ─────────────────────
            (1,    8, 128, 7168),
            (32,   8, 128, 7168),
            (512,  8, 128, 7168),
            (4096, 8, 128, 7168),
            # ── Qwen3-30B-A3B: H=3072, E=128, K=8 ───────────────────────
            (1,    8, 128, 3072),
            (32,   8, 128, 3072),
            (512,  8, 128, 3072),
            (4096, 8, 128, 3072),
        ]),
    ]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MoePermuteNopadBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return 0

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        numel = t.total_tokens * t.top_k
        H = t.hidden_size
        elem_bytes = 2  # bf16
        # hidden_states read [T, H] + perm_h write [T*K, H]
        # + expert_first_token_offset write [E+1]*8 + fwd_idx write [T*K]*4
        # + true_offsets write [E]*4 + true_sizes write [E]*4
        return (t.total_tokens * H + numel * H) * elem_bytes + \
               (t.num_experts + 1) * 8 + numel * 4 + t.num_experts * 4 * 2


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@MoePermuteNopadBenchFixture
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
