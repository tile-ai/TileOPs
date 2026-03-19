"""Benchmark for MoePermuteOp (cutlass path).

Baselines:
  - vLLM moe_permute (optional): only runs when vllm is installed.
  - PyTorch reference: pure Python counting sort + gather.

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_permute.py -vvs
    conda run -n tileops python benchmarks/ops/bench_moe_permute.py
"""

from typing import Optional

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
        moe_permute as _vllm_moe_permute,
    )
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_moe_permute import MoePermuteTest, _ref_moe_permute
from tests.test_base import FixtureBase
from tileops.ops.moe import MoePermuteOp

# ---------------------------------------------------------------------------
# CUPTI warmup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def warmup_cupti():
    if not torch.cuda.is_available():
        return
    dummy = torch.empty(1, device="cuda")
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
    ) as prof:
        for _ in range(2):
            dummy.zero_()
            prof.step()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Benchmark fixture (production-scale configs)
# ---------------------------------------------------------------------------


class MoePermuteBenchFixture(FixtureBase):
    """Production-scale configs for throughput benchmarking.

    Columns: total_tokens, top_k, num_experts, hidden_size
    """
    PARAMS = [
        ("total_tokens, top_k, num_experts, hidden_size", [
            # ── Mixtral-8x7B style: 8 experts, top_k=2 ──────────────────────
            (512,  2,   8,  4096),
            (2048, 2,   8,  4096),
            (4096, 2,   8,  4096),
            # ── DeepSeek-V3 style: 256 experts, top_k=8 ─────────────────────
            (512,  8, 256,  7168),
            (2048, 8, 256,  7168),
            (4096, 8, 256,  7168),
            # ── Qwen3 MoE: 128 experts, top_k=8 ─────────────────────────────
            (512,  8, 128,  2048),
            (2048, 8, 128,  2048),
            (4096, 8, 128,  2048),
        ]),
    ]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MoePermuteBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return None

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        numel = t.total_tokens * t.top_k
        H = t.hidden_size
        elem_bytes = 2  # bf16
        # hidden_states read [T, H] + permuted_hidden write [T*K, H]
        # + expert_first_token_offset write [E+1]*8 + inv_permuted_idx write [T*K]*4
        return (t.total_tokens * H + numel * H) * elem_bytes + \
               (t.num_experts + 1) * 8 + numel * 4


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@MoePermuteBenchFixture
def test_moe_permute_bench(
    total_tokens: int, top_k: int, num_experts: int, hidden_size: int
) -> None:
    dtype = torch.bfloat16
    test = MoePermuteTest(total_tokens, top_k, num_experts, hidden_size, dtype)
    bm = MoePermuteBenchmark(test)
    hidden_states, topk_ids = test.gen_inputs()

    # TileOPs
    op = MoePermuteOp(total_tokens, top_k, num_experts, hidden_size, dtype)
    op(hidden_states, topk_ids)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, hidden_states, topk_ids)
    BenchmarkReport.record("moe_permute", locals(), result, tag="tileops")

    # PyTorch reference baseline
    def _ref_fn(hidden_states, topk_ids):
        return _ref_moe_permute(hidden_states, topk_ids, num_experts)

    _ref_fn(hidden_states, topk_ids)  # warmup
    torch.cuda.synchronize()

    result_ref = bm.profile(_ref_fn, hidden_states, topk_ids)
    BenchmarkReport.record("moe_permute", locals(), result_ref, tag="pytorch-ref")

    # vLLM baseline (optional)
    if _VLLM_AVAILABLE:
        def _vllm_fn(hidden_states, topk_ids):
            return _vllm_moe_permute(
                hidden_states, None, topk_ids, num_experts, top_k
            )

        _vllm_fn(hidden_states, topk_ids)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(_vllm_fn, hidden_states, topk_ids)
        BenchmarkReport.record("moe_permute", locals(), result_vllm, tag="vllm")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
