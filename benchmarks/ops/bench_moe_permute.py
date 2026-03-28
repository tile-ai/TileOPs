"""Benchmark for MoePermutePaddedOp.

Baselines:
  - PyTorch reference: pure Python counting sort + gather.

Note: vLLM moe_permute is not included as a baseline because after the
padded-layout refactor, TileOPs' MoePermutePaddedOp outputs a padding-aligned
buffer ([padded_batch_sum, H]) and fwd_idx, while vLLM's moe_permute
outputs a non-padded buffer ([T*K, H]) with different index semantics.
The two are no longer functionally equivalent and cannot be meaningfully
compared in isolation; use bench_moe_qwen3_moe.py for end-to-end comparison.

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_permute.py -vvs
    conda run -n tileops python benchmarks/ops/bench_moe_permute.py
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_moe_permute import MoePermuteTest, _ref_moe_permute
from tests.test_base import FixtureBase
from tileops.ops.moe import MoePermutePaddedOp

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
        return 0

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        numel = t.total_tokens * t.top_k
        H = t.hidden_size
        elem_bytes = 2  # bf16
        # hidden_states read [T, H] + perm_h_pad write [T*K, H] (actual tokens only, pads are zero-init)
        # + expert_first_token_offset write [E+1]*8 + fwd_idx write [T*K]*4
        # + padded_offsets write [E]*4 + padded_sizes write [E]*4
        return (t.total_tokens * H + numel * H) * elem_bytes + \
               (t.num_experts + 1) * 8 + numel * 4 + t.num_experts * 4 * 2


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
    op = MoePermutePaddedOp(total_tokens, top_k, num_experts, hidden_size, dtype)
    op(hidden_states, topk_ids)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, hidden_states, topk_ids)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # PyTorch reference baseline
    def _ref_fn(hidden_states, topk_ids):
        return _ref_moe_permute(hidden_states, topk_ids, num_experts)

    _ref_fn(hidden_states, topk_ids)  # warmup
    torch.cuda.synchronize()

    result_ref = bm.profile(_ref_fn, hidden_states, topk_ids)
    BenchmarkReport.record(op, locals(), result_ref, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
