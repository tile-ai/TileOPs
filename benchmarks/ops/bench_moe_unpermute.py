"""Benchmark for MoeUnpermuteOp.

Baselines:
  - PyTorch vectorized: index_select + weighted scatter_add.

Note: vLLM moe_unpermute is not included as a baseline because after the
padded-layout refactor, TileOPs' MoeUnpermuteOp accepts a padding-aligned
input buffer ([padded_batch_sum, H]) and fwd_idx (flat_idx → padded_slot),
while vLLM's moe_unpermute accepts a non-padded buffer ([T*K, H]) with
different index semantics. The two are no longer functionally equivalent
and cannot be meaningfully compared in isolation; use bench_moe_qwen3_moe.py
for end-to-end comparison.

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_unpermute.py -vvs
    conda run -n tileops python benchmarks/ops/bench_moe_unpermute.py
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_moe_unpermute import MoeUnpermuteTest
from tests.test_base import FixtureBase
from tileops.ops.moe import MoeUnpermuteOp

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


class MoeUnpermuteBenchFixture(FixtureBase):
    """Production-scale configs for throughput benchmarking.

    Columns: total_tokens, top_k, hidden_size
    """
    PARAMS = [
        ("total_tokens, top_k, hidden_size", [
            # ── Mixtral-8x7B style: top_k=2 ─────────────────────────────────
            (512,  2,  4096),
            (2048, 2,  4096),
            (4096, 2,  4096),
            # ── DeepSeek-V3 style: top_k=8 ──────────────────────────────────
            (512,  8,  7168),
            (2048, 8,  7168),
            (4096, 8,  7168),
            # ── Qwen3 MoE: top_k=8 ──────────────────────────────────────────
            (512,  8,  2048),
            (2048, 8,  2048),
            (4096, 8,  2048),
        ]),
    ]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MoeUnpermuteBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # multiply + add per element per expert slot: 2 * T*K * H
        return 2 * t.total_tokens * t.top_k * t.hidden_size

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        numel = t.total_tokens * t.top_k
        H = t.hidden_size
        elem_bytes = 2  # bf16
        # mm2_pad read [padded_batch_sum, H] ≈ [T*K, H] in standalone bench (padded_batch_sum=T*K)
        # + output write [T, H] + fwd_idx read [T*K]*4 + topk_weights read [T*K]*4
        return (numel * H + t.total_tokens * H) * elem_bytes + numel * 4 + numel * 4


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@MoeUnpermuteBenchFixture
def test_moe_unpermute_bench(total_tokens: int, top_k: int, hidden_size: int) -> None:
    dtype = torch.bfloat16
    test = MoeUnpermuteTest(total_tokens, top_k, hidden_size, dtype)
    bm = MoeUnpermuteBenchmark(test)
    mm2_pad, fwd_idx, topk_weights = test.gen_inputs()

    # TileOPs
    op = MoeUnpermuteOp(total_tokens, top_k, hidden_size, dtype)
    op(mm2_pad, fwd_idx, topk_weights)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, mm2_pad, fwd_idx, topk_weights)
    BenchmarkReport.record("moe_unpermute", locals(), result, tag="tileops")

    # PyTorch vectorized baseline: index_select + weighted scatter_add
    def _torch_fn(mm2_pad, fwd_idx, topk_weights):
        # gather rows at fwd_idx positions, weight, then scatter-add per token
        src = mm2_pad[fwd_idx.long()].float()            # [T*K, H]
        w = topk_weights.flatten().unsqueeze(1)          # [T*K, 1]
        weighted = src * w                               # [T*K, H], float32
        out = torch.zeros(total_tokens, hidden_size, dtype=torch.float32, device=mm2_pad.device)
        token_idx = torch.arange(total_tokens, device=mm2_pad.device).repeat_interleave(top_k)
        out.scatter_add_(0, token_idx.unsqueeze(1).expand_as(weighted), weighted)
        return out.to(mm2_pad.dtype)

    _torch_fn(mm2_pad, fwd_idx, topk_weights)  # warmup
    torch.cuda.synchronize()

    result_torch = bm.profile(_torch_fn, mm2_pad, fwd_idx, topk_weights)
    BenchmarkReport.record("moe_unpermute", locals(), result_torch, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
