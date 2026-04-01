"""Benchmark for MoeUnpermuteOp.

Baselines:
  - vLLM moe_unpermute (optional): vLLM's CUDA kernel.
  - PyTorch reference: gather + weighted view+sum.

Note: vLLM uses inv_permuted_idx (reverse mapping) while TileOPs uses fwd_idx
(forward mapping); inv_permuted_idx is derived from fwd_idx before benchmarking.

Real model configurations:
  Model              H     K
  Kimi K2          7168   8
  DeepSeek-V3      7168   8
  Qwen3-235B-A22B  7168   8
  Qwen3-30B-A3B    3072   8

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_unpermute.py -vvs
    conda run -n tileops python benchmarks/ops/bench_moe_unpermute.py
"""

from typing import Optional

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_unpermute
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_moe_unpermute import MoeUnpermuteTest
from tests.test_base import FixtureBase
from tileops.ops.moe import MoeUnpermuteOp

# ---------------------------------------------------------------------------
# Benchmark fixture (production-scale configs)
# ---------------------------------------------------------------------------


class MoeUnpermuteBenchFixture(FixtureBase):
    """Production-scale configs for throughput benchmarking.

    Columns: total_tokens, top_k, hidden_size
    """
    PARAMS = [
        ("total_tokens, top_k, hidden_size", [
            # ── Kimi K2 / DeepSeek-V3 / Qwen3-235B-A22B: H=7168, K=8 ────
            (1,    8, 7168),
            (32,   8, 7168),
            (512,  8, 7168),
            (4096, 8, 7168),
            # ── Qwen3-30B-A3B: H=3072, K=8 ───────────────────────────────
            (1,    8, 3072),
            (32,   8, 3072),
            (512,  8, 3072),
            (4096, 8, 3072),
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

    # vLLM baseline (optional)
    if _VLLM_AVAILABLE:
        # vLLM uses inv_permuted_idx (reverse mapping: padded_slot → flat_idx)
        # Compute from fwd_idx (forward mapping: flat_idx → padded_slot)
        numel = total_tokens * top_k
        inv_permuted_idx = torch.empty(numel, dtype=torch.int32, device=fwd_idx.device)
        inv_permuted_idx[fwd_idx.long()] = torch.arange(numel, dtype=torch.int32, device=fwd_idx.device)
        out_vllm = torch.empty(total_tokens, hidden_size, dtype=mm2_pad.dtype, device=mm2_pad.device)

        def _vllm_fn(mm2_pad, fwd_idx, topk_weights):
            moe_unpermute(out_vllm, mm2_pad, topk_weights, inv_permuted_idx)
            return out_vllm

        _vllm_fn(mm2_pad, fwd_idx, topk_weights)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(_vllm_fn, mm2_pad, fwd_idx, topk_weights)
        BenchmarkReport.record("moe_unpermute", locals(), result_vllm, tag="vllm")
    else:
        # Fallback: PyTorch vectorized baseline (gather + weighted sum)
        fwd_idx_long = fwd_idx.long()
        topk_weights_f32 = topk_weights.float()

        def _torch_fn(mm2_pad, fwd_idx, topk_weights):
            gathered = mm2_pad[fwd_idx_long].float()                  # [T*K, H]
            weighted_sum = (gathered.view(total_tokens, top_k, hidden_size)
                            * topk_weights_f32.unsqueeze(-1)).sum(dim=1)  # [T, H]
            return weighted_sum.to(mm2_pad.dtype)

        _torch_fn(mm2_pad, fwd_idx, topk_weights)  # warmup
        torch.cuda.synchronize()

        result_torch = bm.profile(_torch_fn, mm2_pad, fwd_idx, topk_weights)
        BenchmarkReport.record("moe_unpermute", locals(), result_torch, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
