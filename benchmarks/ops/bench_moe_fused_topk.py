"""Benchmark for FusedTopKOp.

Baselines:
  - vLLM fused_topk (optional): only runs when vllm is installed.
  - PyTorch reference: torch.softmax/sigmoid + torch.topk.

Real model configurations:
  Model              E    K  scoring   renorm
  Kimi K2          384   8  sigmoid   True
  DeepSeek-V3      256   8  sigmoid   True
  Qwen3-235B-A22B  128   8  softmax   False
  Qwen3-30B-A3B    128   8  softmax   False

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_fused_topk.py -vvs
"""

from typing import Optional

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe import fused_topk as _vllm_fused_topk
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_moe_fused_topk import FusedTopKTest, _ref_fused_topk
from tests.test_base import FixtureBase
from tileops.ops.moe import FusedTopKOp

# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class FusedTopKBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return t.num_tokens * t.num_experts * 2 * (1 + t.top_k)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return (
            t.num_tokens * t.num_experts * 2
            + t.num_tokens * t.top_k * 4
            + t.num_tokens * t.top_k * 4
        )


class FusedTopKBenchFixture(FixtureBase):
    PARAMS = [
        ("num_tokens, num_experts, top_k, scoring_func, renormalize", [
            (1,    384, 8, "sigmoid", True),
            (32,   384, 8, "sigmoid", True),
            (512,  384, 8, "sigmoid", True),
            (4096, 384, 8, "sigmoid", True),
            (1,    256, 8, "sigmoid", True),
            (32,   256, 8, "sigmoid", True),
            (512,  256, 8, "sigmoid", True),
            (4096, 256, 8, "sigmoid", True),
            (1,    128, 8, "softmax", False),
            (32,   128, 8, "softmax", False),
            (512,  128, 8, "softmax", False),
            (4096, 128, 8, "softmax", False),
        ]),
    ]


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@FusedTopKBenchFixture
def test_fused_topk_bench(
    num_tokens: int, num_experts: int, top_k: int, scoring_func: str, renormalize: bool
) -> None:
    dtype = torch.bfloat16
    test = FusedTopKTest(num_tokens, num_experts, top_k, scoring_func, renormalize, dtype)
    bm = FusedTopKBenchmark(test)
    gating_output = test.gen_inputs()

    # TileOPs
    op = FusedTopKOp(num_tokens, num_experts, top_k, scoring_func, renormalize)
    op(gating_output)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, gating_output)
    BenchmarkReport.record("fused_topk", locals(), result, tag="tileops")

    # vLLM baseline (optional)
    has_external = False
    if _VLLM_AVAILABLE and scoring_func in ("softmax", "sigmoid"):
        has_external = True
        hidden_dummy = torch.empty(num_tokens, 1, device=gating_output.device)
        # Cast bf16->f32 inside the timed call to match TileOPs' input conditions.
        def _vllm_fn(gating_output):
            return _vllm_fused_topk(
                hidden_states=hidden_dummy,
                gating_output=gating_output.float(),
                topk=top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
            )

        _vllm_fn(gating_output)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(_vllm_fn, gating_output)
        BenchmarkReport.record("fused_topk", locals(), result_vllm, tag="vllm")

    # Fallback: torch reference baseline (only when no external baselines)
    if not has_external:
        def _ref_fn(gating_output):
            return _ref_fused_topk(gating_output, top_k, scoring_func, renormalize)

        _ref_fn(gating_output)  # warmup
        torch.cuda.synchronize()

        result_ref = bm.profile(_ref_fn, gating_output)
        BenchmarkReport.record("fused_topk", locals(), result_ref, tag="torch-ref")
