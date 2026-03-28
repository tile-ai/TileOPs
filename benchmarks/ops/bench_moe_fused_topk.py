"""Benchmark for FusedTopKOp.

Baselines:
  - vLLM fused_topk (optional): only runs when vllm is installed.
  - PyTorch reference: torch.softmax/sigmoid + torch.topk.

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

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport, _shared_cupti_session
from tests.ops.test_moe_fused_topk import FusedTopKTest, _ref_fused_topk
from tests.test_base import FixtureBase
from tileops.ops.moe import FusedTopKOp

# ---------------------------------------------------------------------------
# CUPTI warmup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def warmup_cupti():
    if True:  # bench_kernel manages its own profiler; no external warmup needed
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
# Benchmark fixture
# ---------------------------------------------------------------------------


class FusedTopKBenchFixture(FixtureBase):
    """Production-scale configs for throughput benchmarking.

    Columns: num_tokens, num_experts, top_k, scoring_func, renormalize
    """
    PARAMS = [
        ("num_tokens, num_experts, top_k, scoring_func, renormalize", [
            # ── Qwen3-MoE: 128 experts, top_k=8, softmax ──────────────────
            (512,  128, 8, "softmax", False),
            (2048, 128, 8, "softmax", False),
            (4096, 128, 8, "softmax", False),
            # ── Qwen3.5-MoE: 256 experts, top_k=8, softmax + renorm ───────
            (512,  256, 8, "softmax", True),
            (2048, 256, 8, "softmax", True),
            (4096, 256, 8, "softmax", True),
            # ── DeepSeek-V3/GLM-4: 256 experts, top_k=8, sigmoid + renorm ─
            (512,  256, 8, "sigmoid", True),
            (2048, 256, 8, "sigmoid", True),
            (4096, 256, 8, "sigmoid", True),
        ]),
    ]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class FusedTopKBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Approx: scoring (2*E ops/token) + K-pass argmax (2*E*K comparisons/token)
        return t.num_tokens * t.num_experts * 2 * (1 + t.top_k)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # gating_output read [T, E] bf16 + topk_weights write [T, K] f32 + topk_ids write [T, K] int32
        return (
            t.num_tokens * t.num_experts * 2
            + t.num_tokens * t.top_k * 4
            + t.num_tokens * t.top_k * 4
        )


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

    # PyTorch reference baseline
    def _ref_fn(gating_output):
        return _ref_fused_topk(gating_output, top_k, scoring_func, renormalize)

    _ref_fn(gating_output)  # warmup
    torch.cuda.synchronize()

    result_ref = bm.profile(_ref_fn, gating_output)
    BenchmarkReport.record("fused_topk", locals(), result_ref, tag="pytorch-ref")

    # vLLM baseline (optional)
    if _VLLM_AVAILABLE and scoring_func in ("softmax", "sigmoid"):
        hidden_dummy = torch.empty(num_tokens, 1, device=gating_output.device)
        # Cast bf16→f32 inside the timed call to match TileOPs' input conditions.
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
