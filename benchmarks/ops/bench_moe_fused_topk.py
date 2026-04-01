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
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.moe import FusedTopKOp

_OP_NAME = "moe_fused_topk"

# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class FusedTopKBenchmark(BenchmarkBase):

    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            t = self.test
            self._roofline_cache = eval_roofline(
                _OP_NAME,
                num_tokens=t.num_tokens,
                num_experts=t.num_experts,
                top_k=t.top_k,
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
                w["num_tokens"], w["num_experts"], w["top_k"],
                w["scoring_func"], w["renormalize"],
                id=f"{label}-{dtype_str}",
            ))
    return params


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, scoring_func, renormalize",
    _manifest_params(),
)
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
