"""Benchmark for Qwen3MoEOp (full fused MoE FFN).

Baselines:
  - PyTorch per-expert reference (always): per-expert GEMM loop, token-major order.
  - vLLM fused_experts (optional): only runs when vllm is installed and scoring_func=softmax.

FLOPs:
  Each selected expert runs:
    gate+up:  T*K * 2*F*H * 2  (multiply-add)
    silu+mul: T*K * F * 3       (sigmoid + mul + mul)
    down:     T*K * H*F * 2     (multiply-add)
  Total ≈ T * K * (4*F*H + 3*F) ≈ T*K * 4*F*H  (dominated by GEMMs)

Memory:
  Reads:   hidden_states [T,H] + w_gate_up [E,2F,H] + w_down [E,H,F]  (bf16)
  Writes:  output [T,H]  (bf16)
  ≈ (T*H + E*(2F+H)*H) * 2  bytes

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_qwen3_moe.py -vvs
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as _vllm_fused_experts
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import FusedTopKOp, Qwen3MoEOp


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
# Test / fixture types
# ---------------------------------------------------------------------------


class Qwen3MoEBenchTest(TestBase):
    def __init__(
        self,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        scoring_func,
        renormalize,
        dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev
        )
        gating = torch.randn(
            self.num_tokens, self.num_experts, dtype=self.dtype, device=dev
        )
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        return hidden, gating, w_gate_up, w_down

    def ref_program(self, *args):
        return None


class Qwen3MoEBenchFixture(FixtureBase):
    """Production-scale configs.

    Columns: num_tokens, num_experts, top_k, hidden_size, ffn_size,
             scoring_func, renormalize, dtype
    """
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " scoring_func, renormalize, dtype",
            [
                # ── Qwen3-MoE: 128 experts, softmax, no renorm ──────────────
                (512,  128, 8, 2048, 1024, "softmax", False, torch.bfloat16),
                (2048, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16),
                (4096, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16),
                # ── Qwen3.5-MoE: 256 experts, softmax, renorm ───────────────
                (512,  256, 8, 2048, 1024, "softmax", True,  torch.bfloat16),
                (2048, 256, 8, 2048, 1024, "softmax", True,  torch.bfloat16),
                (4096, 256, 8, 2048, 1024, "softmax", True,  torch.bfloat16),
                # ── DeepSeek-V3 style: sigmoid, renorm ──────────────────────
                (512,  256, 8, 2048, 1024, "sigmoid", True,  torch.bfloat16),
                (2048, 256, 8, 2048, 1024, "sigmoid", True,  torch.bfloat16),
                (4096, 256, 8, 2048, 1024, "sigmoid", True,  torch.bfloat16),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class Qwen3MoEBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Gate+up: T*K × (2*F×H) MACs = T*K * 2*F*H * 2 FLOPs
        # Down:    T*K × (H×F)   MACs = T*K *   H*F * 2 FLOPs
        # silu+mul: minor, dominated by GEMMs
        return (
            t.num_tokens * t.top_k
            * (2 * t.ffn_size * t.hidden_size * 2   # gate+up
               + t.hidden_size * t.ffn_size * 2)    # down
        )

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        elem = 2  # bf16 = 2 bytes
        # Weights (dominant): w_gate_up + w_down
        w = (t.num_experts * 2 * t.ffn_size * t.hidden_size
             + t.num_experts * t.hidden_size * t.ffn_size) * elem
        # Activations: hidden read + output write
        a = t.num_tokens * t.hidden_size * elem * 2
        return w + a


# ---------------------------------------------------------------------------
# PyTorch per-expert reference
# ---------------------------------------------------------------------------


def _ref_qwen3_moe_bench(
    hidden_states, gating, w_gate_up, w_down,
    topk_weights, topk_ids,
):
    """Memory-efficient per-expert reference (same as test reference)."""
    T, H = hidden_states.shape
    E = w_gate_up.shape[0]
    ffn_dim = w_gate_up.shape[1] // 2
    output = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    topk_ids_i64 = topk_ids.to(torch.int64)
    for e in range(E):
        mask = (topk_ids_i64 == e)
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        h = hidden_states[t_idx].float()
        gate_up = h @ w_gate_up[e].float().t()
        act = F.silu(gate_up[:, :ffn_dim]) * gate_up[:, ffn_dim:]
        down = act @ w_down[e].float().t()
        weights = topk_weights[t_idx, k_idx].float().unsqueeze(-1)
        output.index_add_(0, t_idx, down * weights)
    return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@Qwen3MoEBenchFixture
def test_qwen3_moe_bench(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    scoring_func, renormalize, dtype,
) -> None:
    test = Qwen3MoEBenchTest(
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        scoring_func, renormalize, dtype,
    )
    bm = Qwen3MoEBenchmark(test)
    hidden, gating, w_gate_up, w_down = test.gen_inputs()

    # Pre-compute routing once (shared across all implementations)
    fk = FusedTopKOp(num_tokens, num_experts, top_k, scoring_func, renormalize)
    topk_weights, topk_ids = fk(gating)

    # ── TileOPs ──────────────────────────────────────────────────────────────
    op = Qwen3MoEOp(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        scoring_func=scoring_func,
        renormalize=renormalize,
        dtype=dtype,
    )
    op(hidden, gating, w_gate_up, w_down)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, hidden, gating, w_gate_up, w_down)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # ── PyTorch per-expert baseline ───────────────────────────────────────────
    def _ref_fn(hidden, gating, w_gate_up, w_down):
        return _ref_qwen3_moe_bench(hidden, gating, w_gate_up, w_down,
                                    topk_weights, topk_ids)

    _ref_fn(hidden, gating, w_gate_up, w_down)  # warmup
    torch.cuda.synchronize()

    result_ref = bm.profile(_ref_fn, hidden, gating, w_gate_up, w_down)
    BenchmarkReport.record(op, locals(), result_ref, tag="torch")

    # ── vLLM baseline (softmax only, optional) ───────────────────────────────
    if _VLLM_AVAILABLE and scoring_func == "softmax":
        def _vllm_fn(hidden, gating, w_gate_up, w_down):
            return _vllm_fused_experts(
                hidden, w_gate_up, w_down,
                topk_weights, topk_ids,
            )

        _vllm_fn(hidden, gating, w_gate_up, w_down)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(_vllm_fn, hidden, gating, w_gate_up, w_down)
        BenchmarkReport.record(op, locals(), result_vllm, tag="vllm")
