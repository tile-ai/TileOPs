"""Benchmark for KimiMoENopadOp (Kimi K2 / DeepSeekV3-variant MoE FFN).

Baselines:
  - tileops-padded: KimiMoEPaddedOp (block_m-aligned layout, same semantics)
  - vllm:           vLLM fused_experts with pre-computed Kimi K2 routing
                    (optional; runs when vllm is installed)

FLOPs (same formula as Qwen3MoE):
  Gate+up: T*K * 2*F*H * 2  FLOPs
  Down:    T*K *   H*F * 2  FLOPs
  Total ≈  T*K * 4*F*H

Memory (dominant: weight reads):
  w_gate_up [E, 2F, H] + w_down [E, H, F]  (bf16)
  + activations: hidden [T,H] + output [T,H]
  ≈ (E*(2F+H)*H + T*H) * 2 * 2  bytes

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_kimi_moe.py -vvs
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

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport, _shared_cupti_session
from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import FusedTopKOp, KimiMoENopadOp, KimiMoEPaddedOp

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
# Test / fixture types
# ---------------------------------------------------------------------------


class KimiMoEBenchTest(TestBase):
    def __init__(
        self,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        routed_scaling_factor,
        dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
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
        # correction_bias: float32, ~N(0, 0.1) as in production
        correction_bias = torch.randn(
            self.num_experts, dtype=torch.float32, device=dev
        ) * 0.1
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        return hidden, gating, correction_bias, w_gate_up, w_down

    def ref_program(self, *args):
        return None


class KimiMoEBenchFixture(FixtureBase):
    """Kimi K2 production-scale configs: E=384, K=6, H=7168, F=2048."""
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, dtype",
            [
                # ── Kimi K2: E=384, K=6, H=7168, F=2048 ────────────────────
                (512,  384, 6, 7168, 2048, 2.872, torch.bfloat16),
                (2048, 384, 6, 7168, 2048, 2.872, torch.bfloat16),
                (4096, 384, 6, 7168, 2048, 2.872, torch.bfloat16),
                # ── Smaller H for ablation ───────────────────────────────────
                (512,  384, 6, 2048, 1024, 2.872, torch.bfloat16),
                (2048, 384, 6, 2048, 1024, 2.872, torch.bfloat16),
                (4096, 384, 6, 2048, 1024, 2.872, torch.bfloat16),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class KimiMoEBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Gate+up: T*K × (2*F×H) MACs = T*K * 2*F*H * 2 FLOPs
        # Down:    T*K × (H×F)   MACs = T*K *   H*F * 2 FLOPs
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
# Benchmark test
# ---------------------------------------------------------------------------


@KimiMoEBenchFixture
def test_kimi_moe_bench(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    routed_scaling_factor, dtype,
) -> None:
    test = KimiMoEBenchTest(
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        routed_scaling_factor, dtype,
    )
    bm = KimiMoEBenchmark(test)
    hidden, gating, correction_bias, w_gate_up, w_down = test.gen_inputs()

    # Pre-compute routing once (used for vLLM baseline; TileOPs includes routing)
    fk = FusedTopKOp(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        scoring_func="sigmoid",
        renormalize=True,
        with_correction_bias=True,
    )
    topk_weights, topk_ids = fk(gating, correction_bias)

    # ── TileOPs nopad ─────────────────────────────────────────────────────────
    op = KimiMoENopadOp(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        routed_scaling_factor=routed_scaling_factor,
        dtype=dtype,
    )
    op(hidden, gating, correction_bias, w_gate_up, w_down)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, hidden, gating, correction_bias, w_gate_up, w_down)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # ── TileOPs padded ────────────────────────────────────────────────────────
    op_pad = KimiMoEPaddedOp(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        routed_scaling_factor=routed_scaling_factor,
        dtype=dtype,
    )
    op_pad(hidden, gating, correction_bias, w_gate_up, w_down)
    torch.cuda.synchronize()

    result_pad = bm.profile(op_pad, hidden, gating, correction_bias, w_gate_up, w_down)
    BenchmarkReport.record(op_pad, locals(), result_pad, tag="tileops-padded")

    # ── vLLM baseline (optional) ──────────────────────────────────────────────
    # Uses pre-computed Kimi K2 routing (FusedTopKOp) + vLLM fused_experts for
    # a fair GEMM/SwiGLU/unpermute comparison.
    if _VLLM_AVAILABLE:
        def _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down):
            return _vllm_fused_experts(
                hidden, w_gate_up, w_down,
                topk_weights, topk_ids,
            )

        _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(
            _vllm_fn, hidden, gating, correction_bias, w_gate_up, w_down
        )
        BenchmarkReport.record(op, locals(), result_vllm, tag="vllm")
    else:
        # torch-ref baseline: memory-efficient per-expert GEMM loop
        def _ref_fn(hidden, gating, correction_bias, w_gate_up, w_down):
            T, H = hidden.shape
            E = w_gate_up.shape[0]
            ffn_dim = w_gate_up.shape[1] // 2
            output = torch.zeros(T, H, dtype=torch.float32, device=hidden.device)
            ids_i64 = topk_ids.to(torch.int64)
            for e in range(E):
                mask = (ids_i64 == e)
                if not mask.any():
                    continue
                t_idx, k_idx = mask.nonzero(as_tuple=True)
                h = hidden[t_idx].float()
                gate_up = h @ w_gate_up[e].float().t()
                act = F.silu(gate_up[:, :ffn_dim]) * gate_up[:, ffn_dim:]
                down = act @ w_down[e].float().t()
                weights = topk_weights[t_idx, k_idx].float().unsqueeze(-1)
                output.index_add_(0, t_idx, down * weights)
            return (output * routed_scaling_factor).to(hidden.dtype)

        _ref_fn(hidden, gating, correction_bias, w_gate_up, w_down)  # warmup
        torch.cuda.synchronize()

        result_ref = bm.profile(
            _ref_fn, hidden, gating, correction_bias, w_gate_up, w_down
        )
        BenchmarkReport.record(op, locals(), result_ref, tag="torch-ref")
