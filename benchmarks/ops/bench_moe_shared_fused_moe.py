"""Benchmark for SharedFusedMoE — FusedMoE with shared expert support.

Covers Kimi K2 configuration (the primary model with shared experts):

  Model    H     F     E    K  Fs     scoring   renorm  bias   scale
  Kimi K2  7168  2048  384  8  18432  sigmoid   True    True   2.827

Baselines:
  - vllm:      vLLM fused_topk + fused_experts + F.linear shared MLP
  - torch-ref: per-expert GEMM loop + manual shared MLP (fallback when vLLM absent)

FLOPs:
  Routed:  T*K * 6*F*H   (gate+up + down)
  Shared:  T   * 6*Fs*H  (gate+up + down)
  Total  = T*K*6*F*H + T*6*Fs*H

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_shared_fused_moe.py -vvs
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk as _vllm_fused_topk,
    )
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.moe import FusedTopKOp, SharedFusedMoE
from workloads.workload_base import FixtureBase, WorkloadBase

# ---------------------------------------------------------------------------
# Test / fixture types
# ---------------------------------------------------------------------------


class SharedFusedMoEBenchTest(WorkloadBase):
    def __init__(
        self,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        shared_ffn_size,
        scoring_func,
        renormalize,
        with_correction_bias,
        routed_scaling_factor,
        dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.shared_ffn_size = shared_ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
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
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev) * 0.1
            if self.with_correction_bias else None
        )
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        # Shared expert weights: gate+up concatenated [2*Fs, H], down [H, Fs]
        shared_w_gate_up = torch.randn(
            self.shared_ffn_size * 2, self.hidden_size, dtype=self.dtype, device=dev
        ) * 0.02
        shared_w_down = torch.randn(
            self.hidden_size, self.shared_ffn_size, dtype=self.dtype, device=dev
        ) * 0.02
        return hidden, gating, correction_bias, w_gate_up, w_down, shared_w_gate_up, shared_w_down


class SharedFusedMoEBenchFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size, shared_ffn_size,"
            " scoring_func, renormalize, with_correction_bias,"
            " routed_scaling_factor, dtype",
            [
                # ── Kimi K2: E=384, K=8, H=7168, F=2048, Fs=18432, sigmoid+bias ──
                pytest.param(
                    1,    384, 8, 7168, 2048, 18432, "sigmoid", True, True, 2.827, torch.bfloat16,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    32,   384, 8, 7168, 2048, 18432, "sigmoid", True, True, 2.827, torch.bfloat16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    512,  384, 8, 7168, 2048, 18432, "sigmoid", True, True, 2.827, torch.bfloat16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    2048, 384, 8, 7168, 2048, 18432, "sigmoid", True, True, 2.827, torch.bfloat16,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    4096, 384, 8, 7168, 2048, 18432, "sigmoid", True, True, 2.827, torch.bfloat16,
                    marks=pytest.mark.full,
                ),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class SharedFusedMoEBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        routed = t.num_tokens * t.top_k * (
            2 * t.ffn_size * t.hidden_size * 2   # gate+up
            + t.hidden_size * t.ffn_size * 2      # down
        )
        shared = t.num_tokens * (
            2 * t.shared_ffn_size * t.hidden_size * 2   # gate+up
            + t.hidden_size * t.shared_ffn_size * 2      # down
        )
        return routed + shared

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = 2  # bf16 = 2 bytes
        routed_w = (
            t.num_experts * 2 * t.ffn_size * t.hidden_size
            + t.num_experts * t.hidden_size * t.ffn_size
        ) * elem
        shared_w = (
            2 * t.shared_ffn_size * t.hidden_size
            + t.hidden_size * t.shared_ffn_size
        ) * elem
        act = t.num_tokens * t.hidden_size * elem * 2
        return routed_w + shared_w + act


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@SharedFusedMoEBenchFixture
def test_shared_fused_moe_bench(
    num_tokens, num_experts, top_k, hidden_size, ffn_size, shared_ffn_size,
    scoring_func, renormalize, with_correction_bias,
    routed_scaling_factor, dtype,
) -> None:
    test = SharedFusedMoEBenchTest(
        num_tokens, num_experts, top_k, hidden_size, ffn_size, shared_ffn_size,
        scoring_func, renormalize, with_correction_bias,
        routed_scaling_factor, dtype,
    )
    bm = SharedFusedMoEBenchmark(test)
    hidden, gating, correction_bias, w_gate_up, w_down, shared_w_gate_up, shared_w_down = test.gen_inputs()

    # ── TileOPs ───────────────────────────────────────────────────────────────
    op = SharedFusedMoE(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        scoring_func=scoring_func,
        renormalize=renormalize,
        with_correction_bias=with_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        layout="nopad",
        dtype=dtype,
        shared_ffn_size=shared_ffn_size,
    )
    op(hidden, gating, w_gate_up, w_down, correction_bias,
       shared_w_gate_up=shared_w_gate_up, shared_w_down=shared_w_down)  # warmup / JIT compile
    torch.cuda.synchronize()

    def _tileops_fn(hidden, gating, w_gate_up, w_down, correction_bias,
                    shared_w_gate_up, shared_w_down):
        return op(hidden, gating, w_gate_up, w_down, correction_bias,
                  shared_w_gate_up=shared_w_gate_up, shared_w_down=shared_w_down)

    result = bm.profile(
        _tileops_fn, hidden, gating, w_gate_up, w_down, correction_bias,
        shared_w_gate_up, shared_w_down,
    )
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # ── vLLM baseline (optional) ──────────────────────────────────────────────
    if _VLLM_AVAILABLE:
        gating_f32 = gating.float()
        # vLLM shared expert: separate gate/up weights [Fs, H]
        sw_gate = shared_w_gate_up[:shared_ffn_size]   # [Fs, H]
        sw_up   = shared_w_gate_up[shared_ffn_size:]   # [Fs, H]
        sw_d    = shared_w_down                         # [H, Fs]

        def _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down,
                     shared_w_gate_up, shared_w_down):
            tw, tids, _ = _vllm_fused_topk(
                hidden_states=hidden,
                gating_output=gating_f32,
                topk=top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
            )
            routed_out = _vllm_fused_experts(hidden, w_gate_up, w_down, tw, tids)
            if routed_scaling_factor != 1.0:
                routed_out = routed_out * routed_scaling_factor
            # Shared expert: gate+up GEMM → SiLU → down GEMM
            gate = F.linear(hidden, sw_gate)   # [T, Fs]
            up   = F.linear(hidden, sw_up)     # [T, Fs]
            act  = F.silu(gate) * up
            shared_out = F.linear(act, sw_d)   # [T, H]
            return shared_out, routed_out

        _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down,
                 shared_w_gate_up, shared_w_down)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(
            _vllm_fn, hidden, gating, correction_bias, w_gate_up, w_down,
            shared_w_gate_up, shared_w_down,
        )  # all positional — OK
        BenchmarkReport.record(op, locals(), result_vllm, tag="vllm")
    else:
        # torch-ref: per-expert GEMM loop + manual shared MLP
        fk = FusedTopKOp(
            num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
            scoring_func=scoring_func, renormalize=renormalize,
            with_correction_bias=with_correction_bias,
        )
        topk_weights, topk_ids = fk(gating, correction_bias)
        output_buf = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden.device)
        ids_i64 = topk_ids.to(torch.int64)

        def _ref_fn(hidden, gating, correction_bias, w_gate_up, w_down,
                    shared_w_gate_up, shared_w_down):
            E = w_gate_up.shape[0]
            ffn_dim = w_gate_up.shape[1] // 2
            output_buf.zero_()
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
                output_buf.index_add_(0, t_idx, down * weights)
            routed_out = (output_buf * routed_scaling_factor).to(hidden.dtype)
            # Shared expert
            sw_gate = shared_w_gate_up[:shared_ffn_size]
            sw_up   = shared_w_gate_up[shared_ffn_size:]
            gate = F.linear(hidden, sw_gate)
            up   = F.linear(hidden, sw_up)
            act  = F.silu(gate) * up
            shared_out = F.linear(act, shared_w_down)
            return shared_out, routed_out

        _ref_fn(hidden, gating, correction_bias, w_gate_up, w_down,
                shared_w_gate_up, shared_w_down)  # warmup
        torch.cuda.synchronize()

        result_ref = bm.profile(
            _ref_fn, hidden, gating, correction_bias, w_gate_up, w_down,
            shared_w_gate_up, shared_w_down,
        )
        BenchmarkReport.record(op, locals(), result_ref, tag="torch-ref")
