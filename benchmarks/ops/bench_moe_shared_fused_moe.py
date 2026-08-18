"""Benchmark for SharedFusedMoE — FusedMoE with shared expert support.

Covers Kimi K2 configuration (the primary model with shared experts):

  Model    H     F     E    K  Fs     scoring   renorm  bias   scale
  Kimi K2  7168  2048  384  8  18432  sigmoid   True    True   2.827

Baselines:
  - vllm: fused_topk + fused_experts + F.linear shared MLP. Absent without vLLM
    installed -- no row is recorded rather than a slower stand-in.

FLOPs:
  Routed:  T*K * 6*F*H   (gate+up + down)
  Shared:  T   * 6*Fs*H  (gate+up + down)
  Total  = T*K*6*F*H + T*6*Fs*H
"""

import warnings
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
from tileops.ops.moe import SharedFusedMoE
from workloads.moe import SharedFusedMoeWorkload
from workloads.workload_base import FixtureBase

# Test / fixture types


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


# Benchmark class


class SharedFusedMoEBenchmark(BenchmarkBase[SharedFusedMoeWorkload]):

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
        # A call reads only the experts its tokens routed to. With T tokens each picking
        # K of E, and this workload drawing gating at random, the expected number of
        # distinct experts is E * (1 - (1 - K/E)^T): K of them at T=1, all E once T*K
        # covers the pool. Counting all E regardless is right in the prefill limit and
        # overstates decode traffic by E/K -- 48x on the Kimi K2 config below.
        touched = t.num_experts * (
            1.0 - (1.0 - t.top_k / t.num_experts) ** t.num_tokens
        )
        routed_w = touched * 3 * t.ffn_size * t.hidden_size * elem
        shared_w = 3 * t.shared_ffn_size * t.hidden_size * elem
        act = t.num_tokens * t.hidden_size * elem * 2
        return routed_w + shared_w + act


# Benchmark test


@SharedFusedMoEBenchFixture
def test_shared_fused_moe_bench(
    num_tokens, num_experts, top_k, hidden_size, ffn_size, shared_ffn_size,
    scoring_func, renormalize, with_correction_bias,
    routed_scaling_factor, dtype,
) -> None:
    test = SharedFusedMoeWorkload(
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
        routed_scaling_factor=routed_scaling_factor,
        shared_ffn_size=shared_ffn_size,
    )
    op(hidden, gating, w_gate_up, w_down, correction_bias,
       shared_w_gate_up=shared_w_gate_up, shared_w_down=shared_w_down)  # warmup / JIT compile
    torch.cuda.synchronize()

    def _tileops_fn(hidden, gating, w_gate_up, w_down, correction_bias,
                    shared_w_gate_up, shared_w_down):
        return op(hidden, gating, w_gate_up, w_down, correction_bias,
                  shared_w_gate_up=shared_w_gate_up, shared_w_down=shared_w_down)

    functors = {"tileops": _tileops_fn}

    # ── vLLM baseline (optional) ──────────────────────────────────────────────
    if _VLLM_AVAILABLE:
        # vLLM shared expert: separate gate/up weights [Fs, H]
        sw_gate = shared_w_gate_up[:shared_ffn_size]   # [Fs, H]
        sw_up   = shared_w_gate_up[shared_ffn_size:]   # [Fs, H]
        sw_d    = shared_w_down                         # [H, Fs]

        def _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down,
                     shared_w_gate_up, shared_w_down):
            tw, tids, _ = _vllm_fused_topk(
                hidden_states=hidden,
                gating_output=gating.float(),
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

        functors["vllm"] = (_vllm_fn, (hidden, gating, correction_bias, w_gate_up, w_down, shared_w_gate_up, shared_w_down, ))
    else:
        # No baseline rather than a misleading one. The per-expert Python loop this used
        # to time is a correctness reference: it upcasts to fp32 and index_add_s one
        # expert at a time, so "TileOPs is 30x faster" said nothing about either.
        warnings.warn(
            "vLLM is not installed; recording no baseline for SharedFusedMoE. "
            "Install vllm to compare against fused_topk + fused_experts.",
            stacklevel=2,
        )

    bm.compare(functors, hidden, gating, w_gate_up, w_down, correction_bias, shared_w_gate_up, shared_w_down, record_as=op, params=locals())
