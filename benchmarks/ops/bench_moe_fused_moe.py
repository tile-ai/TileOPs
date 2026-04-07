"""Benchmark for FusedMoe -- unified routed MoE FFN operator.

Covers real model configurations in both decode (T small) and prefill (T large) regimes:

  Model              H     F     E    K  scoring   renorm  bias   scale
  Kimi K2          7168  2048  384   8  sigmoid   True    True   2.827
  DeepSeek-V3      7168  2048  256   8  sigmoid   True    False  1.0
  Qwen3-235B-A22B  7168  2048  128   8  softmax   False   False  1.0
  Qwen3-30B-A3B    3072  1536  128   8  softmax   False   False  1.0

Baselines:
  - tileops-padded: FusedMoe(layout="padded")
  - vllm:           vLLM fused_topk + fused_experts (complete flow)
  - torch-ref:      per-expert GEMM loop (fallback when vLLM is absent)

FLOPs (same formula for both model families):
  Gate+up: T*K * 2*F*H * 2  FLOPs
  Down:    T*K *   H*F * 2  FLOPs
  Total =  T*K * 6*F*H

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_fused_moe.py -vvs
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

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.moe import FusedMoe, FusedTopKOp
from workloads.base import FixtureBase, WorkloadBase

# ---------------------------------------------------------------------------
# Test / fixture types
# ---------------------------------------------------------------------------


class FusedMoeBenchTest(WorkloadBase):
    def __init__(
        self,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
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
        return hidden, gating, correction_bias, w_gate_up, w_down

    def ref_program(self, *args):
        return None


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class FusedMoeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return (
            t.num_tokens * t.top_k
            * (2 * t.ffn_size * t.hidden_size * 2
               + t.hidden_size * t.ffn_size * 2)
        )

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = 2
        w = (t.num_experts * 2 * t.ffn_size * t.hidden_size
             + t.num_experts * t.hidden_size * t.ffn_size) * elem
        a = t.num_tokens * t.hidden_size * elem * 2
        return w + a


class FusedMoeBenchFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " scoring_func, renormalize, with_correction_bias,"
            " routed_scaling_factor, dtype",
            [
                (1,    384, 8, 7168, 2048, "sigmoid", True, True, 2.827, torch.bfloat16),
                (32,   384, 8, 7168, 2048, "sigmoid", True, True, 2.827, torch.bfloat16),
                (512,  384, 8, 7168, 2048, "sigmoid", True, True, 2.827, torch.bfloat16),
                (4096, 384, 8, 7168, 2048, "sigmoid", True, True, 2.827, torch.bfloat16),
                (1,    256, 8, 7168, 2048, "sigmoid", True, False, 1.0, torch.bfloat16),
                (32,   256, 8, 7168, 2048, "sigmoid", True, False, 1.0, torch.bfloat16),
                (512,  256, 8, 7168, 2048, "sigmoid", True, False, 1.0, torch.bfloat16),
                (4096, 256, 8, 7168, 2048, "sigmoid", True, False, 1.0, torch.bfloat16),
                (1,    128, 8, 7168, 2048, "softmax", False, False, 1.0, torch.bfloat16),
                (32,   128, 8, 7168, 2048, "softmax", False, False, 1.0, torch.bfloat16),
                (512,  128, 8, 7168, 2048, "softmax", False, False, 1.0, torch.bfloat16),
                (4096, 128, 8, 7168, 2048, "softmax", False, False, 1.0, torch.bfloat16),
                (1,    128, 8, 3072, 1536, "softmax", False, False, 1.0, torch.bfloat16),
                (32,   128, 8, 3072, 1536, "softmax", False, False, 1.0, torch.bfloat16),
                (512,  128, 8, 3072, 1536, "softmax", False, False, 1.0, torch.bfloat16),
                (4096, 128, 8, 3072, 1536, "softmax", False, False, 1.0, torch.bfloat16),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@FusedMoeBenchFixture
def test_fused_moe_bench(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    scoring_func, renormalize, with_correction_bias,
    routed_scaling_factor, dtype,
) -> None:
    test = FusedMoeBenchTest(
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        scoring_func, renormalize, with_correction_bias,
        routed_scaling_factor, dtype,
    )
    bm = FusedMoeBenchmark(test)
    hidden, gating, correction_bias, w_gate_up, w_down = test.gen_inputs()

    # -- TileOPs nopad -----------------------------------------------------
    op = FusedMoe(
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
    )
    op(hidden, gating, w_gate_up, w_down, correction_bias)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, hidden, gating, w_gate_up, w_down, correction_bias)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # -- TileOPs padded ----------------------------------------------------
    op_pad = FusedMoe(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        scoring_func=scoring_func,
        renormalize=renormalize,
        with_correction_bias=with_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        layout="padded",
        dtype=dtype,
    )
    op_pad(hidden, gating, w_gate_up, w_down, correction_bias)
    torch.cuda.synchronize()

    result_pad = bm.profile(op_pad, hidden, gating, w_gate_up, w_down, correction_bias)
    BenchmarkReport.record(op_pad, locals(), result_pad, tag="tileops-padded")

    # -- vLLM baseline (optional) ------------------------------------------
    if _VLLM_AVAILABLE:
        gating_f32 = gating.float()

        def _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down):
            # vLLM: routing + GEMM (complete flow like TileOPs)
            tw, tids, _ = _vllm_fused_topk(
                hidden_states=hidden,
                gating_output=gating_f32,
                topk=top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
                correction_bias=correction_bias,
            )
            out = _vllm_fused_experts(hidden, w_gate_up, w_down, tw, tids)
            if routed_scaling_factor != 1.0:
                out = out * routed_scaling_factor
            return out

        _vllm_fn(hidden, gating, correction_bias, w_gate_up, w_down)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(
            _vllm_fn, hidden, gating, correction_bias, w_gate_up, w_down
        )
        BenchmarkReport.record(op, locals(), result_vllm, tag="vllm")
    else:
        # torch-ref baseline: memory-efficient per-expert GEMM loop
        fk = FusedTopKOp(num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
                         scoring_func=scoring_func, renormalize=renormalize,
                         with_correction_bias=with_correction_bias)
        topk_weights, topk_ids = fk(gating, correction_bias)
        output_buf = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden.device)
        ids_i64 = topk_ids.to(torch.int64)

        def _ref_fn(hidden, gating, correction_bias, w_gate_up, w_down):
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
            return (output_buf * routed_scaling_factor).to(hidden.dtype)

        _ref_fn(hidden, gating, correction_bias, w_gate_up, w_down)  # warmup
        torch.cuda.synchronize()

        result_ref = bm.profile(
            _ref_fn, hidden, gating, correction_bias, w_gate_up, w_down
        )
        BenchmarkReport.record(op, locals(), result_ref, tag="torch-ref")
