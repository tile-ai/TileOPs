"""Benchmark for FusedMoeSharedExpertFwdOp — FusedMoE with shared expert support.

Workload shapes come from the op's manifest ``workloads`` (via
``load_workloads``); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``) and a vLLM baseline.

Coverage: Kimi K2, DeepSeek-V3 and GLM-4.5, each at a small-route, a decode and
a prefill token count — all three route with sigmoid and a correction bias.

Baselines:
  - vllm: fused_topk_bias + fused_experts + F.linear shared MLP. Absent without
    vLLM installed -- no row is recorded rather than a slower stand-in.
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias as _vllm_fused_topk_bias,
    )

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark_base import (
    ManifestBenchmark,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops.moe import FusedMoeSharedExpertFwdOp
from workloads.moe import FusedMoeSharedExpertWorkload


def _fused_moe_shared_expert_args(w: dict, dtype: torch.dtype) -> tuple:
    """Positional args for one shared-MoE case, in the order the test declares them."""
    return (
        w["num_tokens"],
        w["num_experts"],
        w["top_k"],
        w["hidden_size"],
        w["ffn_size"],
        w.get("shared_ffn_size"),
        w["scoring_func"],
        bool(w.get("renormalize", False)),
        "correction_bias_shape" in w,
        float(w.get("routed_scaling_factor", 1.0)),
        dtype,
    )


_FWD_PARAMS = workload_params(
    load_workloads(FusedMoeSharedExpertFwdOp), _fused_moe_shared_expert_args
)


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, ffn_size, shared_ffn_size,"
    " scoring_func, renormalize, with_correction_bias,"
    " routed_scaling_factor, dtype",
    _FWD_PARAMS,
)
def test_fused_moe_shared_expert_bench(
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
) -> None:
    test = FusedMoeSharedExpertWorkload(
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
    )
    hidden, gating, correction_bias, w_gate_up, w_down, shared_w_gate_up, shared_w_down = (
        test.gen_inputs()
    )

    # ── TileOPs ───────────────────────────────────────────────────────────────
    op = FusedMoeSharedExpertFwdOp(
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
    bm = ManifestBenchmark(op, test)
    tileops_out = op(
        hidden,
        gating,
        w_gate_up,
        w_down,
        correction_bias,
        shared_w_gate_up=shared_w_gate_up,
        shared_w_down=shared_w_down,
    )  # warmup / JIT compile
    torch.cuda.synchronize()

    def _tileops_fn(
        hidden, gating, w_gate_up, w_down, correction_bias, shared_w_gate_up, shared_w_down
    ):
        return op(
            hidden,
            gating,
            w_gate_up,
            w_down,
            correction_bias,
            shared_w_gate_up=shared_w_gate_up,
            shared_w_down=shared_w_down,
        )

    functors = {"tileops": _tileops_fn}

    # ── vLLM baseline (optional) ──────────────────────────────────────────────
    if _VLLM_AVAILABLE and shared_ffn_size is not None:
        # vLLM shared expert: separate gate/up weights [Fs, H]
        sw_gate = shared_w_gate_up[:shared_ffn_size]  # [Fs, H]
        sw_up = shared_w_gate_up[shared_ffn_size:]  # [Fs, H]
        sw_d = shared_w_down  # [H, Fs]

        def _vllm_fn(
            hidden, gating, correction_bias, w_gate_up, w_down, shared_w_gate_up, shared_w_down
        ):
            tw, tids = _vllm_fused_topk_bias(
                hidden_states=hidden,
                gating_output=gating.float(),
                scoring_func=scoring_func,
                e_score_correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
            )
            routed_out = _vllm_fused_experts(hidden, w_gate_up, w_down, tw, tids)
            # Shared expert: gate+up GEMM → SiLU → down GEMM
            gate = F.linear(hidden, sw_gate)  # [T, Fs]
            up = F.linear(hidden, sw_up)  # [T, Fs]
            act = F.silu(gate) * up
            shared_out = F.linear(act, sw_d)  # [T, H]
            return shared_out, routed_out

        vllm_out = _vllm_fn(
            hidden, gating, correction_bias, w_gate_up, w_down, shared_w_gate_up, shared_w_down
        )  # warmup
        torch.cuda.synchronize()
        for actual, expected in zip(tileops_out, vllm_out, strict=True):
            torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=1e-1)

        functors["vllm"] = (
            _vllm_fn,
            (
                hidden,
                gating,
                correction_bias,
                w_gate_up,
                w_down,
                shared_w_gate_up,
                shared_w_down,
            ),
        )
    else:
        # No baseline rather than a misleading one: the per-expert Python loop is a
        # correctness reference, upcasting to fp32 and index_add_ing one expert at a
        # time, so timing against it measures neither implementation.
        warnings.warn(
            "No baseline recorded for FusedMoeSharedExpertFwdOp: vLLM is not installed, or the "
            "row is routed-only and the vLLM path here always builds a shared expert.",
            stacklevel=2,
        )

    bm.compare(
        functors,
        hidden,
        gating,
        w_gate_up,
        w_down,
        correction_bias,
        shared_w_gate_up,
        shared_w_down,
    )
