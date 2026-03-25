"""Correctness test: KimiMoENopadOp vs vLLM fused_experts.

Verifies that TileOPs KimiMoENopadOp produces numerically equivalent output
to vLLM's fused_experts given identical routing.

Routing alignment:
    Both sides use the same FusedTopKOp result (sigmoid + correction_bias +
    renormalize).  vLLM receives pre-computed topk_weights / topk_ids, so
    only the GEMM + SwiGLU + unpermute paths are compared.

routed_scaling_factor:
    vLLM fused_experts has no concept of routed_scaling_factor; it is applied
    manually to the vLLM output before comparison.

Skip condition:
    The entire module is skipped when vLLM is not installed.

Usage:
    conda run -n tileops python -m pytest tests/ops/test_moe_kimi_moe_vllm.py -vvs
"""

import pytest
import torch

vllm_fused_experts = pytest.importorskip(
    "vllm.model_executor.layers.fused_moe.fused_moe",
    reason="vllm not installed",
).fused_experts

from tests.test_base import FixtureBase  # noqa: E402
from tileops.ops.moe import FusedTopKOp, KimiMoENopadOp  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


class KimiVllmFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, dtype",
            [
                # ── smoke ────────────────────────────────────────────────────
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 2.872, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-scale-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 2.872, torch.float16,
                    marks=pytest.mark.smoke, id="smoke-scale-fp16",
                ),
                # ── Kimi K2 shape (reduced H for speed) ─────────────────────
                pytest.param(
                    64, 384, 6, 64, 32, 2.872, torch.bfloat16,
                    marks=pytest.mark.smoke, id="kimi-k2-smoke-bf16",
                ),
                pytest.param(
                    512, 384, 6, 256, 128, 2.872, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-small-bf16",
                ),
                pytest.param(
                    2048, 384, 6, 256, 128, 2.872, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-medium-bf16",
                ),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Check logic
# ---------------------------------------------------------------------------


def _check(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    routed_scaling_factor, dtype,
):
    torch.manual_seed(42)
    dev = "cuda"

    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=dev)
    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device=dev)
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device=dev) * 0.1
    w_gate_up = torch.randn(
        num_experts, ffn_size * 2, hidden_size, dtype=dtype, device=dev
    ) * 0.02
    w_down = torch.randn(
        num_experts, hidden_size, ffn_size, dtype=dtype, device=dev
    ) * 0.02

    # Pre-compute routing — shared by both sides
    fk = FusedTopKOp(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        scoring_func="sigmoid",
        renormalize=True,
        with_correction_bias=True,
    )
    topk_weights, topk_ids = fk(gating, correction_bias)

    # ── TileOPs ──────────────────────────────────────────────────────────────
    op = KimiMoENopadOp(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        routed_scaling_factor=routed_scaling_factor,
        with_correction_bias=True,
        dtype=dtype,
    )
    out_tileops = op(hidden, gating, correction_bias, w_gate_up, w_down)

    # ── vLLM ─────────────────────────────────────────────────────────────────
    # vLLM fused_experts expects topk_ids as int32
    out_vllm = vllm_fused_experts(
        hidden, w_gate_up, w_down,
        topk_weights, topk_ids.to(torch.int32),
    )
    if routed_scaling_factor != 1.0:
        out_vllm = out_vllm * routed_scaling_factor

    # ── Compare ──────────────────────────────────────────────────────────────
    torch.testing.assert_close(
        out_tileops.float(), out_vllm.float(),
        rtol=1e-2, atol=1e-2,
        msg=f"TileOPs vs vLLM mismatch "
            f"[T={num_tokens}, E={num_experts}, K={top_k}, "
            f"H={hidden_size}, F={ffn_size}, scale={routed_scaling_factor}, {dtype}]",
    )
    print(
        f"PASS [T={num_tokens}, E={num_experts}, K={top_k}, "
        f"H={hidden_size}, F={ffn_size}, scale={routed_scaling_factor}, {dtype}]"
    )


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@KimiVllmFixture
def test_kimi_moe_vs_vllm(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    routed_scaling_factor, dtype,
) -> None:
    _check(
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        routed_scaling_factor, dtype,
    )
