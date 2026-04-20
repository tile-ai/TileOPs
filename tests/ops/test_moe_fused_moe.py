"""Tests for FusedMoe — unified routed MoE FFN operator.

Covers:
  - Qwen3 config: softmax, renormalize=False/True
  - Kimi K2 config: sigmoid, correction_bias, routed_scaling_factor=2.827
  - layout="nopad" vs layout="padded" numerical agreement
  - expert_map local filtering (EP simulation without All-to-All)
  - vLLM correctness (optional, skipped when vLLM is not installed)
  - correction_bias routing precision (weights from original sigmoid, not biased)
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase
from tileops.ops.moe import FusedMoe, FusedTopKOp

# ---------------------------------------------------------------------------
# vLLM optional import
# ---------------------------------------------------------------------------

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_moe_ffn(
    hidden_states: torch.Tensor,   # [T, H]
    w_gate_up: torch.Tensor,       # [E, 2*F, H]
    w_down: torch.Tensor,          # [E, H, F]
    topk_weights: torch.Tensor,    # [T, K] float32
    topk_ids: torch.Tensor,        # [T, K] int64
) -> torch.Tensor:
    """PyTorch reference: per-expert GEMM (memory-efficient, no O(T*K*2F*H) alloc)."""
    T, H = hidden_states.shape
    E = w_gate_up.shape[0]
    ffn_size = w_gate_up.shape[1] // 2

    output = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        h = hidden_states[t_idx].float()
        gate_up = h @ w_gate_up[e].float().t()
        act = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]
        down = act @ w_down[e].float().t()
        weights = topk_weights[t_idx, k_idx].float().unsqueeze(-1)
        output.index_add_(0, t_idx, down * weights)
    return output.to(hidden_states.dtype)


def _ref_kimi_routing(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch Kimi K2 routing: sigmoid → biased topk → gather original weights."""
    scores = gating_output.float().sigmoid()
    tmp = (scores + correction_bias.float().unsqueeze(0)) if correction_bias is not None else scores
    topk_ids = tmp.topk(top_k, dim=-1, sorted=False).indices
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


# ---------------------------------------------------------------------------
# Test fixture — Qwen3 config
# ---------------------------------------------------------------------------


class Qwen3Fixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " scoring_func, renormalize, dtype",
            [
                pytest.param(
                    32, 8, 2, 64, 32, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-softmax-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, "softmax", True, torch.float16,
                    marks=pytest.mark.smoke, id="smoke-softmax-renorm-fp16",
                ),
            ],
        )
    ]


@Qwen3Fixture
def test_fused_moe_qwen3(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    scoring_func, renormalize, dtype,
) -> None:
    torch.manual_seed(42)
    dev = "cuda"
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=dev)
    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device=dev)
    w_gate_up = torch.randn(
        num_experts, ffn_size * 2, hidden_size, dtype=dtype, device=dev
    ) * 0.02
    w_down = torch.randn(
        num_experts, hidden_size, ffn_size, dtype=dtype, device=dev
    ) * 0.02

    op_nopad = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func=scoring_func, renormalize=renormalize,
        layout="nopad", dtype=dtype,
    )
    op_padded = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func=scoring_func, renormalize=renormalize,
        layout="padded", dtype=dtype,
    )

    out_nopad = op_nopad(hidden, gating, w_gate_up, w_down)
    out_padded = op_padded(hidden, gating, w_gate_up, w_down)

    assert out_nopad.shape == (num_tokens, hidden_size)
    assert out_nopad.dtype == dtype

    # Reference using the same FusedTopKOp routing
    fk = FusedTopKOp(num_tokens, num_experts, top_k, scoring_func, renormalize)
    topk_weights, topk_ids = fk(gating)
    ref = _ref_moe_ffn(hidden, w_gate_up, w_down, topk_weights, topk_ids.long())

    torch.testing.assert_close(out_nopad.float(), ref.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_padded.float(), ref.float(), rtol=1e-2, atol=1e-2)

    if _VLLM_AVAILABLE and scoring_func == "softmax":
        out_vllm = _vllm_fused_experts(
            hidden.float(), w_gate_up.float(), w_down.float(),
            topk_weights, topk_ids,
        ).to(dtype)
        torch.testing.assert_close(out_nopad.float(), out_vllm.float(), rtol=1e-2, atol=1e-2)

    print(
        f"PASS [T={num_tokens}, E={num_experts}, K={top_k}, "
        f"H={hidden_size}, F={ffn_size}, fn={scoring_func}, "
        f"renorm={renormalize}, {dtype}]"
    )


# ---------------------------------------------------------------------------
# Test fixture — Kimi K2 config
# ---------------------------------------------------------------------------


class KimiFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, with_correction_bias, dtype",
            [
                pytest.param(
                    32, 8, 2, 64, 32, 2.827, True, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-bias-scale-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 2.827, True, torch.float16,
                    marks=pytest.mark.smoke, id="smoke-bias-scale-fp16",
                ),
            ],
        )
    ]


@KimiFixture
def test_fused_moe_kimi(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    routed_scaling_factor, with_correction_bias, dtype,
) -> None:
    torch.manual_seed(42)
    dev = "cuda"
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=dev)
    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device=dev)
    correction_bias = (
        torch.randn(num_experts, dtype=torch.float32, device=dev) * 0.1
        if with_correction_bias else None
    )
    w_gate_up = torch.randn(
        num_experts, ffn_size * 2, hidden_size, dtype=dtype, device=dev
    ) * 0.02
    w_down = torch.randn(
        num_experts, hidden_size, ffn_size, dtype=dtype, device=dev
    ) * 0.02

    op_nopad = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func="sigmoid", renormalize=True,
        with_correction_bias=with_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        layout="nopad", dtype=dtype,
    )
    op_padded = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func="sigmoid", renormalize=True,
        with_correction_bias=with_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        layout="padded", dtype=dtype,
    )

    out_nopad = op_nopad(hidden, gating, w_gate_up, w_down, correction_bias)
    out_padded = op_padded(hidden, gating, w_gate_up, w_down, correction_bias)

    assert out_nopad.shape == (num_tokens, hidden_size)
    assert out_nopad.dtype == dtype

    # Reference using FusedTopKOp for consistent routing
    fk = FusedTopKOp(
        num_tokens, num_experts, top_k, "sigmoid", True,
        with_correction_bias=with_correction_bias,
    )
    topk_weights, topk_ids = fk(gating, correction_bias)
    ref = _ref_moe_ffn(hidden, w_gate_up, w_down, topk_weights, topk_ids.long())
    if routed_scaling_factor != 1.0:
        ref = ref * routed_scaling_factor

    torch.testing.assert_close(out_nopad.float(), ref.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_padded.float(), ref.float(), rtol=1e-2, atol=1e-2)

    print(
        f"PASS [T={num_tokens}, E={num_experts}, K={top_k}, "
        f"H={hidden_size}, F={ffn_size}, scale={routed_scaling_factor}, "
        f"bias={with_correction_bias}, {dtype}]"
    )


# ---------------------------------------------------------------------------
# expert_map local filtering test (EP simulation without All-to-All)
# ---------------------------------------------------------------------------
