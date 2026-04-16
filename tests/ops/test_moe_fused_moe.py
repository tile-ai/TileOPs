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
                pytest.param(
                    32, 8, 2, 64, 32, "sigmoid", True, torch.bfloat16,
                    marks=pytest.mark.full, id="smoke-sigmoid-renorm-bf16",
                ),
                pytest.param(
                    512, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen3-small",
                ),
                pytest.param(
                    2048, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen3-medium",
                ),
                pytest.param(
                    512, 256, 8, 2048, 1024, "softmax", True, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen35-small",
                ),
                pytest.param(
                    512, 256, 8, 2048, 1024, "sigmoid", True, torch.bfloat16,
                    marks=pytest.mark.full, id="deepseek-small",
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
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, False, torch.bfloat16,
                    marks=pytest.mark.full, id="smoke-nobias-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, True, torch.bfloat16,
                    marks=pytest.mark.full, id="smoke-bias-bf16",
                ),
                pytest.param(
                    64, 384, 8, 64, 32, 2.827, True, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-smoke-bf16",
                ),
                pytest.param(
                    512, 384, 8, 256, 128, 2.827, True, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-small-bf16",
                ),
                pytest.param(
                    2048, 384, 8, 256, 128, 2.827, True, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-medium-bf16",
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


@pytest.mark.full
def test_expert_map_local_filter() -> None:
    """Simulate EP=2 on a single GPU: each rank owns half the experts.

    Verification: sum of outputs from rank-0 and rank-1 (with their disjoint
    expert_maps) equals the output without expert_map.
    """
    torch.manual_seed(42)
    dev = "cuda"
    T, E, K, H, F = 32, 8, 2, 64, 32
    dtype = torch.bfloat16

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    # Rank 0 owns experts 0..3, rank 1 owns experts 4..7
    expert_map_rank0 = torch.full((E,), -1, dtype=torch.int32, device=dev)
    expert_map_rank0[:E // 2] = torch.arange(E // 2, dtype=torch.int32, device=dev)

    expert_map_rank1 = torch.full((E,), -1, dtype=torch.int32, device=dev)
    expert_map_rank1[E // 2:] = torch.arange(E // 2, dtype=torch.int32, device=dev)

    # Full output (no expert_map)
    op_full = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F, dtype=dtype,
    )
    out_full = op_full(hidden, gating, w_gate_up, w_down)

    # Rank-0 partial output (local experts 0..3)
    op_r0 = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F, dtype=dtype,
        expert_map=expert_map_rank0,
    )
    out_r0 = op_r0(hidden, gating, w_gate_up[:E // 2], w_down[:E // 2])

    # Rank-1 partial output (local experts 4..7)
    op_r1 = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F, dtype=dtype,
        expert_map=expert_map_rank1,
    )
    out_r1 = op_r1(hidden, gating, w_gate_up[E // 2:], w_down[E // 2:])

    # Sum of partial outputs should match full output
    out_sum = (out_r0.float() + out_r1.float()).to(dtype)
    torch.testing.assert_close(out_sum.float(), out_full.float(), rtol=1e-2, atol=1e-2)
    print(f"PASS expert_map local filter [T={T}, E={E}, K={K}, H={H}, F={F}]")


# ---------------------------------------------------------------------------
# correction_bias routing precision
# ---------------------------------------------------------------------------


@pytest.mark.full
def test_correction_bias_routing_precision() -> None:
    """Verify correction_bias is used only for top-k selection, not for weights.

    Constructs a case where biased and unbiased top-k differ, then checks:
      - topk_ids match the biased-sort order
      - topk_weights use ORIGINAL (unbiased) sigmoid scores, renormalized
    """
    torch.manual_seed(7)
    T, E, K = 4, 8, 2
    dev = "cuda"

    logits = torch.randn(T, E, dtype=torch.float32, device=dev)
    bias = torch.randn(E, dtype=torch.float32, device=dev)

    ref_weights, ref_ids = _ref_kimi_routing(logits, bias, K)

    op = FusedTopKOp(
        num_tokens=T, num_experts=E, top_k=K,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
    )
    tw, ti = op(logits, bias)

    ref_ids_sorted = ref_ids.sort(dim=-1).values
    tile_ids_sorted = ti.long().sort(dim=-1).values
    assert torch.equal(ref_ids_sorted, tile_ids_sorted), (
        f"Expert selection mismatch:\n  ref={ref_ids_sorted}\n  got={tile_ids_sorted}"
    )
    torch.testing.assert_close(tw, ref_weights, rtol=1e-4, atol=1e-4)
    print("PASS correction_bias precision test")


# ---------------------------------------------------------------------------
# vLLM alignment (optional)
# ---------------------------------------------------------------------------


class VllmFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, dtype",
            [
                pytest.param(
                    32, 8, 2, 64, 32, 2.827, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-scale-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, torch.bfloat16,
                    marks=pytest.mark.full, id="smoke-bf16",
                ),
                pytest.param(
                    64, 384, 8, 64, 32, 2.827, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-smoke-bf16",
                ),
                pytest.param(
                    512, 384, 8, 256, 128, 2.827, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-small-bf16",
                ),
            ],
        )
    ]


@VllmFixture
def test_fused_moe_vs_vllm(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    routed_scaling_factor, dtype,
) -> None:
    if not _VLLM_AVAILABLE:
        pytest.skip("vllm not installed")

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

    fk = FusedTopKOp(
        num_tokens, num_experts, top_k, "sigmoid", True, with_correction_bias=True,
    )
    topk_weights, topk_ids = fk(gating, correction_bias)

    op = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
        routed_scaling_factor=routed_scaling_factor,
        layout="nopad", dtype=dtype,
    )
    out_tileops = op(hidden, gating, w_gate_up, w_down, correction_bias)

    out_vllm = _vllm_fused_experts(hidden, w_gate_up, w_down, topk_weights, topk_ids)
    if routed_scaling_factor != 1.0:
        out_vllm = out_vllm * routed_scaling_factor

    torch.testing.assert_close(
        out_tileops.float(), out_vllm.float(), rtol=1e-2, atol=1e-2,
    )
    print(
        f"PASS vs vLLM [T={num_tokens}, E={num_experts}, K={top_k}, "
        f"H={hidden_size}, F={ffn_size}, scale={routed_scaling_factor}, {dtype}]"
    )
