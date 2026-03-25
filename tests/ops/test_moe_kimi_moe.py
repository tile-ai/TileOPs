"""Tests for KimiMoENopadOp and KimiMoEPaddedOp.

Tests cover:
  - correction_bias: bias is used for top-k selection, original sigmoid score
    for topk_weights.
  - routed_scaling_factor: output is multiplied by the factor.
  - shared_experts_fn: shared expert output is added to routed output.
  - End-to-end forward pass matching a pure PyTorch reference.
  - KimiMoEPaddedOp gives the same results as KimiMoENopadOp.

Routing note:
  The reference implementation and KimiMoENopadOp both use FusedTopKOp with
  with_correction_bias=True for expert selection, so tie-breaking is identical
  and only the GEMM/SwiGLU computation is verified.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase
from tileops.ops.moe import FusedTopKOp, KimiMoENopadOp, KimiMoEPaddedOp

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _ref_kimi_routing(
    gating_output: torch.Tensor,   # [T, E] any float dtype
    correction_bias: torch.Tensor | None,  # [E] float32 or None
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch Kimi K2 routing reference.

    sigmoid → (sigmoid + bias) → topk selection → gather original sigmoid
    scores → renormalize.

    Returns:
        topk_weights: [T, K] float32 (original sigmoid, renormalized)
        topk_ids:     [T, K] int64
    """
    logits_f32 = gating_output.float()
    scores = logits_f32.sigmoid()                       # [T, E]

    if correction_bias is not None:
        tmp_scores = scores + correction_bias.float().unsqueeze(0)  # [T, E]
    else:
        tmp_scores = scores

    # topk on biased scores for selection
    topk_ids = tmp_scores.topk(top_k, dim=-1, sorted=False).indices  # [T, K] int64

    # gather weights from ORIGINAL (unbiased) sigmoid scores
    topk_weights = scores.gather(1, topk_ids)          # [T, K] float32

    # renormalize
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


def _ref_moe_ffn(
    hidden_states: torch.Tensor,   # [T, H]
    w_gate_up: torch.Tensor,       # [E, 2*F, H]
    w_down: torch.Tensor,          # [E, H, F]
    topk_weights: torch.Tensor,    # [T, K] float32
    topk_ids: torch.Tensor,        # [T, K] int64
) -> torch.Tensor:
    """PyTorch reference: per-expert GEMM avoiding O(T*K*2F*H) allocation."""
    T, H = hidden_states.shape
    E = w_gate_up.shape[0]
    ffn_size = w_gate_up.shape[1] // 2

    output = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)

    for e in range(E):
        mask = (topk_ids == e)           # [T, K] bool
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        h = hidden_states[t_idx].float()
        gate_up = h @ w_gate_up[e].float().t()             # [n_e, 2*F]
        act = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]  # [n_e, F]
        down = act @ w_down[e].float().t()                 # [n_e, H]
        weights = topk_weights[t_idx, k_idx].float().unsqueeze(-1)
        output.index_add_(0, t_idx, down * weights)

    return output.to(hidden_states.dtype)


def _ref_kimi_moe(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float = 1.0,
    shared_experts_fn=None,
) -> torch.Tensor:
    """Full Kimi K2 MoE reference using FusedTopKOp routing for consistency.

    Uses FusedTopKOp (same as KimiMoENopadOp) for routing so that expert
    assignments are identical; differences only come from GEMM / SwiGLU.
    """
    fk = FusedTopKOp(
        num_tokens=hidden_states.shape[0],
        num_experts=gating_output.shape[1],
        top_k=top_k,
        scoring_func="sigmoid",
        renormalize=True,
        with_correction_bias=correction_bias is not None,
    )
    topk_weights, topk_ids = fk(gating_output, correction_bias)
    topk_ids_i64 = topk_ids.to(torch.int64)

    output = _ref_moe_ffn(
        hidden_states, w_gate_up, w_down, topk_weights, topk_ids_i64
    )

    if routed_scaling_factor != 1.0:
        output = output * routed_scaling_factor

    if shared_experts_fn is not None:
        output = output + shared_experts_fn(hidden_states)

    return output


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------


class KimiMoETest:
    def __init__(
        self,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        routed_scaling_factor,
        with_correction_bias,
        with_shared_expert,
        dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
        self.with_correction_bias = with_correction_bias
        self.with_shared_expert = with_shared_expert
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


class KimiMoEFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, with_correction_bias, with_shared_expert, dtype",
            [
                # ── smoke: small shapes, fast ──────────────────────────────
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, False, False, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-nobias-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, True, False, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-bias-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 2.872, True, False, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-bias-scale-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, True, True, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-bias-shared-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 2.872, True, False, torch.float16,
                    marks=pytest.mark.smoke, id="smoke-bias-scale-fp16",
                ),
                # ── Kimi K2 config: E=384, K=6 ─────────────────────────────
                pytest.param(
                    64, 384, 6, 64, 32, 2.872, True, False, torch.bfloat16,
                    marks=pytest.mark.smoke, id="kimi-k2-smoke-bf16",
                ),
                pytest.param(
                    512, 384, 6, 256, 128, 2.872, True, False, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-small-bf16",
                ),
                pytest.param(
                    512, 384, 6, 256, 128, 2.872, True, True, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-shared-bf16",
                ),
                pytest.param(
                    2048, 384, 6, 256, 128, 2.872, True, False, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-medium-bf16",
                ),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Check logic
# ---------------------------------------------------------------------------


def _check(test: KimiMoETest) -> None:
    hidden, gating, correction_bias, w_gate_up, w_down = test.gen_inputs()

    # Shared expert: small linear layer for testing
    shared_w = None
    if test.with_shared_expert:
        torch.manual_seed(0)
        shared_w = torch.randn(
            test.hidden_size, test.hidden_size, dtype=test.dtype, device="cuda"
        ) * 0.01

    # ── TileOPs nopad ─────────────────────────────────────────────────────
    op_nopad = KimiMoENopadOp(
        num_tokens=test.num_tokens,
        num_experts=test.num_experts,
        top_k=test.top_k,
        hidden_size=test.hidden_size,
        ffn_size=test.ffn_size,
        routed_scaling_factor=test.routed_scaling_factor,
        with_correction_bias=test.with_correction_bias,
        dtype=test.dtype,
    )
    # shared_w is non-None iff with_shared_expert=True; no None guard needed inside.
    shared_fn_arg = (lambda x: x.float() @ shared_w.float().t()) if test.with_shared_expert else None  # noqa: E731

    out_nopad = op_nopad(
        hidden, gating, correction_bias, w_gate_up, w_down,
        shared_experts_fn=shared_fn_arg,
    )

    assert out_nopad.shape == (test.num_tokens, test.hidden_size), (
        f"nopad shape mismatch: {out_nopad.shape}"
    )
    assert out_nopad.dtype == test.dtype, f"nopad dtype mismatch: {out_nopad.dtype}"

    # ── PyTorch reference ─────────────────────────────────────────────────
    ref = _ref_kimi_moe(
        hidden, gating, correction_bias, w_gate_up, w_down,
        top_k=test.top_k,
        routed_scaling_factor=test.routed_scaling_factor,
        shared_experts_fn=shared_fn_arg,
    )
    torch.testing.assert_close(out_nopad.float(), ref.float(), rtol=1e-2, atol=1e-2)

    # ── TileOPs padded (same results as nopad) ────────────────────────────
    op_padded = KimiMoEPaddedOp(
        num_tokens=test.num_tokens,
        num_experts=test.num_experts,
        top_k=test.top_k,
        hidden_size=test.hidden_size,
        ffn_size=test.ffn_size,
        routed_scaling_factor=test.routed_scaling_factor,
        with_correction_bias=test.with_correction_bias,
        dtype=test.dtype,
    )
    out_padded = op_padded(
        hidden, gating, correction_bias, w_gate_up, w_down,
        shared_experts_fn=shared_fn_arg,
    )

    assert out_padded.shape == (test.num_tokens, test.hidden_size), (
        f"padded shape mismatch: {out_padded.shape}"
    )
    torch.testing.assert_close(out_padded.float(), ref.float(), rtol=1e-2, atol=1e-2)

    tag = (
        f"[T={test.num_tokens}, E={test.num_experts}, K={test.top_k}, "
        f"H={test.hidden_size}, F={test.ffn_size}, "
        f"scale={test.routed_scaling_factor}, bias={test.with_correction_bias}, "
        f"shared={test.with_shared_expert}, {test.dtype}]"
    )
    print(f"PASS {tag}")


# ---------------------------------------------------------------------------
# correction_bias precision: verify gather from original sigmoid (not biased)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_correction_bias_routing_precision() -> None:
    """Verify that correction_bias is used only for selection, not for weights.

    Constructs a case where the biased and unbiased top-k differ, then checks:
      - topk_ids match the biased-sort order
      - topk_weights are the ORIGINAL (unbiased) sigmoid scores, renormalized
    """
    torch.manual_seed(7)
    T, E, K = 4, 8, 2
    dev = "cuda"

    logits = torch.randn(T, E, dtype=torch.float32, device=dev)
    bias = torch.randn(E, dtype=torch.float32, device=dev)

    # Reference
    ref_weights, ref_ids = _ref_kimi_routing(logits, bias, K)

    # TileOPs
    op = FusedTopKOp(
        num_tokens=T,
        num_experts=E,
        top_k=K,
        scoring_func="sigmoid",
        renormalize=True,
        with_correction_bias=True,
    )
    tileops_weights, tileops_ids = op(logits, bias)

    # Expert selection must match
    ref_ids_sorted = ref_ids.sort(dim=-1).values
    tile_ids_sorted = tileops_ids.long().sort(dim=-1).values
    assert torch.equal(ref_ids_sorted, tile_ids_sorted), (
        f"Expert selection mismatch:\n  ref={ref_ids_sorted}\n  got={tile_ids_sorted}"
    )

    # Weights must match original sigmoid (not biased)
    torch.testing.assert_close(
        tileops_weights, ref_weights, rtol=1e-4, atol=1e-4,
        msg="topk_weights should reflect original sigmoid, not biased scores",
    )
    print("PASS correction_bias precision test")


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@KimiMoEFixture
def test_kimi_moe(
    num_tokens,
    num_experts,
    top_k,
    hidden_size,
    ffn_size,
    routed_scaling_factor,
    with_correction_bias,
    with_shared_expert,
    dtype,
) -> None:
    test = KimiMoETest(
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        routed_scaling_factor, with_correction_bias, with_shared_expert, dtype,
    )
    _check(test)


# ---------------------------------------------------------------------------
# vLLM correctness test (skipped when vLLM is not installed)
# ---------------------------------------------------------------------------
#
# Routing alignment: both sides use the same FusedTopKOp result so only the
# GEMM + SwiGLU + unpermute paths differ.
# routed_scaling_factor: applied manually to the vLLM output before comparison
# (vLLM fused_experts has no such concept).


class KimiVllmFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, dtype",
            [
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


@KimiVllmFixture
def test_kimi_moe_vs_vllm(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    routed_scaling_factor, dtype,
) -> None:
    vllm_fused_experts = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        reason="vllm not installed",
    ).fused_experts

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
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        scoring_func="sigmoid",
        renormalize=True,
        with_correction_bias=True,
    )
    topk_weights, topk_ids = fk(gating, correction_bias)

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

    out_vllm = vllm_fused_experts(hidden, w_gate_up, w_down, topk_weights, topk_ids)
    if routed_scaling_factor != 1.0:
        out_vllm = out_vllm * routed_scaling_factor

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
