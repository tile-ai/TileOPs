"""Tests for Qwen3MoEOp.

Reference implementations:
  - PyTorch: scatter gather + bmm + silu_and_mul (always available, used in CI)
  - vLLM:    fused_experts (optional, only when vllm is installed)

Test cases cover:
  - Qwen3-MoE style: softmax, renormalize=False
  - Qwen3.5-MoE style: softmax, renormalize=True
  - DeepSeek-V3 style: sigmoid, renormalize=True
  - Various (T, E, K, H, F) shapes including small smoke configs
  - bf16 and fp16 dtypes

Routing note:
  Both the reference and Qwen3MoEOp use FusedTopKOp for expert selection so
  tie-breaking is identical and only the GEMM/SwiGLU computation is tested here.
  FusedTopKOp correctness is tested independently in test_moe_fused_topk.py.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase
from tileops.ops.moe import FusedTopKOp, Qwen3MoEOp

# ---------------------------------------------------------------------------
# vLLM optional import
# ---------------------------------------------------------------------------

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as _vllm_fused_experts
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_qwen3_moe(
    hidden_states: torch.Tensor,   # [T, H]
    w_gate_up: torch.Tensor,       # [E, 2*F, H]
    w_down: torch.Tensor,          # [E, H, F]
    topk_weights: torch.Tensor,    # [T, K] float32 (already scored & renormed)
    topk_ids: torch.Tensor,        # [T, K] int32
) -> torch.Tensor:
    """PyTorch reference: per-expert GEMM to avoid O(T*K*2F*H) memory allocation."""
    T, H = hidden_states.shape
    E = w_gate_up.shape[0]
    ffn_size = w_gate_up.shape[1] // 2

    output = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    topk_ids_i32 = topk_ids.to(torch.int64)

    for e in range(E):
        mask = (topk_ids_i32 == e)           # [T, K] bool
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)  # token indices, k-slot indices
        h = hidden_states[t_idx].float()             # [n_e, H]
        gate_up = h @ w_gate_up[e].float().t()       # [n_e, 2*F]
        act = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]  # [n_e, F]
        down = act @ w_down[e].float().t()           # [n_e, H]
        weights = topk_weights[t_idx, k_idx].float().unsqueeze(-1)  # [n_e, 1]
        output.index_add_(0, t_idx, down * weights)

    return output.to(hidden_states.dtype)


def _vllm_qwen3_moe(
    hidden_states: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """vLLM fused_experts reference using pre-computed routing."""
    return _vllm_fused_experts(
        hidden_states.float(),
        w_gate_up.float(),
        w_down.float(),
        topk_weights,
        topk_ids,
    ).to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------


class Qwen3MoETest:
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


class Qwen3MoEFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " scoring_func, renormalize, dtype",
            [
                # ── smoke ──────────────────────────────────────────────────
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
                    marks=pytest.mark.smoke, id="smoke-sigmoid-renorm-bf16",
                ),
                # ── Qwen3-MoE: 128 experts, softmax, no renorm ─────────────
                pytest.param(
                    512, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen3-small",
                ),
                pytest.param(
                    2048, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen3-medium",
                ),
                # ── Qwen3.5-MoE: 256 experts, softmax, renorm ──────────────
                pytest.param(
                    512, 256, 8, 2048, 1024, "softmax", True, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen35-small",
                ),
                # ── DeepSeek-V3 style: sigmoid, renorm ─────────────────────
                pytest.param(
                    512, 256, 8, 2048, 1024, "sigmoid", True, torch.bfloat16,
                    marks=pytest.mark.full, id="deepseek-small",
                ),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Check logic
# ---------------------------------------------------------------------------


def _check(test: Qwen3MoETest) -> None:
    hidden, gating, w_gate_up, w_down = test.gen_inputs()

    op = Qwen3MoEOp(
        num_tokens=test.num_tokens,
        num_experts=test.num_experts,
        top_k=test.top_k,
        hidden_size=test.hidden_size,
        ffn_size=test.ffn_size,
        scoring_func=test.scoring_func,
        renormalize=test.renormalize,
        dtype=test.dtype,
    )

    out = op(hidden, gating, w_gate_up, w_down)

    assert out.shape == (test.num_tokens, test.hidden_size), (
        f"shape mismatch: {out.shape}"
    )
    assert out.dtype == test.dtype, f"dtype mismatch: {out.dtype}"

    # Compute routing once with FusedTopKOp so that reference and Qwen3MoEOp
    # use the same expert assignments.  This isolates GEMM/SwiGLU correctness
    # from tie-breaking differences between scoring implementations.
    fk = FusedTopKOp(
        num_tokens=test.num_tokens,
        num_experts=test.num_experts,
        top_k=test.top_k,
        scoring_func=test.scoring_func,
        renormalize=test.renormalize,
    )
    topk_weights, topk_ids = fk(gating)

    # ── PyTorch reference (always) ────────────────────────────────────────
    ref = _ref_qwen3_moe(hidden, w_gate_up, w_down, topk_weights, topk_ids)
    # bf16/fp16 accumulation errors across T*K paths → loose tolerance
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)

    # ── vLLM reference (optional) ─────────────────────────────────────────
    if _VLLM_AVAILABLE and test.scoring_func == "softmax":
        vllm_ref = _vllm_qwen3_moe(
            hidden, w_gate_up, w_down, topk_weights, topk_ids,
        )
        torch.testing.assert_close(out.float(), vllm_ref.float(), rtol=1e-2, atol=1e-2)

    tag = (
        f"[T={test.num_tokens}, E={test.num_experts}, K={test.top_k}, "
        f"H={test.hidden_size}, F={test.ffn_size}, "
        f"fn={test.scoring_func}, renorm={test.renormalize}, {test.dtype}]"
    )
    vllm_tag = " +vLLM" if (_VLLM_AVAILABLE and test.scoring_func == "softmax") else ""
    print(f"PASS{vllm_tag} {tag}")


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@Qwen3MoEFixture
def test_qwen3_moe(
    num_tokens,
    num_experts,
    top_k,
    hidden_size,
    ffn_size,
    scoring_func,
    renormalize,
    dtype,
) -> None:
    test = Qwen3MoETest(
        num_tokens, num_experts, top_k, hidden_size, ffn_size,
        scoring_func, renormalize, dtype,
    )
    _check(test)
