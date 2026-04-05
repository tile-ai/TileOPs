"""Tests for FusedTopKOp.

Reference implementation: torch.softmax/sigmoid + torch.topk + optional renormalize.

Test cases cover:
  - softmax scoring (Qwen3/Qwen2 style)
  - sigmoid scoring (DeepSeek-V3/GLM-4 style)
  - renormalize=True / False
  - Various (num_tokens, num_experts, top_k) shapes
  - bf16 and fp16 input dtypes
  - top_k=1, top_k=8
"""

import pytest
import torch

from tests.test_base import FixtureBase
from tileops.ops.moe import FusedTopKOp
from workloads.ops.moe_fused_topk import FusedTopKTest


def fused_topk_torch(
    gating_output: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = gating_output.to(torch.float32)
    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        raise ValueError(f"Unknown scoring_func: {scoring_func}")

    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.int()

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------


class FusedTopKFixture(FixtureBase):
    PARAMS = [
        ("num_tokens, num_experts, top_k, scoring_func, renormalize, dtype", [
            # smoke cases must be first
            pytest.param(32,   128, 8, "softmax", False, torch.bfloat16, marks=pytest.mark.smoke, id="smoke-softmax-bf16"),
            pytest.param(32,   128, 8, "softmax", False, torch.float16,  marks=pytest.mark.smoke, id="smoke-softmax-fp16"),
            pytest.param(32,   256, 8, "sigmoid", True,  torch.bfloat16, marks=pytest.mark.smoke, id="smoke-sigmoid-renorm"),
            # E not divisible by 32 — exercises padding path (expert_idx >= num_experts)
            pytest.param(32,   100, 4, "softmax", False, torch.bfloat16, marks=pytest.mark.smoke, id="smoke-e100-pad"),
            pytest.param(32,    33, 2, "sigmoid", False, torch.bfloat16, marks=pytest.mark.smoke, id="smoke-e33-pad"),
            # softmax, no renorm (Qwen3-MoE style)
            pytest.param(512,  128, 8, "softmax", False, torch.bfloat16, marks=pytest.mark.full, id="qwen3-small"),
            pytest.param(2048, 128, 8, "softmax", False, torch.bfloat16, marks=pytest.mark.full, id="qwen3-medium"),
            pytest.param(4096, 128, 8, "softmax", False, torch.bfloat16, marks=pytest.mark.full, id="qwen3-large"),
            # softmax + renorm (Qwen3.5-MoE style)
            pytest.param(512,  256, 8, "softmax", True,  torch.bfloat16, marks=pytest.mark.full, id="qwen35-small"),
            pytest.param(2048, 256, 8, "softmax", True,  torch.bfloat16, marks=pytest.mark.full, id="qwen35-medium"),
            # sigmoid, no renorm
            pytest.param(512,  256, 8, "sigmoid", False, torch.bfloat16, marks=pytest.mark.full, id="sigmoid-no-renorm"),
            # sigmoid + renorm (DeepSeek-V3/GLM-4 style)
            pytest.param(512,  256, 8, "sigmoid", True,  torch.bfloat16, marks=pytest.mark.full, id="sigmoid-renorm"),
            # top_k=1
            pytest.param(512,   64, 1, "softmax", False, torch.bfloat16, marks=pytest.mark.full, id="top-k-1"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _check(test: FusedTopKTest) -> None:
    gating = test.gen_inputs()
    op = FusedTopKOp(
        num_tokens=test.num_tokens,
        num_experts=test.num_experts,
        top_k=test.top_k,
        scoring_func=test.scoring_func,
        renormalize=test.renormalize,
    )
    ref_w, ref_ids = fused_topk_torch(gating, test.top_k, test.scoring_func, test.renormalize)
    out_w, out_ids = op(gating)

    assert out_w.shape == (test.num_tokens, test.top_k), f"weights shape mismatch: {out_w.shape}"
    assert out_ids.shape == (test.num_tokens, test.top_k), f"ids shape mismatch: {out_ids.shape}"
    assert out_w.dtype == torch.float32, f"weights dtype must be float32, got {out_w.dtype}"
    assert out_ids.dtype == torch.int32, f"ids dtype must be int32, got {out_ids.dtype}"

    # weights must match (sorted, handles tie-breaking differences in expert ordering)
    ref_w_sorted = ref_w.sort(dim=-1).values
    out_w_sorted = out_w.sort(dim=-1).values
    torch.testing.assert_close(out_w_sorted, ref_w_sorted, rtol=1e-3, atol=1e-3)

    # topk_ids must select experts whose scores are valid top-k scores.
    # When two experts have identical scores (fp32 ties), either is a valid selection;
    # we verify that each selected weight >= the (K+1)-th largest weight.
    gating_f32 = gating.to(torch.float32)
    if test.scoring_func == "softmax":
        all_scores = torch.softmax(gating_f32, dim=-1)
    else:
        all_scores = torch.sigmoid(gating_f32)
    # Use raw (non-renormalized) scores as threshold — renormalize only scales weights,
    # it doesn't change which experts are valid top-k selections.
    raw_ref_w = all_scores.gather(1, ref_ids.long())
    raw_ref_w_sorted = raw_ref_w.sort(dim=-1).values
    for i in range(test.num_tokens):
        sel_ids = out_ids[i].tolist()
        assert len(set(sel_ids)) == test.top_k, (
            f"token {i}: duplicate expert ids: {sorted(sel_ids)}"
        )
        # Each selected score >= min selected score (basic sanity, not strict tie check)
        if test.top_k < test.num_experts:
            kth_val = raw_ref_w_sorted[i, 0].item()  # min raw score among ref top-k
            for eid in sel_ids:
                got_score = all_scores[i, eid].item()
                assert got_score >= kth_val - 1e-4, (
                    f"token {i}: expert {eid} score {got_score:.6f} < min ref score "
                    f"{kth_val:.6f}"
                )

    tag = (f"[T={test.num_tokens}, E={test.num_experts}, K={test.top_k}, "
           f"fn={test.scoring_func}, renorm={test.renormalize}, {test.dtype}]")
    print(f"PASS {tag}")


@FusedTopKFixture
def test_fused_topk(
    num_tokens, num_experts, top_k, scoring_func, renormalize, dtype
) -> None:
    test = FusedTopKTest(num_tokens, num_experts, top_k, scoring_func, renormalize, dtype)
    _check(test)
