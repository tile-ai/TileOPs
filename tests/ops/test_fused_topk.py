"""Tests for FusedTopKFwdOp.

Reference: ``FusedTopKWorkload.ref_program`` (softmax/sigmoid/sqrt-softplus, top-k, optional
renormalize, grouped routing, scaling factor, caller-given experts).

Test cases cover:
  - softmax scoring (Qwen3/Qwen2 style)
  - sigmoid scoring (DeepSeek-V3/GLM-4 style)
  - sqrt-softplus scoring with a correction bias (DeepSeek-V4-Flash style)
  - renormalize=True / False
  - Various (num_tokens, num_experts, top_k) shapes
  - bf16 and fp16 input dtypes
  - top_k=1, top_k=8
  - grouped routing: the biased top-2 group score and the unbiased max, groups spanning
    whole warp rows and groups that straddle them, E off the warp width
  - routed_scaling_factor with and without grouping
  - caller-given experts (DeepSeek-V4-Flash hash routing) under each scoring function
  - sigmoid logits below the float32 normal range: zero scores, NaN renormalized weights
  - ties at the cut on negative biased scores, and rows tied on NaN or -inf scores
  - a kept group whose selection reaches its -inf biased scores, and groups scoring -inf or NaN
  - softmax rows made NaN by a NaN or +inf logit or by a row of -inf
"""

import math
import re

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.moe import FusedTopKFwdOp
from tileops.backend import OpNotAvailableError
from workloads.device import run_device
from workloads.moe import FusedTopKWorkload
from workloads.workload_base import manifest_call


class FusedTopKFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, scoring_func, renormalize, dtype",
            [
                # smoke cases must be first
                pytest.param(
                    32,
                    128,
                    8,
                    "softmax",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-softmax-bf16",
                ),
                pytest.param(
                    32,
                    128,
                    8,
                    "softmax",
                    False,
                    torch.float16,
                    marks=pytest.mark.smoke,
                    id="smoke-softmax-fp16",
                ),
                pytest.param(
                    32,
                    256,
                    8,
                    "sigmoid",
                    True,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-sigmoid-renorm",
                ),
                # E not divisible by 32 — exercises padding path (expert_idx >= num_experts)
                pytest.param(
                    32,
                    100,
                    4,
                    "softmax",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-e100-pad",
                ),
                pytest.param(
                    32,
                    33,
                    2,
                    "sigmoid",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-e33-pad",
                ),
                # sqrt-softplus without a bias.
                pytest.param(
                    32,
                    64,
                    4,
                    "sqrtsoftplus",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-sqrtsoftplus",
                ),
                # top_k = E: every expert is selected, the weights renormalize to one.
                pytest.param(
                    32,
                    64,
                    64,
                    "softmax",
                    True,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="top-k-equals-e",
                ),
                # softmax, no renorm (Qwen3-MoE style)
                pytest.param(
                    512,
                    128,
                    8,
                    "softmax",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="qwen3-small",
                ),
                pytest.param(
                    2048,
                    128,
                    8,
                    "softmax",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="qwen3-medium",
                ),
                pytest.param(
                    4096,
                    128,
                    8,
                    "softmax",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="qwen3-large",
                ),
                # softmax + renorm (Qwen3.5-MoE style)
                pytest.param(
                    512,
                    256,
                    8,
                    "softmax",
                    True,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="qwen35-small",
                ),
                pytest.param(
                    2048,
                    256,
                    8,
                    "softmax",
                    True,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="qwen35-medium",
                ),
                # sigmoid, no renorm
                pytest.param(
                    512,
                    256,
                    8,
                    "sigmoid",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="sigmoid-no-renorm",
                ),
                # sigmoid + renorm (DeepSeek-V3/GLM-4 style)
                pytest.param(
                    512,
                    256,
                    8,
                    "sigmoid",
                    True,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="sigmoid-renorm",
                ),
                # top_k=1
                pytest.param(
                    512,
                    64,
                    1,
                    "softmax",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="top-k-1",
                ),
            ],
        ),
    ]


def _check(test: FusedTopKWorkload) -> None:
    op = FusedTopKFwdOp(**test.call.arguments())
    TestBase.check(test, op, *test.gen_inputs())


@FusedTopKFixture
def test_fused_topk(num_tokens, num_experts, top_k, scoring_func, renormalize, dtype) -> None:
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": str(dtype).removeprefix("torch.")},
        T=num_tokens,
        E=num_experts,
        top_k=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
    )
    _check(FusedTopKWorkload(call))


class GroupedTopKFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, num_expert_group, topk_group, scoring_func,"
            " with_bias, renormalize, routed_scaling_factor, dtype",
            [
                # DeepSeek-V3: groups of 32 experts scored by their biased top-2.
                pytest.param(
                    64,
                    256,
                    8,
                    8,
                    4,
                    "sigmoid",
                    True,
                    True,
                    2.5,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-ds-v3",
                ),
                # DeepSeek-V2: softmax groups scored by their max, groups of 20 experts, a scale
                # without renormalization.
                pytest.param(
                    64,
                    160,
                    6,
                    8,
                    3,
                    "softmax",
                    False,
                    False,
                    16.0,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-ds-v2-softmax-max",
                ),
                # Sigmoid groups scored by their max.
                pytest.param(
                    64,
                    256,
                    8,
                    8,
                    4,
                    "sigmoid",
                    False,
                    True,
                    1.0,
                    torch.float16,
                    marks=pytest.mark.smoke,
                    id="smoke-sigmoid-nobias-fp16",
                ),
                # E off a multiple of 32.
                pytest.param(
                    32,
                    100,
                    4,
                    4,
                    2,
                    "sigmoid",
                    True,
                    False,
                    1.5,
                    torch.float32,
                    marks=pytest.mark.smoke,
                    id="smoke-e100-pad-fp32",
                ),
                # A group count off a power of two.
                pytest.param(
                    32,
                    96,
                    4,
                    3,
                    2,
                    "sigmoid",
                    True,
                    True,
                    1.0,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-three-groups",
                ),
                # A scale without grouping.
                pytest.param(
                    32,
                    128,
                    8,
                    1,
                    1,
                    "softmax",
                    False,
                    True,
                    2.5,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-ungrouped-scaled",
                ),
                # DeepSeek-V4-Flash: biased sqrt-softplus scores of float32 logits.
                pytest.param(
                    32,
                    64,
                    6,
                    1,
                    1,
                    "sqrtsoftplus",
                    True,
                    True,
                    1.5,
                    torch.float32,
                    marks=pytest.mark.smoke,
                    id="smoke-sqrtsoftplus-bias",
                ),
                # Every group kept.
                pytest.param(
                    32,
                    64,
                    8,
                    4,
                    4,
                    "sigmoid",
                    True,
                    True,
                    1.0,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-all-groups-kept",
                ),
                # 32 biased groups of two experts; the one kept group holds exactly top_k experts.
                pytest.param(
                    32,
                    64,
                    2,
                    32,
                    1,
                    "sigmoid",
                    True,
                    True,
                    1.0,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-g32-kept-holds-top-k",
                ),
                # A power-of-two group count above 32.
                pytest.param(
                    32,
                    128,
                    8,
                    64,
                    8,
                    "sigmoid",
                    True,
                    True,
                    1.0,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-g64-butterfly",
                ),
                # Two groups, one token.
                pytest.param(
                    1,
                    64,
                    4,
                    2,
                    1,
                    "sigmoid",
                    False,
                    True,
                    1.0,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-two-groups-nobias",
                ),
                # Groups of one expert without a bias (num_expert_group == E).
                pytest.param(
                    32,
                    64,
                    8,
                    64,
                    8,
                    "softmax",
                    False,
                    False,
                    1.0,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-groups-of-one-softmax",
                ),
                # An odd token count.
                pytest.param(
                    201,
                    256,
                    8,
                    8,
                    4,
                    "sigmoid",
                    True,
                    True,
                    2.5,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-ds-v3-tail-block",
                ),
                pytest.param(
                    4096,
                    256,
                    8,
                    8,
                    4,
                    "sigmoid",
                    True,
                    True,
                    2.5,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="ds-v3-t4096",
                ),
                # More selected experts than warp lanes, the second lane slot partly filled.
                pytest.param(
                    32,
                    40,
                    36,
                    1,
                    1,
                    "sigmoid",
                    True,
                    True,
                    1.0,
                    torch.float32,
                    marks=pytest.mark.full,
                    id="biased-top-k-over-warp",
                ),
            ],
        ),
    ]


@GroupedTopKFixture
def test_grouped_topk(
    num_tokens,
    num_experts,
    top_k,
    num_expert_group,
    topk_group,
    scoring_func,
    with_bias,
    renormalize,
    routed_scaling_factor,
    dtype,
) -> None:
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": str(dtype).removeprefix("torch.")},
        T=num_tokens,
        E=num_experts,
        top_k=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        some=["correction_bias"] if with_bias else [],
    )
    _check(FusedTopKWorkload(call))


class GivenExpertsFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, scoring_func, renormalize, dtype",
            [
                # DeepSeek-V4-Flash's hash-routed layers, a partial last block.
                pytest.param(
                    201,
                    64,
                    6,
                    "sqrtsoftplus",
                    True,
                    torch.float32,
                    marks=pytest.mark.smoke,
                    id="smoke-dsv4-flash-hash-tail-block",
                ),
                # Renormalized softmax: the shift is the given logits' max.
                pytest.param(
                    32,
                    64,
                    8,
                    "softmax",
                    True,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-softmax-renorm-bf16",
                ),
                # Unrenormalized softmax: normalized over the whole row, E off the warp width.
                pytest.param(
                    32,
                    100,
                    4,
                    "softmax",
                    False,
                    torch.float16,
                    marks=pytest.mark.smoke,
                    id="smoke-softmax-row-sum-e100-fp16",
                ),
                # Unrenormalized sigmoid.
                pytest.param(
                    32,
                    64,
                    8,
                    "sigmoid",
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-sigmoid",
                ),
                # More given experts than warp lanes: a lane holds two.
                pytest.param(
                    32,
                    64,
                    40,
                    "sqrtsoftplus",
                    True,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                    id="smoke-top-k-over-warp",
                ),
            ],
        ),
    ]


@GivenExpertsFixture
def test_fused_topk_given_experts(
    num_tokens, num_experts, top_k, scoring_func, renormalize, dtype
) -> None:
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": str(dtype).removeprefix("torch.")},
        T=num_tokens,
        E=num_experts,
        top_k=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=1.5,
        some=["expert_ids"],
    )
    _check(FusedTopKWorkload(call))


@pytest.mark.smoke
def test_given_experts_ignore_the_selection_inputs() -> None:
    """A bias and a grouping only steer a selection, so given experts ignore them, even a
    softmax bias and a grouping that cannot split E, which a selection refuses."""
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": "float32"},
        T=32,
        E=64,
        top_k=6,
        scoring_func="softmax",
        renormalize=True,
        num_expert_group=3,
        topk_group=2,
        some=["expert_ids", "correction_bias"],
    )
    _check(FusedTopKWorkload(call))


@pytest.mark.smoke
def test_sqrtsoftplus_keeps_float32_precision_at_extreme_logits() -> None:
    """Every logit from -92 to 200 scores within 1e-4 of torch's float32 score, where
    log(1 + exp(x)) rounds a softplus below 2^-24 to zero and overflows past x = 88, and
    sqrt(softplus(x)) flushes to zero below x = -87 with a subnormal softplus."""
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": "float32"},
        T=64,
        E=64,
        top_k=64,
        scoring_func="sqrtsoftplus",
        renormalize=False,
        some=["expert_ids"],
    )
    workload = FusedTopKWorkload(call)
    _, _, expert_ids = workload.gen_inputs()
    gating = torch.linspace(-92, 200, 64 * 64, device=run_device()).view(64, 64)
    TestBase.check(workload, FusedTopKFwdOp(64, "sqrtsoftplus"), gating, None, expert_ids)


@pytest.mark.smoke
@pytest.mark.parametrize("path", ["selected", "biased", "given"])
def test_sigmoid_scores_a_logit_below_the_normal_range_zero(path) -> None:
    """On each kernel path, sigmoid scores 0 for logits of -88, whose torch float32 score is
    subnormal, and of -100, whose is 0, and a renormalized row of only such logits returns NaN
    weights."""
    device = run_device()
    gating = torch.tensor([[-88.0], [-100.0]], device=device).repeat(1, 64)
    bias = torch.zeros(64, device=device) if path == "biased" else None
    ids = torch.arange(4, dtype=torch.int32, device=device).view(2, 2) if path == "given" else None
    for renormalize, expected in ((False, 0.0), (True, float("nan"))):
        weights, _ = FusedTopKFwdOp(2, "sigmoid", renormalize)(gating, bias, expert_ids=ids)
        torch.testing.assert_close(
            weights, torch.full_like(weights, expected), rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_fused_topk_refuses_a_softmax_selection_with_a_bias() -> None:
    """No in-tree kernel adds a correction bias to softmax scores; the op refuses before
    building."""
    gating = torch.randn(4, 64, device=run_device())
    bias = torch.randn(64, device=run_device())
    op = FusedTopKFwdOp(4)
    with pytest.raises(OpNotAvailableError, match="not softmax ones"):
        op(gating, bias)
    assert not op._entries_built("fused_topk")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "num_experts, num_expert_group, topk_group, top_k, scale, rule",
    [
        pytest.param(
            256, 3, 1, 8, 1.0, "E % num_expert_group == 0", id="group-count-not-dividing-e"
        ),
        pytest.param(256, 0, 1, 8, 1.0, "1 <= num_expert_group", id="no-groups"),
        pytest.param(256, 8, 0, 8, 1.0, "1 <= topk_group", id="no-group-kept"),
        pytest.param(
            256, 8, 9, 8, 1.0, "topk_group <= num_expert_group", id="more-kept-than-groups"
        ),
        pytest.param(
            64,
            32,
            1,
            4,
            1.0,
            "top_k <= topk_group * (E // num_expert_group)",
            id="kept-under-top-k",
        ),
        pytest.param(64, 64, 8, 8, 1.0, "E // num_expert_group >= 2", id="biased-group-of-one"),
        pytest.param(64, 1, 1, 0, 1.0, "0 < top_k", id="no-expert-selected"),
        pytest.param(64, 1, 1, 65, 1.0, "top_k <= E", id="top-k-over-e"),
        pytest.param(256, 8, 4, 8, 0.0, "routed_scaling_factor > 0", id="zero-scale"),
    ],
)
def test_grouped_topk_rejects_an_invalid_configuration(
    num_experts, num_expert_group, topk_group, top_k, scale, rule
) -> None:
    """Each invalid grouping configuration is refused, naming the rule it breaks."""
    gating = torch.randn(4, num_experts, dtype=torch.bfloat16, device=run_device())
    bias = torch.randn(num_experts, dtype=torch.float32, device=run_device())
    with pytest.raises(ValueError, match=re.escape(rule)):
        FusedTopKFwdOp(
            top_k,
            "sigmoid",
            True,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            routed_scaling_factor=scale,
        )(gating, bias)


@pytest.mark.smoke
def test_spelling_out_the_ungrouped_defaults_changes_nothing() -> None:
    """One group, one kept group and a unit scale return the ungrouped call's bits."""
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": "bfloat16"},
        T=32,
        E=256,
        top_k=8,
        scoring_func="sigmoid",
        renormalize=True,
        some=["correction_bias"],
    )
    workload = FusedTopKWorkload(call)
    gating, bias, _ids = workload.gen_inputs()
    default = FusedTopKFwdOp(8, "sigmoid", True)
    spelled = FusedTopKFwdOp(
        8, "sigmoid", True, num_expert_group=1, topk_group=1, routed_scaling_factor=1.0
    )
    TestBase.check(workload, default, gating, bias)
    for got, want in zip(spelled(gating, bias), default(gating, bias), strict=True):
        assert torch.equal(got, want)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "tied, kept",
    [
        pytest.param((200, 69, 37, 5), (5, 37), id="slots-of-one-lane"),
        pytest.param((200, 40, 9), (9, 40), id="across-lanes"),
    ],
)
def test_negative_tied_selection_scores_keep_the_lower_experts(tied, kept) -> None:
    """Experts tied on a negative biased score at the cut: the lower ones are kept."""
    device = run_device()
    gating = torch.zeros(3, 256, device=device)
    bias = torch.full((256,), -2.0, device=device)
    bias[list(tied)] = -1.0
    weights, ids = FusedTopKFwdOp(2, "sigmoid", renormalize=True)(gating, bias)
    assert ids.sort(dim=-1).values.tolist() == [list(kept)] * 3
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5), rtol=0, atol=0)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_rows_of_tied_minus_inf_scores_select_distinct_experts() -> None:
    """A row whose biased scores all tie at -inf still selects top_k distinct experts of the
    row, the lowest ones, and never the padding past E."""
    device = run_device()
    gating = torch.zeros(2, 33, device=device)
    _, ids = FusedTopKFwdOp(2, "sigmoid")(gating, torch.full((33,), -math.inf, device=device))
    assert ids.sort(dim=-1).values.tolist() == [[0, 1]] * 2


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_grouped_selection_past_the_kept_group_ties_at_minus_inf() -> None:
    """The kept group holds 2 finite biased scores and top_k is 3: the third expert ties at
    -inf with every expert of the dropped group, and the lowest, expert 0, is selected."""
    device = run_device()
    gating = torch.zeros(2, 64, device=device)
    bias = torch.full((64,), -1.0, device=device)
    bias[32:64] = float("-inf")
    bias[32:34] = 0.0
    op = FusedTopKFwdOp(3, "sigmoid", num_expert_group=2, topk_group=1)
    weights, ids = op(gating, bias)
    assert ids.sort(dim=-1).values.tolist() == [[0, 32, 33]] * 2
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5), rtol=0, atol=0)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "num_expert_group, topk_group, top_k, bias, kept",
    [
        # Group scores [-inf, -inf, 1]: the second kept group is group 0, tied at -inf.
        pytest.param(
            3,
            2,
            1,
            [100.0] + [-math.inf] * 31 + [90.0] + [-math.inf] * 31 + [0.0, 0.0] + [-math.inf] * 30,
            [0],
            id="minus-inf-group-picked-serially",
        ),
        # Group scores [inf - inf = NaN, 21]: only the NaN group is kept, so its -inf expert 1
        # is selected, tied at -inf with the dropped group's experts and lower.
        pytest.param(
            2, 1, 2, [math.inf, -math.inf, 10.0, 10.0], [0, 1], id="nan-group-across-lanes"
        ),
        # Group scores [NaN, 21, 1].
        pytest.param(
            3,
            1,
            2,
            [math.inf, -math.inf, 10.0, 10.0, 0.0, 0.0],
            [0, 1],
            id="nan-group-picked-serially",
        ),
    ],
)
def test_groups_rank_non_finite_scores_as_torch_topk(
    num_expert_group, topk_group, top_k, bias, kept
) -> None:
    """Group scores of -inf or NaN rank as ``torch.topk`` ranks them: NaN first, ties to the
    lower group, and a -inf group is kept when no other group is left."""
    device = run_device()
    correction_bias = torch.tensor(bias, device=device)
    gating = torch.zeros(2, correction_bias.numel(), device=device)
    op = FusedTopKFwdOp(top_k, "sigmoid", num_expert_group=num_expert_group, topk_group=topk_group)
    weights, ids = op(gating, correction_bias)
    assert ids.sort(dim=-1).values.tolist() == [kept] * 2
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5), rtol=0, atol=0)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "logit, renormalize, num_expert_group",
    [
        pytest.param(math.inf, False, 2, id="plus-inf-grouped"),
        pytest.param(math.nan, True, 1, id="nan-renormalized"),
        pytest.param(-math.inf, False, 1, id="row-of-minus-inf"),
    ],
)
def test_softmax_rows_with_non_finite_logits_tie_as_the_reference(
    logit, renormalize, num_expert_group
) -> None:
    """A NaN or +inf logit, or a row of -inf, makes the whole softmax row NaN: every expert
    ties, the lowest are selected and their weights are NaN."""
    device = run_device()
    gating = torch.full((2, 64), logit if logit == -math.inf else 0.0, device=device)
    gating[:, 40] = logit
    op = FusedTopKFwdOp(
        2, "softmax", renormalize=renormalize, num_expert_group=num_expert_group, topk_group=1
    )
    weights, ids = op(gating)
    assert ids.sort(dim=-1).values.tolist() == [[0, 1]] * 2
    assert weights.isnan().all()


@pytest.mark.smoke
def test_sigmoid_bias_cancellation_keeps_the_absolute_routing_tolerance() -> None:
    """Biased scores cancelled to near zero differ from torch's by more than 1e-5 of
    themselves; the 1e-6 absolute term admits the kernel's choice among them."""
    device = run_device()
    generator = torch.Generator(device=device).manual_seed(137)
    gating = torch.randn(1, 32, device=device, generator=generator) * 3
    bias = -gating.sigmoid()[0]
    bias[:16] = torch.nextafter(bias[:16], torch.full_like(bias[:16], -float("inf")))
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": "float32"},
        T=1,
        E=32,
        top_k=1,
        scoring_func="sigmoid",
        renormalize=False,
        some=["correction_bias"],
    )
    TestBase.check(FusedTopKWorkload(call), FusedTopKFwdOp(1, "sigmoid"), gating, bias)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_underflowed_softmax_experts_tie_on_the_transformed_score() -> None:
    gating = torch.full((16, 128), -500.0, device=run_device(), dtype=torch.bfloat16)
    gating[:, 126] = 0
    gating[:, 127] = -256
    weights, ids = FusedTopKFwdOp(8, "softmax", renormalize=True)(gating)
    expected_ids = ids.new_tensor([0, 1, 2, 3, 4, 5, 6, 126]).expand_as(ids)
    assert torch.equal(ids.sort(dim=-1).values, expected_ids)
    assert torch.equal(weights, (ids == 126).to(weights.dtype))
