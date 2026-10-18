"""Tests for FusedTopKFwdOp.

Reference: ``FusedTopKWorkload.ref_program`` (softmax/sigmoid, top-k, optional renormalize,
grouped routing, scaling factor).

Test cases cover:
  - softmax scoring (Qwen3/Qwen2 style)
  - sigmoid scoring (DeepSeek-V3/GLM-4 style)
  - renormalize=True / False
  - Various (num_tokens, num_experts, top_k) shapes
  - bf16 and fp16 input dtypes
  - top_k=1, top_k=8
  - grouped routing: the biased top-2 group score and the unbiased max, groups spanning
    whole warp rows and groups that straddle them, E off the warp width
  - routed_scaling_factor with and without grouping
"""

import re

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.moe import FusedTopKFwdOp
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
    inputs = [t for t in test.gen_inputs() if t is not None]
    op = FusedTopKFwdOp(**test.call.arguments({}))
    TestBase.check(test, op, *inputs)


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
    gating, bias = workload.gen_inputs()
    default = FusedTopKFwdOp(8, "sigmoid", True)
    spelled = FusedTopKFwdOp(
        8, "sigmoid", True, num_expert_group=1, topk_group=1, routed_scaling_factor=1.0
    )
    TestBase.check(workload, default, gating, bias)
    for got, want in zip(spelled(gating, bias), default(gating, bias), strict=True):
        assert torch.equal(got, want)


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
