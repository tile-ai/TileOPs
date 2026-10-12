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

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.moe import FusedTopKFwdOp
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
                # DeepSeek-V3: a group is one warp row of 32 experts, scored by its biased top-2.
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
                # DeepSeek-V2: softmax without bias scores a group by its max; groups of 20
                # straddle warp rows; the scale lands on the row-sum divide.
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
                # Sigmoid without bias: the max group score on the sigmoid path.
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
                # E = 100 is off the warp width: padding slots belong to no group.
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
                # Three groups: a group count off a power of two merges per group.
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
                # The scale alone, on the ungrouped kernel.
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
def test_spelling_out_the_ungrouped_defaults_changes_nothing() -> None:
    """Spelling out num_expert_group=1, topk_group=1 and a unit scale is the ungrouped call:
    the same kernel build, so the same bits."""
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
