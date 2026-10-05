"""Tests for FusedTopKFwdOp.

Reference: ``FusedTopKWorkload.ref_program`` (softmax/sigmoid, top-k, optional renormalize).

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
    gating, _ = test.gen_inputs()
    op = FusedTopKFwdOp(**test.call.arguments({}))
    TestBase.check(test, op, gating)


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
