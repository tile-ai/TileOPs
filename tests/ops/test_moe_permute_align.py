"""Op-level tests for MoEPermuteAlignFwdOp.

Verifies that the op correctly routes tokens to experts and pads each
expert's slot count to the GEMM block_size boundary.

Reference: SGLang moe_align_block_size
  python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoEPermuteAlignFwdOp
from workloads.device import run_device
from workloads.moe import MoEPermuteAlignWorkload, moe_call


class MoEPermuteAlignFixture(FixtureBase):
    PARAMS = [
        (
            "total_tokens, top_k, num_experts, block_size",
            [
                pytest.param(
                    4,
                    2,
                    4,
                    4,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="moe")],
                    id="tiny-bs4",
                ),
                pytest.param(16, 2, 8, 16, marks=pytest.mark.full, id="small-bs16"),
                pytest.param(128, 4, 8, 64, marks=pytest.mark.full, id="medium-bs64"),
                pytest.param(1024, 8, 64, 128, marks=pytest.mark.full, id="large-bs128"),
                pytest.param(1, 2, 4, 4, marks=pytest.mark.full, id="single-token"),
                # top_k=1: each token is routed to exactly one expert
                pytest.param(8, 1, 4, 4, marks=pytest.mark.full, id="top-k-1"),
                # small-batch path (numel < 1024, num_experts <= 64)
                pytest.param(100, 2, 8, 16, marks=pytest.mark.full, id="sb-numel200"),
                pytest.param(300, 2, 8, 16, marks=pytest.mark.full, id="sb-numel600"),
                pytest.param(400, 2, 8, 16, marks=pytest.mark.full, id="sb-numel800"),
                pytest.param(100, 6, 64, 64, marks=pytest.mark.full, id="sb-numel600-maxexp"),
                # dispatch boundary: numel=1023 (last small-batch) vs numel=1024 (first large-batch)
                pytest.param(511, 2, 8, 16, marks=pytest.mark.full, id="sb-boundary-1022"),
                pytest.param(512, 2, 8, 16, marks=pytest.mark.full, id="lb-boundary-1024"),
                # More experts than the alignment block has threads.
                pytest.param(64, 8, 1100, 64, marks=pytest.mark.full, id="experts-past-threads"),
            ],
        ),
    ]


# Custom comparator


@MoEPermuteAlignFixture
def test_permute_align_op(total_tokens: int, top_k: int, num_experts: int, block_size: int) -> None:
    call = moe_call(
        "MoEPermuteAlignFwdOp",
        T=total_tokens,
        K=top_k,
        num_experts=num_experts,
        block_size=block_size,
    )
    test = MoEPermuteAlignWorkload(call)
    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    inputs = test.gen_inputs()

    TestBase.check(test, op, *inputs)


@pytest.mark.smoke
def test_permute_align_sentinel_padding() -> None:
    """Padding slots must be filled with sentinel value (numel).

    Uses 3 tokens (not a multiple of block_size=4) to force non-trivial padding.
    """
    total_tokens, top_k, num_experts, block_size = 3, 2, 4, 4
    numel = total_tokens * top_k
    topk_ids = torch.randint(
        0, num_experts, (total_tokens, top_k), dtype=torch.int32, device=run_device()
    )

    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    sorted_ids, _, num_post_pad = op(topk_ids)

    n = num_post_pad.item()
    padding_mask = sorted_ids[:n] >= numel
    assert (sorted_ids[:n][padding_mask] == numel).all(), (
        "Padding slots must equal sentinel (numel)"
    )


@pytest.mark.smoke
def test_permute_align_expert_ids_range() -> None:
    """All expert_ids must be in [0, num_experts)."""
    total_tokens, top_k, num_experts, block_size = 16, 4, 8, 16
    topk_ids = torch.randint(
        0, num_experts, (total_tokens, top_k), dtype=torch.int32, device=run_device()
    )

    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    _, expert_ids, num_post_pad = op(topk_ids)

    n = num_post_pad.item()
    num_blocks = n // block_size
    eids = expert_ids[:num_blocks].cpu()
    assert (eids >= 0).all() and (eids < num_experts).all(), (
        f"expert_ids out of range [0, {num_experts}): {eids}"
    )


@pytest.mark.smoke
def test_permute_align_skewed_distribution() -> None:
    """All tokens routed to expert 0 — stress-tests Step 3 loop bound.

    With a uniform loop bound of ceil(max_num_blocks / num_experts), expert 0
    would only write the first few expert_ids entries and leave the rest
    uninitialised. This test catches that regression.
    """
    total_tokens, top_k, num_experts, block_size = 32, 4, 8, 16
    # All tokens go to expert 0
    topk_ids = torch.zeros((total_tokens, top_k), dtype=torch.int32, device=run_device())

    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    workload = MoEPermuteAlignWorkload(
        moe_call(
            "MoEPermuteAlignFwdOp",
            T=total_tokens,
            K=top_k,
            num_experts=num_experts,
            block_size=block_size,
        )
    )
    TestBase.check(workload, op, topk_ids)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_permute_align_builds_one_kernel_per_routed_count() -> None:
    """The routed count comes from each call, so a second count builds a second kernel."""
    op = MoEPermuteAlignFwdOp(num_experts=8, block_size=16)
    for tokens in (4, 4, 6):
        op(torch.randint(0, 8, (tokens, 2), dtype=torch.int32, device=run_device()))
    assert len(op.built_kernels("permute_align")) == 2
