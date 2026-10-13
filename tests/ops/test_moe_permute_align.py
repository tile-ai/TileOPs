"""Op-level tests for MoEPermuteAlignFwdOp.

Verifies that the op correctly routes tokens to experts and pads each
expert's slot count to the GEMM block_size boundary.

Reference: SGLang moe_align_block_size
  python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py
"""

import re

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops._signature_codegen import CheckError
from tileops.ops.moe import MoEPermuteAlignFwdOp
from workloads.device import run_device
from workloads.moe import MoEPermuteAlignWorkload
from workloads.workload_base import manifest_call


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
                pytest.param(1024, 8, 64, 128, marks=pytest.mark.full, id="large-bs128"),
                pytest.param(1, 2, 4, 4, marks=pytest.mark.full, id="single-token"),
                # top_k=1: each token is routed to exactly one expert
                pytest.param(8, 1, 4, 4, marks=pytest.mark.full, id="top-k-1"),
                # A few hundred routes over 8, 32 and 33 experts.
                pytest.param(100, 2, 8, 16, marks=pytest.mark.full, id="sb-numel200"),
                pytest.param(100, 6, 32, 64, marks=pytest.mark.full, id="sb-numel600-maxexp"),
                pytest.param(100, 6, 33, 64, marks=pytest.mark.full, id="experts-33"),
                # dispatch boundary: numel=1023 (last small-batch) vs numel=1024 (first large-batch)
                pytest.param(1023, 1, 8, 16, marks=pytest.mark.full, id="routes-1023"),
                pytest.param(512, 2, 8, 16, marks=pytest.mark.full, id="lb-boundary-1024"),
                # 1024 experts, then 1100.
                pytest.param(64, 8, 1024, 64, marks=pytest.mark.full, id="experts-1024"),
                pytest.param(64, 8, 1100, 64, marks=pytest.mark.full, id="experts-past-threads"),
                # No routes, over many experts and over few.
                pytest.param(0, 8, 128, 64, marks=pytest.mark.full, id="no-routes"),
                pytest.param(0, 8, 8, 16, marks=pytest.mark.full, id="no-routes-few-experts"),
                # One route over many experts.
                pytest.param(1, 1, 128, 64, marks=pytest.mark.full, id="one-route"),
                # Every route on the only expert.
                pytest.param(2048, 1, 1, 64, marks=pytest.mark.full, id="one-expert"),
                # A block height of one, then one that is not a power of two.
                pytest.param(64, 8, 128, 1, marks=pytest.mark.full, id="bs1"),
                pytest.param(64, 8, 128, 48, marks=pytest.mark.full, id="bs-not-pow2"),
                # 1032 routes, which no power of two above 8 divides.
                pytest.param(129, 8, 128, 64, marks=pytest.mark.full, id="routes-1032"),
                # 512 experts, then 513.
                pytest.param(64, 8, 512, 64, marks=pytest.mark.full, id="experts-512"),
                pytest.param(64, 8, 513, 64, marks=pytest.mark.full, id="experts-513"),
            ],
        ),
    ]


# Custom comparator


@MoEPermuteAlignFixture
def test_permute_align_op(total_tokens: int, top_k: int, num_experts: int, block_size: int) -> None:
    call = manifest_call(
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


@pytest.mark.parametrize(
    "num_experts",
    [
        pytest.param(4, marks=pytest.mark.smoke, id="few-experts"),
        pytest.param(128, marks=pytest.mark.full, id="many-experts"),
    ],
)
def test_permute_align_sentinel_padding(num_experts: int) -> None:
    """Every slot that holds no route is numel, and every block past the padded length names
    num_experts, over few experts and over many.

    Six routes in blocks of 4 leave padding inside the ranges. The workload verification reads
    neither tail, so this is what checks them.
    """
    total_tokens, top_k, block_size = 3, 2, 4
    numel = total_tokens * top_k
    topk_ids = torch.randint(
        0, num_experts, (total_tokens, top_k), dtype=torch.int32, device=run_device()
    )

    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    sorted_ids, expert_ids, num_post_pad = op(topk_ids)

    routed = sorted_ids < numel
    assert int(routed.sum()) == numel, "every route occupies one slot"
    assert (sorted_ids[~routed] == numel).all(), "a slot without a route holds numel"
    assert (expert_ids[num_post_pad.item() // block_size :] == num_experts).all(), (
        "a block past the padded length names num_experts"
    )


@pytest.mark.parametrize(
    "total_tokens, top_k, num_experts, block_size",
    [
        pytest.param(32, 4, 8, 16, marks=pytest.mark.smoke, id="few-experts"),
        pytest.param(256, 8, 384, 64, marks=pytest.mark.full, id="many-experts"),
    ],
)
def test_permute_align_skewed_distribution(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    """Every route on expert 0, over few experts and over many."""
    topk_ids = torch.zeros((total_tokens, top_k), dtype=torch.int32, device=run_device())

    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    workload = MoEPermuteAlignWorkload(
        manifest_call(
            "MoEPermuteAlignFwdOp",
            T=total_tokens,
            K=top_k,
            num_experts=num_experts,
            block_size=block_size,
        )
    )
    TestBase.check(workload, op, topk_ids)


@pytest.mark.parametrize(
    "num_experts, block_size, rule",
    [
        pytest.param(0, 64, "num_experts > 0", marks=pytest.mark.smoke, id="no-experts"),
        pytest.param(8, 0, "block_size > 0", marks=pytest.mark.full, id="empty-block"),
    ],
)
def test_permute_align_rejects_nonpositive_params(
    num_experts: int, block_size: int, rule: str
) -> None:
    """The op refuses an expert count or block height below one when it is built."""
    with pytest.raises(CheckError, match=re.escape(rule)):
        MoEPermuteAlignFwdOp(num_experts, block_size)


@pytest.mark.smoke
def test_permute_align_tuned_mode() -> None:
    """A tuned op returns a verified result, and the same result on the call after."""
    total_tokens, top_k, num_experts, block_size = 64, 8, 128, 64
    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    op.request_tune()
    workload = MoEPermuteAlignWorkload(
        manifest_call(
            "MoEPermuteAlignFwdOp",
            T=total_tokens,
            K=top_k,
            num_experts=num_experts,
            block_size=block_size,
        )
    )
    inputs = workload.gen_inputs()
    TestBase.check(workload, op, *inputs)
    TestBase.check(workload, op, *inputs)
