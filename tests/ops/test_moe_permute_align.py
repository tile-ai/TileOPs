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
from tileops._ops._signature_codegen import CheckError
from tileops._ops.moe import MoEPermuteAlignFwdOp
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
                # A few hundred routes over 32 and 33 experts, either side of the one-warp
                # small-batch limit.
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


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    "num_experts",
    [pytest.param(128, marks=pytest.mark.smoke), pytest.param(384, marks=pytest.mark.full)],
)
def test_sparse_routes_across_expert_blocks(num_experts) -> None:
    from tileops._kernels.build_target import building_on
    from tileops._kernels.moe.permute_align import _make_sparse_routes_kernel
    from tileops.backend import BUILTIN

    # The first expert spans two blocks. Other active experts belong to different CTAs.
    ids = torch.tensor(
        [1] * 65 + [num_experts // 2] * 33 + [num_experts - 1] * 31,
        dtype=torch.int32,
        device=run_device(),
    )
    workload = MoEPermuteAlignWorkload(
        manifest_call("MoEPermuteAlignFwdOp", T=129, K=1, num_experts=num_experts, block_size=64)
    )
    op = MoEPermuteAlignFwdOp(num_experts, 64, target=BUILTIN)
    TestBase.check(workload, op, ids.view(-1, 1))
    (kernel,) = op.iter_kernels()
    with building_on(ids.device):
        assert kernel._fused_fn is _make_sparse_routes_kernel(129, num_experts)


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
