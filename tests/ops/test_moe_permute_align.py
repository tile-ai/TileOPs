"""Smoke test for MoePermuteAlignFwdOp."""

import math

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoePermuteAlignFwdOp
from workloads.moe import MoePermuteAlignTest as _MoePermuteAlignTestWorkload


def _ref_permute_align(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = topk_ids.numel()
    flat = topk_ids.flatten().tolist()

    counts = [0] * num_experts
    for eid in flat:
        counts[eid] += 1

    cumsum = [0] * (num_experts + 1)
    for i in range(num_experts):
        padded = math.ceil(counts[i] / block_size) * block_size
        cumsum[i + 1] = cumsum[i] + padded

    total_padded = cumsum[num_experts]
    sorted_token_ids = [numel] * total_padded
    slot = list(cumsum[:-1])
    for flat_idx, eid in enumerate(flat):
        sorted_token_ids[slot[eid]] = flat_idx
        slot[eid] += 1

    expert_ids_list = []
    for block_start in range(0, total_padded, block_size):
        for eid in range(num_experts):
            if cumsum[eid] <= block_start < cumsum[eid + 1]:
                expert_ids_list.append(eid)
                break

    device = topk_ids.device
    return (
        torch.tensor(sorted_token_ids, dtype=torch.int32, device=device),
        torch.tensor(expert_ids_list, dtype=torch.int32, device=device),
        torch.tensor([total_padded], dtype=torch.int32, device=device),
    )


class MoePermuteAlignTest(_MoePermuteAlignTestWorkload, TestBase):
    def ref_program(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _ref_permute_align(topk_ids, self.block_size, self.num_experts)


class MoePermuteAlignFixture(FixtureBase):
    PARAMS = [
        (
            "total_tokens, top_k, num_experts, block_size",
            [
                pytest.param(
                    16, 2, 8, 16, marks=pytest.mark.smoke, id="small-bs16"
                ),
            ],
        ),
    ]


def _permute_align_compare(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    outputs_ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    block_size: int,
    num_experts: int,
    numel: int,
) -> None:
    sorted_ids, expert_ids, num_post_pad = outputs
    ref_sorted, ref_expert, ref_num = outputs_ref

    n = ref_num.item()
    num_blocks = n // block_size

    assert num_post_pad.item() == ref_num.item()
    assert torch.equal(expert_ids[:num_blocks].cpu(), ref_expert[:num_blocks].cpu())

    got_sorted = sorted_ids[:n].cpu().tolist()
    ref_sorted_list = ref_sorted[:n].cpu().tolist()
    ref_eids = ref_expert[:num_blocks].cpu().tolist()
    for eid in range(num_experts):
        got_tokens = sorted(
            tok
            for block, block_eid in enumerate(ref_eids)
            if block_eid == eid
            for tok in got_sorted[block * block_size : (block + 1) * block_size]
            if tok < numel
        )
        ref_tokens = sorted(
            tok
            for block, block_eid in enumerate(ref_eids)
            if block_eid == eid
            for tok in ref_sorted_list[block * block_size : (block + 1) * block_size]
            if tok < numel
        )
        assert got_tokens == ref_tokens

    padding_mask = sorted_ids[:n].cpu() >= numel
    assert (sorted_ids[:n].cpu()[padding_mask] == numel).all()


@MoePermuteAlignFixture
def test_permute_align_op(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    numel = total_tokens * top_k
    test = MoePermuteAlignTest(total_tokens, top_k, num_experts, block_size)
    op = MoePermuteAlignFwdOp(numel, num_experts, block_size)
    inputs = test.gen_inputs()

    outputs = tuple(op(*inputs))
    outputs_ref = tuple(test.ref_program(*inputs))

    _permute_align_compare(outputs, outputs_ref, block_size, num_experts, numel)
