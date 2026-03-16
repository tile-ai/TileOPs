"""Op-level tests for MoePermuteAlignOp.

Verifies that the op correctly routes tokens to experts and pads each
expert's slot count to the GEMM block_size boundary.

Reference: SGLang moe_align_block_size
  python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py
"""

import math
from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoePermuteAlignOp


# ---------------------------------------------------------------------------
# Reference implementation (pure Python / PyTorch)
# ---------------------------------------------------------------------------


def _ref_permute_align(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Python reference for permute_align.

    Args:
        topk_ids: [total_tokens, top_k] int32 expert indices (0-indexed).
        block_size: GEMM tile size M dimension.
        num_experts: total number of experts.

    Returns:
        sorted_token_ids: [max_num_tokens_padded] int32
        expert_ids:       [num_blocks] int32
        num_tokens_post_pad: [1] int32 scalar
    """
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

    num_blocks = total_padded // block_size
    expert_ids_list = []
    for b in range(num_blocks):
        block_start = b * block_size
        lo, hi = 0, num_experts - 1
        eid = num_experts - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if cumsum[mid] <= block_start < cumsum[mid + 1]:
                eid = mid
                break
            elif block_start < cumsum[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        expert_ids_list.append(eid)

    device = topk_ids.device
    return (
        torch.tensor(sorted_token_ids, dtype=torch.int32, device=device),
        torch.tensor(expert_ids_list, dtype=torch.int32, device=device),
        torch.tensor([total_padded], dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


class MoePermuteAlignFixture(FixtureBase):
    PARAMS = [
        ("total_tokens, top_k, num_experts, block_size", [
            pytest.param(4,   2,   4,   4,   marks=pytest.mark.smoke, id="tiny-bs4"),
            pytest.param(16,  2,   8,   16,  marks=pytest.mark.full,  id="small-bs16"),
            pytest.param(128, 4,   8,   64,  marks=pytest.mark.full,  id="medium-bs64"),
            pytest.param(1024,8,   64,  128, marks=pytest.mark.full,  id="large-bs128"),
            pytest.param(1,   2,   4,   4,   marks=pytest.mark.full,  id="single-token"),
            pytest.param(8,   1,   4,   4,   marks=pytest.mark.full,  id="one-expert"),
        ]),
    ]


# ---------------------------------------------------------------------------
# TestBase subclass
# ---------------------------------------------------------------------------


class MoePermuteAlignTest(TestBase):

    def __init__(self, total_tokens: int, top_k: int, num_experts: int, block_size: int):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32, device="cuda",
        )
        return (topk_ids,)

    def ref_program(
        self, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _ref_permute_align(topk_ids, self.block_size, self.num_experts)


# ---------------------------------------------------------------------------
# Custom comparator
# ---------------------------------------------------------------------------


def _permute_align_compare(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    outputs_ref: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Order-insensitive comparison for permute_align outputs.

    sorted_token_ids comparison is per-expert token-set (parallel atomicAdd
    makes intra-expert ordering non-deterministic).
    """
    sorted_ids, expert_ids, num_post_pad = outputs
    ref_sorted, ref_expert, ref_num = outputs_ref

    numel = sorted_ids[sorted_ids < sorted_ids.max().item() + 1].numel()  # sentinel = numel
    # Determine numel from expert_ids size and block_size (passed via closure below)
    # We use a simple heuristic: any value >= ref num_post_pad is sentinel
    n = ref_num.item()
    num_blocks = n // _block_size  # injected by the test function

    assert num_post_pad.item() == ref_num.item(), (
        f"num_tokens_post_pad mismatch: got {num_post_pad.item()}, "
        f"expected {ref_num.item()}"
    )
    assert torch.equal(expert_ids[:num_blocks].cpu(), ref_expert[:num_blocks].cpu()), (
        f"expert_ids mismatch:\n  got: {expert_ids[:num_blocks].cpu()}"
        f"\n  ref: {ref_expert[:num_blocks].cpu()}"
    )

    got_sorted = sorted_ids[:n].cpu().tolist()
    eids = expert_ids[:num_blocks].cpu().tolist()
    sentinel = ref_num.item()  # numel is sentinel value
    for e in range(_num_experts):
        got_tokens = sorted(
            tok
            for b, eid in enumerate(eids)
            if eid == e
            for tok in got_sorted[b * _block_size:(b + 1) * _block_size]
            if tok < sentinel
        )
        ref_tokens = sorted(
            tok
            for b, eid in enumerate(eids)
            if eid == e
            for tok in ref_sorted[:n].cpu().tolist()[b * _block_size:(b + 1) * _block_size]
            if tok < sentinel
        )
        assert got_tokens == ref_tokens, (
            f"Expert {e} token set mismatch:\n  got: {got_tokens}\n  ref: {ref_tokens}"
        )

    padding_mask = sorted_ids[:n].cpu() >= sentinel
    assert (sorted_ids[:n].cpu()[padding_mask] == sentinel).all(), (
        "Padding slots must equal sentinel (numel)"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Module-level state used by the comparator closure
_block_size: int = 0
_num_experts: int = 0


@MoePermuteAlignFixture
def test_permute_align_op(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    global _block_size, _num_experts
    _block_size = block_size
    _num_experts = num_experts

    test = MoePermuteAlignTest(total_tokens, top_k, num_experts, block_size)
    op = MoePermuteAlignOp(total_tokens * top_k, num_experts, block_size)
    inputs = test.gen_inputs()

    outputs = op(*inputs)
    outputs_ref = test.ref_program(*inputs)

    # Normalize to tuples for comparator
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    if isinstance(outputs_ref, torch.Tensor):
        outputs_ref = (outputs_ref,)

    _permute_align_compare(tuple(outputs), tuple(outputs_ref))
    print(f"All checks passed for MoePermuteAlignOp [{total_tokens}tok, top{top_k}, "
          f"E={num_experts}, bs={block_size}].")


@pytest.mark.smoke
def test_permute_align_sentinel_padding() -> None:
    """Padding slots must be filled with sentinel value (numel)."""
    total_tokens, top_k, num_experts, block_size = 3, 2, 4, 4
    numel = total_tokens * top_k
    topk_ids = torch.randint(0, num_experts, (total_tokens, top_k),
                              dtype=torch.int32, device="cuda")

    op = MoePermuteAlignOp(numel, num_experts, block_size)
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
    topk_ids = torch.randint(0, num_experts, (total_tokens, top_k),
                              dtype=torch.int32, device="cuda")

    op = MoePermuteAlignOp(total_tokens * top_k, num_experts, block_size)
    _, expert_ids, num_post_pad = op(topk_ids)

    n = num_post_pad.item()
    num_blocks = n // block_size
    eids = expert_ids[:num_blocks].cpu()
    assert (eids >= 0).all() and (eids < num_experts).all(), (
        f"expert_ids out of range [0, {num_experts}): {eids}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
