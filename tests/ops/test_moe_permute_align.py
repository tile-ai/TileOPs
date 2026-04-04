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
from workloads.ops.moe_permute_align import MoePermuteAlignTest as _MoePermuteAlignTestWorkload


def _ref_permute_align(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Python reference for permute_align."""
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


class MoePermuteAlignTest(_MoePermuteAlignTestWorkload, TestBase):
    def ref_program(
        self, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _ref_permute_align(topk_ids, self.block_size, self.num_experts)


# ---------------------------------------------------------------------------
# Reference implementation (pure Python / PyTorch)
# ---------------------------------------------------------------------------


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
            # top_k=1: each token is routed to exactly one expert
            pytest.param(8,   1,   4,   4,   marks=pytest.mark.full,  id="top-k-1"),
            # small-batch path (numel < 1024, num_experts <= 64)
            pytest.param(100, 2,   8,   16,  marks=pytest.mark.full,  id="sb-numel200"),
            pytest.param(300, 2,   8,   16,  marks=pytest.mark.full,  id="sb-numel600"),
            pytest.param(400, 2,   8,   16,  marks=pytest.mark.full,  id="sb-numel800"),
            pytest.param(100, 6,  64,   64,  marks=pytest.mark.full,  id="sb-numel600-maxexp"),
            # dispatch boundary: numel=1023 (last small-batch) vs numel=1024 (first large-batch)
            pytest.param(511, 2,   8,   16,  marks=pytest.mark.full,  id="sb-boundary-1022"),
            pytest.param(512, 2,   8,   16,  marks=pytest.mark.full,  id="lb-boundary-1024"),
        ]),
    ]


# ---------------------------------------------------------------------------
# TestBase subclass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Custom comparator
# ---------------------------------------------------------------------------


def _permute_align_compare(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    outputs_ref: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    block_size: int,
    num_experts: int,
    numel: int,
) -> None:
    """Order-insensitive comparison for permute_align outputs.

    sorted_token_ids comparison is per-expert token-set (parallel atomicAdd
    makes intra-expert ordering non-deterministic).

    Args:
        outputs: (sorted_token_ids, expert_ids, num_tokens_post_pad) from kernel.
        outputs_ref: same tuple from reference implementation.
        block_size: GEMM tile size used to compute num_blocks.
        num_experts: number of experts.
        numel: total (token, expert) assignments; also the sentinel value.
    """
    sorted_ids, expert_ids, num_post_pad = outputs
    ref_sorted, ref_expert, ref_num = outputs_ref

    n = ref_num.item()
    num_blocks = n // block_size

    assert num_post_pad.item() == ref_num.item(), (
        f"num_tokens_post_pad mismatch: got {num_post_pad.item()}, "
        f"expected {ref_num.item()}"
    )
    assert torch.equal(expert_ids[:num_blocks].cpu(), ref_expert[:num_blocks].cpu()), (
        f"expert_ids mismatch:\n  got: {expert_ids[:num_blocks].cpu()}"
        f"\n  ref: {ref_expert[:num_blocks].cpu()}"
    )

    got_sorted = sorted_ids[:n].cpu().tolist()
    # Use the reference expert_ids (verified equal above) for both slices so
    # the per-expert token sets are computed consistently.
    ref_eids = ref_expert[:num_blocks].cpu().tolist()
    for e in range(num_experts):
        got_tokens = sorted(
            tok
            for b, eid in enumerate(ref_eids)
            if eid == e
            for tok in got_sorted[b * block_size:(b + 1) * block_size]
            if tok < numel
        )
        ref_tokens = sorted(
            tok
            for b, eid in enumerate(ref_eids)
            if eid == e
            for tok in ref_sorted[:n].cpu().tolist()[b * block_size:(b + 1) * block_size]
            if tok < numel
        )
        assert got_tokens == ref_tokens, (
            f"Expert {e} token set mismatch:\n  got: {got_tokens}\n  ref: {ref_tokens}"
        )

    # Padding slots must all equal sentinel
    padding_mask = sorted_ids[:n].cpu() >= numel
    assert (sorted_ids[:n].cpu()[padding_mask] == numel).all(), (
        "Padding slots must equal sentinel (numel)"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@MoePermuteAlignFixture
def test_permute_align_op(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    numel = total_tokens * top_k
    test = MoePermuteAlignTest(total_tokens, top_k, num_experts, block_size)
    op = MoePermuteAlignOp(numel, num_experts, block_size)
    inputs = test.gen_inputs()

    outputs = tuple(op(*inputs))
    outputs_ref = tuple(test.ref_program(*inputs))

    _permute_align_compare(outputs, outputs_ref, block_size, num_experts, numel)
    print(f"All checks passed for MoePermuteAlignOp [{total_tokens}tok, top{top_k}, "
          f"E={num_experts}, bs={block_size}].")


@pytest.mark.smoke
def test_permute_align_sentinel_padding() -> None:
    """Padding slots must be filled with sentinel value (numel).

    Uses 3 tokens (not a multiple of block_size=4) to force non-trivial padding.
    """
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


@pytest.mark.smoke
def test_permute_align_skewed_distribution() -> None:
    """All tokens routed to expert 0 — stress-tests Step 3 loop bound.

    With a uniform loop bound of ceil(max_num_blocks / num_experts), expert 0
    would only write the first few expert_ids entries and leave the rest
    uninitialised. This test catches that regression.
    """
    total_tokens, top_k, num_experts, block_size = 32, 4, 8, 16
    numel = total_tokens * top_k
    # All tokens go to expert 0
    topk_ids = torch.zeros((total_tokens, top_k), dtype=torch.int32, device="cuda")

    op = MoePermuteAlignOp(numel, num_experts, block_size)
    outputs = tuple(op(topk_ids))
    outputs_ref = tuple(_ref_permute_align(topk_ids, block_size, num_experts))

    _permute_align_compare(outputs, outputs_ref, block_size, num_experts, numel)
    print("All checks passed for skewed distribution (all tokens -> expert 0).")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
