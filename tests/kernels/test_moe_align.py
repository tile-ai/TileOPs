"""Tests for moe_align_kernel.

Reference: SGLang moe_align_block_size
  python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py

The kernel takes topk_ids [total_tokens, top_k] and produces:
  - sorted_token_ids [max_num_tokens_padded]: flat token indices sorted by expert,
      padded with sentinel value (numel) to align each expert's count to block_size
  - expert_ids [num_blocks]: which expert each GEMM block belongs to
  - num_tokens_post_pad [1]: total padded token count (scalar)
"""

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# Reference implementation (pure Python/PyTorch)
# ---------------------------------------------------------------------------


def _ref_moe_align(topk_ids: torch.Tensor, block_size: int,
                   num_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Python reference for moe_align_block_size.

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

    # Count tokens per expert
    counts = [0] * num_experts
    for eid in flat:
        counts[eid] += 1

    # Compute padded cumsum (exclusive prefix sum of padded counts)
    cumsum = [0] * (num_experts + 1)
    for i in range(num_experts):
        padded = math.ceil(counts[i] / block_size) * block_size
        cumsum[i + 1] = cumsum[i] + padded

    total_padded = cumsum[num_experts]

    # Allocate sorted_token_ids filled with sentinel (numel)
    sorted_token_ids = [numel] * total_padded

    # Fill sorted_token_ids: for each token, place its flat index at the
    # next available slot for its expert
    slot = list(cumsum[:-1])  # current write position per expert
    for flat_idx, eid in enumerate(flat):
        sorted_token_ids[slot[eid]] = flat_idx
        slot[eid] += 1

    # Build expert_ids: one entry per GEMM block
    num_blocks = total_padded // block_size
    expert_ids_list = []
    for b in range(num_blocks):
        block_start = b * block_size
        # binary search: find which expert owns this block
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
# Fixtures
# ---------------------------------------------------------------------------


class MoeAlignFixture:
    PARAMS = [
        # (total_tokens, top_k, num_experts, block_size)
        pytest.param(4, 2, 4, 4, marks=pytest.mark.smoke, id="tiny-bs4"),
        pytest.param(16, 2, 8, 16, marks=pytest.mark.smoke, id="small-bs16"),
        pytest.param(128, 4, 8, 64, marks=pytest.mark.full, id="medium-bs64"),
        pytest.param(1024, 8, 64, 128, marks=pytest.mark.full, id="large-bs128"),
        # edge: single token
        pytest.param(1, 2, 4, 4, marks=pytest.mark.full, id="single-token"),
        # edge: all tokens go to one expert
        pytest.param(8, 1, 4, 4, marks=pytest.mark.full, id="one-expert"),
    ]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_and_compare(total_tokens: int, top_k: int, num_experts: int, block_size: int) -> None:
    """Generate random topk_ids, run kernel and reference, compare outputs."""
    from tileops.kernels.moe import moe_align_kernel  # noqa: PLC0415

    topk_ids = torch.randint(0, num_experts, (total_tokens, top_k), dtype=torch.int32, device="cuda")

    kernel = moe_align_kernel(topk_ids.numel(), num_experts, block_size)
    sorted_ids, expert_ids, num_post_pad = kernel(topk_ids)

    ref_sorted, ref_expert, ref_num = _ref_moe_align(topk_ids, block_size, num_experts)

    # num_tokens_post_pad must match exactly
    assert num_post_pad.item() == ref_num.item(), (
        f"num_tokens_post_pad mismatch: got {num_post_pad.item()}, expected {ref_num.item()}"
    )

    numel = topk_ids.numel()
    n = ref_num.item()
    num_blocks = n // block_size

    # expert_ids must match exactly
    assert torch.equal(expert_ids[:num_blocks].cpu(), ref_expert[:num_blocks].cpu()), (
        f"expert_ids mismatch:\n  got: {expert_ids[:num_blocks].cpu()}\n  ref: {ref_expert[:num_blocks].cpu()}"
    )

    # sorted_token_ids: for each expert, the set of real token indices must match.
    # We do not require a specific intra-expert ordering because the parallel
    # sort kernel uses atomicAdd whose order is non-deterministic.
    got_sorted = sorted_ids[:n].cpu().tolist()
    ref_sorted_n = ref_sorted[:n].cpu().tolist()

    # Build per-expert token sets from got and ref using expert_ids
    eids = expert_ids[:num_blocks].cpu().tolist()
    for e in range(num_experts):
        got_tokens = sorted(
            tok
            for b, eid in enumerate(eids)
            if eid == e
            for tok in got_sorted[b * block_size:(b + 1) * block_size]
            if tok < numel
        )
        ref_tokens = sorted(
            tok
            for b, eid in enumerate(eids)
            if eid == e
            for tok in ref_sorted_n[b * block_size:(b + 1) * block_size]
            if tok < numel
        )
        assert got_tokens == ref_tokens, (
            f"Expert {e} token set mismatch:\n  got: {got_tokens}\n  ref: {ref_tokens}"
        )

    # Sentinel slots must be filled with numel
    got_flat = sorted_ids[:n].cpu()
    padding_mask = got_flat >= numel
    assert (got_flat[padding_mask] == numel).all(), "Padding slots must equal sentinel (numel)"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_tokens, top_k, num_experts, block_size", MoeAlignFixture.PARAMS)
def test_moe_align_correctness(total_tokens: int, top_k: int, num_experts: int,
                               block_size: int) -> None:
    _run_and_compare(total_tokens, top_k, num_experts, block_size)


@pytest.mark.smoke
def test_moe_align_sentinel_padding() -> None:
    """Padding slots must be filled with sentinel value (numel = total_tokens * top_k)."""
    from tileops.kernels.moe import moe_align_kernel  # noqa: PLC0415

    total_tokens, top_k, num_experts, block_size = 3, 2, 4, 4
    topk_ids = torch.randint(0, num_experts, (total_tokens, top_k), dtype=torch.int32, device="cuda")
    numel = topk_ids.numel()

    kernel = moe_align_kernel(numel, num_experts, block_size)
    sorted_ids, _, num_post_pad = kernel(topk_ids)

    n = num_post_pad.item()
    # Padding slots (real token indices >= numel) must equal sentinel
    padding_mask = sorted_ids[:n] >= numel
    assert (sorted_ids[:n][padding_mask] == numel).all(), "Padding slots must equal sentinel (numel)"


@pytest.mark.smoke
def test_moe_align_expert_ids_range() -> None:
    """All expert_ids must be in [0, num_experts)."""
    from tileops.kernels.moe import moe_align_kernel  # noqa: PLC0415

    total_tokens, top_k, num_experts, block_size = 16, 4, 8, 16
    topk_ids = torch.randint(0, num_experts, (total_tokens, top_k), dtype=torch.int32, device="cuda")

    kernel = moe_align_kernel(topk_ids.numel(), num_experts, block_size)
    _, expert_ids, num_post_pad = kernel(topk_ids)

    n = num_post_pad.item()
    num_blocks = n // block_size
    eids = expert_ids[:num_blocks].cpu()
    assert (eids >= 0).all() and (eids < num_experts).all(), (
        f"expert_ids out of range [0, {num_experts}): {eids}"
    )
