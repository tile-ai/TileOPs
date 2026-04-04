"""Op-level tests for MoePermutePaddedOp (cutlass path).

Verifies:
  - perm_h_pad: correct gather of hidden_states rows into padded expert layout
  - expert_first_token_offset: correct exclusive prefix-sum (int64)
  - padded_offsets / padded_sizes: correct block_m-aligned padding layout
  - fwd_idx consistency: perm_h_pad[fwd_idx[flat_idx]] == hidden_states[flat_idx // K]
"""


import math
from typing import Optional, Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoePermutePaddedOp
from workloads.ops.moe_permute import MoePermuteTest as _MoePermuteTestWorkload


def _ref_moe_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int = 64,
    perm_h_pad_buf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for moe_permute."""
    T, H = hidden_states.shape
    K = topk_ids.shape[1]
    numel = T * K
    flat_ids = topk_ids.flatten().cpu().tolist()
    dev = hidden_states.device

    counts = [0] * num_experts
    for eid in flat_ids:
        counts[eid] += 1

    offsets = [0] * (num_experts + 1)
    for e in range(num_experts):
        offsets[e + 1] = offsets[e] + counts[e]

    padded_offsets_list = []
    padded_sizes_list = []
    padded_start = 0
    for e in range(num_experts):
        padded_offsets_list.append(padded_start)
        ps = math.ceil(counts[e] / block_m) * block_m if counts[e] > 0 else 0
        padded_sizes_list.append(ps)
        padded_start += ps
    padded_batch_sum = max(padded_start, 1)

    write_ptr = list(offsets[:-1])
    slot_to_padded_list = [0] * numel
    slot_to_row = [0] * numel
    fwd_idx_list = [0] * numel

    for flat_idx, eid in enumerate(flat_ids):
        slot = write_ptr[eid]
        within_expert = slot - offsets[eid]
        padded_slot = padded_offsets_list[eid] + within_expert
        slot_to_padded_list[slot] = padded_slot
        slot_to_row[slot] = flat_idx // K
        fwd_idx_list[flat_idx] = padded_slot
        write_ptr[eid] += 1

    if perm_h_pad_buf is not None and perm_h_pad_buf.shape[0] >= padded_batch_sum:
        perm_h_pad = perm_h_pad_buf[:padded_batch_sum]
        perm_h_pad.zero_()
    else:
        perm_h_pad = torch.zeros(padded_batch_sum, H, dtype=hidden_states.dtype, device=dev)
    for slot in range(numel):
        perm_h_pad[slot_to_padded_list[slot]] = hidden_states[slot_to_row[slot]]

    expert_first_token_offset = torch.tensor(offsets, dtype=torch.int64, device=dev)
    padded_offsets_t = torch.tensor(padded_offsets_list, dtype=torch.int32, device=dev)
    padded_sizes_t = torch.tensor(padded_sizes_list, dtype=torch.int32, device=dev)
    fwd_idx_t = torch.tensor(fwd_idx_list, dtype=torch.int32, device=dev)

    return perm_h_pad, padded_offsets_t, padded_sizes_t, expert_first_token_offset, fwd_idx_t


class MoePermuteTest(_MoePermuteTestWorkload, TestBase):
    def ref_program(self, hidden_states, topk_ids):
        return _ref_moe_permute(hidden_states, topk_ids, self.num_experts)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


class MoePermuteFixture(FixtureBase):
    PARAMS = [
        ("total_tokens, top_k, num_experts, hidden_size, dtype", [
            pytest.param(4,    2,   4,   64,  torch.bfloat16, marks=pytest.mark.smoke, id="tiny-bf16"),
            pytest.param(4,    2,   4,   64,  torch.float16,  marks=pytest.mark.smoke, id="tiny-fp16"),
            pytest.param(16,   2,   8,   128, torch.bfloat16, marks=pytest.mark.full,  id="small"),
            pytest.param(128,  4,   8,   256, torch.bfloat16, marks=pytest.mark.full,  id="medium"),
            pytest.param(1024, 8,   128, 128, torch.bfloat16, marks=pytest.mark.full,  id="qwen3-scale"),
            pytest.param(1,    2,   4,   64,  torch.bfloat16, marks=pytest.mark.full,  id="single-token"),
            pytest.param(8,    1,   4,   64,  torch.bfloat16, marks=pytest.mark.full,  id="top-k-1"),
            # skewed: all tokens to expert 0
            pytest.param(32,   4,   8,   64,  torch.bfloat16, marks=pytest.mark.full,  id="skewed"),
        ]),
    ]


# ---------------------------------------------------------------------------
# TestBase subclass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


def _compare(hidden_states, topk_ids, outputs, outputs_ref, num_experts):
    perm_h_pad, padded_offsets, padded_sizes, offsets, fwd_idx = outputs
    _, ref_p_offsets, ref_p_sizes, ref_offsets, _ = outputs_ref

    K = topk_ids.shape[1]
    numel = topk_ids.numel()

    # expert_first_token_offset must match exactly
    assert torch.equal(offsets.cpu(), ref_offsets.cpu()), (
        f"expert_first_token_offset mismatch:\n  got: {offsets.cpu()}\n  ref: {ref_offsets.cpu()}"
    )

    # padded_sizes and padded_offsets must match exactly (deterministic arithmetic)
    assert torch.equal(padded_sizes.cpu(), ref_p_sizes.cpu()), (
        f"padded_sizes mismatch:\n  got: {padded_sizes.cpu()}\n  ref: {ref_p_sizes.cpu()}"
    )
    assert torch.equal(padded_offsets.cpu(), ref_p_offsets.cpu()), (
        f"padded_offsets mismatch:\n  got: {padded_offsets.cpu()}\n  ref: {ref_p_offsets.cpu()}"
    )

    # Key consistency check: perm_h_pad[fwd_idx[flat_idx]] == hidden_states[flat_idx // K]
    # This validates both perm_h_pad layout and fwd_idx regardless of intra-expert ordering.
    token_rows = torch.arange(numel, device=hidden_states.device) // K  # [T*K]
    gathered = perm_h_pad[fwd_idx.long()]                                # [T*K, H]
    assert torch.equal(gathered, hidden_states[token_rows]), (
        "fwd_idx/perm_h_pad mismatch: perm_h_pad[fwd_idx] != hidden_states[flat_idx // K]"
    )

    # Padding rows within each expert's block must be zero
    for e in range(num_experts):
        p_start = padded_offsets[e].item()
        count = offsets[e + 1].item() - offsets[e].item()
        p_size = padded_sizes[e].item()
        pad_start = p_start + count
        pad_end = p_start + p_size
        if pad_start < pad_end:
            assert torch.all(perm_h_pad[pad_start:pad_end] == 0), (
                f"Expert {e} padding rows are not zero"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@MoePermuteFixture
def test_moe_permute_op(total_tokens, top_k, num_experts, hidden_size, dtype):
    test = MoePermuteTest(total_tokens, top_k, num_experts, hidden_size, dtype)
    op = MoePermutePaddedOp(total_tokens, top_k, num_experts, hidden_size, dtype)
    hidden_states, topk_ids = test.gen_inputs()

    outputs = op(hidden_states, topk_ids)
    outputs_ref = test.ref_program(hidden_states, topk_ids)

    _compare(hidden_states, topk_ids, outputs, outputs_ref, num_experts)
    print(f"PASS [{total_tokens}tok, top{top_k}, E={num_experts}, H={hidden_size}, {dtype}]")


@pytest.mark.smoke
def test_moe_permute_skewed():
    """All tokens routed to expert 0."""
    T, K, E, H = 32, 4, 8, 64
    hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros((T, K), dtype=torch.int32, device="cuda")

    op = MoePermutePaddedOp(T, K, E, H, torch.bfloat16)
    outputs = op(hidden_states, topk_ids)
    outputs_ref = _ref_moe_permute(hidden_states, topk_ids, E)

    _compare(hidden_states, topk_ids, outputs, outputs_ref, E)
    print("PASS skewed (all tokens → expert 0)")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
