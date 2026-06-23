"""Op-level tests for MoePermuteNopadFwdOp (tight, non-padded permute).

Verifies:
  - perm_h: correct gather of hidden_states rows into tight expert layout
  - true_offsets / true_sizes: tight per-expert start and count (int32)
  - expert_first_token_offset: exclusive prefix-sum (int64)
  - fwd_idx consistency: perm_h[fwd_idx[flat_idx]] == hidden_states[flat_idx // K]
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoePermuteNopadFwdOp
from workloads.moe import MoePermuteTest as _MoePermuteTestWorkload


def _ref_moe_permute_nopad(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for moe_permute (tight, no padding)."""
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

    write_ptr = list(offsets[:-1])
    slot_to_row = [0] * numel
    fwd_idx_list = [0] * numel

    for flat_idx, eid in enumerate(flat_ids):
        slot = write_ptr[eid]
        slot_to_row[slot] = flat_idx // K
        fwd_idx_list[flat_idx] = slot
        write_ptr[eid] += 1

    perm_h = torch.empty(numel, H, dtype=hidden_states.dtype, device=dev)
    for slot in range(numel):
        perm_h[slot] = hidden_states[slot_to_row[slot]]

    true_offsets_t = torch.tensor(offsets[:-1], dtype=torch.int32, device=dev)
    true_sizes_t = torch.tensor(counts, dtype=torch.int32, device=dev)
    expert_first_token_offset = torch.tensor(offsets, dtype=torch.int64, device=dev)
    fwd_idx_t = torch.tensor(fwd_idx_list, dtype=torch.int32, device=dev)

    return perm_h, true_offsets_t, true_sizes_t, expert_first_token_offset, fwd_idx_t


class MoePermuteNopadTest(_MoePermuteTestWorkload, TestBase):
    def ref_program(self, hidden_states, topk_ids):
        return _ref_moe_permute_nopad(hidden_states, topk_ids, self.num_experts)


def _compare(hidden_states, topk_ids, outputs, outputs_ref, num_experts):
    perm_h, true_offsets, true_sizes, offsets, fwd_idx = outputs
    _, ref_true_offsets, ref_true_sizes, ref_offsets, _ = outputs_ref

    K = topk_ids.shape[1]
    numel = topk_ids.numel()

    # expert_first_token_offset (int64) must match exactly
    assert torch.equal(offsets.cpu(), ref_offsets.cpu()), (
        f"expert_first_token_offset mismatch:\n  got: {offsets.cpu()}\n  ref: {ref_offsets.cpu()}"
    )

    # true_offsets / true_sizes (int32) must match exactly
    assert torch.equal(true_offsets.cpu(), ref_true_offsets.cpu()), (
        f"true_offsets mismatch:\n  got: {true_offsets.cpu()}\n  ref: {ref_true_offsets.cpu()}"
    )
    assert torch.equal(true_sizes.cpu(), ref_true_sizes.cpu()), (
        f"true_sizes mismatch:\n  got: {true_sizes.cpu()}\n  ref: {ref_true_sizes.cpu()}"
    )

    # Tight output: exactly numel rows, no padding gaps.
    assert perm_h.shape[0] == numel, (
        f"perm_h must have exactly {numel} rows (tight layout), got {perm_h.shape[0]}"
    )

    # Key consistency: perm_h[fwd_idx[flat_idx]] == hidden_states[flat_idx // K].
    # This validates both perm_h layout and fwd_idx regardless of intra-expert ordering.
    token_rows = torch.arange(numel, device=hidden_states.device) // K
    gathered = perm_h[fwd_idx.long()]
    assert torch.equal(gathered, hidden_states[token_rows]), (
        "fwd_idx/perm_h mismatch: perm_h[fwd_idx] != hidden_states[flat_idx // K]"
    )


class MoePermuteNopadFixture(FixtureBase):
    PARAMS = [
        ("total_tokens, top_k, num_experts, hidden_size, dtype", [
            pytest.param(4,  2, 4, 64,  torch.bfloat16, marks=pytest.mark.smoke, id="tiny-bf16"),
            pytest.param(4,  2, 4, 64,  torch.float16,  marks=pytest.mark.smoke, id="tiny-fp16"),
            pytest.param(16, 2, 8, 128, torch.bfloat16, marks=pytest.mark.full,  id="small"),
            # numel = total_tokens * top_k = 10 is NOT a multiple of the gather
            # kernel's ROWS_PER_BLOCK (8), so the last block's `if slot < numel`
            # out-of-bounds guard is actually exercised — every other shape here
            # has numel divisible by 8 and never hits the partial tail.
            pytest.param(5,  2, 4, 64,  torch.bfloat16, marks=pytest.mark.full,  id="partial-tail-numel10"),
        ]),
    ]


@MoePermuteNopadFixture
def test_moe_permute_nopad_op(total_tokens, top_k, num_experts, hidden_size, dtype):
    test = MoePermuteNopadTest(total_tokens, top_k, num_experts, hidden_size, dtype)
    op = MoePermuteNopadFwdOp(total_tokens, top_k, num_experts, hidden_size, dtype)
    hidden_states, topk_ids = test.gen_inputs()

    outputs = op(hidden_states, topk_ids)
    outputs_ref = test.ref_program(hidden_states, topk_ids)

    _compare(hidden_states, topk_ids, outputs, outputs_ref, num_experts)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
