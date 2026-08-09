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
from workloads.moe import MoePermuteWorkload


class MoePermuteNopadTest(MoePermuteWorkload, TestBase):
    pass


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
    op = MoePermuteNopadFwdOp(num_experts=num_experts)
    hidden_states, topk_ids = test.gen_inputs()

    outputs = op(hidden_states, topk_ids)
    outputs_ref = test.ref_program(hidden_states, topk_ids)

    _compare(hidden_states, topk_ids, outputs, outputs_ref, num_experts)
    flops, nbytes = op.eval_roofline()
    assert flops == 0
    assert nbytes > 0


@pytest.mark.smoke
def test_moe_permute_nopad_explicit_shape_mismatch_raises() -> None:
    hidden_states = torch.randn(4, 16, dtype=torch.float16, device="cuda")
    topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32, device="cuda")
    op = MoePermuteNopadFwdOp(
        total_tokens=5,
        top_k=2,
        num_experts=4,
        hidden_size=16,
    )
    with pytest.raises(ValueError, match="Expected total_tokens"):
        op(hidden_states, topk_ids)


@pytest.mark.smoke
def test_moe_permute_nopad_cpu_input_raises() -> None:
    hidden_states = torch.randn(4, 16, dtype=torch.float16)
    topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32)
    op = MoePermuteNopadFwdOp(num_experts=4)
    with pytest.raises(ValueError, match="hidden_states must be a CUDA tensor"):
        op(hidden_states, topk_ids)


@pytest.mark.smoke
def test_moe_permute_nopad_invalid_dtype_raises() -> None:
    hidden_states = torch.randn(4, 16, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32, device="cuda")
    op = MoePermuteNopadFwdOp(num_experts=4)
    with pytest.raises(ValueError, match="Expected hidden_states.dtype"):
        op(hidden_states, topk_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
