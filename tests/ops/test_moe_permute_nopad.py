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
    op = MoePermuteNopadFwdOp(num_experts=num_experts, num_experts_local=num_experts)
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
        num_experts=4,
        num_experts_local=4,
        total_tokens=5,
        top_k=2,
        hidden_size=16,
    )
    with pytest.raises(ValueError, match="Expected total_tokens"):
        op(hidden_states, topk_ids)


@pytest.mark.smoke
def test_moe_permute_nopad_without_a_map_builds_the_map_free_scan() -> None:
    """No map means the map-free scan, whatever num_experts_local says.

    The map-reading scan loads ``expert_map[global_eid]`` twice per token-expert
    pair. Selecting it for a call that passed no map would pay that for nothing.
    """
    op = MoePermuteNopadFwdOp(num_experts=4, num_experts_local=4)
    hidden_states = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, 4, (8, 2), dtype=torch.int32, device="cuda")

    op(hidden_states, topk_ids)

    (kernel,) = op.iter_kernels()
    assert not kernel.expert_parallel


@pytest.mark.smoke
def test_moe_permute_nopad_partial_local_without_a_map_raises() -> None:
    """Owning a slice of the table but naming no map is unsatisfiable."""
    op = MoePermuteNopadFwdOp(num_experts=4, num_experts_local=2)
    hidden_states = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, 4, (8, 2), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="needs an expert_map"):
        op(hidden_states, topk_ids)


@pytest.mark.smoke
def test_moe_permute_nopad_ep_counts_only_local_experts() -> None:
    """With a map, the op lays out num_experts_local slots and drops the rest.

    Every token routed to a local expert reaches a slot; a token routed elsewhere
    gets fwd_idx == -1.
    """
    E, E_local, T, K, H = 4, 2, 8, 2, 64
    expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
    expert_map[:E_local] = torch.arange(E_local, dtype=torch.int32, device="cuda")
    op = MoePermuteNopadFwdOp(num_experts=E, num_experts_local=E_local)
    hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")

    _, _, true_sizes, _, fwd_idx = op(hidden_states, topk_ids, expert_map)

    assert true_sizes.shape[0] == E_local
    local_pairs = int((topk_ids.flatten() < E_local).sum())
    assert int(true_sizes.sum()) == local_pairs
    assert int((fwd_idx >= 0).sum()) == local_pairs


@pytest.mark.smoke
def test_moe_permute_nopad_ep_rejects_a_non_dense_map() -> None:
    """A map whose local ids are not 0..E_local-1 must raise, not miscount.

    ``[-1, 0, 2, -1]`` marks two experts local while naming id 2, which the
    kernel's two counters cannot hold: the tokens routed there used to be
    dropped silently.
    """
    E, E_local, T, K, H = 4, 2, 8, 2, 64
    expert_map = torch.tensor([-1, 0, 2, -1], dtype=torch.int32, device="cuda")
    op = MoePermuteNopadFwdOp(num_experts=E, num_experts_local=E_local)
    hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="exactly once each"):
        op(hidden_states, topk_ids, expert_map)


@pytest.mark.smoke
def test_moe_permute_nopad_ep_rechecks_an_edited_map() -> None:
    """Editing a valid map into an invalid one must be caught on the next call."""
    E, E_local, T, K, H = 4, 2, 8, 2, 64
    expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
    expert_map[:E_local] = torch.arange(E_local, dtype=torch.int32, device="cuda")
    op = MoePermuteNopadFwdOp(num_experts=E, num_experts_local=E_local)
    hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")

    op(hidden_states, topk_ids, expert_map)

    expert_map[2] = 2
    with pytest.raises(ValueError, match="exactly once each"):
        op(hidden_states, topk_ids, expert_map)


@pytest.mark.smoke
def test_moe_permute_nopad_cpu_input_raises() -> None:
    hidden_states = torch.randn(4, 16, dtype=torch.float16)
    topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32)
    op = MoePermuteNopadFwdOp(num_experts=4, num_experts_local=4)
    with pytest.raises(ValueError, match="hidden_states must be a CUDA tensor"):
        op(hidden_states, topk_ids)


@pytest.mark.smoke
def test_moe_permute_nopad_invalid_dtype_raises() -> None:
    hidden_states = torch.randn(4, 16, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32, device="cuda")
    op = MoePermuteNopadFwdOp(num_experts=4, num_experts_local=4)
    with pytest.raises(ValueError, match="Expected hidden_states.dtype"):
        op(hidden_states, topk_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
