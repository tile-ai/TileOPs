"""Op-level tests for MoePermuteOp (cutlass path).

Verifies:
  - permuted_hidden_states: correct gather of hidden_states rows
  - expert_first_token_offset: correct exclusive prefix-sum (int64)
  - inv_permuted_idx: correct inverse mapping (permuted pos → original flat pos)
"""

from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoePermuteOp

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _ref_moe_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for moe_permute.

    Args:
        hidden_states: [T, H]
        topk_ids: [T, K] int32
        num_experts: E

    Returns:
        permuted_hidden_states:    [T*K, H]
        expert_first_token_offset: [E+1] int64
        inv_permuted_idx:          [T*K] int32
    """
    T, H = hidden_states.shape
    K = topk_ids.shape[1]
    numel = T * K
    flat_ids = topk_ids.flatten().cpu().tolist()

    # Count tokens per expert
    counts = [0] * num_experts
    for eid in flat_ids:
        counts[eid] += 1

    # Exclusive prefix-sum → expert_first_token_offset
    offsets = [0] * (num_experts + 1)
    for e in range(num_experts):
        offsets[e + 1] = offsets[e] + counts[e]

    # Scatter: assign each flat index to its expert slot
    write_ptr = list(offsets[:-1])
    permuted_idx = [0] * numel      # permuted pos → flat index
    inv_permuted_idx = [0] * numel  # same in reference (slot → flat_idx)
    for flat_idx, eid in enumerate(flat_ids):
        slot = write_ptr[eid]
        permuted_idx[slot] = flat_idx
        inv_permuted_idx[slot] = flat_idx
        write_ptr[eid] += 1

    # Gather: hidden_states row for flat_idx is flat_idx // K
    dev = hidden_states.device
    perm_idx_t = torch.tensor(permuted_idx, dtype=torch.int64, device=dev)
    src_rows = perm_idx_t // K
    permuted_hidden = hidden_states[src_rows]  # [T*K, H]

    expert_first_token_offset = torch.tensor(offsets, dtype=torch.int64, device=dev)
    inv_perm = torch.tensor(inv_permuted_idx, dtype=torch.int32, device=dev)

    return permuted_hidden, expert_first_token_offset, inv_perm


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


class MoePermuteTest(TestBase):

    def __init__(self, total_tokens, top_k, num_experts, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self):
        hidden_states = torch.randn(
            self.total_tokens, self.hidden_size, dtype=self.dtype, device="cuda"
        )
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32, device="cuda",
        )
        return hidden_states, topk_ids

    def ref_program(self, hidden_states, topk_ids):
        return _ref_moe_permute(hidden_states, topk_ids, self.num_experts)


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


def _compare(outputs, outputs_ref, num_experts):
    perm_h, offsets, inv_idx = outputs
    ref_perm_h, ref_offsets, ref_inv_idx = outputs_ref

    # expert_first_token_offset must match exactly
    assert torch.equal(offsets.cpu(), ref_offsets.cpu()), (
        f"expert_first_token_offset mismatch:\n  got: {offsets.cpu()}\n  ref: {ref_offsets.cpu()}"
    )

    # For each expert slot range, the set of gathered rows must match
    # (intra-expert order may differ due to atomic scatter)
    for e in range(num_experts):
        start = ref_offsets[e].item()
        end = ref_offsets[e + 1].item()
        if start == end:
            continue
        got_rows = torch.sort(perm_h[start:end].cpu().float(), dim=0).values
        ref_rows = torch.sort(ref_perm_h[start:end].cpu().float(), dim=0).values
        assert torch.allclose(got_rows, ref_rows, atol=0), (
            f"Expert {e} permuted_hidden_states mismatch"
        )

    # inv_permuted_idx: per-expert slot sets must match (order-insensitive)
    for e in range(num_experts):
        start = ref_offsets[e].item()
        end = ref_offsets[e + 1].item()
        if start == end:
            continue
        got_set = sorted(inv_idx[start:end].cpu().tolist())
        ref_set = sorted(ref_inv_idx[start:end].cpu().tolist())
        assert got_set == ref_set, (
            f"Expert {e} inv_permuted_idx mismatch:\n  got: {got_set}\n  ref: {ref_set}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@MoePermuteFixture
def test_moe_permute_op(total_tokens, top_k, num_experts, hidden_size, dtype):
    test = MoePermuteTest(total_tokens, top_k, num_experts, hidden_size, dtype)
    op = MoePermuteOp(total_tokens, top_k, num_experts, hidden_size, dtype)
    hidden_states, topk_ids = test.gen_inputs()

    outputs = op(hidden_states, topk_ids)
    outputs_ref = test.ref_program(hidden_states, topk_ids)

    _compare(outputs, outputs_ref, num_experts)
    print(f"PASS [{total_tokens}tok, top{top_k}, E={num_experts}, H={hidden_size}, {dtype}]")


@pytest.mark.smoke
def test_moe_permute_skewed():
    """All tokens routed to expert 0."""
    T, K, E, H = 32, 4, 8, 64
    hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros((T, K), dtype=torch.int32, device="cuda")

    op = MoePermuteOp(T, K, E, H, torch.bfloat16)
    outputs = op(hidden_states, topk_ids)
    outputs_ref = _ref_moe_permute(hidden_states, topk_ids, E)

    _compare(outputs, outputs_ref, E)
    print("PASS skewed (all tokens → expert 0)")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
