import math
from typing import Optional, Tuple

import torch

from workloads.base import WorkloadBase


def _ref_moe_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int = 64,
    perm_h_pad_buf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for moe_permute.

    Args:
        hidden_states: [T, H]
        topk_ids: [T, K] int32
        num_experts: E
        block_m: alignment for padded sizes

    Returns:
        perm_h_pad:                [padded_batch_sum, H]
        padded_offsets:            [E] int32
        padded_sizes:              [E] int32
        expert_first_token_offset: [E+1] int64
        fwd_idx:                   [T*K] int32
    """
    T, H = hidden_states.shape
    K = topk_ids.shape[1]
    numel = T * K
    flat_ids = topk_ids.flatten().cpu().tolist()
    dev = hidden_states.device

    # Count tokens per expert
    counts = [0] * num_experts
    for eid in flat_ids:
        counts[eid] += 1

    # Exclusive prefix-sum → expert_first_token_offset
    offsets = [0] * (num_experts + 1)
    for e in range(num_experts):
        offsets[e + 1] = offsets[e] + counts[e]

    # Padded layout
    padded_offsets_list = []
    padded_sizes_list = []
    padded_start = 0
    for e in range(num_experts):
        padded_offsets_list.append(padded_start)
        ps = math.ceil(counts[e] / block_m) * block_m if counts[e] > 0 else 0
        padded_sizes_list.append(ps)
        padded_start += ps
    padded_batch_sum = max(padded_start, 1)

    # Scatter: assign each flat_idx to its expert slot
    write_ptr = list(offsets[:-1])
    slot_to_padded_list = [0] * numel
    slot_to_row = [0] * numel      # non-padded slot → token row
    fwd_idx_list = [0] * numel

    for flat_idx, eid in enumerate(flat_ids):
        slot = write_ptr[eid]
        within_expert = slot - offsets[eid]
        padded_slot = padded_offsets_list[eid] + within_expert
        slot_to_padded_list[slot] = padded_slot
        slot_to_row[slot] = flat_idx // K
        fwd_idx_list[flat_idx] = padded_slot
        write_ptr[eid] += 1

    # Build perm_h_pad (zeros, write at padded positions)
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


class MoePermuteTest(WorkloadBase):

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
