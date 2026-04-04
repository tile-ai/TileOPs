import math
from typing import Tuple

import torch

from workloads.base import WorkloadBase


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
    # Sentinel value is numel (matches kernel: sorted_token_ids[pad_slot] = numel)
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


class MoePermuteAlignTest(WorkloadBase):

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
