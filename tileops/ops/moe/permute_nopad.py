"""MoE permute op (no-pad variant): counting sort + tight gather without block_m padding."""

from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.permute_nopad import MoePermuteNopadKernel

from ..op import Op

__all__ = ["MoePermuteNopadFwdOp"]


class MoePermuteNopadFwdOp(Op):
    """Route tokens to tight (non-padded) expert-contiguous layout.

    Unlike MoePermutePaddedFwdOp, the output perm_h has exactly T*K rows with no
    inter-expert padding, enabling smaller intermediate tensors throughout
    the MoE pipeline.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        num_experts: Total number of experts E (global count).
        hidden_size: Hidden dimension H.
        dtype: Data type of hidden_states (bf16 or fp16).
        expert_map: Optional [E_global] int32 tensor mapping global expert ids
            to local ids (-1 = not on this rank).  When provided, only local
            token-expert pairs are counted; non-local positions get fwd_idx = -1.
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = MoePermuteNopadFwdOp(num_tokens=4, top_k=2, num_experts=8, hidden_size=128)
        >>> perm_h, offsets, sizes, expert_offset, fwd_idx = op(hidden_states, topk_ids)
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["permute_nopad_kernel"](
            num_tokens, top_k, num_experts, hidden_size, dtype, expert_map
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"permute_nopad_kernel": MoePermuteNopadKernel}

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute without padding.

        Args:
            hidden_states: [T, H] input activations (bf16/fp16).
            topk_ids: [T, K] int32 expert assignments (global ids).

        Returns:
            perm_h:                    [T*K, H] tight hidden states
            true_offsets:              [E_local] int32 tight start per local expert
            true_sizes:                [E_local] int32 true token count per local expert
            expert_first_token_offset: [E_local+1] int64 non-padded prefix-sum
            fwd_idx:                   [T*K] int32 forward mapping: flat_idx → tight slot
                                       (-1 for non-local pairs when expert_map is set)
        """
        return self.kernel(hidden_states, topk_ids)
