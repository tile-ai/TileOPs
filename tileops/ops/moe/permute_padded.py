"""MoE permute op (cutlass path): counting sort + padded gather tokens by expert."""

from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import MoePermutePaddedKernel

from ..op_base import Op

__all__ = ["MoePermutePaddedFwdOp"]


class MoePermutePaddedFwdOp(Op):
    """Route tokens to block_m-aligned padded expert layout for NT grouped-GEMM.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        num_experts: Total number of experts E.
        hidden_size: Hidden dimension H.
        dtype: Data type of hidden_states (bf16 or fp16).
        block_m: Block size for expert row-start alignment (default: 64).
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = MoePermutePaddedFwdOp(num_tokens=4, top_k=2, num_experts=8, hidden_size=128)
        >>> perm_h_pad, p_offsets, p_sizes, offsets, fwd_idx = op(hidden_states, topk_ids)
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        block_m: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.block_m = block_m

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["permute_kernel"](
            num_tokens, top_k, num_experts, hidden_size, dtype, block_m
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"permute_kernel": MoePermutePaddedKernel}

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = self.dtype.itemsize
        flops = 0
        mem_bytes = (
            (self.num_tokens * self.hidden_size
             + self.num_tokens * self.top_k * self.hidden_size)
            * elem_bytes
            + (self.num_experts + 1) * 8
            + self.num_tokens * self.top_k * 4
            + self.num_experts * 8
        )
        return flops, mem_bytes

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute with padded output.

        Args:
            hidden_states: [T, H] input activations (bf16/fp16).
            topk_ids: [T, K] int32 expert assignments.

        Returns:
            perm_h_pad:                [padded_batch_sum, H]
            padded_offsets:            [E] int32 padded start per expert
            padded_sizes:              [E] int32 block_m-aligned sizes per expert
            expert_first_token_offset: [E+1] int64 non-padded prefix-sum
            fwd_idx:                   [T*K] int32 forward mapping: flat_idx → padded slot
        """
        return self.kernel(hidden_states, topk_ids)
