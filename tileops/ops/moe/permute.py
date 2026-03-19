"""MoE permute op (cutlass path): counting sort + gather tokens by expert."""

from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.moe import MoePermuteKernel

from ..op import Op

__all__ = ["MoePermuteOp"]


class MoePermuteOp(Op):
    """Route tokens to expert-contiguous layout for MoE cutlass grouped GEMM.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        num_experts: Total number of experts E.
        hidden_size: Hidden dimension H.
        dtype: Data type of hidden_states (bf16 or fp16).
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = MoePermuteOp(num_tokens=4, top_k=2, num_experts=8, hidden_size=128)
        >>> perm_h, offsets, inv_idx = op(hidden_states, topk_ids)
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["permute_kernel"](
            num_tokens, top_k, num_experts, hidden_size, dtype
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"permute_kernel": MoePermuteKernel}

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute.

        Args:
            hidden_states: [T, H] input activations (bf16/fp16).
            topk_ids: [T, K] int32 expert assignments.

        Returns:
            permuted_hidden_states:    [T*K, H]
            expert_first_token_offset: [E+1] int64
            inv_permuted_idx:          [T*K] int32
        """
        return self.kernel(hidden_states, topk_ids)
