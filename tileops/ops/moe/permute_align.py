"""MoE permute-align op: routes tokens to experts and pads to tile boundary."""

from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.moe import MoePermuteAlignKernel

from ..op import Op

__all__ = ["MoePermuteAlignOp"]


class MoePermuteAlignOp(Op):
    """Route tokens to experts and pad each expert's token count to block_size.

    Takes ``topk_ids`` and produces the three index arrays required by MoE
    grouped GEMM: sorted token indices, per-block expert ids, and the total
    padded token count.

    Args:
        numel: Total (token, expert) assignments = total_tokens * top_k.
        num_experts: Number of experts.
        block_size: GEMM tile size (M dimension).
        kernel_map: Optional kernel override dict.
        tune: Whether to autotune the kernel.

    Example:
        >>> op = MoePermuteAlignOp(numel=32, num_experts=8, block_size=16)
        >>> sorted_ids, expert_ids, num_post_pad = op(topk_ids)
    """

    def __init__(
        self,
        numel: int,
        num_experts: int,
        block_size: int,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.numel = numel
        self.num_experts = num_experts
        self.block_size = block_size

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["permute_align_kernel"](
            numel, num_experts, block_size
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"permute_align_kernel": MoePermuteAlignKernel}

    def forward(
        self, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run permute-align.

        Args:
            topk_ids: [total_tokens, top_k] int32 expert indices (0-indexed).

        Returns:
            sorted_token_ids: [max_num_tokens_padded] int32
            expert_ids:       [num_blocks] int32
            num_tokens_post_pad: [1] int32
        """
        return self.kernel(topk_ids)
