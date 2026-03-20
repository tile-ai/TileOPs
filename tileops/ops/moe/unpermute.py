"""MoE unpermute op (cutlass path): scatter-add expert outputs back to token order."""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.moe import MoeUnpermuteKernel

from ..op import Op

__all__ = ["MoeUnpermuteOp"]


class MoeUnpermuteOp(Op):
    """Scatter expert outputs back to original token order with weighted reduction.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        dtype: Data type of mm2_out and output (bf16 or fp16).
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = MoeUnpermuteOp(num_tokens=4, top_k=2, hidden_size=128)
        >>> output = op(mm2_out, inv_permuted_idx, topk_weights)
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["unpermute_kernel"](
            num_tokens, top_k, hidden_size, dtype
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"unpermute_kernel": MoeUnpermuteKernel}

    def forward(
        self,
        mm2_out: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            mm2_out: [T*K, H] bf16/fp16 down-proj output.
            inv_permuted_idx: [T*K] int32 inverse mapping from moe_permute.
            topk_weights: [T, K] float32 routing weights.

        Returns:
            output: [T, H] bf16/fp16
        """
        return self.kernel(mm2_out, inv_permuted_idx, topk_weights)
