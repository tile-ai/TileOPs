"""MoE unpermute op (cutlass path): scatter-add padded expert outputs back to token order."""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import MoeUnpermuteKernel

from ..op import Op

__all__ = ["MoeUnpermuteFwdOp"]


class MoeUnpermuteFwdOp(Op):
    """Scatter padded expert outputs back to original token order with weighted reduction.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        dtype: Data type of mm2_pad and output (bf16 or fp16).
        padded_batch_sum: Size of the padded mm2_pad buffer (first dim of mm2_pad).
            Must be >= T*K. When used with MoePermuteOp, pass the padded_batch_sum
            value returned by the kernel (T*K + E*block_m upper bound).
            Defaults to num_tokens * top_k for standalone testing only — do NOT
            use the default when mm2_pad comes from MoePermuteOp, as the padded
            buffer will be larger and the kernel will index out of bounds.
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = MoeUnpermuteFwdOp(num_tokens=4, top_k=2, hidden_size=128, padded_batch_sum=512)
        >>> output = op(mm2_pad, fwd_idx, topk_weights)
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        padded_batch_sum: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.padded_batch_sum = padded_batch_sum if padded_batch_sum is not None else num_tokens * top_k

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["unpermute_kernel"](
            num_tokens, top_k, hidden_size, self.padded_batch_sum, dtype
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"unpermute_kernel": MoeUnpermuteKernel}

    def forward(
        self,
        mm2_pad: torch.Tensor,
        fwd_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            mm2_pad: [padded_batch_sum, H] bf16/fp16 down-proj output (padded layout).
            fwd_idx: [T*K] int32 forward mapping: flat_idx → padded slot.
            topk_weights: [T, K] float32 routing weights.

        Returns:
            output: [T, H] bf16/fp16
        """
        return self.kernel(mm2_pad, fwd_idx, topk_weights)
