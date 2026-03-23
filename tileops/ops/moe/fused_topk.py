"""MoE fused top-k routing operator."""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.moe.fused_topk import FusedTopKKernel

from ..op import Op

__all__ = ["FusedTopKOp"]


class FusedTopKOp(Op):
    """MoE top-k routing operator.

    Applies scoring (softmax or sigmoid) to router logits and selects the
    top-k experts per token.

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Number of experts E.
        top_k: Number of experts to select per token K.
        scoring_func: "softmax" (Qwen3/Qwen2) or "sigmoid" (DeepSeek-V3/GLM-4).
        renormalize: If True, normalize top-k weights to sum to 1.
        kernel_map: Optional kernel map override.
        config: Optional kernel config dict.

    Example:
        >>> op = FusedTopKOp(num_tokens=512, num_experts=128, top_k=8)
        >>> topk_weights, topk_ids = op(gating_output)
        >>> # topk_weights: [512, 8] float32
        >>> # topk_ids:     [512, 8] int32
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        config: Optional[dict] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["fused_topk_kernel"](
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            config=config,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_topk_kernel": FusedTopKKernel}

    def forward(
        self,
        gating_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run top-k routing.

        Args:
            gating_output: [T, E] router logits (bf16, fp16, or float32).

        Returns:
            topk_weights: [T, K] float32.
            topk_ids:     [T, K] int32.
        """
        return self.kernel(gating_output)
