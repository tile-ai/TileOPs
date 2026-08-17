"""MoE fused top-k routing operator."""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.fused_topk import FusedTopKKernel

from ..op_base import Op

__all__ = ["FusedTopKOp"]


class FusedTopKOp(Op):
    """MoE top-k routing operator.

    Applies scoring (softmax or sigmoid) to router logits and selects the
    top-k experts per token.

    Args:
        num_tokens: Optional committed number of input tokens T. Preferred
            API infers it from ``gating_output.shape[0]``.
        num_experts: Optional committed number of experts E. Preferred API
            infers it from ``gating_output.shape[1]``.
        top_k: Number of experts to select per token K.
        scoring_func: "softmax" (Qwen3/Qwen2) or "sigmoid" (DeepSeek-V3/GLM-4/Kimi K2).
        renormalize: If True, normalize top-k weights to sum to 1.
        scoring_func: see above. Passing ``correction_bias`` to ``forward``
            requires ``"sigmoid"``: the bias is added to sigmoid scores for
            selection only, and the output weights stay the original scores.
        kernel_map: Optional kernel map override.
        config: Optional kernel config dict.

    Example:
        >>> op = FusedTopKOp(top_k=8)
        >>> topk_weights, topk_ids = op(gating_output)
        >>> # topk_weights: [512, 8] float32
        >>> # topk_ids:     [512, 8] int32
    """

    def __init__(
        self,
        num_tokens: Optional[int] = None,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        config: Optional[dict] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self._committed_num_tokens = num_tokens
        self._committed_num_experts = num_experts
        self.scoring_func = scoring_func
        self.renormalize = renormalize

        self.dispatch_kernel(kernel_map)
        self.config = config

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_topk_kernel": FusedTopKKernel}

    def _get_kernel(
        self, num_tokens: int, num_experts: int, top_k: int,
        device_index: int | None, with_correction_bias: bool,
    ) -> Kernel:
        key = (num_tokens, num_experts, top_k, device_index, with_correction_bias)
        return self.get_or_build_kernel(
            "fused_topk_kernel",
            key=key,
            build=lambda: self.kernel_map["fused_topk_kernel"](
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                scoring_func=self.scoring_func,
                renormalize=self.renormalize,
                with_correction_bias=with_correction_bias,
                config=self.config,
            ),
        )

    def forward(
        self,
        gating_output: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run top-k routing.

        Args:
            gating_output: [T, E] router logits (bf16, fp16, or float32).
            correction_bias: [E] float32 per-expert bias, or None. Passing it
                requires scoring_func="sigmoid".

        Returns:
            topk_weights: [T, K] float32.
            topk_ids:     [T, K] int32.
        """
        if not gating_output.is_cuda:
            raise ValueError("gating_output must be a CUDA tensor")
        if correction_bias is not None and not correction_bias.is_cuda:
            raise ValueError("correction_bias must be a CUDA tensor")
        if correction_bias is not None and gating_output.device != correction_bias.device:
            raise ValueError(
                f"Expected gating_output and correction_bias to be on the same device, "
                f"got {gating_output.device} and {correction_bias.device}"
            )
        if gating_output.ndim != 2:
            raise ValueError(
                f"Expected gating_output to be 2D [T, E], got {gating_output.ndim}D"
            )
        if gating_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "Expected gating_output.dtype to be torch.float16, "
                f"torch.bfloat16, or torch.float32, got {gating_output.dtype}"
            )
        num_tokens, num_experts = gating_output.shape
        if (
            self._committed_num_tokens is not None
            and num_tokens != self._committed_num_tokens
        ):
            raise ValueError(
                f"Expected num_tokens={self._committed_num_tokens}, got {num_tokens}"
            )
        if (
            self._committed_num_experts is not None
            and num_experts != self._committed_num_experts
        ):
            raise ValueError(
                f"Expected num_experts={self._committed_num_experts}, got {num_experts}"
            )
        if self.top_k is None:
            raise ValueError("top_k must be provided at construction time")
        if self.top_k > num_experts:
            raise ValueError(
                f"top_k={self.top_k} cannot exceed num_experts={num_experts}"
            )
        if correction_bias is not None:
            if self.scoring_func != "sigmoid":
                raise ValueError(
                    "correction_bias requires scoring_func='sigmoid', got "
                    f"{self.scoring_func!r}"
                )
            if correction_bias.shape != (num_experts,):
                raise ValueError(
                    f"Expected correction_bias shape {(num_experts,)}, "
                    f"got {tuple(correction_bias.shape)}"
                )
            if correction_bias.dtype != torch.float32:
                raise ValueError(
                    f"Expected correction_bias.dtype torch.float32, got {correction_bias.dtype}"
                )

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.dtype = gating_output.dtype
        kernel = self._get_kernel(
            num_tokens, num_experts, self.top_k, gating_output.device.index,
            correction_bias is not None,
        )
        return kernel(gating_output, correction_bias)
