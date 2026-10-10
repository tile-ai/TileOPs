"""MoE fused top-k routing operator."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.moe import FusedTopKCall, FusedTopKFwdInterface, FusedTopKKernel
from tileops.ops.op_base import Op

__all__ = ["FusedTopKFwdOp"]


class FusedTopKFwdOp(Op):
    """Mixture-of-Experts (MoE) top-k routing operator.

    Applies scoring (softmax or sigmoid) to router logits and selects the
    top-k experts per token.

    Example:
        ```python linenums="1"
        op = FusedTopKFwdOp(top_k=8)
        topk_weights, topk_ids = op(gating_output)
        # topk_weights: [512, 8] float32
        # topk_ids:     [512, 8] int32
        ```
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"fused_topk": FusedTopKKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "fused_topk": FusedTopKFwdInterface
    }

    def __init__(
        self,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            top_k: Number of experts to select per token K.
            scoring_func: "softmax" (Qwen3/Qwen2) or "sigmoid" (DeepSeek-V3/GLM-4/Kimi K2).
                Passing ``correction_bias`` to ``forward`` requires ``"sigmoid"``: the
                bias is added to sigmoid scores for selection only, and the output
                weights stay the original scores.
            renormalize: If True, normalize top-k weights to sum to 1.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel map override.
            tune: Whether to autotune the kernel.
        """
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

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
        num_tokens, num_experts = gating_output.shape
        call = FusedTopKCall(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            renormalize=self.renormalize,
            with_correction_bias=correction_bias is not None,
            dtype=gating_output.dtype,
            device=gating_output.device,
        )
        kernel = self.kernel_for("fused_topk", call)
        return kernel(gating_output, correction_bias)
