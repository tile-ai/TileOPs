"""MoE fused top-k routing operator."""

from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.moe import FusedTopKCall, FusedTopKFwdInterface, FusedTopKKernel
from tileops.ops.op_base import Op

__all__ = ["FusedTopKFwdOp"]


class FusedTopKFwdOp(Op):
    """Mixture-of-Experts (MoE) top-k routing operator.

    Applies scoring (softmax or sigmoid) to router logits and selects the
    top-k experts per token. Scoring and selection run in float32 whatever the
    logits dtype, with fast-math transcendentals, so a score can differ from
    torch's float32 score by a few ulp: an expert, or a group, whose selection
    score is within 1e-5 relative plus 1e-6 absolute of the last one kept may be
    kept in its place; a selection that is not that close matches the float32
    reference, ties to the lower index. The weights are
    within 1e-4 (relative) of the float32 scores of the experts returned.

    With ``num_expert_group > 1`` it is DeepSeek grouped routing, as vLLM's
    ``grouped_topk``: the E experts split into ``num_expert_group`` groups of
    ``E / num_expert_group`` consecutive experts, each group is scored by the sum of
    its two best biased scores when ``correction_bias`` is passed and by its best
    score otherwise, the ``topk_group`` best groups are kept (ties to the lower
    group index), and the top-k experts are selected inside them. The returned
    weights are the unbiased scores of the selected experts, renormalized when
    ``renormalize``, times ``routed_scaling_factor``. ``num_expert_group=1`` selects
    over all experts.

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
        num_expert_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            top_k: Number of experts to select per token K.
            scoring_func: "softmax" (Qwen3/Qwen2) or "sigmoid" (DeepSeek-V3/GLM-4/Kimi K2).
                Passing ``correction_bias`` to ``forward`` requires ``"sigmoid"``: the
                bias is added to sigmoid scores for selection only, and the output
                weights stay the original scores.
            renormalize: If True, normalize top-k weights to sum to 1.
            num_expert_group: Groups of consecutive experts the selection is restricted
                by; must divide E. 1 (the default) selects over all experts. DeepSeek-V3: 8.
            topk_group: Groups kept per token, at most ``num_expert_group``; they must hold
                at least ``top_k`` experts. DeepSeek-V3: 4.
            routed_scaling_factor: Positive multiplier on the returned weights, applied
                after renormalization. DeepSeek-V3: 2.5.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        super().__init__(target=target)

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
            topk_weights: [T, K] float32, times ``routed_scaling_factor``.
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
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            dtype=gating_output.dtype,
            device=gating_output.device,
        )
        kernel = self.kernel_for("fused_topk", call)
        return kernel(gating_output, correction_bias)
