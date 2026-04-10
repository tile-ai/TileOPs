"""Unified routed MoE FFN operator — FusedMoe.

Corresponds to vLLM's FusedMoE layer: combines FusedTopKOp (routing) and
FusedMoeExpertsFwdOp (permute + GEMM + SwiGLU + GEMM + unpermute) into a single
forward pass.

Does NOT handle shared experts (handled by an upper SharedFusedMoe layer,
analogous to vLLM's SharedFusedMoE).

Supports both Qwen3-style (softmax) and Kimi K2-style (sigmoid + correction_bias)
routing via constructor parameters, replacing the model-specific ops
(Qwen3MoENopadOp, KimiMoENopadOp, etc.).

Layout variants:
    layout="nopad"   — tight T*K layout, GPU tile scheduler (default, fastest)
    layout="padded"  — block_m-aligned padding (comparison baseline)

EP note:
    For single-GPU / TP-only usage pass expert_map=None (default).
    For multi-GPU EP with local filtering, pass expert_map [E_global] int32.
    All-to-All communication is NOT performed here; use an external framework
    (vLLM / SGLang / Megatron) to handle dispatch/combine and call
    FusedMoeExpertsFwdOp directly with pre-dispatched tokens.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.ops.moe.fused_moe_experts import FusedMoeExpertsFwdOp, FusedMoeExpertsPaddedFwdOp
from tileops.ops.moe.fused_topk import FusedTopKOp

from ..op import Op

__all__ = ["FusedMoe"]


class FusedMoe(Op):
    """Unified routed MoE FFN Op (routing + GEMM), analogous to vLLM FusedMoE.

    Does NOT process shared experts.  Shared expert handling belongs to the
    upper layer (SharedFusedMoe / model-specific MoE).

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Total number of experts E (global count).
        top_k: Experts selected per token K.
        hidden_size: Model hidden dimension H.
        ffn_size: Per-expert intermediate dimension F.
        scoring_func: Router scoring function — "softmax" (Qwen3/Qwen2) or
            "sigmoid" (Kimi K2 / DeepSeek-V3).  Default "softmax".
        renormalize: Renormalize top-k weights to sum to 1 after selection.
            Default False.
        with_correction_bias: Whether the router uses a per-expert correction
            bias (Kimi K2 style).  When True, forward() accepts correction_bias.
            Default False.
        routed_scaling_factor: Multiplier applied to the routed expert output
            after unpermute (Kimi K2: 2.827; default 1.0 = no scaling).
        layout: "nopad" (default, tight T*K layout) or "padded"
            (block_m-aligned padding, comparison baseline).
        dtype: Activation and weight dtype (bf16 or fp16).
        expert_map: Optional [E_global] int32 tensor for local EP filtering.
            Maps global expert ids to local ids (-1 = not on this rank).
            All-to-All communication is NOT performed here.
            Raises NotImplementedError for layout="padded" with non-None value.
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        with_correction_bias: bool = False,
        routed_scaling_factor: float = 1.0,
        layout: str = "nopad",
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.layout = layout
        self.dtype = dtype

        self._fused_topk = FusedTopKOp(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=with_correction_bias,
        )

        if layout not in ("nopad", "padded"):
            raise ValueError(f"Unknown layout {layout!r}; expected 'nopad' or 'padded'")

        experts_cls = FusedMoeExpertsFwdOp if layout == "nopad" else FusedMoeExpertsPaddedFwdOp
        self._experts = experts_cls(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            expert_map=expert_map,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def forward(
        self,
        hidden_states: torch.Tensor,                    # [T, H]  bf16/fp16
        gating_output: torch.Tensor,                    # [T, E]  bf16/fp16/fp32
        w_gate_up: torch.Tensor,                        # [E, 2*F, H]
        w_down: torch.Tensor,                           # [E, H, F]
        correction_bias: Optional[torch.Tensor] = None, # [E] float32, or None
    ) -> torch.Tensor:                                  # [T, H]
        """Run routed MoE FFN.

        Args:
            hidden_states: [T, H] input token activations.
            gating_output: [T, E] router logits (pre-softmax/sigmoid scores).
            w_gate_up: [E, 2*F, H] gate+up projection weights.
            w_down: [E, H, F] down projection weights.
            correction_bias: [E] float32 per-expert correction bias for top-k
                selection (Kimi K2 style).  Required when with_correction_bias=True.

        Returns:
            output: [T, H] routed MoE FFN output, same dtype as hidden_states.
                Does NOT include shared expert contributions.
        """
        topk_weights, topk_ids = self._fused_topk(gating_output, correction_bias)
        return self._experts(hidden_states, w_gate_up, w_down, topk_weights, topk_ids)
