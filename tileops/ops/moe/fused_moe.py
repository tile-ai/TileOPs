"""Unified routed MoE FFN operator — FusedMoe.

Combines FusedTopKOp (routing) and MoEExpertsModular (permute + GEMM + unpermute)
with MoEPrepareAndFinalize (quantization + EP communication) via the ABC protocol.

Backwards-compatible: existing callers pass num_tokens / num_experts / ... as before.
When prepare_finalize / experts are None, defaults are created from the layout param.

Does NOT handle shared experts (handled by SharedFusedMoe).
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops.moe.abc import MoEExpertsModular, MoEPrepareAndFinalize
from tileops.ops.moe.experts.nopad import MoEExpertsNopadFwdOp
from tileops.ops.moe.experts.padded import MoEExpertsPaddedFwdOp
from tileops.ops.moe.fused_topk import FusedTopKOp
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP

from ..op_base import Op

__all__ = ["FusedMoe"]


class FusedMoe(Op):
    """Unified routed MoE FFN Op (routing + GEMM), analogous to vLLM FusedMoE.

    Does NOT process shared experts.  Shared expert handling belongs to the
    upper layer (SharedFusedMoe / model-specific MoE).

    Args:
        num_tokens: T — number of input tokens.
        num_experts: E — total number of experts (global count).
        top_k: K — experts selected per token.
        hidden_size: H — model hidden dimension.
        ffn_size: F — per-expert intermediate dimension.
        scoring_func: "softmax" (Qwen3) or "sigmoid" (Kimi K2 / DeepSeek-V3).
        renormalize: Renormalize top-k weights to sum to 1. Default False.
        with_correction_bias: Accept correction_bias in forward(). Default False.
        routed_scaling_factor: Multiplier on expert output (Kimi K2: 2.827).
        layout: "nopad" (default) or "padded". Ignored when experts is provided.
        dtype: Activation and weight dtype.
        expert_map: [E_global] int32 for EP local filtering. None = no EP.
        prepare_finalize: Override the PrepareAndFinalize implementation.
            Default: MoEPrepareAndFinalizeNoDPEP (no EP, no quantization).
        experts: Override the Experts implementation.
            Default: MoEExpertsNopadFwdOp (layout="nopad") or MoEExpertsPaddedFwdOp.
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
        prepare_finalize: Optional[MoEPrepareAndFinalize] = None,
        experts: Optional[MoEExpertsModular] = None,
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
        self.expert_map = expert_map

        self._fused_topk = FusedTopKOp(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=with_correction_bias,
        )

        self._prepare: MoEPrepareAndFinalize = (
            prepare_finalize if prepare_finalize is not None
            else MoEPrepareAndFinalizeNoDPEP()
        )

        if experts is not None:
            self._experts: MoEExpertsModular = experts
        else:
            if layout not in ("nopad", "padded"):
                raise ValueError(f"Unknown layout {layout!r}; expected 'nopad' or 'padded'")
            experts_cls = MoEExpertsNopadFwdOp if layout == "nopad" else MoEExpertsPaddedFwdOp
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
        hidden_states: torch.Tensor,                    # [T, H]
        gating_output: torch.Tensor,                    # [T, E]
        w_gate_up: torch.Tensor,                        # [E, 2*F, H]
        w_down: torch.Tensor,                           # [E, H, F]
        correction_bias: Optional[torch.Tensor] = None, # [E] float32, or None
    ) -> torch.Tensor:                                  # [T, H]
        # 1. Routing
        topk_weights, topk_ids = self._fused_topk(gating_output, correction_bias)

        # 2. Prepare (quantization / EP dispatch)
        r = self._prepare.prepare(
            hidden_states, topk_weights, topk_ids,
            self.num_experts, expert_map=self.expert_map,
        )

        # 3. Expert GEMM
        T_prime = r.hidden_q.shape[0]
        ws1_shape, ws2_shape = self._experts.workspace_shapes(
            T_prime, self.ffn_size, self.hidden_size,
            self.top_k, self.num_experts,
        )
        ws1 = hidden_states.new_empty(ws1_shape)
        ws2 = hidden_states.new_empty(ws2_shape)

        output = hidden_states.new_empty(hidden_states.shape)
        expert_out_shape = self._experts.output_shape(T_prime, self.hidden_size)
        expert_out = output if expert_out_shape == tuple(hidden_states.shape) else hidden_states.new_empty(expert_out_shape)
        self._experts.apply(
            expert_out, r.hidden_q, w_gate_up, w_down,
            r.topk_weights, r.topk_ids,
            self.num_experts, expert_map=self.expert_map,
            workspace1=ws1, workspace2=ws2,
        )

        # 4. Finalize (weighted reduction / EP gather)
        self._prepare.finalize(
            output, expert_out,
            r.topk_weights, r.topk_ids,
            self._experts.make_weighted_reduce(),
        )
        return output
