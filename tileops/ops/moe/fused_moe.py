"""Routed Mixture-of-Experts (MoE) FFN operators.

Two manifest identities share a single composite implementation:

- ``FusedMoeFwdOp`` — routing + expert FFN without a routing correction bias
  (Qwen3 / DeepSeek-V3 style).
- ``FusedMoeFwdCbFwdOp`` — same flow but accepts a per-expert correction bias
  applied during top-k selection (Kimi K2 style).

The shared core (`FusedMoe`) wires `FusedTopKOp` (routing),
`FusedMoEPrepareAndFinalize` (quantization / EP dispatch), and an
`FusedMoEExpertsModular` implementation (permute + GEMM + unpermute). Shared
expert handling belongs to `SharedFusedMoE`.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops.moe.abc import FusedMoEExpertsModular, FusedMoEPrepareAndFinalize
from tileops.ops.moe.experts.nopad import FusedMoEExpertsNopadPersistent3WGFwdOp
from tileops.ops.moe.experts.padded import FusedMoEExpertsPaddedFwdOp
from tileops.ops.moe.fused_topk import FusedTopKOp
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP

from ..op_base import Op

__all__ = ["FusedMoe", "FusedMoeFwdCbFwdOp", "FusedMoeFwdOp"]


class FusedMoe(Op):
    """Shared composite implementation for routed MoE FFN ops.

    Concrete manifest identities (`FusedMoeFwdOp`, `FusedMoeFwdCbFwdOp`)
    subclass this to pin the correction-bias variant; both share the
    routing-and-expert pipeline below.

    Args:
        num_tokens: T -- number of input tokens.
        num_experts: E -- total number of experts (global count).
        top_k: K -- experts selected per token.
        hidden_size: H -- model hidden dimension.
        ffn_size: F -- per-expert intermediate dimension.
        scoring_func: "softmax" (Qwen3) or "sigmoid" (Kimi K2 / DeepSeek-V3).
        renormalize: Renormalize top-k weights to sum to 1.
        with_correction_bias: Accept `correction_bias` during routing.
        routed_scaling_factor: Multiplier on expert output (Kimi K2: 2.827).
        layout: "nopad" (default) or "padded". Ignored when `experts` is provided.
        dtype: Activation and weight dtype.
        expert_map: [E_global] int32 for Expert Parallel local filtering.
        prepare_finalize: Override the PrepareAndFinalize implementation.
        experts: Override the Experts implementation.
        kernel_map: Override the dispatched kernel map.
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
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
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

        self.dispatch_kernel(kernel_map)

        self._fused_topk = FusedTopKOp(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=with_correction_bias,
            kernel_map=kernel_map,
        )

        self._prepare: FusedMoEPrepareAndFinalize = (
            prepare_finalize if prepare_finalize is not None
            else MoEPrepareAndFinalizeNoDPEP()
        )

        if prepare_finalize is not None and experts is None:
            raise ValueError(
                "prepare_finalize may change the dispatched token count (T'); "
                "you must also supply a matching experts= instance sized for T'."
            )

        if experts is not None:
            self._experts: FusedMoEExpertsModular = experts
        else:
            if layout not in ("nopad", "padded"):
                raise ValueError(f"Unknown layout {layout!r}; expected 'nopad' or 'padded'")
            experts_cls = FusedMoEExpertsNopadPersistent3WGFwdOp if layout == "nopad" else FusedMoEExpertsPaddedFwdOp
            self._experts = experts_cls(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                routed_scaling_factor=routed_scaling_factor,
                dtype=dtype,
                expert_map=expert_map,
                kernel_map=kernel_map,
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
        correction_bias: Optional[torch.Tensor] = None, # [E] float32
    ) -> torch.Tensor:                                  # [T, H]
        topk_weights, topk_ids = self._fused_topk(gating_output, correction_bias)

        r = self._prepare.prepare(
            hidden_states, topk_weights, topk_ids,
            self.num_experts, expert_map=self.expert_map,
        )

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
        self._experts.forward(
            expert_out, r.hidden_q, w_gate_up, w_down,
            r.topk_weights, r.topk_ids,
            expert_map=self.expert_map,
            workspace1=ws1, workspace2=ws2,
            num_experts=self.num_experts,
        )

        self._prepare.finalize(
            output, expert_out,
            r.topk_weights, r.topk_ids,
            self._experts.make_weighted_reduce(),
        )
        return output


class FusedMoeFwdOp(FusedMoe):
    """Routed MoE FFN without a routing correction bias.

    Covers Qwen3 (softmax) and DeepSeek-V3 (sigmoid) style configurations
    where top-k is computed directly from the gating scores.
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
        routed_scaling_factor: float = 1.0,
        layout: str = "nopad",
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        super().__init__(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=False,
            routed_scaling_factor=routed_scaling_factor,
            layout=layout,
            dtype=dtype,
            expert_map=expert_map,
            prepare_finalize=prepare_finalize,
            experts=experts,
            kernel_map=kernel_map,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(hidden_states, gating_output, w_gate_up, w_down, None)


class FusedMoeFwdCbFwdOp(FusedMoe):
    """Routed MoE FFN with a per-expert routing correction bias.

    Covers Kimi K2 style configurations: top-k is selected from
    ``sigmoid(score) + correction_bias`` while the final weights use the
    original (unbiased) sigmoid scores, renormalized.
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str = "sigmoid",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        layout: str = "nopad",
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        super().__init__(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=True,
            routed_scaling_factor=routed_scaling_factor,
            layout=layout,
            dtype=dtype,
            expert_map=expert_map,
            prepare_finalize=prepare_finalize,
            experts=experts,
            kernel_map=kernel_map,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states, gating_output, w_gate_up, w_down, correction_bias,
        )
