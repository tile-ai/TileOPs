"""Routed Mixture-of-Experts (MoE) FFN operators.

``FusedMoeFwdOp`` is routing + expert FFN. Passing ``correction_bias`` adds the
per-expert bias during top-k selection (Kimi K2 style); withholding it selects
straight from the gating scores (Qwen3 / DeepSeek-V3 style).

The shared core (`FusedMoe`) wires `FusedTopKFwdOp` (routing),
`FusedMoEPrepareAndFinalize` (quantization / EP dispatch), and an
`FusedMoEExpertsModular` implementation (permute + GEMM + unpermute). Shared
expert handling belongs to `FusedMoeSharedExpertFwdOp`.
"""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.ops.moe.abc import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalize,
)
from tileops.ops.moe.fused_topk import FusedTopKFwdOp
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP
from tileops.ops.moe.routed_expert import FusedMoEExpertsFwdOp
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["FusedMoe", "FusedMoeFwdOp"]


class FusedMoe(Op):
    """Shared composite implementation for routed MoE FFN ops.

    The concrete manifest identity (`FusedMoeFwdOp`) subclasses this; the
    routing-and-expert pipeline below is shared with `FusedMoeSharedExpertFwdOp`.
    """

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "route_select": FusedTopKFwdOp,
        "routed_experts": FusedMoEExpertsFwdOp,
    }
    execution_parameters: ClassVar[tuple[str, ...]] = ("prepare_finalize", "experts")

    def _build_pipeline(
        self,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize],
        experts: Optional[FusedMoEExpertsModular],
    ) -> None:
        """Hold the routing sub-op, the prepare/finalize stage and the experts.

        Raises:
            ValueError: An injected ``experts`` names no ``activation``, or one that
                conflicts with a non-default ``activation`` passed here.
        """
        self._fused_topk = self.delegate_for(
            "route_select",
            None,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            renormalize=self.renormalize,
        )
        self._prepare: FusedMoEPrepareAndFinalize = (
            prepare_finalize if prepare_finalize is not None else MoEPrepareAndFinalizeNoDPEP()
        )
        if experts is not None:
            # A missing attribute on a third-party implementation would otherwise let a
            # non-matching `activation` pass silently, producing a wrong-activation pipeline.
            if not hasattr(experts, "activation"):
                raise ValueError(
                    f"injected experts instance ({type(experts).__name__}) is missing the "
                    "required `.activation` attribute naming the activation it applies"
                )
            # The default cannot be told from an omitted argument, so only a conflicting
            # non-default value is refused.
            if self.activation != "silu_and_mul" and self.activation != experts.activation:
                raise ValueError(
                    f"activation conflicts with the injected experts instance: got "
                    f"activation={self.activation!r}, experts.activation={experts.activation!r}"
                )
            self.activation = experts.activation
        self._experts: FusedMoEExpertsModular = self.delegate_for(
            "routed_experts",
            None,
            experts,
            routed_scaling_factor=self.routed_scaling_factor,
            activation=self.activation,
        )

    def roofline_inputs(self) -> dict[str, int]:
        """The experts this call read, which its weight reads follow."""
        from tileops.perf.formulas import fused_moe_active_experts

        return {"active_experts": fused_moe_active_experts(self.last_call)}

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["D"])

    def _routed(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        correction_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Route, prepare, run the experts and finalize; the ``[T, H]`` routed output."""
        topk_weights, topk_ids = self._fused_topk(gating_output, correction_bias)
        r = self._prepare.prepare(hidden_states, topk_weights, topk_ids, w_gate_up.shape[0])
        output = hidden_states.new_empty(hidden_states.shape)
        expert_out_shape = self._experts.output_shape(r.hidden_q.shape[0], hidden_states.shape[1])
        expert_out = (
            output
            if expert_out_shape == tuple(hidden_states.shape)
            else hidden_states.new_empty(expert_out_shape)
        )
        self._experts(expert_out, r.hidden_q, w_gate_up, w_down, r.topk_weights, r.topk_ids)
        self._prepare.finalize(
            output, expert_out, r.topk_weights, r.topk_ids, self._experts.make_weighted_reduce()
        )
        return output


class FusedMoeFwdOp(FusedMoe):
    """Routed MoE FFN.

    Covers Qwen3 (softmax) and DeepSeek-V3 (sigmoid) style configurations where
    top-k comes straight from the gating scores, and Kimi K2 style ones where a
    per-expert ``correction_bias`` is passed: top-k is then selected from
    ``sigmoid(score) + correction_bias`` while the final weights use the
    original (unbiased) scores, renormalized.
    """

    def __init__(
        self,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        *,
        activation: str = "silu_and_mul",
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            top_k: K -- experts selected per token.
            scoring_func: "softmax" (Qwen3) or "sigmoid" (Kimi K2 / DeepSeek-V3).
            renormalize: Renormalize top-k weights to sum to 1.
            routed_scaling_factor: Multiplier on expert output (Kimi K2: 2.827).
            activation: Gated activation applied to gate_up.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
                The sub-ops it builds are given the same one.
            kernel_map: Kernel overrides handed to the sub-ops.
            tune: Whether the sub-ops' kernels tune themselves when built.
            prepare_finalize: Override the PrepareAndFinalize implementation.
            experts: Override the Experts implementation.
        """
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._build_pipeline(prepare_finalize, experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Route each token to its top-k experts and return the ``[T, H]`` weighted sum."""
        return self._routed(hidden_states, gating_output, w_gate_up, w_down, correction_bias)
