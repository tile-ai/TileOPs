"""Routed Mixture-of-Experts (MoE) FFN operators.

``FusedMoEFwdOp`` is routing + expert FFN. Passing ``correction_bias`` adds the
per-expert bias during top-k selection (Kimi K2 / DeepSeek-V3 style); withholding it
selects straight from the gating scores (Qwen3 style). ``num_expert_group`` /
``topk_group`` restrict the selection to the best expert groups (DeepSeek-V3).

The shared core (`FusedMoE`) wires `FusedTopKFwdOp` (routing),
`FusedMoEPrepareAndFinalize` (quantization / EP dispatch), and an
`FusedMoEExpertsModular` implementation (permute + GEMM + unpermute). Shared
expert handling belongs to `FusedMoESharedExpertFwdOp`.
"""

from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.ops.moe.abc import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalize,
)
from tileops.ops.moe.fused_topk import FusedTopKFwdOp
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP
from tileops.ops.moe.routed_expert import FusedMoEExpertsFwdOp
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["FusedMoE", "FusedMoEFwdOp"]


class FusedMoE(Op):
    """Shared composite implementation for routed MoE FFN ops.

    The concrete manifest identity (`FusedMoEFwdOp`) subclasses this; the
    routing-and-expert pipeline below is shared with `FusedMoESharedExpertFwdOp`.
    """

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "route_select": FusedTopKFwdOp,
        "routed_experts": FusedMoEExpertsFwdOp,
    }
    injected_parameters: ClassVar[tuple[str, ...]] = ("prepare_finalize", "experts")

    def _settle_activation(self, experts: Optional[FusedMoEExpertsModular]) -> None:
        """Take ``activation`` from injected ``experts``, before ``Op.__init__`` checks it.

        Raises:
            ValueError: An injected ``experts`` names no ``activation``, or one that
                conflicts with a non-default ``activation`` passed here.
        """
        if experts is None:
            return
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

    def _build_pipeline(
        self,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize],
        experts: Optional[FusedMoEExpertsModular],
    ) -> None:
        """Hold the routing sub-op, the prepare/finalize stage and the experts.

        The routing keeps its own ``routed_scaling_factor`` at 1: the experts' epilogue
        applies the op's factor, once.
        """
        self._fused_topk = self.delegate_for(
            "route_select",
            None,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
        )
        self._prepare: FusedMoEPrepareAndFinalize = (
            prepare_finalize if prepare_finalize is not None else MoEPrepareAndFinalizeNoDPEP()
        )
        self._experts: FusedMoEExpertsModular = self.delegate_for(
            "routed_experts",
            None,
            experts,
            routed_scaling_factor=self.routed_scaling_factor,
            activation=self.activation,
        )

    def roofline_data_terms(self) -> dict[str, int]:
        """The experts this call read, which its weight reads follow."""
        from tileops.perf.formulas import fused_moe_active_experts

        return {"active_experts": fused_moe_active_experts(self.last_call)}

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.indices["D"])

    def _routed(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        correction_bias: Optional[torch.Tensor],
        b_gate_up: Optional[torch.Tensor] = None,
        b_down: Optional[torch.Tensor] = None,
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
        self._experts(
            expert_out,
            r.hidden_q,
            w_gate_up,
            w_down,
            r.topk_weights,
            r.topk_ids,
            b_gate_up,
            b_down,
        )
        self._prepare.finalize(
            output, expert_out, r.topk_weights, r.topk_ids, self._experts.make_weighted_reduce()
        )
        return output


class FusedMoEFwdOp(FusedMoE):
    """Routed Mixture-of-Experts (MoE) FFN.

    Covers Qwen3 (softmax) style configurations where top-k comes straight from
    the gating scores, and Kimi K2 / DeepSeek-V3 style ones where a per-expert
    ``correction_bias`` is passed: top-k is then selected from
    ``sigmoid(score) + correction_bias`` while the final weights use the
    original (unbiased) scores, renormalized.

    ``num_expert_group > 1`` routes as DeepSeek-V3 does: the experts split into
    ``num_expert_group`` groups of consecutive experts, each scored by the sum of its
    two best biased scores when ``correction_bias`` is passed and by its best score
    otherwise, and top-k is selected inside the ``topk_group`` best groups. A group or
    an expert within 1e-5 relative plus 1e-6 absolute of the last one kept may be kept
    in its place, as in ``FusedTopKFwdOp``, and the output is then that of the routing
    taken. The output is multiplied by ``routed_scaling_factor`` once.

    gpt-oss is ``scoring_func="softmax"``, ``renormalize=True``, ``activation="swigluoai"``
    and both expert biases. ``activation``, and the row order of ``w_gate_up`` and
    ``b_gate_up`` it reads, are as ``MoEGroupedGemmFwdOp`` defines them; the optional expert
    biases ``b_gate_up`` and ``b_down`` join each route as ``FusedMoEExpertsFwdOp`` states.
    """

    def __init__(
        self,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        num_expert_group: int = 1,
        topk_group: int = 1,
        *,
        activation: str = "silu_and_mul",
        target: Target = None,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            top_k: K -- experts selected per token.
            scoring_func: "softmax" (Qwen3) or "sigmoid" (Kimi K2 / DeepSeek-V3).
            renormalize: Renormalize top-k weights to sum to 1.
            routed_scaling_factor: Multiplier on expert output (Kimi K2: 2.827,
                DeepSeek-V3: 2.5).
            num_expert_group: Groups of consecutive experts the routing is restricted
                by; must divide E. 1 selects over all experts. DeepSeek-V3: 8.
            topk_group: Groups kept per token, holding at least ``top_k`` experts.
                DeepSeek-V3: 4.
            activation: Gated activation applied to gate_up: ``"silu_and_mul"`` or
                ``"gelu_and_mul"`` over a stacked ``w_gate_up``, ``"swigluoai"`` over an
                interleaved one.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
                The sub-ops it builds are given the same one.
            prepare_finalize: Override the PrepareAndFinalize implementation.
            experts: Override the Experts implementation.
        """
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.activation = activation
        self._settle_activation(experts)
        super().__init__(target=target)
        self._build_pipeline(prepare_finalize, experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
        b_gate_up: Optional[torch.Tensor] = None,
        b_down: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Route each token to its top-k experts and return the ``[T, H]`` weighted sum.

        Args:
            hidden_states: ``[T, H]`` tokens.
            gating_output: ``[T, E]`` float32 router logits.
            w_gate_up: ``[E, 2F, H]`` gate and up weights, stacked or interleaved as
                ``activation`` reads them.
            w_down: ``[E, H, F]`` down weights.
            correction_bias: Optional ``[E]`` float32 bias added to the sigmoid scores for
                the selection only.
            b_gate_up: Optional ``[E, 2F]`` gate/up bias in ``w_gate_up``'s row order.
            b_down: Optional ``[E, H]`` down bias.
        """
        return self._routed(
            hidden_states, gating_output, w_gate_up, w_down, correction_bias, b_gate_up, b_down
        )
