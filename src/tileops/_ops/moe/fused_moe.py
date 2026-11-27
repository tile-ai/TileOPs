"""Routed Mixture-of-Experts (MoE) FFN operators."""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._ops.moe.fused_topk import FusedTopKFwdOp
from tileops._ops.moe.routed_expert import FusedMoEExpertsFwdOp
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["FusedMoE", "FusedMoEFwdOp"]


class FusedMoE(Op):
    """Routing (``FusedTopKFwdOp``) then experts (``FusedMoEExpertsFwdOp``), shared by
    ``FusedMoEFwdOp`` and ``FusedMoESharedExpertFwdOp``."""

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "route_select": FusedTopKFwdOp,
        "routed_experts": FusedMoEExpertsFwdOp,
    }
    injected_parameters: ClassVar[tuple[str, ...]] = ("experts",)

    def __init__(
        self,
        top_k: int,
        scoring_func: str,
        renormalize: bool,
        routed_scaling_factor: float,
        num_expert_group: int,
        topk_group: int,
        *,
        activation: str,
        target: Target,
        experts: FusedMoEExpertsFwdOp | None,
    ):
        """Set the routing parameters and build the routing sub-op and the experts.

        Raises:
            ValueError: The injected ``experts`` applies an activation that conflicts with a
                non-default ``activation`` passed here.
        """
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        # The default cannot be told from an omitted argument, so only a conflicting
        # non-default value is refused.
        if experts is not None:
            if activation != "silu_and_mul" and activation != experts.activation:
                raise ValueError(
                    f"activation conflicts with the injected experts instance: got "
                    f"activation={activation!r}, experts.activation={experts.activation!r}"
                )
            activation = experts.activation
        self.activation = activation
        super().__init__(target=target)
        self._fused_topk()
        self._experts(experts)

    def _fused_topk(self) -> Op:
        return self.delegate_for(
            "route_select",
            None,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
        )

    def _experts(self, given: FusedMoEExpertsFwdOp | None = None) -> FusedMoEExpertsFwdOp:
        """Return the experts stage; *given* is an implementation the caller injected."""
        return self.delegate_for(
            "routed_experts",
            None,
            given,
            routed_scaling_factor=self.routed_scaling_factor,
            activation=self.activation,
        )

    def roofline_data_terms(self) -> dict[str, int]:
        """The experts this call read, which its weight reads follow."""
        from tileops._perf.formulas import fused_moe_active_experts

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
        correction_bias: torch.Tensor | None,
        b_gate_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route, then run the experts; the ``[T, H]`` routed output."""
        topk_weights, topk_ids = self._fused_topk()(gating_output, correction_bias)
        return self._experts()(
            hidden_states, w_gate_up, w_down, topk_weights, topk_ids, b_gate_up, b_down
        )


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
    taken. The output is multiplied by ``routed_scaling_factor`` once (the routing itself
    is not scaled).

    gpt-oss is ``scoring_func="softmax"``, ``renormalize=True``, ``activation="swigluoai"``
    and both expert biases. ``activation``, and the row order of ``w_gate_up`` and
    ``b_gate_up`` it reads, are as ``MoEGroupedGEMMFwdOp`` defines them; the optional expert
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
        experts: FusedMoEExpertsFwdOp | None = None,
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
            experts: A configured ``FusedMoEExpertsFwdOp`` to run the experts; ``None``
                builds one from ``activation`` and ``routed_scaling_factor``.
        """
        super().__init__(
            top_k,
            scoring_func,
            renormalize,
            routed_scaling_factor,
            num_expert_group,
            topk_group,
            activation=activation,
            target=target,
            experts=experts,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
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

        Returns:
            A new tensor $[T \\times H]$: the routed experts' weighted sum for each token.
        """
        return self._routed(
            hidden_states, gating_output, w_gate_up, w_down, correction_bias, b_gate_up, b_down
        )
