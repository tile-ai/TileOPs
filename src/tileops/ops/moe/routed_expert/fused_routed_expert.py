"""FusedMoEExperts implementation with indexed and tight backends."""

from __future__ import annotations

from typing import ClassVar, Mapping

from torch import Tensor

from tileops.backend import Target
from tileops.ops.moe.abc import FusedMoEExpertsModular, WeightedReduce, WeightedReduceNoOp
from tileops.ops.moe.contracts import ContiguousLayoutSpec, RoutingEpilogueSpec
from tileops.ops.moe.routed_expert.indexed_routed_expert import IndexedExpertMLPFwdOp
from tileops.ops.moe.staged import MoEExpertMLPFwdOp, MoEPostPermuteFwdOp, MoEPrePermuteFwdOp
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["FusedMoEExpertsFwdOp"]


_TIGHT = ContiguousLayoutSpec.tight_physical_psum()


class FusedMoEExpertsFwdOp(FusedMoEExpertsModular):
    """Expert MLP with indexed small-route and tight grouped backends.

    ``output[t] = scale * sum_k w[t, k] * (down_k(act(gate_up_k(h[t]) + b_gate_up[k]))
    + b_down[k])`` over each token's routed experts ``k``, where ``scale`` is
    ``routed_scaling_factor``. Each bias joins its GEMM's result before the activation or the
    cast; the routing weight then scales the biased down output, as gpt-oss and vLLM's fused
    experts apply it. The biases are optional and
    independent of each other and of the activation.

    ``act`` is the gated activation ``activation`` names, and ``w_gate_up`` and
    ``b_gate_up`` hold gate and up rows in the order it reads, both as
    ``MoEGroupedGemmFwdOp`` defines them.

    forward() writes ``(T, H)``: reduction is done internally by the
    PostPermute/Unpermute stage, so make_weighted_reduce() returns
    WeightedReduceNoOp. A call with few routes per expert (``T * K <= 2 * E``) on
    ``silu_and_mul`` with no bias, ``H % 128 == 0`` and ``F % 256 == 0`` runs the indexed
    small-route op; every other call runs the staged pipeline. The two moduli are the
    widths the indexed kernels are tiled for, not a limit of what they compute.
    """

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "pre_permute": MoEPrePermuteFwdOp,
        "expert_mlp": MoEExpertMLPFwdOp,
        "post_permute": MoEPostPermuteFwdOp,
        "indexed_small_route": IndexedExpertMLPFwdOp,
    }

    def __init__(
        self,
        routed_scaling_factor: float = 1.0,
        *,
        activation: str = "silu_and_mul",
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            routed_scaling_factor: Scalar applied to the final reduced output.
            activation: Gated activation applied to gate_up: 'silu_and_mul' or
                'gelu_and_mul' over a stacked ``w_gate_up``, 'swigluoai' over an
                interleaved one.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        super().__init__(target=target)
        self._expert_mlp = self.delegate_for(
            "expert_mlp", None, layout=_TIGHT, activation=activation
        )
        self._post_permute = self.delegate_for(
            "post_permute",
            None,
            layout=_TIGHT,
            epilogue=RoutingEpilogueSpec(routed_scaling_factor=routed_scaling_factor),
        )
        self._indexed_mlp = (
            self.delegate_for(
                "indexed_small_route", None, routed_scaling_factor=routed_scaling_factor
            )
            if activation == "silu_and_mul"
            else None
        )

    def roofline_data_terms(self) -> dict[str, int]:
        """The experts this call's routing selected, which its weight reads follow."""
        from tileops.perf.formulas import routed_active_experts

        return {"active_experts": routed_active_experts(self.last_call)}

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.indices["D"])

    def output_shape(self, T_prime: int, H: int) -> tuple[int, int]:
        return (T_prime, H)

    def make_weighted_reduce(self) -> WeightedReduce:
        return WeightedReduceNoOp()

    def forward(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        b_gate_up: Tensor | None = None,
        b_down: Tensor | None = None,
    ) -> None:
        """Run the local expert pipeline, writing the reduced result into ``output``.

        Args:
            output: ``[T, H]`` buffer the reduced result is written into.
            hidden_states: ``[T, H]`` tokens.
            w_gate_up: ``[E, 2F, H]`` gate and up weights, stacked or interleaved as
                ``activation`` reads them.
            w_down: ``[E, H, F]`` down weights.
            topk_weights: ``[T, K]`` float32 routing weights.
            topk_ids: ``[T, K]`` int32 expert of each route.
            b_gate_up: Optional ``[E, 2F]`` gate/up bias in ``w_gate_up``'s row order.
            b_down: Optional ``[E, H]`` down bias.
        """
        tokens, top_k = topk_ids.shape
        experts, ffn2, hidden = w_gate_up.shape
        # The indexed kernels add no bias.
        if (
            self._indexed_mlp is not None
            and b_gate_up is None
            and b_down is None
            and hidden % 128 == 0
            and (ffn2 // 2) % 256 == 0
            and tokens * top_k <= 2 * experts
        ):
            self._indexed_mlp(output, hidden_states, w_gate_up, w_down, topk_weights, topk_ids)
            return
        pre_permute = self.delegate_for(
            "pre_permute", experts, layout=_TIGHT, num_local_experts=experts
        )
        expert_input, physical_ends, inverse_indices = pre_permute(hidden_states, topk_ids)
        expert_output = self._expert_mlp(
            expert_input, w_gate_up, w_down, physical_ends, b_gate_up=b_gate_up, b_down=b_down
        )
        self._post_permute(expert_output, topk_weights, inverse_indices, out=output)
