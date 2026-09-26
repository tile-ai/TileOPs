"""FusedMoEExperts implementation with indexed and tight backends."""

from __future__ import annotations

from typing import ClassVar, Dict, Mapping, Optional

from torch import Tensor

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ...op_base import Op
from ..abc import FusedMoEExpertsModular, WeightedReduce, WeightedReduceNoOp
from ..contracts import ContiguousLayoutSpec, RoutingEpilogueSpec
from ..staged import MoeExpertMLPFwdOp, MoePostPermuteFwdOp, MoePrePermuteFwdOp
from .indexed_routed_expert import IndexedExpertMLPFwdOp

__all__ = ["FusedMoEExpertsFwdOp"]


_TIGHT = ContiguousLayoutSpec.tight_physical_psum()


class FusedMoEExpertsFwdOp(FusedMoEExpertsModular):
    """Expert MLP with indexed small-route and tight grouped backends.

    forward() writes ``(T, H)``: reduction is done internally by the
    PostPermute/Unpermute stage, so make_weighted_reduce() returns
    WeightedReduceNoOp. A call with few routes per expert (``T * K <= 2 * E``) on
    ``silu_and_mul`` with ``H % 128 == 0`` and ``F % 256 == 0`` runs the indexed
    small-route op; every other call runs the staged pipeline.
    """

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "pre_permute": MoePrePermuteFwdOp,
        "expert_mlp": MoeExpertMLPFwdOp,
        "post_permute": MoePostPermuteFwdOp,
        "indexed_small_route": IndexedExpertMLPFwdOp,
    }

    def __init__(
        self,
        routed_scaling_factor: float = 1.0,
        *,
        activation: str = "silu_and_mul",
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            routed_scaling_factor: Scalar applied to the final reduced output.
            activation: Gated activation applied to gate_up: 'silu_and_mul' or
                'gelu_and_mul'.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel overrides handed to the sub-ops.
            tune: Whether the sub-ops' kernels tune themselves when built.
        """
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
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

    def roofline_inputs(self) -> dict[str, int]:
        """The experts this call's routing selected, which its weight reads follow."""
        from tileops.perf.formulas import routed_active_experts

        return {"active_experts": routed_active_experts(self.last_call)}

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["D"])

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
    ) -> None:
        """Run the local expert pipeline, writing the reduced result into ``output``."""
        tokens, top_k = topk_ids.shape
        experts, ffn2, hidden = w_gate_up.shape
        if (
            self._indexed_mlp is not None
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
        expert_output = self._expert_mlp(expert_input, w_gate_up, w_down, physical_ends)
        self._post_permute(expert_output, topk_weights, inverse_indices, out=output)
