"""Indexed small-route expert MLP implementation."""

from __future__ import annotations

from typing import ClassVar, Mapping

import torch
from torch import Tensor

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.moe import (
    IndexedExpertCall,
    IndexedExpertDownFwdInterface,
    IndexedExpertDownKernel,
    IndexedExpertGateUpFwdInterface,
    IndexedExpertGateUpKernel,
    IndexedRouteStatsFwdInterface,
    IndexedRouteStatsKernel,
    IndexedWeightedReduceFwdInterface,
    IndexedWeightedReduceKernel,
)
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["IndexedExpertMLPFwdOp"]


class IndexedExpertMLPFwdOp(Op):
    """Route-major expert MLP with device-side reuse dispatch.

    It computes what ``FusedMoEExpertsFwdOp`` computes for ``silu_and_mul`` without biases,
    route by route, where ``MoEExpertMLPFwdOp`` takes rows already grouped by expert and
    returns one unweighted result row per input row.

    Each of the ``T * K`` routes is a row of its expert's GEMM, dispatched per route
    rather than per expert segment. Routes that share an expert are grouped, and only the
    leader copies that expert's weights, so the minimum DRAM traffic the roofline prices
    is one read per distinct expert. That pays off while the routes
    are few; :class:`FusedMoEExpertsFwdOp` picks this op over the staged pipeline on the
    shapes where it does.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "route_stats": IndexedRouteStatsKernel,
        "expert_gate_up": IndexedExpertGateUpKernel,
        "expert_down": IndexedExpertDownKernel,
        "weighted_reduce": IndexedWeightedReduceKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "route_stats": IndexedRouteStatsFwdInterface,
        "expert_gate_up": IndexedExpertGateUpFwdInterface,
        "expert_down": IndexedExpertDownFwdInterface,
        "weighted_reduce": IndexedWeightedReduceFwdInterface,
    }

    def __init__(
        self,
        routed_scaling_factor: float = 1.0,
        *,
        target: Target = None,
    ) -> None:
        """Fix the scalar applied to the reduced output; the route extents come per call.

        Args:
            routed_scaling_factor: Scalar applied to the final reduced output.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.routed_scaling_factor = routed_scaling_factor
        super().__init__(target=target)

    def roofline_data_terms(self) -> dict[str, int]:
        """The experts this call's routing selected, which its weight reads follow."""
        from tileops._perf.formulas import routed_active_experts

        return {"active_experts": routed_active_experts(self.last_call)}

    def roof_key(self) -> str:
        return tensor_core_roof(self.last_call.indices["D"])

    def forward(
        self,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        *,
        out: Tensor | None = None,
    ) -> Tensor:
        """Run the routed expert MLP and reduce each token's routes.

        Args:
            hidden_states: $[T \\times H]$ tokens.
            w_gate_up: $[E \\times 2F \\times H]$ stacked gate and up weights.
            w_down: $[E \\times H \\times F]$ down weights.
            topk_weights: $[T \\times K]$ float32 routing weights.
            topk_ids: $[T \\times K]$ int32 expert of each route.
            out: Optional $[T \\times H]$ buffer the result is written into.

        Returns:
            The $[T \\times H]$ weighted and reduced result: ``out`` when given, else a new
            tensor.
        """
        tokens, top_k = topk_ids.shape
        experts, ffn2, hidden = w_gate_up.shape
        # The spec also sizes the route metadata and the intermediate rows.
        call = self.call_spec(
            "expert_gate_up",
            lambda: IndexedExpertCall(
                num_tokens=tokens,
                top_k=top_k,
                num_experts=experts,
                hidden_size=hidden,
                ffn_size=ffn2 // 2,
                dtype=hidden_states.dtype,
                routed_scaling_factor=self.routed_scaling_factor,
                device=hidden_states.device,
            ),
        )
        metadata = None
        if call.grouped_dispatch:
            size = IndexedRouteStatsFwdInterface.output_size(call)
            metadata = torch.empty(size, dtype=torch.int32, device=topk_ids.device)
            self.kernel_for("route_stats", lambda: call)(topk_ids, metadata)
        hidden_rows = hidden_states.new_empty(tokens, top_k, call.ffn_size)
        route_output = hidden_states.new_empty(tokens, top_k, hidden)
        gate_up = self.kernel_for("expert_gate_up", lambda: call)
        gate_up(hidden_states, w_gate_up, topk_ids, metadata, out=hidden_rows)
        down = self.kernel_for("expert_down", lambda: call)
        down(hidden_rows, w_down, topk_ids, metadata, out=route_output)
        output = hidden_states.new_empty(tokens, hidden) if out is None else out
        self.kernel_for("weighted_reduce", lambda: call)(route_output, topk_weights, output)
        return output
