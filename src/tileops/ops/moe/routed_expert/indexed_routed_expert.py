"""Indexed small-route expert MLP implementation."""

from __future__ import annotations

from typing import ClassVar, Mapping

import torch
from torch import Tensor

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.moe import (
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
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["IndexedExpertMLPFwdOp"]


class IndexedExpertMLPFwdOp(Op):
    """Route-major expert MLP with device-side reuse dispatch.

    Each of the ``T * K`` routes is a row of its expert's GEMM, dispatched per route
    rather than per expert segment. Routes that share an expert are grouped, and only the
    leader copies that expert's weights, so the minimum DRAM traffic the roofline prices
    is one read per distinct expert. That pays off while the routes
    are few; :class:`FusedMoEExpertsFwdOp` picks this op over the staged pipeline on the
    shapes where it does. The indexed kernels require SM90.
    """

    compile_boundary: ClassVar[bool] = True
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
        kernel_map: dict | None = None,
        tune: bool = False,
    ) -> None:
        """Fix the scalar applied to the reduced output; the route extents come per call.

        Args:
            routed_scaling_factor: Scalar applied to the final reduced output.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional dispatch override mapping kernel keys to ``Kernel``
                subclasses.
            tune: Whether the kernels tune themselves when built.
        """
        self.routed_scaling_factor = routed_scaling_factor
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def roofline_inputs(self) -> dict[str, int]:
        """The experts this call's routing selected, which its weight reads follow."""
        from tileops.perf.formulas import routed_active_experts

        return {"active_experts": routed_active_experts(self.last_call)}

    def compute_roof(self) -> str:
        return tensor_core_roof(self.last_call.ix["D"])

    def forward(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
    ) -> None:
        """Write the weighted and reduced expert result into ``output``."""
        return self._call_boundary(output, hidden_states, w_gate_up, w_down, topk_weights, topk_ids)

    def _eager_forward(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
    ) -> None:
        tokens, top_k = topk_ids.shape
        experts, ffn2, hidden = w_gate_up.shape
        call = IndexedExpertCall(
            num_tokens=tokens,
            top_k=top_k,
            num_experts=experts,
            hidden_size=hidden,
            ffn_size=ffn2 // 2,
            dtype=hidden_states.dtype,
            routed_scaling_factor=self.routed_scaling_factor,
            device=hidden_states.device,
        )
        metadata = None
        if call.grouped_dispatch:
            stats = self.kernel_for("route_stats", (topk_ids,), call)
            metadata = torch.empty(stats.output_size, dtype=torch.int32, device=topk_ids.device)
            stats(topk_ids, metadata)
        hidden_rows = hidden_states.new_empty(tokens, top_k, call.ffn_size)
        route_output = hidden_states.new_empty(tokens, top_k, hidden)
        gate_up = self.kernel_for(
            "expert_gate_up", (hidden_states, w_gate_up, topk_ids, metadata, hidden_rows), call
        )
        gate_up(hidden_states, w_gate_up, topk_ids, metadata, out=hidden_rows)
        down = self.kernel_for(
            "expert_down", (hidden_rows, w_down, topk_ids, metadata, route_output), call
        )
        down(hidden_rows, w_down, topk_ids, metadata, out=route_output)
        self.kernel_for("weighted_reduce", (route_output, topk_weights, output), call)(
            route_output, topk_weights, output
        )
