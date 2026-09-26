"""Indexed small-route expert MLP implementation."""

from __future__ import annotations

from typing import ClassVar, Mapping

import torch
from torch import Tensor

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.moe.indexed_expert_gemm import (
    IndexedExpertGemmTemplate,
    IndexedRouteStatsKernel,
    IndexedWeightedReduceKernel,
)
from tileops.perf.profile import tensor_core_roof
from tileops.utils import get_sm_version

from ...op_base import Op
from ..contracts import ContiguousLayoutSpec, RoutingEpilogueSpec
from ..staged import MoeExpertMLPFwdOp, MoePostPermuteFwdOp, MoePrePermuteFwdOp

__all__ = ["IndexedExpertMLPFwdOp"]


_TIGHT = ContiguousLayoutSpec.tight_physical_psum()


class IndexedExpertMLPFwdOp(Op):
    """Route-major expert MLP with device-side reuse dispatch.

    Each of the ``T * K`` routes is a row of its expert's GEMM, dispatched per route
    rather than per expert segment. Routes that share an expert are grouped, and only the
    leader copies that expert's weights, so the minimum DRAM traffic the roofline prices
    is one read per distinct expert. That pays off while the routes
    are few; :class:`FusedMoEExpertsFwdOp` picks this op over the staged pipeline on the
    shapes where it does. The indexed kernels require SM90; on another card the op runs
    the staged pipeline instead.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "route_stats": IndexedRouteStatsKernel,
        "expert_gemm": IndexedExpertGemmTemplate,
        "weighted_reduce": IndexedWeightedReduceKernel,
    }
    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "pre_permute": MoePrePermuteFwdOp,
        "expert_mlp": MoeExpertMLPFwdOp,
        "post_permute": MoePostPermuteFwdOp,
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
                subclasses, also handed to the staged ops this one falls back to.
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

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation per role, built per route extents and operand dtype."""
        return call, lambda: getattr(self, f"_build_{role}")(*call)

    def _build_route_stats(self, tokens: int, top_k: int, experts: int, hidden, ffn, dtype):
        """The route statistics pass that groups a multi-token call's routes by expert."""
        return self.kernel_map["route_stats"](tokens, top_k, experts)

    def _build_expert_gemm(self, tokens: int, top_k: int, experts: int, hidden, ffn, dtype):
        """The gate/up GEMM with its fused activation, and the down GEMM, built together."""
        dispatch_mode = "grouped" if tokens > 1 else "direct"
        expert_gemm = self.kernel_map["expert_gemm"]
        gate = expert_gemm(
            tokens,
            experts,
            top_k,
            ffn,
            hidden,
            dtype,
            activation="silu_and_mul",
            dispatch_mode=dispatch_mode,
        )
        down = expert_gemm(
            tokens,
            experts,
            top_k,
            hidden,
            ffn,
            dtype,
            route_input=True,
            dispatch_mode=dispatch_mode,
        )
        return gate, down

    def _build_weighted_reduce(self, tokens: int, top_k: int, experts, hidden: int, ffn, dtype):
        """The weighted reduction over each token's routes."""
        return self.kernel_map["weighted_reduce"](
            tokens, top_k, hidden, dtype, self.routed_scaling_factor
        )

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

    def _staged(self, experts: int) -> tuple[Op, Op, Op]:
        """The staged pipeline for *experts* local experts, which serves a card without SM90."""
        epilogue = RoutingEpilogueSpec(routed_scaling_factor=self.routed_scaling_factor)
        return (
            self.delegate_for("pre_permute", experts, layout=_TIGHT, num_local_experts=experts),
            self.delegate_for("expert_mlp", None, layout=_TIGHT, activation="silu_and_mul"),
            self.delegate_for("post_permute", None, layout=_TIGHT, epilogue=epilogue),
        )

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
        ffn = ffn2 // 2
        if get_sm_version(hidden_states.device.index) != 90:
            pre, mlp, post = self._staged(experts)
            expert_input, physical_ends, inverse_indices = pre(hidden_states, topk_ids)
            expert_output = mlp(expert_input, w_gate_up, w_down, physical_ends)
            post(expert_output, topk_weights, inverse_indices, out=output)
            return
        inputs = (output, hidden_states, w_gate_up, w_down, topk_weights, topk_ids)
        call = (tokens, top_k, experts, hidden, ffn, hidden_states.dtype)
        gate, down = self.kernel_for("expert_gemm", inputs, call)
        reduce = self.kernel_for("weighted_reduce", inputs, call)
        hidden_rows = hidden_states.new_empty(tokens, top_k, ffn)
        route_output = hidden_states.new_empty(tokens, top_k, hidden)
        if tokens == 1:
            metadata = topk_ids.reshape(-1)[:1]
        else:
            stats = self.kernel_for("route_stats", inputs, call)
            size = IndexedRouteStatsKernel.required_output_size(tokens, top_k, experts)
            metadata = torch.empty(size, dtype=torch.int32, device=topk_ids.device)
            stats(topk_ids, metadata)
        gate(hidden_states, w_gate_up, topk_ids, metadata, out=hidden_rows)
        down(hidden_rows, w_down, topk_ids, metadata, out=route_output)
        reduce(route_output, topk_weights, output)
