"""Indexed small-route expert MLP implementation."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from tileops.kernels.kernel_base import Entry
from tileops.kernels.moe.indexed_expert_gemm import (
    IndexedExpertGemmTemplate,
    IndexedRouteStatsKernel,
    IndexedWeightedReduceKernel,
)
from tileops.perf.profile import tensor_core_roof
from tileops.utils import get_sm_version

from ..._compile_boundary_codegen import OperatorSpec
from ...op_base import Op
from ..abc import _validate_fused_moe_experts_dtypes
from ..contracts import ContiguousLayoutSpec, RoutingEpilogueSpec
from ..staged import MoeExpertMLPFwdOp, MoePostPermuteFwdOp, MoePrePermuteFwdOp

__all__ = ["IndexedExpertMLPFwdOp"]


class IndexedExpertMLPFwdOp(Op):
    """Route-major expert MLP with device-side reuse dispatch.

    Each of the ``T * K`` routes is a row of its expert's GEMM, dispatched per route
    rather than per expert segment. Routes that share an expert are grouped, and only the
    leader copies that expert's weights, so the minimum DRAM traffic the roofline prices
    is one read per distinct expert. That pays off while the routes
    are few; :class:`FusedMoEExpertsFwdOp` picks this op over the staged pipeline on the
    shapes where it does, and requires SM90.
    """

    # The op writes the caller's ``output`` buffer and returns nothing, so its single
    # operator is the writing one.
    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec.writes_out("output"),)

    def roofline_inputs(self) -> dict[str, int]:
        """The experts this call's routing selected, which its weight reads follow."""
        from tileops.perf.formulas import routed_expert_active_experts

        return {"active_experts": routed_expert_active_experts(self)}

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        routed_scaling_factor: float = 1.0,
        kernel_map: dict | None = None,
    ) -> None:
        """Fix the route extents and the scalar applied to the reduced output.

        Args:
            num_tokens: Number of input tokens T (rows of ``hidden_states``).
            num_experts: Number of local compute experts E.
            top_k: Number of experts each token is routed to (K).
            hidden_size: Model hidden dimension H.
            ffn_size: Per-expert FFN intermediate dimension F.
            routed_scaling_factor: Scalar applied to the final reduced output.
            kernel_map: Optional dispatch override mapping kernel keys to ``Kernel``
                subclasses, forwarded to the staged ops this one falls back to.
        """
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
        # The indexed kernels are SM90-only, and the card is a fact of the call. Choosing
        # inside the operator keeps the traced graph one node on any card, so the staged
        # pipeline it falls back to is built here.
        layout = ContiguousLayoutSpec.tight_physical_psum()
        self._pre_permute = MoePrePermuteFwdOp(
            layout=layout, num_local_experts=num_experts, kernel_map=kernel_map
        )
        self._expert_mlp = MoeExpertMLPFwdOp(layout, "silu_and_mul", kernel_map=kernel_map)
        self._post_permute = MoePostPermuteFwdOp(
            layout=layout,
            epilogue=RoutingEpilogueSpec(routed_scaling_factor=routed_scaling_factor),
            kernel_map=kernel_map,
        )
        self.dispatch_kernel(kernel_map)

    def kernel_delegates(self) -> tuple[Op, ...]:
        """The staged ops this one falls back to off SM90."""
        return (self._pre_permute, self._expert_mlp, self._post_permute)

    @property
    def default_kernel_map(self) -> dict:
        return {
            "route_stats": IndexedRouteStatsKernel,
            "expert_gemm": IndexedExpertGemmTemplate,
            "weighted_reduce": IndexedWeightedReduceKernel,
        }

    def _infer_output_shapes(
        self,
        output_shape: tuple[int, ...],
        hidden_states_shape: tuple[int, ...],
        w_gate_up_shape: tuple[int, ...],
        w_down_shape: tuple[int, ...],
        topk_weights_shape: tuple[int, ...],
        topk_ids_shape: tuple[int, ...],
        workspace1_shape: tuple[int, ...],
        workspace2_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: the caller's buffer holds one row per token."""
        return {"output": tuple(hidden_states_shape)}

    def workspace_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The two scratch buffers the caller allocates, in elements."""
        routes = self.num_tokens * self.top_k
        metadata_elements = 0
        if self.num_tokens > 1:
            dtype_ratio = torch.int32.itemsize // torch.float16.itemsize
            metadata_elements = dtype_ratio * IndexedRouteStatsKernel.required_output_size(
                self.num_tokens, self.top_k, self.num_experts
            )
        return (
            (routes * self.ffn_size + metadata_elements,),
            (routes * self.hidden_size,),
        )

    def _validate_dtypes(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        workspace1: Tensor,
        workspace2: Tensor,
    ) -> None:
        self.dtype = hidden_states.dtype
        _validate_fused_moe_experts_dtypes(
            hidden_states.dtype,
            output,
            hidden_states,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            workspace1,
            workspace2,
        )

    def _validate_workspaces(self, workspace1: Tensor, workspace2: Tensor) -> None:
        """Hold the two scratch buffers to the sizes the route extents imply."""
        expected1, expected2 = self.workspace_shapes()
        if tuple(workspace1.shape) != expected1 or tuple(workspace2.shape) != expected2:
            raise ValueError(
                f"indexed workspaces must have shapes {expected1} and {expected2}, got "
                f"{tuple(workspace1.shape)} and {tuple(workspace2.shape)}"
            )

    def entry_for(self, role: str, call: torch.dtype) -> Entry:
        """One implementation, built per operand dtype; the route extents are the op's."""
        return call, lambda: self._build(call)

    def _build(self, dtype: torch.dtype) -> tuple:
        """The route statistics pass, the two GEMMs and the weighted reduction."""
        grouped = self.num_tokens > 1
        dispatch_mode = "grouped" if grouped else "direct"
        route_stats = self.kernel_map["route_stats"]
        expert_gemm = self.kernel_map["expert_gemm"]
        stats = route_stats(self.num_tokens, self.top_k, self.num_experts) if grouped else None
        gate = expert_gemm(
            self.num_tokens,
            self.num_experts,
            self.top_k,
            self.ffn_size,
            self.hidden_size,
            dtype,
            activation="silu_and_mul",
            dispatch_mode=dispatch_mode,
        )
        down = expert_gemm(
            self.num_tokens,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.ffn_size,
            dtype,
            route_input=True,
            dispatch_mode=dispatch_mode,
        )
        reduce = self.kernel_map["weighted_reduce"](
            self.num_tokens,
            self.top_k,
            self.hidden_size,
            dtype,
            self.routed_scaling_factor,
        )
        return stats, gate, down, reduce

    def forward(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        workspace1: Tensor,
        workspace2: Tensor,
    ) -> None:
        """Write the weighted and reduced expert result into ``output``."""
        self._wrapped(
            output,
            hidden_states,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            workspace1,
            workspace2,
            self._instance_key,
        )
        # Outside the wrapped call, which the compiled path would trace, and after it,
        # so a call the validation inside rejects leaves no routing for the roofline.
        self._roofline_topk_ids = topk_ids

    def _eager_forward(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        workspace1: Tensor,
        workspace2: Tensor,
    ) -> None:
        self._validate_dtypes(
            output,
            hidden_states,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            workspace1,
            workspace2,
        )
        self._validate_workspaces(workspace1, workspace2)
        if get_sm_version(hidden_states.device.index) != 90:
            expert_input, physical_ends, inverse_indices = self._pre_permute(
                hidden_states, topk_ids
            )
            expert_output = self._expert_mlp(expert_input, w_gate_up, w_down, physical_ends)
            self._post_permute(expert_output, topk_weights, inverse_indices, out=output)
            return
        stats, gate, down, reduce = self.kernel_for(
            "indexed_mlp",
            (
                output,
                hidden_states,
                w_gate_up,
                w_down,
                topk_weights,
                topk_ids,
                workspace1,
                workspace2,
            ),
            hidden_states.dtype,
        )
        routes = self.num_tokens * self.top_k
        hidden_elements = routes * self.ffn_size
        hidden = workspace1[:hidden_elements].view(self.num_tokens, self.top_k, self.ffn_size)
        route_output = workspace2.view(self.num_tokens, self.top_k, self.hidden_size)
        if stats is None:
            metadata = topk_ids.reshape(-1)[:1]
        else:
            metadata = workspace1[hidden_elements:].view(torch.int32)
            stats(topk_ids, metadata)
        gate(hidden_states, w_gate_up, topk_ids, metadata, out=hidden)
        down(hidden, w_down, topk_ids, metadata, out=route_output)
        reduce(route_output, topk_weights, output)

    def compute_roof(self) -> str:
        return tensor_core_roof(self.dtype)
