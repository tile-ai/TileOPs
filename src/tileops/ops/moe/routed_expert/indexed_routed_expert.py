"""Indexed small-route expert MLP implementation."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from tileops.kernels.moe.indexed_expert_gemm import (
    IndexedExpertGemmTemplate,
    IndexedRouteStatsKernel,
    IndexedWeightedReduceKernel,
)
from tileops.ops.compile_boundary import get_instance
from tileops.perf.profile import tensor_core_roof
from tileops.utils import get_sm_version

from ...op_base import Op
from ..abc import _validate_fused_moe_experts_dtypes
from ..staged import MoeExpertMLPFwdOp, MoePostPermuteFwdOp, MoePrePermuteFwdOp

__all__: list[str] = []


class _IndexedExpertMLPFwdOp(Op):
    """Route-major expert MLP with device-side reuse dispatch."""

    compile_op_names: ClassVar[tuple[str, ...]] = ("tileops::moe_indexed_expert_mlp_fwd",)

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        routed_scaling_factor: float,
        fallback_pre_permute: MoePrePermuteFwdOp,
        fallback_expert_mlp: MoeExpertMLPFwdOp,
        fallback_post_permute: MoePostPermuteFwdOp,
    ) -> None:
        """Configure route shapes, output scaling, and the non-SM90 fallback."""
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
        self._fallback_pre_permute = fallback_pre_permute
        self._fallback_expert_mlp = fallback_expert_mlp
        self._fallback_post_permute = fallback_post_permute
        self.dispatch_kernel()

    @property
    def default_kernel_map(self) -> dict:
        return {}

    def _infer_output_shapes(
        self, hidden_states_shape: tuple[int, ...]
    ) -> dict[str, tuple[int, ...]]:
        return {"output": hidden_states_shape}

    def eval_roofline(self) -> tuple[int, int]:
        if self.dtype is None:
            raise RuntimeError("eval_roofline requires a prior forward call")
        flops = self.num_tokens * self.top_k * 6 * self.ffn_size * self.hidden_size
        nbytes = (
            self.num_experts * 3 * self.ffn_size * self.hidden_size
            + 2 * self.num_tokens * self.hidden_size
        ) * self.dtype.itemsize
        return int(flops), int(nbytes)

    def workspace_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        routes = self.num_tokens * self.top_k
        metadata_elements = 0
        if self.num_tokens > 1:
            metadata_elements = 2 * IndexedRouteStatsKernel.required_output_size(
                self.num_tokens, self.top_k
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
        expected1, expected2 = self.workspace_shapes()
        if tuple(workspace1.shape) != expected1 or tuple(workspace2.shape) != expected2:
            raise ValueError(
                f"indexed workspaces must have shapes {expected1} and {expected2}, got "
                f"{tuple(workspace1.shape)} and {tuple(workspace2.shape)}"
            )

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
        _moe_indexed_expert_mlp_fwd(
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
        if get_sm_version(hidden_states.device.index) != 90:
            expert_input, physical_ends, inverse_indices = self._fallback_pre_permute(
                hidden_states, topk_ids
            )
            expert_output = self._fallback_expert_mlp(
                expert_input, w_gate_up, w_down, physical_ends
            )
            self._fallback_post_permute(expert_output, topk_weights, inverse_indices, out=output)
            return
        dispatch_by_reuse = self.num_tokens > 1

        def build():
            stats = (
                IndexedRouteStatsKernel(self.num_tokens, self.top_k) if dispatch_by_reuse else None
            )
            gate_direct = IndexedExpertGemmTemplate(
                self.num_tokens,
                self.num_experts,
                self.top_k,
                self.ffn_size,
                self.hidden_size,
                hidden_states.dtype,
                activation="silu_and_mul",
                dispatch_by_reuse=dispatch_by_reuse,
            )
            down_direct = IndexedExpertGemmTemplate(
                self.num_tokens,
                self.num_experts,
                self.top_k,
                self.hidden_size,
                self.ffn_size,
                hidden_states.dtype,
                route_input=True,
                dispatch_by_reuse=dispatch_by_reuse,
            )
            gate_grouped = None
            down_grouped = None
            if dispatch_by_reuse:
                gate_grouped = IndexedExpertGemmTemplate(
                    self.num_tokens,
                    self.num_experts,
                    self.top_k,
                    self.ffn_size,
                    self.hidden_size,
                    hidden_states.dtype,
                    activation="silu_and_mul",
                    dispatch_by_reuse=True,
                    dispatch_mode="grouped",
                )
                down_grouped = IndexedExpertGemmTemplate(
                    self.num_tokens,
                    self.num_experts,
                    self.top_k,
                    self.hidden_size,
                    self.ffn_size,
                    hidden_states.dtype,
                    route_input=True,
                    dispatch_by_reuse=True,
                    dispatch_mode="grouped",
                )
            return (
                stats,
                gate_direct,
                gate_grouped,
                down_direct,
                down_grouped,
                IndexedWeightedReduceKernel(
                    self.num_tokens,
                    self.top_k,
                    self.hidden_size,
                    hidden_states.dtype,
                    self.routed_scaling_factor,
                ),
            )

        stats, gate_direct, gate_grouped, down_direct, down_grouped, reduce = (
            self.get_or_build_kernel(
                "indexed_mlp",
                inputs=(hidden_states, w_gate_up, w_down, topk_weights, topk_ids),
                key=hidden_states.dtype,
                build=build,
            )
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
        gate_direct(hidden_states, w_gate_up, topk_ids, metadata, out=hidden)
        if gate_grouped is not None:
            gate_grouped(hidden_states, w_gate_up, topk_ids, metadata, out=hidden)
        down_direct(hidden, w_down, topk_ids, metadata, out=route_output)
        if down_grouped is not None:
            down_grouped(hidden, w_down, topk_ids, metadata, out=route_output)
        reduce(route_output, topk_weights, output)

    def compute_roof(self) -> str:
        return tensor_core_roof(self.dtype)


@torch.library.custom_op(
    "tileops::moe_indexed_expert_mlp_fwd",
    mutates_args=("output", "workspace1", "workspace2"),
)
def _moe_indexed_expert_mlp_fwd(
    output: Tensor,
    hidden_states: Tensor,
    w_gate_up: Tensor,
    w_down: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    workspace1: Tensor,
    workspace2: Tensor,
    instance_key: str,
) -> None:
    get_instance(instance_key)._eager_forward(
        output,
        hidden_states,
        w_gate_up,
        w_down,
        topk_weights,
        topk_ids,
        workspace1,
        workspace2,
    )
