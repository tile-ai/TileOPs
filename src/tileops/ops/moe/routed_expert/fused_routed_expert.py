"""FusedMoEExperts implementation with indexed and tight backends."""

from __future__ import annotations

from typing import ClassVar, Dict, Mapping, Optional

from torch import Tensor

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ...op_base import Op
from ..abc import (
    FusedMoEExpertsModular,
    WeightedReduce,
    WeightedReduceNoOp,
    _validate_fused_moe_experts_dtypes,
)
from ..contracts import ContiguousLayoutSpec, RoutingEpilogueSpec
from ..staged import MoeExpertMLPFwdOp, MoePostPermuteFwdOp, MoePrePermuteFwdOp
from .indexed_routed_expert import IndexedExpertMLPFwdOp

__all__ = ["FusedMoEExpertsFwdOp"]


class FusedMoEExpertsFwdOp(FusedMoEExpertsModular):
    """Expert MLP with indexed small-route and tight grouped backends.

    forward() output shape is (T, H): reduction is done internally by the
    PostPermute/Unpermute stage, so make_weighted_reduce() returns
    WeightedReduceNoOp.
    """

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "pre_permute": MoePrePermuteFwdOp,
        "expert_mlp": MoeExpertMLPFwdOp,
        "post_permute": MoePostPermuteFwdOp,
        "indexed_small_route": IndexedExpertMLPFwdOp,
    }

    # The tight path is what an instance runs until __init__ selects otherwise,
    # so contract checks that read the op without constructing it see it too.
    _indexed_mlp: IndexedExpertMLPFwdOp | None = None

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
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
        target: Target = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            num_tokens: Number of input tokens T (rows of hidden_states).
            num_experts: Number of local compute experts E.
            top_k: Number of experts each token is routed to (K).
            hidden_size: Model hidden dimension H (GEMM contraction dim for
                gate_up, output dim for down).
            ffn_size: Per-expert FFN intermediate dimension F.
            routed_scaling_factor: Scalar applied to the final reduced output.
                Defaults to 1.0 (no scaling).
            kernel_map: Optional kernel overrides forwarded to the inner Ops.
            activation: Gated activation applied to gate_up: 'silu_and_mul' or
                'gelu_and_mul'.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            tune: Whether the sub-ops' kernels tune themselves when built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        layout = ContiguousLayoutSpec.tight_physical_psum()
        self._expert_mlp = self.delegate_for(
            "expert_mlp", None, layout=layout, activation=activation
        )
        self._pre_permute = self.delegate_for(
            "pre_permute", None, layout=layout, num_local_experts=num_experts
        )
        self._post_permute = self.delegate_for(
            "post_permute",
            None,
            layout=layout,
            epilogue=RoutingEpilogueSpec(routed_scaling_factor=routed_scaling_factor),
        )
        indexed = (
            activation == "silu_and_mul"
            and hidden_size % 128 == 0
            and ffn_size % 256 == 0
            and num_tokens * top_k <= 2 * num_experts
        )
        if indexed:
            self._indexed_mlp = self.delegate_for(
                "indexed_small_route",
                None,
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                routed_scaling_factor=routed_scaling_factor,
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
        # hidden_states is the dtype anchor: the helper requires output,
        # w_gate_up and w_down to agree with it.
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
        """Hold the two scratch buffers to the sizes the chosen backend implies."""
        expected1, expected2 = self.workspace_shapes(
            self.num_tokens,
            self.ffn_size,
            self.hidden_size,
            self.top_k,
            self.num_experts,
        )
        if tuple(workspace1.shape) != expected1 or tuple(workspace2.shape) != expected2:
            raise ValueError(
                f"workspace1 and workspace2 must have shapes {expected1} and {expected2}; "
                f"got {tuple(workspace1.shape)} and {tuple(workspace2.shape)}"
            )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if getattr(self, "_indexed_mlp", None) is not None:
            return self._indexed_mlp.workspace_shapes()
        return ((0,), (0,))

    def output_shape(self, T_prime: int, H: int) -> tuple[int, int]:
        return (T_prime, H)

    def make_weighted_reduce(self) -> WeightedReduce:
        return WeightedReduceNoOp()

    @property
    def default_kernel_map(self) -> dict:
        return {}

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
        """Run the local expert pipeline, writing the reduced result into ``output``."""
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
        if self._indexed_mlp is not None:
            self._indexed_mlp(
                output,
                hidden_states,
                w_gate_up,
                w_down,
                topk_weights,
                topk_ids,
                workspace1,
                workspace2,
            )
            self._roofline_topk_ids = topk_ids
            return
        expert_input, physical_ends, inverse_indices = self._pre_permute(hidden_states, topk_ids)
        expert_output = self._expert_mlp(expert_input, w_gate_up, w_down, physical_ends)
        self._post_permute(expert_output, topk_weights, inverse_indices, out=output)
        # Set once the call has run, so a rejected one leaves no routing for the roofline.
        self._roofline_topk_ids = topk_ids

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
