"""FusedMoEExperts implementation: nopad + 3WG persistent variant.

Registers no operator of its own: a composite is not the unit of replacement,
so its graph is its leaves' operators.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

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
from ..staged import MoePostPermuteFwdOp, MoePrePermuteFwdOp
from .gate_up import MoeGateUpFwdOp
from .moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp

__all__ = ["FusedMoEExpertsNopadPersistent3WGFwdOp"]


class FusedMoEExpertsNopadPersistent3WGFwdOp(FusedMoEExpertsModular):
    """Expert GEMM using tight (T*K rows, no-pad) layout with 3WG persistent kernel.

    Internal pipeline: MoePrePermuteFwdOp -> MoeGateUpFwdOp -> down GEMM ->
    MoePostPermuteFwdOp (weighted reduction included).  Pre/Post use the staged
    rank-grouped contract; the middle two ops retain their existing ABI until
    the ExpertMLP migration.

    forward() output shape is (T, H): reduction is done internally by
    MoePostPermuteFwdOp, so make_weighted_reduce() returns WeightedReduceNoOp.

    Example:
        ```python linenums="1"
        experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=512, num_experts=128, top_k=8,
            hidden_size=7168, ffn_size=2048,
        )
        ```
    """

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
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            num_tokens: Number of input tokens T (rows of hidden_states).
            num_experts: Number of local experts E. Global placement is resolved
                by EPDispatch before this local compute boundary.
            top_k: Number of experts each token is routed to (K).
            hidden_size: Model hidden dimension H (GEMM contraction dim for
                gate_up, output dim for down).
            ffn_size: Per-expert FFN intermediate dimension F.
            routed_scaling_factor: Scalar applied to the final reduced output.
                Defaults to 1.0 (no scaling).
            kernel_map: Optional kernel overrides forwarded to the inner Ops.
            activation: Gated activation applied to gate_up: 'silu_and_mul' or
                'gelu_and_mul'.
        """
        self.dispatch_kernel(kernel_map)
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.activation = activation
        self._routed_scaling_factor = routed_scaling_factor
        numel = num_tokens * top_k

        self._gate_up = MoeGateUpFwdOp(
            numel=numel,
            num_experts=num_experts,
            ffn=ffn_size,
            k=hidden_size,
            activation=activation,
            kernel_map=kernel_map,
        )
        self._gemm_down = MoeGroupedGemmNopadFwdOp(
            numel=numel,
            num_experts=num_experts,
            n=hidden_size,
            k=ffn_size,
            kernel_map=kernel_map,
        )
        layout = ContiguousLayoutSpec.tight_physical_psum()
        self._post_permute = MoePostPermuteFwdOp(
            layout=layout,
            epilogue=RoutingEpilogueSpec(
                routed_scaling_factor=routed_scaling_factor,
            ),
            kernel_map=kernel_map,
        )
        self._pre_permute = MoePrePermuteFwdOp(
            layout=layout,
            num_local_experts=num_experts,
            kernel_map=kernel_map,
        )

    def kernel_delegates(self) -> tuple[Op, ...]:
        return (self._pre_permute, self._gate_up, self._gemm_down, self._post_permute)

    def eval_roofline(self) -> tuple[int, int]:
        """Manifest ``roofline``: three F x H weight planes per local expert."""
        if self.dtype is None:
            raise ValueError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() to bind dtype"
            )
        flops = self.num_tokens * self.top_k * 6 * self.ffn_size * self.hidden_size
        nbytes = (
            self.num_experts * 3 * self.ffn_size * self.hidden_size
            + 2 * self.num_tokens * self.hidden_size
        ) * self.dtype.itemsize
        return int(flops), int(nbytes)

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
        self._reject_non_empty_workspaces(workspace1, workspace2)

    def _reject_non_empty_workspaces(
        self,
        workspace1: Tensor,
        workspace2: Tensor,
    ) -> None:
        """workspace_shapes() returns ((0,), (0,)); anything else is a mismatch."""
        if workspace1.numel() != 0 or workspace2.numel() != 0:
            raise ValueError(
                "workspace1 and workspace2 must be empty (numel == 0) for "
                f"{type(self).__name__}; got "
                f"workspace1.numel()={workspace1.numel()}, "
                f"workspace2.numel()={workspace2.numel()}."
            )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return ((0,), (0,))

    def output_shape(self, T_prime: int, H: int) -> tuple[int, int]:
        return (T_prime, H)

    def make_weighted_reduce(self) -> WeightedReduce:
        return WeightedReduceNoOp()

    @property
    def default_kernel_map(self) -> dict:
        # All sub-kernels are owned by the inner Ops (permute / GEMM / activation / unpermute).
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
        """Run the expert pipeline, writing the reduced result into ``output``.

        ``topk_ids`` are local expert IDs. EPDispatch owns global placement.
        """
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
        expert_input, physical_ends, inverse_indices = self._pre_permute(hidden_states, topk_ids)
        # Temporary bridge to the existing grouped-GEMM ABI.  The staged
        # PrePermute contract stays layout-based; this conversion disappears when
        # ExpertMLP/GroupedGemm migrate in the next round.
        true_offsets = torch.cat((physical_ends.new_zeros(1), physical_ends[:-1]))
        true_sizes = physical_ends - true_offsets
        act = self._gate_up(expert_input, w_gate_up, true_sizes, true_offsets)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        # Unpermute reduces into ``output`` directly and folds
        # ``routed_scaling_factor`` into its prim_func — no separate copy/scale.
        self._post_permute(mm2, topk_weights, inverse_indices, out=output)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
