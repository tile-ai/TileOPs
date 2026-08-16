"""FusedMoEExperts implementation: nopad + 3WG persistent variant.

Registers no operator of its own: a composite is not the unit of replacement,
so its graph is its leaves' operators.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from tileops.kernels.kernel_base import Kernel

from ...op_base import Op
from ..abc import (
    FusedMoEExpertsModular,
    WeightedReduce,
    WeightedReduceNoOp,
    _validate_fused_moe_experts_dtypes,
)
from .gate_up import MoeGateUpFwdOp
from .moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from .permute_nopad import MoePermuteNopadFwdOp
from .unpermute import MoeUnpermuteFwdOp

__all__ = ["FusedMoEExpertsNopadPersistent3WGFwdOp"]


class FusedMoEExpertsNopadPersistent3WGFwdOp(FusedMoEExpertsModular):
    """Expert GEMM using tight (T*K rows, no-pad) layout with 3WG persistent kernel.

    Internal pipeline: MoePermuteNopadFwdOp -> MoeGateUpFwdOp -> down GEMM ->
    MoeUnpermuteFwdOp (weighted reduction included). Every stage is built at
    construction, so ``forward`` resolves nothing and stays traceable.

    forward() output shape is (T, H): reduction is done internally by
    MoeUnpermuteFwdOp, so make_weighted_reduce() returns WeightedReduceNoOp.

    Args:
        num_tokens: Number of input tokens T (rows of hidden_states).
        num_experts: Total number of experts E in the routing table.
        num_experts_local: Number of those experts this rank owns; the weights
            and both grouped GEMMs are sized by it. Equal to ``num_experts``
            outside expert parallelism.
        top_k: Number of experts each token is routed to (K).
        hidden_size: Model hidden dimension H (GEMM contraction dim for
            gate_up, output dim for down).
        ffn_size: Per-expert FFN intermediate dimension F.
        routed_scaling_factor: Scalar applied to the final reduced output.
            Defaults to 1.0 (no scaling).
        kernel_map: Optional kernel overrides forwarded to the inner Ops.
        activation: Gated activation applied to gate_up: 'silu_and_mul' or
            'gelu_and_mul'.

    Example:
        >>> experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
        ...     num_tokens=512, num_experts=128, num_experts_local=128, top_k=8,
        ...     hidden_size=7168, ffn_size=2048,
        ... )
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        num_experts_local: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        routed_scaling_factor: float = 1.0,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
    ):
        self.dispatch_kernel(kernel_map)
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.num_experts_local = num_experts_local
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.activation = activation
        self._routed_scaling_factor = routed_scaling_factor
        numel = num_tokens * top_k

        self._gate_up = MoeGateUpFwdOp(
            numel=numel, num_experts=num_experts_local,
            ffn=ffn_size, k=hidden_size, activation=activation,
            kernel_map=kernel_map,
        )
        self._gemm_down = MoeGroupedGemmNopadFwdOp(
            numel=numel, num_experts=num_experts_local,
            n=hidden_size, k=ffn_size,
            kernel_map=kernel_map,
        )
        self._unpermute = MoeUnpermuteFwdOp(
            total_tokens=num_tokens, top_k=top_k,
            hidden_size=hidden_size, padded_batch_sum=numel,
            kernel_map=kernel_map,
            routed_scaling_factor=routed_scaling_factor,
        )
        self._permute = MoePermuteNopadFwdOp(
            num_experts=num_experts, num_experts_local=num_experts_local,
            kernel_map=kernel_map,
        )

    def kernel_delegates(self) -> tuple[Op, ...]:
        return (self._permute, self._gate_up, self._gemm_down, self._unpermute)

    def eval_roofline(self) -> tuple[int, int]:
        """Manifest ``roofline``: three F x H weight planes per local expert."""
        if self.dtype is None:
            raise ValueError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "to bind dtype"
            )
        flops = self.num_tokens * self.top_k * 6 * self.ffn_size * self.hidden_size
        nbytes = (
            self.num_experts_local * 3 * self.ffn_size * self.hidden_size
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
        *,
        expert_map: Optional[Tensor] = None,
    ) -> None:
        # hidden_states is the dtype anchor: the helper requires output,
        # w_gate_up and w_down to agree with it.
        self.dtype = hidden_states.dtype
        _validate_fused_moe_experts_dtypes(
            hidden_states.dtype,
            output, hidden_states, w_gate_up, w_down,
            topk_weights, topk_ids, workspace1, workspace2,
        )
        if expert_map is not None and expert_map.dtype != torch.int32:
            raise ValueError(
                f"Expected expert_map.dtype == int32, got {expert_map.dtype}")
        self._reject_non_empty_workspaces(workspace1, workspace2)

    def _reject_non_empty_workspaces(
        self, workspace1: Tensor, workspace2: Tensor,
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
        self, M: int, N: int, K: int, topk: int, num_experts: int,
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
        expert_map: Optional[Tensor] = None,
        *,
        num_experts: int,
    ) -> None:
        """Run the expert pipeline, writing the reduced result into ``output``.

        Args:
            expert_map: [E] int32 global-to-local expert ids under expert
                parallelism; ``None`` when this rank owns every expert. The
                permute stage rejects a map that does not cover exactly the
                local ids.
        """
        self._validate_dtypes(
            output, hidden_states, w_gate_up, w_down,
            topk_weights, topk_ids, workspace1, workspace2,
            expert_map=expert_map,
        )
        perm_h, true_offsets, true_sizes, _, fwd_idx = self._permute(
            hidden_states, topk_ids, expert_map)
        act = self._gate_up(perm_h, w_gate_up, true_sizes, true_offsets)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        # Unpermute reduces into ``output`` directly and folds
        # ``routed_scaling_factor`` into its prim_func — no separate copy/scale.
        self._unpermute(mm2, fwd_idx, topk_weights, out=output)
