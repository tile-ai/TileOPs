"""FusedMoEExperts implementation: nopad + 3WG persistent variant.

Two manifest identities share the pipeline, one per expert-parallel (EP) shape:

- ``FusedMoEExpertsNopadPersistent3WGFwdOp`` — every expert is local.
- ``FusedMoEExpertsNopadPersistent3WGEpFwdOp`` — this rank owns
  ``num_experts_local`` experts, a constructor parameter, and takes the
  global-to-local map as a ``forward`` input.

Neither registers an operator of its own: a composite is not the unit of
replacement, so its graph is its leaves' operators.
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
from .permute_nopad import MoePermuteNopadEpFwdOp, MoePermuteNopadFwdOp
from .unpermute import MoeUnpermuteFwdOp

__all__ = [
    "FusedMoEExpertsNopadPersistent3WGEpFwdOp",
    "FusedMoEExpertsNopadPersistent3WGFwdOp",
]


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
        ...     num_tokens=512, num_experts=128, top_k=8,
        ...     hidden_size=7168, ffn_size=2048,
        ... )
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
        self.dispatch_kernel(kernel_map)
        self._init_pipeline(
            num_tokens=num_tokens,
            num_experts=num_experts,
            num_experts_local=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            routed_scaling_factor=routed_scaling_factor,
            kernel_map=kernel_map,
            activation=activation,
        )
        self._permute = MoePermuteNopadFwdOp(
            num_experts=num_experts, kernel_map=kernel_map,
        )

    def _init_pipeline(
        self,
        *,
        num_tokens: int,
        num_experts: int,
        num_experts_local: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        routed_scaling_factor: float,
        kernel_map: Optional[Dict[str, Kernel]],
        activation: str,
    ) -> None:
        """Build the stages sized by the expert count this rank owns.

        The two grouped GEMMs and the unpermute are the same for both identities;
        only the permute stage differs, and each identity builds its own.
        """
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
    ) -> None:
        # hidden_states is the dtype anchor: the helper requires output,
        # w_gate_up and w_down to agree with it.
        self.dtype = hidden_states.dtype
        _validate_fused_moe_experts_dtypes(
            hidden_states.dtype,
            output, hidden_states, w_gate_up, w_down,
            topk_weights, topk_ids, workspace1, workspace2,
        )
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

    def _run_stages(
        self,
        output: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        permuted: tuple,
    ) -> None:
        """The three stages after the permute, shared by both identities."""
        perm_h, true_offsets, true_sizes, _, fwd_idx = permuted
        act = self._gate_up(perm_h, w_gate_up, true_sizes, true_offsets)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        # Unpermute reduces into ``output`` directly and folds
        # ``routed_scaling_factor`` into its prim_func — no separate copy/scale.
        self._unpermute(mm2, fwd_idx, topk_weights, out=output)

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
        num_experts: int,
    ) -> None:
        self._validate_dtypes(
            output, hidden_states, w_gate_up, w_down,
            topk_weights, topk_ids, workspace1, workspace2,
        )
        self._run_stages(
            output, w_gate_up, w_down, topk_weights,
            self._permute(hidden_states, topk_ids),
        )


class FusedMoEExpertsNopadPersistent3WGEpFwdOp(FusedMoEExpertsNopadPersistent3WGFwdOp):
    """The same pipeline sized for the experts one rank owns under expert parallelism.

    ``num_experts_local`` sizes the permute kernel's output buffers and both
    grouped GEMMs, so it is a constructor parameter. ``expert_map`` holds the
    global-to-local ids, read at launch, so it is a ``forward`` input; the permute
    stage rejects a map that does not cover exactly the local ids.

    Args:
        num_tokens: Number of input tokens T (rows of hidden_states).
        num_experts: Total number of experts E in the routing table.
        num_experts_local: Number of those experts this rank owns.
        top_k: Number of experts each token is routed to (K).
        hidden_size: Model hidden dimension H.
        ffn_size: Per-expert FFN intermediate dimension F.
        routed_scaling_factor: Scalar applied to the final reduced output.
        kernel_map: Optional kernel overrides forwarded to the inner Ops.
        activation: Gated activation applied to gate_up.

    Example:
        >>> experts = FusedMoEExpertsNopadPersistent3WGEpFwdOp(
        ...     num_tokens=512, num_experts=256, num_experts_local=128, top_k=8,
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
        self._init_pipeline(
            num_tokens=num_tokens,
            num_experts=num_experts,
            num_experts_local=num_experts_local,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            routed_scaling_factor=routed_scaling_factor,
            kernel_map=kernel_map,
            activation=activation,
        )
        self._permute = MoePermuteNopadEpFwdOp(
            num_experts=num_experts, num_experts_local=num_experts_local,
            kernel_map=kernel_map,
        )

    def _validate_dtypes(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        expert_map: Tensor,
        workspace1: Tensor,
        workspace2: Tensor,
    ) -> None:
        self.dtype = hidden_states.dtype
        _validate_fused_moe_experts_dtypes(
            hidden_states.dtype,
            output, hidden_states, w_gate_up, w_down,
            topk_weights, topk_ids, workspace1, workspace2,
        )
        if expert_map.dtype != torch.int32:
            raise ValueError(
                f"Expected expert_map.dtype == int32, got {expert_map.dtype}")
        self._reject_non_empty_workspaces(workspace1, workspace2)

    def forward(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        expert_map: Tensor,
        workspace1: Tensor,
        workspace2: Tensor,
        num_experts: int,
    ) -> None:
        self._validate_dtypes(
            output, hidden_states, w_gate_up, w_down,
            topk_weights, topk_ids, expert_map, workspace1, workspace2,
        )
        self._run_stages(
            output, w_gate_up, w_down, topk_weights,
            self._permute(hidden_states, topk_ids, expert_map),
        )
