"""FusedMoEExperts implementation: nopad + 3WG persistent variant."""

from __future__ import annotations

from typing import Dict, Optional

from torch import Tensor

from tileops.kernels.kernel_base import Kernel

from .abc import (
    FusedMoEExpertsModular,
    WeightedReduce,
    WeightedReduceNoOp,
    _validate_fused_moe_experts_dtypes,
)
from .gate_up import MoeGateUpFwdOp
from .moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from .permute_nopad import MoePermuteNopadFwdOp
from .unpermute import MoeUnpermuteFwdOp

__all__ = [
    "FusedMoEExpertsNopadPersistent3WGFwdOp",
]


class FusedMoEExpertsNopadPersistent3WGFwdOp(FusedMoEExpertsModular):
    """Expert GEMM using tight (T*K rows, no-pad) layout with 3WG persistent kernel.

    Internal pipeline: MoePermuteNopadFwdOp → MoeGateUpFwdOp → down GEMM →
    MoeUnpermuteFwdOp (weighted reduction included). Each stage picks its own
    implementation when it is called, from the shapes it was built for and the
    dtype and device of the call.

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
        expert_map: Optional global→local expert id map for expert parallelism
            (EP). Entries < 0 mark experts not owned by this rank.
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
        expert_map: Optional[Tensor] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
    ):
        self.dispatch_kernel(kernel_map)
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        numel = num_tokens * top_k
        num_experts_local = (
            int((expert_map >= 0).sum().item()) if expert_map is not None else num_experts
        )

        self.activation = activation
        self._permute = MoePermuteNopadFwdOp(
            num_experts=num_experts, expert_map=expert_map,
            kernel_map=kernel_map,
        )
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
        self._routed_scaling_factor = routed_scaling_factor

    def _validate_dtypes(
        self,
        output: Tensor,
        hidden_states: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        expert_map: Tensor | None,
        workspace1: Tensor,
        workspace2: Tensor,
    ) -> None:
        # hidden_states is the dtype anchor: the helper requires output,
        # w_gate_up and w_down to agree with it.
        self.dtype = hidden_states.dtype
        _validate_fused_moe_experts_dtypes(
            hidden_states.dtype,
            output, hidden_states, w_gate_up, w_down,
            topk_weights, topk_ids, expert_map, workspace1, workspace2,
        )
        # workspace_shapes() returns ((0,), (0,)) for this implementation; flag
        # callers that pass non-empty workspaces (likely a pipeline mismatch).
        if workspace1.numel() != 0 or workspace2.numel() != 0:
            raise ValueError(
                "workspace1 and workspace2 must be empty (numel == 0) for "
                "FusedMoEExpertsNopadPersistent3WGFwdOp; got "
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
        expert_map: Tensor | None,
        workspace1: Tensor,
        workspace2: Tensor,
        num_experts: int,
    ) -> None:
        self._validate_dtypes(
            output, hidden_states, w_gate_up, w_down,
            topk_weights, topk_ids, expert_map, workspace1, workspace2,
        )
        perm_h, true_offsets, true_sizes, _, fwd_idx = self._permute(hidden_states, topk_ids)
        act = self._gate_up(perm_h, w_gate_up, true_sizes, true_offsets)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        # Unpermute reduces into ``output`` directly and folds
        # ``routed_scaling_factor`` into its prim_func — no separate copy/scale.
        self._unpermute(mm2, fwd_idx, topk_weights, out=output)
