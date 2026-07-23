"""FusedMoEExperts implementation: nopad + 3WG persistent variant."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from tileops.kernels.kernel_base import Kernel

from .abc import (
    FusedMoEExpertsModular,
    WeightedReduce,
    WeightedReduceNoOp,
    _validate_fused_moe_experts_dtypes,
)
from .dispatched_expert import DispatchedExpertMLPFwdOp
from .permute_nopad import MoePermuteNopadFwdOp
from .unpermute import MoeUnpermuteFwdOp

__all__ = [
    "FusedMoEExpertsNopadPersistent3WGFwdOp",
]

class FusedMoEExpertsNopadPersistent3WGFwdOp(FusedMoEExpertsModular):
    """Expert GEMM using tight (T*K rows, no-pad) layout with 3WG persistent kernel.

    Internal pipeline: MoePermuteNopadFwdOp → gate_up GEMM (3WG; activation
    fused into the epilogue when use_fused_activation=True, else a separate
    silu_and_mul/gelu_and_mul step) → down GEMM (3WG) → MoeUnpermuteFwdOp
    (weighted reduction included).

    forward() output shape is (T, H): reduction is done internally by
    MoeUnpermuteFwdOp, so make_weighted_reduce() returns WeightedReduceNoOp.

    Performance note: the 3WG persistent kernel is throughput-tuned for
    prefill-scale workloads; small-batch decode (num_tokens ≲ 512) may run
    a few percent behind tile-scheduler kernels. Decode-heavy deployments
    can pass ``gemm_kernel=MoeGroupedGemmNopadKernel`` to bypass 3WG and
    use the lighter tile-scheduler path explicitly.

    Args:
        num_tokens: Number of input tokens T (rows of hidden_states).
        num_experts: Total number of experts E in the routing table.
        top_k: Number of experts each token is routed to (K).
        hidden_size: Model hidden dimension H (GEMM contraction dim for
            gate_up, output dim for down).
        ffn_size: Per-expert FFN intermediate dimension F.
        routed_scaling_factor: Scalar applied to the final reduced output.
            Defaults to 1.0 (no scaling).
        dtype: Storage dtype for activations and weights. Defaults to bfloat16.
        expert_map: Optional global→local expert id map for expert parallelism
            (EP). Entries < 0 mark experts not owned by this rank.
        gemm_kernel: Optional override for the grouped-GEMM kernel class.
            Defaults to GroupedGemmPersistent3WGKernel; pass
            MoeGroupedGemmNopadKernel to force the tile-scheduler path.
        kernel_map: Optional kernel overrides forwarded to the inner Ops.
        activation: Gated activation applied to gate_up: 'silu_and_mul' or
            'gelu_and_mul'.
        use_fused_activation: If True, fuse the activation into the gate_up
            GEMM epilogue via MoeGroupedGemmPersistent3WGFusedActKernel (avoids
            materializing the [numel, 2*ffn] gate_up in global memory). Falls
            back to the separate activation op (with a logged warning) unless:
            CUDA is available, the device is SM90+, the 3WG kernel is selected,
            activation is silu_and_mul or gelu_and_mul, and ffn_size is a
            multiple of the fused kernel's block_n (128). Default False
            reproduces the unfused pipeline exactly. Performance: at production
            FFN sizes (H200, bf16) this is a small win in both regimes —
            roughly 1.02-1.05x for compute-bound prefill and ~1.05x for
            memory-bound decode — from eliminating the gate_up global
            round-trip; benefits grow as the token count shrinks.

    Example (decode-optimized opt-out):
        from tileops.kernels.moe.moe_grouped_gemm_nopad import (
            MoeGroupedGemmNopadKernel,
        )
        experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=T, num_experts=E, top_k=K,
            hidden_size=H, ffn_size=F,
            gemm_kernel=MoeGroupedGemmNopadKernel,
        )
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        routed_scaling_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[Tensor] = None,
        gemm_kernel: Optional[type] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
        use_fused_activation: bool = False,
    ):
        self.dispatch_kernel(kernel_map)
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        numel = num_tokens * top_k
        num_experts_local = (
            int((expert_map >= 0).sum().item()) if expert_map is not None else num_experts
        )

        self._permute = MoePermuteNopadFwdOp(
            num_experts=num_experts, dtype=dtype, expert_map=expert_map,
            kernel_map=kernel_map,
        )
        self.activation = activation
        self._expert_mlp = DispatchedExpertMLPFwdOp(
            num_pairs=numel,
            num_experts=num_experts_local,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            dtype=dtype,
            gemm_kernel=gemm_kernel,
            kernel_map=kernel_map,
            activation=activation,
            use_fused_activation=use_fused_activation,
        )
        self.use_fused_activation = self._expert_mlp.use_fused_activation
        # Keep these implementation attributes available for callers/tests
        # that inspected the legacy composite op.
        self._gemm_gate_up = self._expert_mlp._gemm_gate_up
        self._activation_op = self._expert_mlp._activation_op
        self._gemm_down = self._expert_mlp._gemm_down
        self._unpermute = MoeUnpermuteFwdOp(
            total_tokens=num_tokens, top_k=top_k,
            hidden_size=hidden_size, dtype=dtype, padded_batch_sum=numel,
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
        _validate_fused_moe_experts_dtypes(
            self.dtype,
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
        mm2 = self._expert_mlp(
            perm_h, w_gate_up, w_down, true_sizes, true_offsets
        )
        # Unpermute reduces into ``output`` directly and folds
        # ``routed_scaling_factor`` into its prim_func — no separate copy/scale.
        self._unpermute(mm2, fwd_idx, topk_weights, out=output)
