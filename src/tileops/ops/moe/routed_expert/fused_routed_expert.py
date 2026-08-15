"""FusedMoEExperts implementation: nopad + 3WG persistent variant."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from torch import Tensor

from tileops.kernels.grouped_gemm import (
    GroupedGemmPersistent3WGKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel
from tileops.kernels.moe.moe_grouped_gemm_persistent_3wg_fused_act import (
    MoeGroupedGemmPersistent3WGFusedActKernel,
)
from tileops.ops.moe._activation import build_activation_op

from .abc import (
    FusedMoEExpertsModular,
    WeightedReduce,
    WeightedReduceNoOp,
    _validate_fused_moe_experts_dtypes,
)
from .moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from .permute_nopad import MoePermuteNopadFwdOp
from .unpermute import MoeUnpermuteFwdOp

__all__ = [
    "FusedMoEExpertsNopadPersistent3WGFwdOp",
]

_logger = logging.getLogger(__name__)

class FusedMoEExpertsNopadPersistent3WGFwdOp(FusedMoEExpertsModular):
    """Expert GEMM using tight (T*K rows, no-pad) layout with 3WG persistent kernel.

    Internal pipeline: MoePermuteNopadFwdOp → gate_up GEMM (3WG; activation
    fused into the epilogue for sparse routed batches, else a separate
    silu_and_mul/gelu_and_mul step) → shape-selected down GEMM (3WG) →
    MoeUnpermuteFwdOp (weighted reduction included).

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
        expert_map: Optional global→local expert id map for expert parallelism
            (EP). Entries < 0 mark experts not owned by this rank.
        gemm_kernel: Optional override for the grouped-GEMM kernel class.
            Defaults to GroupedGemmPersistent3WGKernel; pass
            MoeGroupedGemmNopadKernel to force the tile-scheduler path.
        kernel_map: Optional kernel overrides forwarded to the inner Ops.
        activation: Gated activation applied to gate_up: 'silu_and_mul' or
            'gelu_and_mul'.
        use_fused_activation: If True, force activation fusion into the gate_up
            GEMM epilogue via MoeGroupedGemmPersistent3WGFusedActKernel (avoids
            materializing the [numel, 2*ffn] gate_up in global memory). Raises
            when this op cannot honour it: the 3WG kernel must be the gate_up
            GEMM with no conflicting moe_grouped_gemm_kernel override,
            activation must be silu_and_mul or gelu_and_mul, and ffn_size must
            be a multiple of the fused kernel's block_n (128). Whether the
            device can run the fused kernel is answered when that kernel is
            built, not here. With the default False, production dispatch still
            enables fusion for aligned sparse routed batches when both the
            average routed rows per local expert and the conservative CTA count
            support the decode-specialized schedule. Explicit GEMM overrides
            disable this automatic selection.

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
        numel = num_tokens * top_k
        num_experts_local = (
            int((expert_map >= 0).sum().item()) if expert_map is not None else num_experts
        )

        kernel_cls = gemm_kernel or GroupedGemmPersistent3WGKernel

        # The 3WG kernel serves only aligned shapes, and says which ones; both
        # GEMMs must be served or neither pipeline half can use it.
        gate_up_n = ffn_size * 2
        if kernel_cls is GroupedGemmPersistent3WGKernel:
            reason = (
                GroupedGemmPersistent3WGKernel.unsupported_reason(
                    numel, num_experts_local, gate_up_n, hidden_size)
                or GroupedGemmPersistent3WGKernel.unsupported_reason(
                    numel, num_experts_local, hidden_size, ffn_size)
            )
            if reason is not None:
                _logger.warning(
                    "FusedMoEExpertsNopadPersistent3WGFwdOp: %s — falling back to "
                    "MoeGroupedGemmNopadKernel.", reason,
                )
                kernel_cls = MoeGroupedGemmNopadKernel

        # A caller can steer the gate_up GEMM either via gemm_kernel (already
        # folded into kernel_cls) or via kernel_map["moe_grouped_gemm_kernel"]
        # (merged into the unfused gate_up / down ops below). The fused gate_up
        # wrapper keys off "moe_grouped_gemm_fused_act_kernel" and cannot honor a
        # "moe_grouped_gemm_kernel" override, so enabling fusion alongside a
        # non-3WG override would produce a fused 3WG gate_up next to an
        # overridden down GEMM. That combination is refused below.
        gemm_override = (kernel_map or {}).get("moe_grouped_gemm_kernel")
        # Whether the fused kernel is the faster gate_up pipeline for this shape
        # is the kernel's own answer: it owns the thresholds and the block sizes.
        auto_fuse = (
            not use_fused_activation
            and gemm_kernel is None
            and gemm_override is None
            and kernel_cls is GroupedGemmPersistent3WGKernel
            and activation in ("silu_and_mul", "gelu_and_mul")
            and MoeGroupedGemmPersistent3WGFusedActKernel.uses_decode_schedule(
                numel, num_experts_local, ffn_size, hidden_size,
            )
        )
        self.use_fused_activation = use_fused_activation or auto_fuse
        if use_fused_activation:
            # Fusion is what the caller asked for, so a request this op cannot
            # honour is refused rather than quietly downgraded: a caller who
            # wanted the fused epilogue and silently got the separate one reads
            # the unfused result as the fused one. Each condition is reported on
            # its own, because "not eligible" over six conjuncts says nothing
            # about which one to fix.
            #
            # Whether the device can run the fused kernel is not asked here: the
            # kernel states the architectures it is built for and refuses when it
            # is built.
            if kernel_cls is not GroupedGemmPersistent3WGKernel:
                raise ValueError(
                    "use_fused_activation=True requires the 3WG persistent gate_up GEMM, "
                    f"got {kernel_cls.__name__}")
            if gemm_override is not None and gemm_override is not GroupedGemmPersistent3WGKernel:
                raise ValueError(
                    "use_fused_activation=True cannot honour a moe_grouped_gemm_kernel "
                    f"override ({gemm_override.__name__}): the fused gate_up wrapper keys off "
                    "moe_grouped_gemm_fused_act_kernel, so the override would reach the down "
                    "GEMM alone and leave the pipeline inconsistent")
            if activation not in ("silu_and_mul", "gelu_and_mul"):
                raise ValueError(
                    "use_fused_activation=True supports activation in "
                    f"{{silu_and_mul, gelu_and_mul}}, got {activation!r}")
            reason = MoeGroupedGemmPersistent3WGFusedActKernel.unsupported_reason(
                numel, num_experts_local, ffn_size, hidden_size,
            )
            if reason is not None:
                raise ValueError(f"use_fused_activation=True: {reason}")

        self._permute = MoePermuteNopadFwdOp(
            num_experts=num_experts, expert_map=expert_map,
            kernel_map=kernel_map,
        )
        self.activation = activation
        if self.use_fused_activation:
            from .moe_grouped_gemm_nopad_fused_act import (
                MoeGroupedGemmNopad3WGFusedActFwdOp,
            )
            self._gemm_gate_up = MoeGroupedGemmNopad3WGFusedActFwdOp(
                numel=numel, num_experts=num_experts_local,
                ffn=ffn_size, k=hidden_size, activation=activation,
                kernel_map=kernel_map,
            )
            self._activation_op = None
        else:
            self._gemm_gate_up = MoeGroupedGemmNopadFwdOp(
                numel=numel, num_experts=num_experts_local,
                n=ffn_size * 2, k=hidden_size,
                kernel_map={"moe_grouped_gemm_kernel": kernel_cls, **(kernel_map or {})},
            )
            self._activation_op = build_activation_op(
                activation, M=numel, N=ffn_size, kernel_map=kernel_map,
            )
        self._gemm_down = MoeGroupedGemmNopadFwdOp(
            numel=numel, num_experts=num_experts_local,
            n=hidden_size, k=ffn_size,
            kernel_map={"moe_grouped_gemm_kernel": kernel_cls, **(kernel_map or {})},
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
        gate_up = self._gemm_gate_up(perm_h, w_gate_up, true_sizes, true_offsets)
        act = gate_up if self.use_fused_activation else self._activation_op(gate_up)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        # Unpermute reduces into ``output`` directly and folds
        # ``routed_scaling_factor`` into its prim_func — no separate copy/scale.
        self._unpermute(mm2, fwd_idx, topk_weights, out=output)
