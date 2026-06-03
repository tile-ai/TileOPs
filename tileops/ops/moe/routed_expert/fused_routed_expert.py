"""FusedMoEExperts implementation: nopad + 3WG persistent variant."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import Tensor

from tileops.kernels.grouped_gemm import (
    GroupedGemmPersistent3WGKernel,
)
from tileops.kernels.grouped_gemm.grouped_gemm_persistent_3wg import (
    _DEFAULT_CONFIG as _3WG_DEFAULT_CONFIG,
)
# Imported unconditionally: the eligibility check reads its block_n even when
# use_fused_activation ends up False. The wrapper class itself is deferred to
# the fused branch (imported lazily in __init__).
from tileops.kernels.moe.moe_grouped_gemm_persistent_3wg_fused_act import (
    _DEFAULT_CONFIG as _FUSED_ACT_DEFAULT_CONFIG,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel
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
            GEMM epilogue via MoeGroupedGemmNopad3WGFusedActKernel (avoids
            materializing the [numel, 2*ffn] gate_up in global memory). Falls
            back to the separate activation op (with a logged warning) unless:
            CUDA is available, the device is SM90+, the 3WG kernel is selected,
            activation is silu_and_mul or gelu_and_mul, and ffn_size is a
            multiple of the fused kernel's block_n (128). Default False
            reproduces the unfused pipeline exactly. Performance: at production
            FFN sizes this is roughly break-even for compute-bound prefill (the
            dual-B epilogue is constrained to block_n=128, offsetting the
            eliminated gate_up round-trip) and a small (~3%) win for
            memory-bound decode; benefits grow as the token count shrinks.
            Default False is recommended for prefill-heavy serving.

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

        kernel_cls = gemm_kernel or GroupedGemmPersistent3WGKernel

        # 3WG requires N and K aligned to its default block dimensions.
        # Fall back to tile scheduler kernel for small/unaligned dimensions.
        _3wg_block_n = _3WG_DEFAULT_CONFIG["block_n"]
        _3wg_block_k = _3WG_DEFAULT_CONFIG["block_k"]
        gate_up_n = ffn_size * 2
        if kernel_cls is GroupedGemmPersistent3WGKernel:
            gate_up_ok = (gate_up_n % _3wg_block_n == 0) and (hidden_size % _3wg_block_k == 0)
            down_ok = (hidden_size % _3wg_block_n == 0) and (ffn_size % _3wg_block_k == 0)
            if not (gate_up_ok and down_ok):
                _logger.warning(
                    "FusedMoEExpertsNopadPersistent3WGFwdOp: dims not aligned "
                    "to 3WG block (gate_up_n=%d, hidden_size=%d, ffn_size=%d; "
                    "block_n=%d, block_k=%d) — falling back to "
                    "MoeGroupedGemmNopadKernel.",
                    gate_up_n, hidden_size, ffn_size, _3wg_block_n, _3wg_block_k,
                )
                kernel_cls = MoeGroupedGemmNopadKernel

        self.use_fused_activation = use_fused_activation
        if use_fused_activation:
            fused_block_n = _FUSED_ACT_DEFAULT_CONFIG["block_n"]
            ok = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 9
                and kernel_cls is GroupedGemmPersistent3WGKernel
                and activation in ("silu_and_mul", "gelu_and_mul")
                and (ffn_size % fused_block_n == 0)
            )
            if not ok:
                _logger.warning(
                    "use_fused_activation=True not eligible (requires CUDA + SM90 + "
                    "GroupedGemmPersistent3WGKernel + activation in {silu_and_mul, "
                    "gelu_and_mul} + ffn_size %% %d == 0); falling back to unfused "
                    "activation. ffn_size=%d, activation=%s.",
                    fused_block_n, ffn_size, activation,
                )
                self.use_fused_activation = False

        self._permute = MoePermuteNopadFwdOp(
            total_tokens=num_tokens, top_k=top_k, num_experts=num_experts,
            hidden_size=hidden_size, dtype=dtype, expert_map=expert_map,
            kernel_map=kernel_map,
        )
        self.activation = activation
        if self.use_fused_activation:
            from .moe_grouped_gemm_nopad_fused_act import (
                MoeGroupedGemmNopad3WGFusedActFwdOp,
            )
            self._gemm_gate_up = MoeGroupedGemmNopad3WGFusedActFwdOp(
                numel=numel, num_experts=num_experts_local,
                ffn=ffn_size, k=hidden_size, dtype=dtype, activation=activation,
                kernel_map=kernel_map,
            )
            self._activation_op = None
        else:
            self._gemm_gate_up = MoeGroupedGemmNopadFwdOp(
                numel=numel, num_experts=num_experts_local,
                n=ffn_size * 2, k=hidden_size, dtype=dtype,
                kernel_map={"moe_grouped_gemm_kernel": kernel_cls, **(kernel_map or {})},
            )
            self._activation_op = build_activation_op(
                activation, M=numel, N=ffn_size, dtype=dtype, kernel_map=kernel_map,
            )
        self._gemm_down = MoeGroupedGemmNopadFwdOp(
            numel=numel, num_experts=num_experts_local,
            n=hidden_size, k=ffn_size, dtype=dtype,
            kernel_map={"moe_grouped_gemm_kernel": kernel_cls, **(kernel_map or {})},
        )
        self._unpermute = MoeUnpermuteFwdOp(
            total_tokens=num_tokens, top_k=top_k,
            hidden_size=hidden_size, dtype=dtype, padded_batch_sum=numel,
            kernel_map=kernel_map,
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
        gate_up = self._gemm_gate_up(perm_h, w_gate_up, true_sizes, true_offsets)
        act = gate_up if self.use_fused_activation else self._activation_op(gate_up)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        result = self._unpermute(mm2, fwd_idx, topk_weights)
        if self._routed_scaling_factor != 1.0:
            torch.mul(result, self._routed_scaling_factor, out=output)
        else:
            output.copy_(result)
