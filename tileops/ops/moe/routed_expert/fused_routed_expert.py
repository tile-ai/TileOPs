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

    Internal pipeline: MoePermuteNopadFwdOp → gate_up GEMM (3WG) → SwiGLU →
    down GEMM (3WG) → MoeUnpermuteFwdOp (weighted reduction included).

    forward() output shape is (T, H): reduction is done internally by
    MoeUnpermuteFwdOp, so make_weighted_reduce() returns WeightedReduceNoOp.

    Performance note: the 3WG persistent kernel is throughput-tuned for
    prefill-scale workloads; small-batch decode (num_tokens ≲ 512) may run
    a few percent behind tile-scheduler kernels. Decode-heavy deployments
    can pass ``gemm_kernel=MoeGroupedGemmNopadKernel`` to bypass 3WG and
    use the lighter tile-scheduler path explicitly.

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

        self._permute = MoePermuteNopadFwdOp(
            total_tokens=num_tokens, top_k=top_k, num_experts=num_experts,
            hidden_size=hidden_size, dtype=dtype, expert_map=expert_map,
            kernel_map=kernel_map,
        )
        self._gemm_gate_up = MoeGroupedGemmNopadFwdOp(
            numel=numel, num_experts=num_experts_local,
            n=ffn_size * 2, k=hidden_size, dtype=dtype,
            kernel_map={"moe_grouped_gemm_kernel": kernel_cls, **(kernel_map or {})},
        )
        self.activation = activation
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
        # All sub-kernels are owned by the inner Ops (permute / GEMM / SiLU / unpermute).
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
        act = self._activation_op(gate_up)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        result = self._unpermute(mm2, fwd_idx, topk_weights)
        if self._routed_scaling_factor != 1.0:
            torch.mul(result, self._routed_scaling_factor, out=output)
        else:
            output.copy_(result)
