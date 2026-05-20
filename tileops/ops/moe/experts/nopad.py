"""FusedMoEExpertsNopadPersistent3WGFwdOp — tight (no-pad) layout expert GEMM with 3WG persistent kernel."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from tileops.kernels.grouped_gemm import GroupedGemmPersistent3WGKernel
from tileops.kernels.grouped_gemm.grouped_gemm_persistent_3wg import (
    _DEFAULT_CONFIG as _3WG_DEFAULT_CONFIG,
)
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel
from tileops.ops.elementwise import SiluAndMulFwdOp
from tileops.ops.moe.abc import FusedMoEExpertsModular, WeightedReduce, WeightedReduceNoOp
from tileops.ops.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from tileops.ops.moe.permute_nopad import MoePermuteNopadFwdOp
from tileops.ops.moe.unpermute import MoeUnpermuteFwdOp

__all__ = ["FusedMoEExpertsNopadPersistent3WGFwdOp"]


class FusedMoEExpertsNopadPersistent3WGFwdOp(FusedMoEExpertsModular):
    """Expert GEMM using tight (T*K rows, no-pad) layout with 3WG persistent kernel.

    Internal pipeline: MoePermuteNopadFwdOp → gate_up GEMM (3WG) → SwiGLU →
    down GEMM (3WG) → MoeUnpermuteFwdOp (weighted reduction included).

    forward() output shape is (T, H): reduction is done internally by
    MoeUnpermuteFwdOp, so make_weighted_reduce() returns WeightedReduceNoOp.
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
        kernel_map=None,
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
        self._silu_and_mul = SiluAndMulFwdOp(M=numel, N=ffn_size, dtype=dtype)
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
        allowed = (torch.float16, torch.bfloat16)
        if self.dtype not in allowed:
            raise ValueError(f"self.dtype must be one of {allowed}, got {self.dtype}")
        for name, t in (
            ("output", output),
            ("hidden_states", hidden_states),
            ("w_gate_up", w_gate_up),
            ("w_down", w_down),
        ):
            if t.dtype != self.dtype:
                raise ValueError(
                    f"Expected {name}.dtype == self.dtype ({self.dtype}), got {t.dtype}"
                )
        if topk_weights.dtype != torch.float32:
            raise ValueError(f"Expected topk_weights.dtype == float32, got {topk_weights.dtype}")
        if topk_ids.dtype != torch.int32:
            raise ValueError(f"Expected topk_ids.dtype == int32, got {topk_ids.dtype}")
        if expert_map is not None and expert_map.dtype != torch.int32:
            raise ValueError(f"Expected expert_map.dtype == int32, got {expert_map.dtype}")
        for name, t in (("workspace1", workspace1), ("workspace2", workspace2)):
            if t.dtype not in allowed:
                raise ValueError(f"Expected {name}.dtype in {allowed}, got {t.dtype}")

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
        perm_h, true_offsets, true_sizes, _, fwd_idx = self._permute(hidden_states, topk_ids)
        gate_up = self._gemm_gate_up(perm_h, w_gate_up, true_sizes, true_offsets)
        act = self._silu_and_mul(gate_up)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        result = self._unpermute(mm2, fwd_idx, topk_weights)
        if self._routed_scaling_factor != 1.0:
            torch.mul(result, self._routed_scaling_factor, out=output)
        else:
            output.copy_(result)
