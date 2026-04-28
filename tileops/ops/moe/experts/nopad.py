"""MoEExpertsNopadFwdOp — tight (no-pad) layout expert GEMM."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from tileops.ops.elementwise import SiluAndMulFwdOp
from tileops.ops.moe.abc import MoEExpertsModular, WeightedReduce, WeightedReduceNoOp
from tileops.ops.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from tileops.ops.moe.permute_nopad import MoePermuteNopadFwdOp
from tileops.ops.moe.unpermute import MoeUnpermuteFwdOp

__all__ = ["MoEExpertsNopadFwdOp"]


class MoEExpertsNopadFwdOp(MoEExpertsModular):
    """Expert GEMM using tight (T*K rows, no-pad) layout with GPU tile scheduler.

    Internal pipeline: MoePermuteNopadFwdOp → gate_up GEMM → SwiGLU →
    down GEMM → MoeUnpermuteFwdOp (weighted reduction included).

    apply() output shape is (T, H): reduction is done internally by
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
    ):
        numel = num_tokens * top_k
        num_experts_local = (
            int((expert_map >= 0).sum().item()) if expert_map is not None else num_experts
        )

        self._permute = MoePermuteNopadFwdOp(
            num_tokens=num_tokens, top_k=top_k, num_experts=num_experts,
            hidden_size=hidden_size, dtype=dtype, expert_map=expert_map,
        )
        self._gemm_gate_up = MoeGroupedGemmNopadFwdOp(
            numel=numel, num_experts=num_experts_local,
            n=ffn_size * 2, k=hidden_size, dtype=dtype,
        )
        self._silu_and_mul = SiluAndMulFwdOp(M=numel, N=ffn_size, dtype=dtype)
        self._gemm_down = MoeGroupedGemmNopadFwdOp(
            numel=numel, num_experts=num_experts_local,
            n=hidden_size, k=ffn_size, dtype=dtype,
        )
        self._unpermute = MoeUnpermuteFwdOp(
            num_tokens=num_tokens, top_k=top_k,
            hidden_size=hidden_size, dtype=dtype, padded_batch_sum=numel,
        )
        self._routed_scaling_factor = routed_scaling_factor

    def workspace_shapes(
        self, M: int, N: int, K: int, topk: int, num_experts: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return ((0,), (0,))

    def output_shape(self, T_prime: int, H: int) -> tuple[int, int]:
        return (T_prime, H)

    def make_weighted_reduce(self) -> WeightedReduce:
        return WeightedReduceNoOp()

    def apply(
        self,
        output: Tensor,
        hidden_q: Tensor,
        w1: Tensor,
        w2: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        num_experts: int,
        expert_map: Tensor | None,
        workspace1: Tensor,
        workspace2: Tensor,
    ) -> None:
        perm_h, true_offsets, true_sizes, _, fwd_idx = self._permute(hidden_q, topk_ids)
        gate_up = self._gemm_gate_up(perm_h, w1, true_sizes, true_offsets)
        act = self._silu_and_mul(gate_up)
        mm2 = self._gemm_down(act, w2, true_sizes, true_offsets)
        result = self._unpermute(mm2, fwd_idx, topk_weights)
        if self._routed_scaling_factor != 1.0:
            torch.mul(result, self._routed_scaling_factor, out=output)
        else:
            output.copy_(result)
