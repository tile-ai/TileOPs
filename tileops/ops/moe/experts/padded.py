"""MoEExpertsPaddedFwdOp — block_m-aligned padded layout expert GEMM."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from tileops.kernels.grouped_gemm import _DEFAULT_CONFIGS as _GEMM_DEFAULT_CONFIGS
from tileops.ops.elementwise import SiluAndMulFwdOp
from tileops.ops.grouped_gemm import GroupedGemmOp
from tileops.ops.moe.abc import MoEExpertsModular, WeightedReduce, WeightedReduceNoOp
from tileops.ops.moe.permute_padded import MoePermutePaddedFwdOp
from tileops.ops.moe.unpermute import MoeUnpermuteFwdOp

__all__ = ["MoEExpertsPaddedFwdOp"]

_BLOCK_M: int = _GEMM_DEFAULT_CONFIGS[(False, True)]["block_m"]


class MoEExpertsPaddedFwdOp(MoEExpertsModular):
    """Expert GEMM using block_m-aligned padded layout (reference baseline).

    Internal pipeline: MoePermutePaddedFwdOp → gate_up GEMM → SwiGLU →
    down GEMM → MoeUnpermuteFwdOp (weighted reduction included).

    expert_map is not supported; raises NotImplementedError if non-None.
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
        if expert_map is not None:
            raise NotImplementedError(
                "expert_map is not supported for padded layout. "
                "Use MoEExpertsNopadFwdOp for EP mode."
            )
        numel = num_tokens * top_k
        padded_batch_sum = numel + (num_experts * (_BLOCK_M - 1))

        self._permute = MoePermutePaddedFwdOp(
            num_tokens=num_tokens, top_k=top_k, num_experts=num_experts,
            hidden_size=hidden_size, dtype=dtype, block_m=_BLOCK_M,
        )
        self._gemm_gate_up = GroupedGemmOp(
            batch_sum=padded_batch_sum, batch_count=num_experts,
            n=ffn_size * 2, k=hidden_size, dtype=dtype,
        )
        self._silu_and_mul = SiluAndMulFwdOp(M=padded_batch_sum, N=ffn_size, dtype=dtype)
        self._gemm_down = GroupedGemmOp(
            batch_sum=padded_batch_sum, batch_count=num_experts,
            n=hidden_size, k=ffn_size, dtype=dtype,
        )
        self._unpermute = MoeUnpermuteFwdOp(
            num_tokens=num_tokens, top_k=top_k,
            hidden_size=hidden_size, dtype=dtype, padded_batch_sum=padded_batch_sum,
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
        perm_h_pad, padded_offsets, padded_sizes, _, fwd_idx = self._permute(
            hidden_q, topk_ids
        )
        gate_up_pad = self._gemm_gate_up(
            perm_h_pad, w1, padded_sizes, padded_offsets, padded_offsets
        )
        act_pad = self._silu_and_mul(gate_up_pad)
        mm2_pad = self._gemm_down(
            act_pad, w2, padded_sizes, padded_offsets, padded_offsets
        )
        result = self._unpermute(mm2_pad, fwd_idx, topk_weights)
        if self._routed_scaling_factor != 1.0:
            result = result * self._routed_scaling_factor
        output.copy_(result)
