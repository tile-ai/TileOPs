"""MoEExpertsNopadFwdOp — tight (no-pad) layout expert GEMM."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from tileops.kernels.kernel_base import Kernel
from tileops.ops.elementwise import SiluAndMulFwdOp
from tileops.ops.moe.abc import MoEExpertsModular, WeightedReduce, WeightedReduceNoOp
from tileops.ops.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from tileops.ops.moe.permute_nopad import MoePermuteNopadFwdOp
from tileops.ops.moe.unpermute import MoeUnpermuteFwdOp
from tileops.ops.op_base import Op

__all__ = ["MoEExpertsNopadFwdOp"]


class MoEExpertsNopadFwdOp(MoEExpertsModular, Op):
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
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        self._expert_map = expert_map

        self.dispatch_kernel(kernel_map)

        numel = num_tokens * top_k
        num_experts_local = (
            int((expert_map >= 0).sum().item()) if expert_map is not None else num_experts
        )

        self._permute = MoePermuteNopadFwdOp(
            num_tokens=num_tokens, top_k=top_k, num_experts=num_experts,
            hidden_size=hidden_size, dtype=dtype, expert_map=expert_map,
            kernel_map=kernel_map,
        )
        self._gemm_gate_up = MoeGroupedGemmNopadFwdOp(
            numel=numel, num_experts=num_experts_local,
            n=ffn_size * 2, k=hidden_size, dtype=dtype,
            kernel_map=kernel_map,
        )
        self._silu_and_mul = SiluAndMulFwdOp(
            M=numel, N=ffn_size, dtype=dtype, kernel_map=kernel_map,
        )
        self._gemm_down = MoeGroupedGemmNopadFwdOp(
            numel=numel, num_experts=num_experts_local,
            n=hidden_size, k=ffn_size, dtype=dtype,
            kernel_map=kernel_map,
        )
        self._unpermute = MoeUnpermuteFwdOp(
            total_tokens=num_tokens, top_k=top_k,
            hidden_size=hidden_size, dtype=dtype, padded_batch_sum=numel,
            kernel_map=kernel_map,
        )
        self._routed_scaling_factor = routed_scaling_factor

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def forward(
        self,
        hidden_states: Tensor,   # [T, H]
        w_gate_up: Tensor,       # [E, 2F, H]
        w_down: Tensor,          # [E, H, F]
        topk_weights: Tensor,    # [T, K] float32
        topk_ids: Tensor,        # [T, K] int32
    ) -> Tensor:                  # [T, H]
        """Allocating wrapper around apply(): runs the expert pipeline and returns the output.

        Mirrors FusedMoeExpertsFwdOp.forward() semantics so the manifest signature
        (hidden_states, w_gate_up, w_down, topk_weights, topk_ids) -> output is
        satisfied at the validator's Op-class resolution path.
        """
        perm_h, true_offsets, true_sizes, _, fwd_idx = self._permute(
            hidden_states, topk_ids
        )
        gate_up = self._gemm_gate_up(perm_h, w_gate_up, true_sizes, true_offsets)
        act = self._silu_and_mul(gate_up)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        output = self._unpermute(mm2, fwd_idx, topk_weights)
        if self._routed_scaling_factor != 1.0:
            output = output * self._routed_scaling_factor
        return output

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
