"""Qwen3-style MoE FFN operator — padded layout variant (for comparison).

Uses block_m-aligned padding between experts:
  _padded_batch_sum = T*K + E*block_m

Pipeline:
  FusedTopKOp → MoePermutePaddedOp (padded) → GroupedGemmOp (gate+up) →
  SiluAndMulOp → GroupedGemmOp (down) → MoeUnpermuteOp

This is the reference implementation used to benchmark against the
no-padding variant (Qwen3MoEOp).  The padding inflates intermediate
tensors by ~2× for E=256, costing extra memory bandwidth in SiluAndMul
and the unpermute step.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.grouped_gemm.grouped_gemm import _DEFAULT_CONFIGS as _GEMM_DEFAULT_CONFIGS
from tileops.kernels.kernel import Kernel
from tileops.ops.elementwise import SiluAndMulOp
from tileops.ops.grouped_gemm import GroupedGemmOp
from tileops.ops.moe.fused_topk import FusedTopKOp
from tileops.ops.moe.permute_padded import MoePermutePaddedOp
from tileops.ops.moe.unpermute import MoeUnpermuteOp

from ..op import Op

__all__ = ["Qwen3MoEPaddedOp"]

_BLOCK_M: int = _GEMM_DEFAULT_CONFIGS[(False, True)]["block_m"]


class Qwen3MoEPaddedOp(Op):
    """Qwen3-style MoE FFN with block_m-aligned padded layout.

    Identical semantics to Qwen3MoEOp but uses MoePermutePaddedOp (padded) and
    GroupedGemmOp (general grouped GEMM) instead of the no-pad variants.

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Total number of experts E.
        top_k: Experts selected per token K.
        hidden_size: Model hidden dimension H.
        ffn_size: Per-expert intermediate dimension F.
        scoring_func: "softmax" or "sigmoid".
        renormalize: Renormalize top-k weights to sum to 1.
        dtype: Activation and weight dtype (bf16 or fp16).
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.dtype = dtype
        if expert_map is not None:
            raise NotImplementedError("expert_map is not yet supported.")

        # Padded batch sum: T*K + E*block_m (conservative upper bound)
        numel = num_tokens * top_k
        _padded_batch_sum = numel + num_experts * _BLOCK_M

        self._fused_topk = FusedTopKOp(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
        )
        self._permute = MoePermutePaddedOp(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
            block_m=_BLOCK_M,
        )
        # NT: A[padded, H] @ B[E, 2*F, H]^T → C[padded, 2*F]
        self._gemm_gate_up = GroupedGemmOp(
            batch_sum=_padded_batch_sum,
            batch_count=num_experts,
            n=ffn_size * 2,
            k=hidden_size,
            dtype=dtype,
        )
        self._silu_and_mul = SiluAndMulOp(
            M=_padded_batch_sum,
            N=ffn_size,
            dtype=dtype,
        )
        # NT: A[padded, F] @ B[E, H, F]^T → C[padded, H]
        self._gemm_down = GroupedGemmOp(
            batch_sum=_padded_batch_sum,
            batch_count=num_experts,
            n=hidden_size,
            k=ffn_size,
            dtype=dtype,
        )
        self._unpermute = MoeUnpermuteOp(
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=hidden_size,
            dtype=dtype,
            padded_batch_sum=_padded_batch_sum,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T, H]
        gating_output: torch.Tensor,  # [T, E]
        w_gate_up: torch.Tensor,      # [E, 2*F, H]
        w_down: torch.Tensor,         # [E, H, F]
    ) -> torch.Tensor:
        # Step 1: routing
        topk_weights, topk_ids = self._fused_topk(gating_output)

        # Step 2: padded permute
        #   perm_h_pad:    [padded, H]
        #   padded_offsets: [E] block_m-aligned start per expert
        #   padded_sizes:   [E] block_m-aligned size per expert
        #   fwd_idx:        [T*K] flat_idx → padded slot
        perm_h_pad, padded_offsets, padded_sizes, _, fwd_idx = self._permute(
            hidden_states, topk_ids
        )

        # Step 3: gate+up GEMM (padded layout)
        gate_up_pad = self._gemm_gate_up(
            perm_h_pad, w_gate_up, padded_sizes, padded_offsets, padded_offsets
        )

        # Step 4: SwiGLU
        act_pad = self._silu_and_mul(gate_up_pad)

        # Step 5: down GEMM (padded layout)
        mm2_pad = self._gemm_down(
            act_pad, w_down, padded_sizes, padded_offsets, padded_offsets
        )

        # Step 6: weighted scatter back to token order
        return self._unpermute(mm2_pad, fwd_idx, topk_weights)
