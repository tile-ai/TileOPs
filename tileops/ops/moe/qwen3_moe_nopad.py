"""Qwen3-style MoE FFN operator.

Fuses softmax routing, token permutation, grouped expert GEMM
(gate+up), SwiGLU activation, down projection, and inverse permutation
into a single forward pass using existing TileOPs kernels.

Supported models: Qwen3-MoE, Qwen2-MoE, Mixtral (softmax routing,
SwiGLU experts).

Inputs:
  hidden_states  [T, H]       bf16/fp16 token activations
  gating_output  [T, E]       bf16/fp16/fp32 router logits
  w_gate_up      [E, 2*F, H]  bf16/fp16 gate+up projection weights
  w_down         [E, H, F]    bf16/fp16 down projection weights

Output:
  output         [T, H]       same dtype as hidden_states

Tight layout note:
  MoePermuteNopadOp outputs perm_h with exactly T*K rows — no block_m-aligned
  padding between experts. MoeGroupedGemmNopadOp uses a GPU tile scheduler to map
  each CTA to its (expert, row_offset) in O(1) via a precomputed table.
  fwd_idx[flat_idx] = tight_slot is passed to MoeUnpermuteOp to gather GEMM
  outputs back to token order.

EP note:
  expert_map [E] int32 maps global expert ids to local ids (-1 = remote).
  When expert_map is None all experts are treated as local (single-GPU
  or TP-only mode).  Multi-GPU EP dispatch (All-to-All) is not yet
  implemented; expert_map is stored for future use.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.ops.elementwise import SiluAndMulOp
from tileops.ops.moe.fused_topk import FusedTopKOp
from tileops.ops.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadOp
from tileops.ops.moe.permute_nopad import MoePermuteNopadOp
from tileops.ops.moe.unpermute import MoeUnpermuteOp

from ..op import Op

__all__ = ["Qwen3MoENopadOp"]


class Qwen3MoENopadOp(Op):
    """Qwen3-style MoE FFN: softmax/sigmoid routing + SwiGLU experts.

    Combines FusedTopKOp → MoePermuteNopadOp → MoeGroupedGemmNopadOp (gate+up) →
    SiluAndMulOp → MoeGroupedGemmNopadOp (down) → MoeUnpermuteOp into a single
    forward pass.  All kernels are TileLang-compiled; no CPU loops or
    host–device syncs in the forward path.

    Uses tight (non-padded) layout: intermediate tensors have exactly T*K rows,
    eliminating the conservative T*K + E*block_m allocation overhead.

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Total number of experts E (global count).
        top_k: Experts selected per token K.
        hidden_size: Model hidden dimension H.
        ffn_size: Per-expert intermediate dimension F.
        scoring_func: "softmax" (Qwen3/Qwen2) or "sigmoid" (DeepSeek-V3).
        renormalize: Renormalize top-k weights to sum to 1.
        dtype: Activation and weight dtype (bf16 or fp16).
        expert_map: Optional [E] int32 tensor mapping global expert ids to
            local ids (-1 = not on this rank).  None means all experts are
            local.  Multi-GPU EP dispatch (All-to-All) is not yet implemented;
            passing a non-None value raises NotImplementedError.

    Example:
        >>> op = Qwen3MoEOp(num_tokens=512, num_experts=128, top_k=8,
        ...                  hidden_size=2048, ffn_size=1024)
        >>> output = op(hidden_states, gating_output, w_gate_up, w_down)
        >>> # output: [512, 2048] bf16
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
            raise NotImplementedError(
                "expert_map is reserved for future multi-GPU EP dispatch "
                "(All-to-All) which is not yet implemented. Pass None for "
                "single-GPU or TP-only usage."
            )

        # Tight batch sum: exactly T*K rows, no block_m padding
        numel = num_tokens * top_k

        self._fused_topk = FusedTopKOp(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
        )
        self._permute = MoePermuteNopadOp(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        # NT: A[numel, H] @ B[E, 2*F, H]^T → C[numel, 2*F]
        self._gemm_gate_up = MoeGroupedGemmNopadOp(
            numel=numel,
            num_experts=num_experts,
            n=ffn_size * 2,
            k=hidden_size,
            dtype=dtype,
        )
        # silu(gate) * up: [numel, 2*F] → [numel, F]
        self._silu_and_mul = SiluAndMulOp(
            M=numel,
            N=ffn_size,
            dtype=dtype,
        )
        # NT: A[numel, F] @ B[E, H, F]^T → C[numel, H]
        self._gemm_down = MoeGroupedGemmNopadOp(
            numel=numel,
            num_experts=num_experts,
            n=hidden_size,
            k=ffn_size,
            dtype=dtype,
        )
        self._unpermute = MoeUnpermuteOp(
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=hidden_size,
            dtype=dtype,
            padded_batch_sum=numel,  # tight: no padding
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T, H]  bf16/fp16
        gating_output: torch.Tensor,  # [T, E]  bf16/fp16/fp32
        w_gate_up: torch.Tensor,      # [E, 2*F, H]  bf16/fp16
        w_down: torch.Tensor,         # [E, H, F]    bf16/fp16
    ) -> torch.Tensor:
        """Run Qwen3 MoE FFN.

        Args:
            hidden_states: [T, H] input token activations.
            gating_output: [T, E] router logits.
            w_gate_up: [E, 2*F, H] gate+up projection weights.
            w_down: [E, H, F] down projection weights.

        Returns:
            output: [T, H] MoE FFN output, same dtype as hidden_states.
        """
        # Step 1: Routing — softmax/sigmoid + top-k selection
        topk_weights, topk_ids = self._fused_topk(gating_output)

        # Step 2: Permute tokens into tight expert-contiguous layout.
        #   perm_h:       [T*K, H]  tight hidden states (no inter-expert gaps)
        #   true_offsets: [E] int32  tight start per expert
        #   true_sizes:   [E] int32  true token count per expert
        #   fwd_idx:      [T*K] int32  flat_idx → tight slot
        perm_h, true_offsets, true_sizes, _, fwd_idx = self._permute(
            hidden_states, topk_ids
        )

        # Step 3: Gate+up projection — [tight, H] × [E, 2*F, H]^T → [tight, 2*F]
        gate_up = self._gemm_gate_up(perm_h, w_gate_up, true_sizes, true_offsets)

        # Step 4: SwiGLU — silu(gate) * up → [tight, F]
        act = self._silu_and_mul(gate_up)

        # Step 5: Down projection — [tight, F] × [E, H, F]^T → [tight, H]
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)

        # Step 6: Weighted sum back to token order → [T, H]
        return self._unpermute(mm2, fwd_idx, topk_weights)
