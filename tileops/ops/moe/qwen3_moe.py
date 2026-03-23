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

Padded layout note:
  The NT grouped-GEMM kernel maps each M-tile to an expert by checking
  whether the tile's row-start falls within that expert's row range.
  For correctness, every expert's token block must start on a block_m
  boundary.  MoePermuteOp computes block_m-aligned padded_sizes and
  padded_offsets on the GPU and scatters tokens into perm_h_pad.
  The fwd_idx mapping (flat_idx → padded slot) is passed directly to
  MoeUnpermuteOp to gather the GEMM outputs back to token order.

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
from tileops.ops.grouped_gemm import GroupedGemmOp
from tileops.ops.moe.fused_topk import FusedTopKOp
from tileops.ops.moe.permute import MoePermuteOp
from tileops.ops.moe.unpermute import MoeUnpermuteOp

from ..op import Op

__all__ = ["Qwen3MoEOp"]

# NT kernel default block_m (grouped_gemm.py: (False, True): {"block_m": 64, ...}).
# Each expert's row slice must start on this boundary for correct tile→expert mapping.
_BLOCK_M_NT = 64


class Qwen3MoEOp(Op):
    """Qwen3-style MoE FFN: softmax/sigmoid routing + SwiGLU experts.

    Combines FusedTopKOp → MoePermuteOp → GroupedGemmOp (gate+up) →
    SiluAndMulOp → GroupedGemmOp (down) → MoeUnpermuteOp into a single
    forward pass.  All kernels are TileLang-compiled; no CPU loops or
    host–device syncs in the forward path.

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
            local.  Stored for future multi-GPU EP support; All-to-All
            dispatch is not yet implemented.
        kernel_map: Optional kernel override dict.
        config: Optional sub-op config overrides.

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
        kernel_map: Optional[Dict[str, Kernel]] = None,
        config: Optional[dict] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.dtype = dtype
        # Stored for future EP dispatch; not yet applied in kernel path.
        self.expert_map = expert_map

        batch_sum = num_tokens * top_k
        # Padded batch sum: upper bound for the block_m-aligned layout.
        # actual_padded = Σ ⌈count_i/block_m⌉×block_m ≤ batch_sum + E×block_m.
        self._padded_batch_sum = batch_sum + num_experts * _BLOCK_M_NT

        self._fused_topk = FusedTopKOp(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
        )
        self._permute = MoePermuteOp(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
            block_m=_BLOCK_M_NT,
        )
        # NT: A[padded_batch_sum, H] @ B[E, 2*F, H]^T → C[padded_batch_sum, 2*F]
        self._gemm_gate_up = GroupedGemmOp(
            batch_sum=self._padded_batch_sum,
            batch_count=num_experts,
            n=ffn_size * 2,
            k=hidden_size,
            dtype=dtype,
            transpose_a=False,
            transpose_b=True,
        )
        # silu(gate) * up: [padded_batch_sum, 2*F] → [padded_batch_sum, F]
        self._silu_and_mul = SiluAndMulOp(
            M=self._padded_batch_sum,
            N=ffn_size,
            dtype=dtype,
        )
        # NT: A[padded_batch_sum, F] @ B[E, H, F]^T → C[padded_batch_sum, H]
        self._gemm_down = GroupedGemmOp(
            batch_sum=self._padded_batch_sum,
            batch_count=num_experts,
            n=hidden_size,
            k=ffn_size,
            dtype=dtype,
            transpose_a=False,
            transpose_b=True,
        )
        self._unpermute = MoeUnpermuteOp(
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=hidden_size,
            dtype=dtype,
            padded_batch_sum=self._padded_batch_sum,
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

        # Step 2: Permute tokens into padded expert-contiguous layout.
        #   perm_h_pad:     [padded_batch_sum, H]
        #   padded_offsets: [E] int32  padded start per expert
        #   padded_sizes:   [E] int32  block_m-aligned sizes
        #   expert_offset:  [E+1] int64  non-padded prefix-sum (unused here)
        #   fwd_idx:        [T*K] int32  flat_idx → padded slot
        perm_h_pad, padded_offsets, padded_sizes, _, fwd_idx = self._permute(
            hidden_states, topk_ids
        )

        # Step 3: Gate+up projection — [padded, H] × [E, 2*F, H]^T → [padded, 2*F]
        # Both batch_offsets and batch_padded_offsets are padded_offsets: the NT kernel
        # uses batch_offsets to locate expert boundaries (m_start), and the padded layout
        # already aligns those boundaries to block_m, so both arguments are identical.
        gate_up_pad = self._gemm_gate_up(
            perm_h_pad, w_gate_up, padded_sizes, padded_offsets, padded_offsets
        )

        # Step 4: SwiGLU — silu(gate) * up → [padded, F]
        act_pad = self._silu_and_mul(gate_up_pad)

        # Step 5: Down projection — [padded, F] × [E, H, F]^T → [padded, H]
        mm2_pad = self._gemm_down(
            act_pad, w_down, padded_sizes, padded_offsets, padded_offsets
        )

        # Step 6: Weighted sum back to token order → [T, H]
        return self._unpermute(mm2_pad, fwd_idx, topk_weights)
