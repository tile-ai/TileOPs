"""Kimi K2 / DeepSeekV3-variant MoE FFN operator.

Supports sigmoid routing with per-expert correction bias and a routed scaling
factor, matching the Kimi K2 model architecture.

Key differences from Qwen3MoENopadOp / Qwen3MoEPaddedOp:
  - Routing: sigmoid (not softmax), renormalize=True
  - correction_bias: per-expert bias added to sigmoid scores for top-k
    selection; output weights remain the original (unbiased) sigmoid scores.
  - routed_scaling_factor: multiplied onto the final output after unpermute.
  - shared_experts_fn: optional callable for a shared (dense) expert whose
    output is added to the routed result (e.g., DeepSeekV3 n_shared_experts=1).
    Kimi K2 passes None.

Two layout variants are provided:
  KimiMoENopadOp  — tight layout (T*K rows), GPU tile scheduler, fastest.
  KimiMoEPaddedOp — block_m-aligned padding, reference / comparison baseline.

Inputs:
  hidden_states  [T, H]       bf16/fp16 token activations
  gating_output  [T, E]       bf16/fp16/fp32 router logits
  correction_bias [E]         float32 per-expert bias (or None)
  w_gate_up      [E, 2*F, H]  bf16/fp16 gate+up projection weights
  w_down         [E, H, F]    bf16/fp16 down projection weights

Output:
  output         [T, H]       same dtype as hidden_states
"""

from typing import Callable, Dict, Optional

import torch

from tileops.kernels.grouped_gemm.grouped_gemm import _DEFAULT_CONFIGS as _GEMM_DEFAULT_CONFIGS
from tileops.kernels.kernel import Kernel
from tileops.ops.elementwise import SiluAndMulOp
from tileops.ops.grouped_gemm import GroupedGemmOp
from tileops.ops.moe.fused_topk import FusedTopKOp
from tileops.ops.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadOp
from tileops.ops.moe.permute_nopad import MoePermuteNopadOp
from tileops.ops.moe.permute_padded import MoePermutePaddedOp
from tileops.ops.moe.unpermute import MoeUnpermuteOp

from ..op import Op

__all__ = ["KimiMoENopadOp", "KimiMoEPaddedOp"]

_BLOCK_M: int = _GEMM_DEFAULT_CONFIGS[(False, True)]["block_m"]


class KimiMoENopadOp(Op):
    """Kimi K2 MoE FFN: sigmoid+bias routing + SwiGLU experts, tight layout.

    Combines FusedTopKOp (sigmoid, correction_bias, renormalize=True) →
    MoePermuteNopadOp → MoeGroupedGemmNopadOp (gate+up) → SiluAndMulOp →
    MoeGroupedGemmNopadOp (down) → MoeUnpermuteOp into a single forward pass.

    Uses tight (non-padded) layout: intermediate tensors have exactly T*K rows.
    GPU tile scheduler maps each CTA to its (expert, row_offset) in O(1).

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Total number of experts E.
        top_k: Experts selected per token K.
        hidden_size: Model hidden dimension H.
        ffn_size: Per-expert intermediate dimension F.
        routed_scaling_factor: Multiplier applied to the routed expert output
            after unpermute (Kimi K2: 2.872; default 1.0 = no scaling).
        dtype: Activation and weight dtype (bf16 or fp16).
        expert_map: Reserved for future multi-GPU EP dispatch. Pass None.

    Example:
        >>> op = KimiMoENopadOp(
        ...     num_tokens=512, num_experts=384, top_k=6,
        ...     hidden_size=7168, ffn_size=2048, routed_scaling_factor=2.872,
        ... )
        >>> output = op(hidden_states, gating_output, correction_bias, w_gate_up, w_down)
        >>> # output: [512, 7168] bf16
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        routed_scaling_factor: float = 1.0,
        with_correction_bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
        self.with_correction_bias = with_correction_bias
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
            scoring_func="sigmoid",
            renormalize=True,
            with_correction_bias=with_correction_bias,
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
        hidden_states: torch.Tensor,                       # [T, H]  bf16/fp16
        gating_output: torch.Tensor,                       # [T, E]  bf16/fp16/fp32
        correction_bias: Optional[torch.Tensor],           # [E]     float32, or None
        w_gate_up: torch.Tensor,                           # [E, 2*F, H]
        w_down: torch.Tensor,                              # [E, H, F]
        shared_experts_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run Kimi K2 MoE FFN.

        Args:
            hidden_states: [T, H] input token activations.
            gating_output: [T, E] router logits.
            correction_bias: [E] float32 per-expert correction bias, or None to
                skip bias (equivalent to all-zeros bias).
            w_gate_up: [E, 2*F, H] gate+up projection weights.
            w_down: [E, H, F] down projection weights.
            shared_experts_fn: Optional callable receiving hidden_states [T, H]
                and returning a [T, H] tensor.  Its output is added to the
                routed result (used for n_shared_experts > 0 models such as
                DeepSeekV3).  Kimi K2 passes None.

        Returns:
            output: [T, H] MoE FFN output, same dtype as hidden_states.
        """
        # Step 1: Routing — sigmoid + correction_bias + top-k + renormalize
        topk_weights, topk_ids = self._fused_topk(gating_output, correction_bias)

        # Step 2: Permute tokens into tight expert-contiguous layout
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
        output = self._unpermute(mm2, fwd_idx, topk_weights)

        # Step 7: Apply routed scaling factor (Kimi K2: 2.872)
        if self.routed_scaling_factor != 1.0:
            output = output * self.routed_scaling_factor

        # Step 8: Add shared expert output (DeepSeekV3 n_shared_experts=1)
        if shared_experts_fn is not None:
            output = output + shared_experts_fn(hidden_states).to(output.dtype)

        return output


class KimiMoEPaddedOp(Op):
    """Kimi K2 MoE FFN with block_m-aligned padded layout (comparison baseline).

    Identical semantics to KimiMoENopadOp but uses MoePermutePaddedOp (padded)
    and GroupedGemmOp (general grouped GEMM) instead of the no-pad variants.

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Total number of experts E.
        top_k: Experts selected per token K.
        hidden_size: Model hidden dimension H.
        ffn_size: Per-expert intermediate dimension F.
        routed_scaling_factor: Multiplier applied to the routed output (default 1.0).
        dtype: Activation and weight dtype (bf16 or fp16).
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        routed_scaling_factor: float = 1.0,
        with_correction_bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
        self.with_correction_bias = with_correction_bias
        self.dtype = dtype
        if expert_map is not None:
            raise NotImplementedError("expert_map is not yet supported.")

        numel = num_tokens * top_k
        _padded_batch_sum = numel + num_experts * _BLOCK_M

        self._fused_topk = FusedTopKOp(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func="sigmoid",
            renormalize=True,
            with_correction_bias=with_correction_bias,
        )
        self._permute = MoePermutePaddedOp(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
            block_m=_BLOCK_M,
        )
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
        hidden_states: torch.Tensor,                       # [T, H]
        gating_output: torch.Tensor,                       # [T, E]
        correction_bias: Optional[torch.Tensor],           # [E] or None
        w_gate_up: torch.Tensor,                           # [E, 2*F, H]
        w_down: torch.Tensor,                              # [E, H, F]
        shared_experts_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run Kimi K2 MoE FFN (padded layout).

        Args:
            hidden_states: [T, H] input activations.
            gating_output: [T, E] router logits.
            correction_bias: [E] float32 per-expert bias, or None.
            w_gate_up: [E, 2*F, H] gate+up weights.
            w_down: [E, H, F] down weights.
            shared_experts_fn: Optional shared expert callable.

        Returns:
            output: [T, H] same dtype as hidden_states.
        """
        # Step 1: routing
        topk_weights, topk_ids = self._fused_topk(gating_output, correction_bias)

        # Step 2: padded permute
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
        output = self._unpermute(mm2_pad, fwd_idx, topk_weights)

        # Step 7: Apply routed scaling factor
        if self.routed_scaling_factor != 1.0:
            output = output * self.routed_scaling_factor

        # Step 8: Add shared expert output
        if shared_experts_fn is not None:
            output = output + shared_experts_fn(hidden_states).to(output.dtype)

        return output
