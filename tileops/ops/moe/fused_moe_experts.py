"""Local routed expert GEMM — tight (no-pad) and padded layout variants.

FusedMoeExpertsFwdOp        — tight layout (T*K rows), GPU tile scheduler, fastest.
FusedMoeExpertsPaddedFwdOp  — block_m-aligned padding, reference / comparison baseline.

Both classes share an identical forward signature:

    forward(hidden_states, w_gate_up, w_down, topk_weights, topk_ids) -> output

Routing (FusedTopKOp) is handled by the caller; these classes only perform:
    permute → GEMM(gate+up) → SwiGLU → GEMM(down) → unpermute

EP note:
    expert_map [global_E] int32: maps global expert ids to local ids (-1 = remote).
    When expert_map is None all experts are treated as local (single-GPU or TP-only).
    When expert_map is provided, non-local token-expert pairs are filtered out during
    permute (fwd_idx = -1) and contribute zero in unpermute.  The weights w_gate_up
    and w_down must be pre-sliced to [E_local, ...] by the caller.
    Multi-GPU All-to-All dispatch is NOT handled here; that belongs to the upper layer.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.grouped_gemm import _DEFAULT_CONFIGS as _GEMM_DEFAULT_CONFIGS
from tileops.kernels.kernel import Kernel
from tileops.ops.elementwise import SiluAndMulOp
from tileops.ops.grouped_gemm import GroupedGemmOp
from tileops.ops.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from tileops.ops.moe.permute_nopad import MoePermuteNopadFwdOp
from tileops.ops.moe.permute_padded import MoePermutePaddedFwdOp
from tileops.ops.moe.unpermute import MoeUnpermuteFwdOp

from ..op import Op

__all__ = ["FusedMoeExpertsFwdOp", "FusedMoeExpertsPaddedFwdOp"]

_BLOCK_M: int = _GEMM_DEFAULT_CONFIGS[(False, True)]["block_m"]


class FusedMoeExpertsFwdOp(Op):
    """Local routed expert GEMM, tight layout (T*K rows, no padding).

    Receives pre-computed routing (topk_weights, topk_ids) from the caller and
    runs permute → GEMM(gate+up) → SwiGLU → GEMM(down) → unpermute.  Uses a
    GPU tile scheduler to map each CTA to its (expert, row_offset) in O(1).

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Total number of experts E (global count).
        top_k: Experts selected per token K.
        hidden_size: Model hidden dimension H.
        ffn_size: Per-expert intermediate dimension F.
        routed_scaling_factor: Scalar multiplied onto the output after unpermute
            (Kimi K2: 2.827; default 1.0 = no scaling).
        dtype: Activation and weight dtype (bf16 or fp16).
        expert_map: Optional [E_global] int32 tensor mapping global expert ids to
            local ids (-1 = not on this rank).  When set, non-local token-expert
            pairs are excluded from GEMM (fwd_idx = -1 → zero contribution in
            unpermute).  The caller must pass w_gate_up/w_down sliced to
            [E_local, ...].  All-to-All communication is NOT performed here.
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
        expert_map: Optional[torch.Tensor] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype

        numel = num_tokens * top_k
        num_experts_local = (
            int((expert_map >= 0).sum().item()) if expert_map is not None else num_experts
        )

        self._permute = MoePermuteNopadFwdOp(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
            expert_map=expert_map,
        )
        self._gemm_gate_up = MoeGroupedGemmNopadFwdOp(
            numel=numel,
            num_experts=num_experts_local,
            n=ffn_size * 2,
            k=hidden_size,
            dtype=dtype,
        )
        self._silu_and_mul = SiluAndMulOp(
            M=numel,
            N=ffn_size,
            dtype=dtype,
        )
        self._gemm_down = MoeGroupedGemmNopadFwdOp(
            numel=numel,
            num_experts=num_experts_local,
            n=hidden_size,
            k=ffn_size,
            dtype=dtype,
        )
        self._unpermute = MoeUnpermuteFwdOp(
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=hidden_size,
            dtype=dtype,
            padded_batch_sum=numel,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]
        w_gate_up: torch.Tensor,        # [E, 2*F, H]
        w_down: torch.Tensor,           # [E, H, F]
        topk_weights: torch.Tensor,     # [T, K] float32
        topk_ids: torch.Tensor,         # [T, K] int32 (global expert ids)
    ) -> torch.Tensor:                  # [T, H]
        """Run local routed expert GEMM (tight layout).

        Args:
            hidden_states: [T, H] input token activations.
            w_gate_up: [E, 2*F, H] gate+up projection weights.
            w_down: [E, H, F] down projection weights.
            topk_weights: [T, K] float32 routing weights (pre-normalized).
            topk_ids: [T, K] int32 selected expert indices (global ids).

        Returns:
            output: [T, H] same dtype as hidden_states.
        """
        perm_h, true_offsets, true_sizes, _, fwd_idx = self._permute(
            hidden_states, topk_ids
        )
        gate_up = self._gemm_gate_up(perm_h, w_gate_up, true_sizes, true_offsets)
        act = self._silu_and_mul(gate_up)
        mm2 = self._gemm_down(act, w_down, true_sizes, true_offsets)
        output = self._unpermute(mm2, fwd_idx, topk_weights)

        if self.routed_scaling_factor != 1.0:
            output = output * self.routed_scaling_factor

        return output


class FusedMoeExpertsPaddedFwdOp(Op):
    """Local routed expert GEMM, block_m-aligned padded layout.

    Identical semantics to FusedMoeExpertsFwdOp but uses MoePermutePaddedFwdOp and
    GroupedGemmOp instead of the no-pad variants.  Used as a comparison
    baseline to quantify the benefit of the tight (no-pad) layout.

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Total number of experts E (global count).
        top_k: Experts selected per token K.
        hidden_size: Model hidden dimension H.
        ffn_size: Per-expert intermediate dimension F.
        routed_scaling_factor: Scalar multiplied onto the output (default 1.0).
        dtype: Activation and weight dtype (bf16 or fp16).
        expert_map: Optional [E_global] int32 EP mapping.  Not yet implemented
            for the padded layout; raises NotImplementedError if non-None.
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
        expert_map: Optional[torch.Tensor] = None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype
        if expert_map is not None:
            raise NotImplementedError(
                "expert_map is not yet supported for the padded layout. "
                "Use FusedMoeExpertsFwdOp (nopad) for EP mode."
            )

        numel = num_tokens * top_k
        _padded_batch_sum = numel + (num_experts * (_BLOCK_M - 1))

        self._permute = MoePermutePaddedFwdOp(
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
        self._unpermute = MoeUnpermuteFwdOp(
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
        hidden_states: torch.Tensor,   # [T, H]
        w_gate_up: torch.Tensor,        # [E, 2*F, H]
        w_down: torch.Tensor,           # [E, H, F]
        topk_weights: torch.Tensor,     # [T, K] float32
        topk_ids: torch.Tensor,         # [T, K] int32 (global expert ids)
    ) -> torch.Tensor:                  # [T, H]
        """Run local routed expert GEMM (padded layout).

        Args:
            hidden_states: [T, H] input token activations.
            w_gate_up: [E, 2*F, H] gate+up projection weights.
            w_down: [E, H, F] down projection weights.
            topk_weights: [T, K] float32 routing weights.
            topk_ids: [T, K] int32 selected expert indices.

        Returns:
            output: [T, H] same dtype as hidden_states.
        """
        perm_h_pad, padded_offsets, padded_sizes, _, fwd_idx = self._permute(
            hidden_states, topk_ids
        )
        gate_up_pad = self._gemm_gate_up(
            perm_h_pad, w_gate_up, padded_sizes, padded_offsets, padded_offsets
        )
        act_pad = self._silu_and_mul(gate_up_pad)
        mm2_pad = self._gemm_down(
            act_pad, w_down, padded_sizes, padded_offsets, padded_offsets
        )
        output = self._unpermute(mm2_pad, fwd_idx, topk_weights)

        if self.routed_scaling_factor != 1.0:
            output = output * self.routed_scaling_factor

        return output
