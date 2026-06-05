"""MoE gate_up grouped GEMM with fused activation (no-pad, 3WG)."""
from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import MoeGroupedGemmPersistent3WGFusedActKernel

from ...op_base import Op

__all__ = ["MoeGroupedGemmNopad3WGFusedActFwdOp"]


class MoeGroupedGemmNopad3WGFusedActFwdOp(Op):
    """A[numel,K] @ B[E,2*ffn,K]^T -> act(gate)*up -> C[numel,ffn].

    Wraps MoeGroupedGemmPersistent3WGFusedActKernel: the gate (first ffn rows
    of each expert weight) and up (next ffn rows) projections are computed and
    fused with the activation in the GEMM epilogue, so the [numel, 2*ffn]
    gate_up intermediate never reaches global memory.

    Args:
        numel: T * top_k tight row count.
        num_experts: Number of local experts E.
        ffn: FFN width (output column count); B has 2*ffn rows (gate||up).
        k: Hidden size K.
        dtype: Activation/weight dtype (bf16 or fp16).
        activation: "silu_and_mul" or "gelu_and_mul".
        kernel_map: Optional kernel override.
        tune: Autotune flag.

    Example:
        >>> op = MoeGroupedGemmNopad3WGFusedActFwdOp(
        ...     numel=16384, num_experts=256, ffn=768, k=2048,
        ...     dtype=torch.bfloat16, activation="silu_and_mul")
        >>> C = op(A, B, true_sizes, true_offsets)  # [numel, ffn]
    """

    def __init__(
        self,
        numel: int,
        num_experts: int,
        ffn: int,
        k: int,
        dtype: torch.dtype = torch.bfloat16,
        activation: str = "silu_and_mul",
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.numel = numel
        self.num_experts = num_experts
        self.ffn = ffn
        self.k = k
        self.dtype = dtype
        self.activation = activation
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["moe_grouped_gemm_fused_act_kernel"](
            numel, num_experts, ffn, k, dtype=dtype, activation=activation, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"moe_grouped_gemm_fused_act_kernel": MoeGroupedGemmPersistent3WGFusedActKernel}

    def forward(
        self,
        a: torch.Tensor,            # [numel, K]
        b: torch.Tensor,            # [num_experts, 2*ffn, K]
        true_sizes: torch.Tensor,   # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run the fused gate_up GEMM + activation.

        Args:
            a: [numel, K] tight permuted activations.
            b: [num_experts, 2*ffn, K] gate||up expert weights.
            true_sizes: [E] int32 token count per expert.
            true_offsets: [E] int32 tight start offset per expert in a.

        Returns:
            C: [numel, ffn] activated gate_up output.
        """
        return self.kernel(a, b, true_sizes, true_offsets)
