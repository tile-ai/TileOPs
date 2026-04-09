"""MoE grouped GEMM op (no-pad variant): NT GEMM with precomputed tile scheduling."""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel

from ..op import Op

__all__ = ["MoeGroupedGemmNopadFwdOp"]


class MoeGroupedGemmNopadFwdOp(Op):
    """NT grouped GEMM for MoE without block_m-aligned padding.

    Uses a GPU tile scheduler to map each CTA to its (expert, row_offset) in O(1),
    eliminating the O(E) per-CTA expert scan in standard grouped GEMM.

    Accepts tight A[T*K, K] inputs (no padding between experts) from
    MoePermuteNoPadOp, producing tight C[T*K, N] outputs.

    Args:
        numel: T * top_k, total (token, expert) pairs = tight row count.
        num_experts: Total number of experts E.
        n: Output feature dimension N (e.g. 2*ffn_size or hidden_size).
        k: Input feature dimension K (hidden_size or ffn_size).
        dtype: Activation and weight dtype (bf16 or fp16).
        kernel_map: Optional kernel override dict.
        tune: Whether to autotune.

    Example:
        >>> op = MoeGroupedGemmNopadFwdOp(numel=16384, num_experts=256, n=4096, k=2048,
        ...                       dtype=torch.bfloat16)
        >>> C = op(A, B, true_sizes, true_offsets)  # [numel, N]
    """

    def __init__(
        self,
        numel: int,
        num_experts: int,
        n: int,
        k: int,
        dtype: torch.dtype = torch.bfloat16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.numel = numel
        self.num_experts = num_experts
        self.N = n
        self.K = k
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["moe_grouped_gemm_kernel"](
            numel, num_experts, n, k, dtype=dtype, tune=tune
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"moe_grouped_gemm_kernel": MoeGroupedGemmNopadKernel}

    def forward(
        self,
        a: torch.Tensor,           # [numel, K]
        b: torch.Tensor,           # [num_experts, N, K]
        true_sizes: torch.Tensor,  # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run tile-scheduled NT GEMM.

        Args:
            a: [numel, K] tight permuted activations.
            b: [num_experts, N, K] expert weights (NT: B^T applied).
            true_sizes: [E] int32 true token count per expert.
            true_offsets: [E] int32 tight start offset per expert in a.

        Returns:
            C: [numel, N] GEMM output.
        """
        return self.kernel(a, b, true_sizes, true_offsets)
