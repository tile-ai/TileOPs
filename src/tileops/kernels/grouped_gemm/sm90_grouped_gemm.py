"""The SM90 GEMM template behind ``GroupedGemmFwdOp``'s four layouts."""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.sm90_gemm import GemmType, SM90GemmFwdKernel

__all__ = ["SM90GroupedGemmKernel"]


class SM90GroupedGemmKernel(Kernel):
    """``GroupedGemmFwdOp``'s NT / NN / TN / TT on the SM90 GEMM template.

    An adapter: the op builds a kernel as ``cls(batch_sum, batch_count, n, k, ...)``
    and calls it with the tight group tables, while the template speaks GEMM
    types and psum metadata. The two M-grouped layouts (NT, NN) run
    ``M_GROUPED_TIGHT_PSUM`` on ``batch_offsets + batch_sizes``; an NN ``b`` is
    the transposed view of its ``[E, K, N]`` storage. The two K-grouped layouts
    (TN, TT) run ``K_GROUPED_CONTIGUOUS`` on ``batch_sizes`` over the transposed
    views of ``a`` and ``b``, so a TT ``b`` keeps its ``[K, batch_sum]`` storage
    K-major. ``batch_padded_offsets`` is not read: the template pads nothing.

    Claims bf16 or fp16 operands whose TMA-addressed extents are multiples of 8:
    every contiguous operand extent and the output row pitch. ``GroupedGemmKernel``
    stays the general candidate for the rest.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call) -> bool:
        if call.dtype not in (torch.bfloat16, torch.float16):
            return False
        # ``call.n`` is a's non-group extent, ``call.k`` b's; TT also strides b by numel.
        extents = [call.n, call.k]
        if call.transpose_a and call.transpose_b:
            extents.append(call.numel)
        return all(extent % 8 == 0 for extent in extents)

    def __init__(
        self,
        batch_sum: int,
        batch_count: int,
        n: int,
        k: int,
        dtype: torch.dtype = torch.float16,
        transpose_a: bool = False,
        transpose_b: bool = True,
        tune: bool = False,
    ) -> None:
        """Bind the layout; shapes and the tile come from each call."""
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.n = n
        self.k = k
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        gemm_type = GemmType.K_GROUPED_CONTIGUOUS if transpose_a else GemmType.M_GROUPED_TIGHT_PSUM
        self.inner = SM90GemmFwdKernel(gemm_type, num_groups=batch_count, tune=tune)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One GEMM per group, packed along rows (NT / NN) or along K (TN / TT)."""
        if self.transpose_a:
            # a [batch_sum, M] and b [batch_sum, N] or [N, batch_sum]: the groups split K.
            b_logical = b if self.transpose_b else b.transpose(0, 1)
            return self.inner(a.transpose(0, 1), b_logical, grouped_layout=batch_sizes, out=out)
        b_logical = b if self.transpose_b else b.transpose(1, 2)
        ends = batch_offsets + batch_sizes
        return self.inner(a, b_logical, grouped_layout=ends, out=out)
