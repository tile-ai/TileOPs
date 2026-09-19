"""The persistent kernel behind ``GroupedGemmFwdOp``'s four layouts."""

from typing import Optional

import torch

from tileops.kernels.grouped_gemm.call import GroupedGemmCall
from tileops.kernels.grouped_gemm.heuristics import GemmType
from tileops.kernels.grouped_gemm.template import GemmTemplate
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["GroupedGemmPersistentKernel"]


def grouped_gemm_entry(cls: type, call: GroupedGemmCall, **extra: object) -> Entry:
    """The entry for a grouped-GEMM candidate: both take the same construction arguments.

    The device index is in the identity because the kernel is compiled for the
    architecture it is built on. *extra* carries the construction arguments only
    one candidate takes, and joins the identity so its builds stay apart.
    """
    index = call.device.index if call.device is not None else None
    identity = (
        call.numel,
        call.num_experts,
        call.n,
        call.k,
        call.dtype,
        call.transpose_a,
        call.transpose_b,
        call.tune,
        index,
        *sorted(extra.items()),
    )
    return identity, lambda: cls(
        call.numel,
        call.num_experts,
        call.n,
        call.k,
        call.dtype,
        transpose_a=call.transpose_a,
        transpose_b=call.transpose_b,
        tune=call.tune,
        **extra,
    )


class GroupedGemmPersistentKernel(Kernel):
    """``GroupedGemmFwdOp``'s NT / NN / TN / TT on the SM90 GEMM template.

    An adapter: the op builds a kernel as ``cls(batch_sum, batch_count, n, k, ...)``
    and calls it with the group tables, while the template speaks GEMM types and
    psum metadata. The two M-grouped layouts (NT, NN) take their psum ends from the
    row table the call states: ``batch_offsets + batch_sizes`` for tight rows under
    ``M_GROUPED_TIGHT_PSUM``, or ``batch_padded_offsets + batch_sizes`` for padded
    rows under ``M_GROUPED_ALIGNED_PSUM``, whose groups start on a row block so no
    tile spans two of them. An NN ``b`` is the transposed view of its ``[E, K, N]``
    storage. The two K-grouped layouts (TN, TT) run ``K_GROUPED_CONTIGUOUS`` on
    ``batch_sizes`` over the transposed views of ``a`` and ``b``, so a TT ``b``
    keeps its ``[K, batch_sum]`` storage K-major; groups split K there, which
    carries no row padding.

    Claims bf16 or fp16 operands whose TMA-addressed extents are multiples of 8:
    every contiguous operand extent and the output row pitch. ``GroupedGemmKernel``
    stays the general candidate for the rest of the tight calls.
    """

    supported_archs: list[int] = [90]
    # The row block a padded layout starts its groups on: the template's
    # m_alignment, which an aligned GEMM type takes as its block_m.
    row_block: int = 128

    @classmethod
    def applies(cls, call) -> bool:
        if call.dtype not in (torch.bfloat16, torch.float16):
            return False
        # ``call.n`` is a's non-group extent, ``call.k`` b's; TT also strides b by numel.
        extents = [call.n, call.k]
        if call.transpose_a and call.transpose_b:
            extents.append(call.numel)
        return all(extent % 8 == 0 for extent in extents)

    @classmethod
    def entry_for(cls, call: GroupedGemmCall) -> Entry:
        """A padded call is a separate build: it runs a different GEMM type."""
        return grouped_gemm_entry(cls, call, padded=call.padded)

    def __init__(
        self,
        batch_sum: int,
        batch_count: int,
        n: int,
        k: int,
        dtype: torch.dtype = torch.float16,
        transpose_a: bool = False,
        transpose_b: bool = True,
        padded: bool = False,
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
        self.padded = padded
        if transpose_a:
            gemm_type = GemmType.K_GROUPED_CONTIGUOUS
        elif padded:
            gemm_type = GemmType.M_GROUPED_ALIGNED_PSUM
        else:
            gemm_type = GemmType.M_GROUPED_TIGHT_PSUM
        self.inner = GemmTemplate(
            gemm_type, num_groups=batch_count, m_alignment=self.row_block, tune=tune
        )

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
        # Both M-grouped types read psum ends; the layout the call stated decides
        # which row table they start from.
        starts = batch_padded_offsets if self.padded else batch_offsets
        return self.inner(a, b_logical, grouped_layout=starts + batch_sizes, out=out)
