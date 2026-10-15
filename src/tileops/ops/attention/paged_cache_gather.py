"""Gather ragged rows from a caller-owned paged cache."""

import math
from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention.paged_cache_gather import (
    PagedKVCacheGatherCall,
    PagedKVCacheGatherFwdInterface,
    PagedKVCacheGatherKernel,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op

__all__ = ["PagedKVCacheGatherFwdOp"]


class PagedKVCacheGatherFwdOp(Op):
    """Copy ragged logical cache ranges into a packed destination, optionally dequantizing FP8.

    FP8 values are dequantized by one FP32 multiply and rounded once to the destination
    dtype. A non-contiguous destination uses a contiguous staging buffer and a copy back
    into the caller's view; its previous contents are never read.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "paged_cache_gather_kernel": PagedKVCacheGatherKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "paged_cache_gather": PagedKVCacheGatherFwdInterface
    }

    def __init__(
        self,
        out_dtype: Optional[torch.dtype] = None,
        *,
        target: Target = None,
    ) -> None:
        """Configure cache gathering; shapes and optional inputs are resolved at each call.

        Args:
            out_dtype: Destination dtype for FP8 dequantization; otherwise the cache dtype.
            target: Backend target, or None to resolve from the input device.
        """
        self.out_dtype = out_dtype
        super().__init__(target=target)

    def forward(
        self,
        dst: torch.Tensor,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        seq_starts: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Gather each request's logical cache range into its packed destination rows.

        Args:
            dst: Write-only destination [total_tokens, ...], in out_dtype or the cache dtype.
            cache: Contiguous cache [num_pages, page_size, ...], FP16, BF16 or FP8 E4M3.
            block_table: Physical pages of logical request pages, int32 [batch, width].
            cu_seq_lens: Packed destination boundaries, int32 [batch + 1].
            seq_starts: Optional first logical cache position of each request, int32 [batch].
            scale: FP32 [1] dequantization multiplier, required exactly for an FP8 cache.

        Returns:
            None; every element of dst is written and the cache is unchanged.
        """
        # The signature permits strided destinations. A write-only staging buffer avoids
        # reading their old contents; copying back preserves the caller's view and storage.
        output = (
            dst
            if dst.is_contiguous()
            else torch.empty_like(dst, memory_format=torch.contiguous_format)
        )
        call = PagedKVCacheGatherCall(
            batch=block_table.shape[0],
            num_pages=cache.shape[0],
            page_size=cache.shape[1],
            table_width=block_table.shape[1],
            row_width=math.prod(dst.shape[1:]),
            dtype=cache.dtype,
            out_dtype=dst.dtype,
            has_starts=seq_starts is not None,
            device=dst.device,
        )
        self.kernel_for("paged_cache_gather", call)(
            output,
            cache,
            block_table.contiguous(),
            cu_seq_lens.contiguous(),
            seq_starts.contiguous() if seq_starts is not None else None,
            scale.contiguous() if scale is not None else None,
        )
        if output is not dst:
            dst.copy_(output)
        return None
