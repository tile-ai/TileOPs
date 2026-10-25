"""Gather ragged rows from a caller-owned paged cache."""

import math
from typing import ClassVar, Mapping, Optional

import torch

from tileops._kernels.attention.paged_cache_gather import (
    PagedKVCacheGatherCall,
    PagedKVCacheGatherFwdInterface,
    PagedKVCacheGatherKernel,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["PagedKVCacheGatherFwdOp"]


class PagedKVCacheGatherFwdOp(Op):
    """Copy ragged cache ranges into ``out`` in place, optionally dequantizing FP8.

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
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_starts: Optional[torch.Tensor] = None,
        kv_scale: Optional[torch.Tensor] = None,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Gather each request's logical cache range into its packed destination rows.

        Args:
            kv_cache: Contiguous cache $[NP \\times PS \\times *E]$, FP16, BF16 or FP8 E4M3.
            block_table: Physical pages of logical request pages, int32 $[B \\times W]$.
            cu_seqlens: Packed destination boundaries, int32 $[B + 1]$.
            seq_starts: Optional first logical cache position of each request, int32 $[B]$.
            kv_scale: FP32 $[1]$ dequantization multiplier, required exactly for an FP8 cache.
            out: Required write-only destination $[T_q \\times *E]$, in ``out_dtype`` or the
                cache dtype; $T_q$ is ``cu_seqlens[-1]``, which no input shape carries.

        Returns:
            ``out``, every element written; the cache is unchanged.
        """
        # The signature permits strided destinations. A write-only staging buffer avoids
        # reading their old contents; copying back preserves the caller's view and storage.
        output = (
            out
            if out.is_contiguous()
            else torch.empty_like(out, memory_format=torch.contiguous_format)
        )
        self.kernel_for(
            "paged_cache_gather",
            lambda: PagedKVCacheGatherCall(
                batch=block_table.shape[0],
                num_pages=kv_cache.shape[0],
                page_size=kv_cache.shape[1],
                table_width=block_table.shape[1],
                row_width=math.prod(out.shape[1:]),
                dtype=kv_cache.dtype,
                out_dtype=out.dtype,
                has_starts=seq_starts is not None,
                device=out.device,
            ),
        )(
            output,
            kv_cache,
            block_table.contiguous(),
            cu_seqlens.contiguous(),
            seq_starts.contiguous() if seq_starts is not None else None,
            kv_scale.contiguous() if kv_scale is not None else None,
        )
        if output is not out:
            out.copy_(output)
        return out
