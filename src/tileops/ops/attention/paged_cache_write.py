"""Paged KV cache write operator (host-side Op layer).

Provides:
  - PagedKVCacheWriteFwdOp: k_pages[slot_mapping] = k, v_pages[slot_mapping] = v
"""

from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention.call_spec import (
    PagedKVCacheWriteCall,
    PagedKVCacheWriteFwdInterface,
)
from tileops.kernels.attention.paged_cache_write import PagedKVCacheWriteKernel
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op

__all__ = ["PagedKVCacheWriteFwdOp"]


class PagedKVCacheWriteFwdOp(Op):
    """Scatter new K/V rows into a page pool, the write half of a paged attention cache.

    The slots come from the caller as one flattened row index per token, ``-1`` skipping
    the token, so the pages a call writes need not be consecutive, need not be new, and
    need not belong to one request: a prefix-caching server reuses a page it already holds
    and a chunked prefill writes the pages it was just given. Distinct non-negative slots
    are the caller's obligation, and a call writes nothing outside the slots it names.
    With an FP8 cache the rows are divided by their scale and cast to E4M3FN. NaN is
    preserved; finite overflow and infinities saturate to the largest finite magnitude
    with the same sign.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "paged_cache_write_kernel": PagedKVCacheWriteKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "paged_cache_write": PagedKVCacheWriteFwdInterface
    }

    def __init__(
        self,
        *,
        target: Target = None,
    ) -> None:
        """Build the op; the cache geometry and dtypes arrive with each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Write each token's key and value row into the page-pool row its slot names.

        Args:
            k: New keys, ``(tokens, heads_kv, dim)`` in float16 or bfloat16.
            v: New values, the same shape and dtype.
            k_pages: The key page pool, ``(num_pages, page_size, heads_kv, dim)``, written in
                place. Its dtype is the cache dtype: float16, bfloat16, or float8_e4m3fn.
            v_pages: The value page pool, the same shape and dtype, written in place.
            slot_mapping: ``int64`` ``(tokens,)``; entry ``n`` is
                ``page * page_size + offset``, the flattened row token ``n`` writes, or
                ``-1`` to skip it.
            k_scale: ``float32`` ``(1,)`` divisor of the stored keys, required exactly when
                the cache is float8_e4m3fn.
            v_scale: The same divisor for the stored values.

        Returns:
            None; both page pools are mutated and no other tensor is touched.
        """
        k = k.contiguous()
        v = v.contiguous()
        slot_mapping = slot_mapping.contiguous()
        k_scale = None if k_scale is None else k_scale.contiguous()
        v_scale = None if v_scale is None else v_scale.contiguous()
        # k_pages and v_pages are declared contiguous and mutated, so they are handed over
        # as they arrived: a copy would take the write with it.
        self.kernel_for(
            "paged_cache_write",
            lambda: PagedKVCacheWriteCall(
                heads_kv=k.shape[1],
                dim=k.shape[2],
                num_pages=k_pages.shape[0],
                page_size=k_pages.shape[1],
                dtype=k.dtype,
                cache_dtype=k_pages.dtype,
                device=k.device,
            ),
        )(k, v, k_pages, v_pages, slot_mapping, k_scale, v_scale)
        return None
