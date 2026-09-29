"""The paged GQA prefill slot: one constructor, one call, one result.

    kernel(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
           cu_seqlens_q, cache_seqlens, block_table, max_seqlen_q,
           cos_table, sin_table) -> o

An implementation accepts the whole spec whether or not it reads every field,
and one that appends to the cache does so itself. See
docs/design/ops-design.md § Kernel selection.
"""

from typing import Optional

import torch

from ..kernel_base import Entry, Kernel
from .call_spec import AttentionCall

__all__ = ["PagedPrefillKernel", "page_size_refusal"]


def page_size_refusal(page_size: int) -> Optional[str]:
    """Why the paged kernels cannot index pages of *page_size* tokens by shift, or ``None``."""
    if page_size <= 0 or page_size & (page_size - 1) != 0:
        return "requires a power-of-two page_size"
    return None


class PagedPrefillKernel(Kernel):
    """Base for every implementation of the paged GQA prefill slot."""

    @classmethod
    def applies(cls, call: AttentionCall) -> bool:
        return cls._region_refusal(call) is None

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        return cls._region_refusal(call)

    @classmethod
    def _region_refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why *call* is outside this implementation's region, or ``None``.

        Every implementation indexes pages by shift. A subclass states its own region
        and asks this for the page-size limit.
        """
        return page_size_refusal(call.page_size)

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The cache identity and the thunk that builds this class for *call*.

        Every implementation of the slot takes this constructor, so the base states
        it once. The device index is in the identity because the kernel is compiled
        for the architecture it is built on.
        """
        index = call.device.index if call.device is not None else None
        args = dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            max_pages_per_req=call.max_pages_per_req,
            page_size=call.page_size,
            dim=call.dim,
            is_causal=call.is_causal,
            dtype=call.dtype,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            max_position=call.max_position,
            rotary_dim=call.rotary_dim,
        )
        return (*args.values(), index), lambda: cls(**args, tune=call.tune, device_index=index)

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        max_pages_per_req: int,
        page_size: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        max_position: Optional[int] = None,
        rotary_dim: Optional[int] = None,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.max_pages_per_req = max_pages_per_req
        self.page_size = page_size
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self._build_program()
        self.init_config(config, tune)

    def _build_program(self) -> None:
        """Build whatever the implementation launches beyond its wrapped call."""

    def forward(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
        cos_table: Optional[torch.Tensor] = None,
        sin_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend against the paged cache and return the semantic output only."""
        raise NotImplementedError
