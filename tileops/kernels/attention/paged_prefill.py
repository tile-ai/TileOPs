"""The paged GQA prefill slot: one constructor, one call, one result.

Every implementation — plain cache, FP8 cache, fused rotary position embedding
(RoPE) — is built from the same semantic facts and called the same way:

    kernel(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
           cu_seqlens_q, cache_seqlens, block_table, max_seqlen_q,
           cos_table, sin_table) -> o

Arguments an implementation does not read are facts about the call rather than
about the implementation, so they are still accepted. A fused-RoPE
implementation runs its cache-append pass itself: appending and attending are
two launches of one semantic operation, not two candidates to choose between.
"""

from typing import Optional

import torch

from ..kernel_base import Kernel

__all__ = ["PagedPrefillKernel"]


class PagedPrefillKernel(Kernel):
    """Base for every implementation of the paged GQA prefill slot."""

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
    ) -> None:
        super().__init__()
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
        self._validate_spec()
        self._build_program()
        self.init_config(config, tune)

    def _validate_spec(self) -> None:
        """Reject a spec this implementation cannot honour. Override as needed."""

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
