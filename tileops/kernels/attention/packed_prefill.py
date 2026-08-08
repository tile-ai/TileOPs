"""The packed GQA prefill slot: one constructor, one call, one result.

Every implementation of packed prefill — dense, warp-specialized causal, H200
square causal, variable-length, sliding-window, FP8 — is built from the same
semantic facts and called the same way:

    kernel(q, k, v, cu_seqlens_q, cu_seqlens_kv, q_scale, k_scale, v_scale) -> o

Tensors are packed THD throughout. An implementation that computes on a BSHD
view takes that view itself; one that reads descale tensors expands them itself;
one that produces a log-sum-exp keeps it. None of that reaches the op, which is
why op-side selection needs no per-implementation adapter.

Fields an implementation does not use are still accepted, because they are facts
about the call rather than about the implementation. What differs between
implementations is only the region each serves, which each states in ``applies``.
"""

from typing import Optional, Tuple

import torch

from ..kernel_base import Kernel
from .call_spec import ATTENTION_DTYPES

__all__ = ["PackedPrefillKernel"]


class PackedPrefillKernel(Kernel):
    """Base for every implementation of the packed GQA prefill slot."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        window_size_left: int = -1,
        window_size_right: int = -1,
        accum_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if is_causal and max_seqlen_q > max_seqlen_kv:
            raise ValueError("causal prefill requires max_seqlen_q <= max_seqlen_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dim = dim
        self.is_causal = is_causal
        # For a slot whose output element type is an independent choice, this is
        # the output type; FP8 inputs carry their own.
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.accum_dtype = accum_dtype
        self._validate_spec()
        self._build_program()
        self.init_config(config, tune)

    def _validate_spec(self) -> None:
        """Reject a spec this implementation cannot honour. Override as needed."""

    def _build_program(self) -> None:
        """Build whatever the implementation launches. Override."""
        raise NotImplementedError

    def _bshd(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """View packed THD tensors as BSHD; valid only for a uniform request.

        An implementation whose program is written against BSHD takes this view
        itself, so the op never has to know which layout a kernel prefers.
        """
        return (
            q.view(self.batch, self.max_seqlen_q, self.heads, self.dim),
            k.view(self.batch, self.max_seqlen_kv, self.heads_kv, self.dim),
            v.view(self.batch, self.max_seqlen_kv, self.heads_kv, self.dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run packed prefill and return the semantic output only."""
        raise NotImplementedError

    @staticmethod
    def _is_attention_dtype(dtype: Optional[torch.dtype]) -> bool:
        return dtype in ATTENTION_DTYPES
