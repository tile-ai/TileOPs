"""Shared construction state for GQA prefill kernel families.

Dense and packed/variable-length prefill use the same static configuration,
but they deliberately expose different runtime tensor ABIs:

* :class:`DensePrefillKernel` consumes BSHD tensors.
* :class:`PackedPrefillKernel` consumes packed THD tensors plus cu-seqlens.

Keeping them as sibling families prevents a native Dense implementation from
acquiring synthetic packed inputs merely to reuse constructor bookkeeping.
"""

from typing import Optional

import torch

from ..kernel_base import Kernel

__all__ = ["DensePrefillKernel", "PackedPrefillKernel", "PrefillKernel"]


class PrefillKernel(Kernel):
    """Topology-neutral static configuration shared by prefill kernels."""

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
        fuse_rope: bool = False,
        max_position: Optional[int] = None,
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
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
        self.fuse_rope = fuse_rope
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.accum_dtype = accum_dtype
        self._validate_spec()
        self._build_program()
        self.init_config(config, tune)

    def _validate_spec(self) -> None:
        """Reject a spec this implementation cannot honour. Override as needed."""

    def _build_program(self) -> None:
        """Build whatever the implementation launches. Override."""
        raise NotImplementedError


class DensePrefillKernel(PrefillKernel):
    """Base for native fixed-shape BSHD prefill implementations."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class PackedPrefillKernel(PrefillKernel):
    """Base for packed THD prefill implementations with explicit ranges."""

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
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
