from typing import Literal, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import is_hopper

from .gqa_fwd import (
    GQAPrefillFwdKernel,
    GQAPrefillPagedWithFP8KVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheRopeAppendKernel,
    GQAPrefillPagedWithKVCacheRopeFwdKernel,
    GQAPrefillWithKVCacheFwdKernel,
    GQAPrefillWithKVCacheRopeAppendKernel,
    GQAPrefillWithKVCacheRopeFwdKernel,
)
from .gqa_fwd_fp8 import GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel
from .gqa_prefill_varlen_fwd import GQAPrefillVarlenFwdKernel
from .gqa_sliding_window_fwd import (
    GQASlidingWindowFwdKernel,
    GQASlidingWindowFwdWgmmaPipelinedKernel,
)
from .gqa_sliding_window_varlen_fwd import (
    GQASlidingWindowVarlenFwdKernel,
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)

__all__ = [
    "GQAPrefillKernel",
    "GQAPrefillPagedKernel",
    "GQAPrefillWithKVCacheKernel",
]


class GQAPrefillKernel(Kernel):
    """Logical no-cache GQA prefill kernel.

    This wrapper keeps current concrete kernels intact and dispatches to the
    implementation that matches the requested layout/features. It is an internal
    stepping stone for the GQA taxonomy refactor; public OP contracts stay
    manifest-owned.
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        *,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        dtype: torch.dtype = torch.float16,
        layout: Literal["dense", "ragged"] = "dense",
        seq_len_q: Optional[int] = None,
        seq_len_kv: Optional[int] = None,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        use_swa: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        fp8_tensor_core: bool = False,
        backend: Literal["auto", "basic", "wgmma", "ws", "fa3"] = "auto",
        accum_dtype: torch.dtype = torch.float32,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.use_swa = use_swa
        self.fp8_tensor_core = fp8_tensor_core
        self.backend = backend
        self.dtype = dtype

        if fp8_tensor_core:
            if seq_len_q is None:
                raise ValueError("seq_len_q is required for fp8_tensor_core=True")
            if seq_len_kv is not None and seq_len_kv != seq_len_q:
                raise ValueError("FP8 Tensor Core prefill currently requires seq_len_q == seq_len_kv")
            self.impl = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
                batch, heads, heads_kv, seq_len_q, dim, dtype, tune=tune)
        elif layout == "ragged":
            if use_swa:
                kernel_cls = (
                    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel
                    if is_hopper() else GQASlidingWindowVarlenFwdKernel)
                self.impl = kernel_cls(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    dim=dim,
                    is_causal=is_causal,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    dtype=dtype,
                    accum_dtype=accum_dtype,
                    tune=tune,
                )
            else:
                self.impl = GQAPrefillVarlenFwdKernel(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    dim=dim,
                    is_causal=is_causal,
                    dtype=dtype,
                    sm_scale=sm_scale,
                    softcap=softcap,
                    tune=tune,
                )
        elif layout == "dense":
            if seq_len_q is None or seq_len_kv is None:
                raise ValueError("seq_len_q and seq_len_kv are required for dense prefill")
            if use_swa:
                if seq_len_q != seq_len_kv:
                    raise ValueError("sliding-window dense prefill currently requires seq_len_q == seq_len_kv")
                kernel_cls = GQASlidingWindowFwdWgmmaPipelinedKernel if is_hopper() else GQASlidingWindowFwdKernel
                self.impl = kernel_cls(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    seq_len=seq_len_q,
                    dim=dim,
                    is_causal=is_causal,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    dtype=dtype,
                    tune=tune,
                )
            else:
                if backend not in {"auto", "basic", "wgmma", "ws", "fa3"}:
                    raise ValueError(f"unsupported backend {backend!r}")
                self.impl = GQAPrefillFwdKernel(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    seq_len_q=seq_len_q,
                    seq_len_kv=seq_len_kv,
                    dim=dim,
                    is_causal=is_causal,
                    dtype=dtype,
                    sm_scale=sm_scale,
                    softcap=softcap,
                    tune=tune,
                )
        else:
            raise ValueError(f"unsupported GQA prefill layout {layout!r}")

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.impl(*args, **kwargs)


class GQAPrefillWithKVCacheKernel(Kernel):
    """Logical contiguous KV-cache GQA prefill kernel."""

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        *,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len_new: int,
        seqlen_kv: int,
        dim: int,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        fuse_rope: bool = False,
        max_position: Optional[int] = None,
        rotary_dim: Optional[int] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.fuse_rope = fuse_rope
        self.dtype = dtype
        if fuse_rope:
            if max_position is None or rotary_dim is None:
                raise ValueError("max_position and rotary_dim are required when fuse_rope=True")
            self.append_impl = GQAPrefillWithKVCacheRopeAppendKernel(
                batch, heads_kv, seq_len_new, seqlen_kv, dim, max_position, rotary_dim, dtype,
                tune=tune)
            self.impl = GQAPrefillWithKVCacheRopeFwdKernel(
                batch, heads, heads_kv, seq_len_new, seqlen_kv, dim, max_position, rotary_dim,
                is_causal, dtype, sm_scale=sm_scale, softcap=softcap, tune=tune)
        else:
            self.append_impl = None
            self.impl = GQAPrefillWithKVCacheFwdKernel(
                batch, heads, heads_kv, seq_len_new, seqlen_kv, dim, is_causal, dtype,
                sm_scale=sm_scale, softcap=softcap, tune=tune)

    def forward(self, *args: object, **kwargs: object) -> object:
        if not self.fuse_rope:
            return self.impl(*args, **kwargs)
        if len(args) < 8:
            raise ValueError(
                "fused RoPE contiguous KV-cache prefill expects cos_table and sin_table")
        q, k_new, v_new, k_cache, v_cache, cache_seqlens, cos_table, sin_table = args[:8]
        self.append_impl(k_new, v_new, k_cache, v_cache, cache_seqlens, cos_table, sin_table)
        return self.impl(q, k_new, v_new, k_cache, v_cache, cache_seqlens, cos_table, sin_table)


class GQAPrefillPagedKernel(Kernel):
    """Logical paged KV-cache GQA prefill kernel."""

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        *,
        batch: int,
        heads: int,
        heads_kv: int,
        max_pages_per_req: int,
        page_size: int,
        dim: int,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_dtype: Optional[torch.dtype] = None,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        fuse_rope: bool = False,
        max_position: Optional[int] = None,
        rotary_dim: Optional[int] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.fuse_rope = fuse_rope
        self.cache_dtype = cache_dtype
        self.dtype = dtype
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if cache_dtype is not None and cache_dtype == fp8_dtype:
            if fuse_rope:
                raise ValueError("paged FP8 KV-cache prefill does not support fused RoPE yet")
            self.append_impl = None
            self.impl = GQAPrefillPagedWithFP8KVCacheFwdKernel(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                max_pages_per_req=max_pages_per_req,
                page_size=page_size,
                dim=dim,
                is_causal=is_causal,
                dtype=dtype,
                sm_scale=sm_scale,
                softcap=softcap,
                tune=tune,
            )
        elif fuse_rope:
            if max_position is None or rotary_dim is None:
                raise ValueError("max_position and rotary_dim are required when fuse_rope=True")
            self.append_impl = GQAPrefillPagedWithKVCacheRopeAppendKernel(
                batch=batch,
                heads_kv=heads_kv,
                max_pages_per_req=max_pages_per_req,
                page_size=page_size,
                dim=dim,
                max_position=max_position,
                rotary_dim=rotary_dim,
                dtype=dtype,
                tune=tune,
            )
            self.impl = GQAPrefillPagedWithKVCacheRopeFwdKernel(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                max_pages_per_req=max_pages_per_req,
                page_size=page_size,
                dim=dim,
                max_position=max_position,
                rotary_dim=rotary_dim,
                is_causal=is_causal,
                dtype=dtype,
                sm_scale=sm_scale,
                softcap=softcap,
                tune=tune,
            )
        else:
            self.append_impl = None
            self.impl = GQAPrefillPagedWithKVCacheFwdKernel(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                max_pages_per_req=max_pages_per_req,
                page_size=page_size,
                dim=dim,
                is_causal=is_causal,
                dtype=dtype,
                sm_scale=sm_scale,
                softcap=softcap,
                tune=tune,
            )

    def forward(self, *args: object, **kwargs: object) -> object:
        if not self.fuse_rope:
            return self.impl(*args, **kwargs)
        if len(args) < 11:
            raise ValueError("fused RoPE paged prefill expects cos_table and sin_table")
        (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            max_seqlen_q,
            cos_table,
            sin_table,
        ) = args[:11]
        self.append_impl(
            k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table,
            max_seqlen_q, cos_table, sin_table)
        return self.impl(
            q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table,
            max_seqlen_q, cos_table, sin_table)
