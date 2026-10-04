from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    GQAPrefillPagedWithFP8KVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheRoPEFwdKernel,
)
from tileops.kernels.attention.call_spec import (
    AttentionCall,
    GQAPrefillPagedFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.attention.gqa.parameters import _attention_scale, _rope_rotary_dim, _score_softcap
from tileops.ops.op_base import Op
from tileops.ops.rope import base_freqs
from tileops.perf.profile import tensor_core_roof

__all__ = ["GQAPrefillPagedWithKVCacheFwdOp"]


class GQAPrefillPagedWithKVCacheFwdOp(Op):
    """Packed GQA prefill with paged KV cache append. Layout: THD.

    The current chunk is packed by request. ``cache_seqlens`` stores each
    request's logical KV length before append. ``block_table`` maps logical
    page ids to physical pages in ``k_pages`` / ``v_pages``.

    The in-tree kernels refuse a ``page_size`` that is not a power of two and fused RoPE
    over an FP8 cache.

    By default the op does not check tensor contents. An FP8 cache scale that is not
    finite and positive makes the output non-finite, and with fused RoPE a request whose
    cached plus new tokens exceed ``max_position`` reads the RoPE table out of bounds. The
    caller guarantees both, or passes ``validate_inputs=True``, which checks them on every
    call at the cost of device synchronizations and cannot run inside CUDA Graph capture.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_prefill_paged_with_kv_cache_fwd_kernel": GQAPrefillPagedWithKVCacheFwdKernel,
        "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel": GQAPrefillPagedWithFP8KVCacheFwdKernel,
        "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel": GQAPrefillPagedWithKVCacheRoPEFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gqa_prefill_paged": GQAPrefillPagedFwdInterface
    }

    def eval_roofline_read_bytes(self) -> "int | None":
        """Not derivable here: the call writes part of the pool, not all of it.

        ``k_pages`` and ``v_pages`` are mutated, and the base class takes a
        mutated input's whole extent off ``bytes`` as the write. This call
        appends the new tokens into pages the block table names and leaves the
        rest untouched, so that subtraction would understate the read half.
        """
        return None

    def roofline_inputs(self) -> "dict[str, int]":
        """The distinct cache rows this call reads, which its cache traffic follows."""
        from tileops.perf.formulas import gqa_prefill_paged_cache_rows

        return {"cache_rows": gqa_prefill_paged_cache_rows(self.last_call)}

    def __init__(
        self,
        page_size: int,
        max_seqlen_q: int,
        is_causal: bool = True,
        cache_dtype: Optional[torch.dtype] = None,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        fuse_rope: bool = False,
        rope_base: float = 10000.0,
        max_position: Optional[int] = None,
        rotary_dim: Optional[int] = None,
        *,
        validate_inputs: bool = False,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            page_size: Manifest ``params.page_size``, ``int``.
            max_seqlen_q: Manifest ``params.max_seqlen_q``, the launch bound the kernel
                is built for; callers must keep every request within it.
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``True``.
            cache_dtype: Manifest ``params.cache_dtype``, ``dtype | None``, default ``None``.
            sm_scale: Manifest ``params.sm_scale``, ``float | None``, default ``None``,
                which resolves to ``1 / sqrt(D)`` from each call's head dimension.
            softcap: Manifest ``params.softcap``, ``float | None``, default ``None``.
            fuse_rope: Manifest ``params.fuse_rope``, ``bool``, default ``False``.
            rope_base: Manifest ``params.rope_base``, ``float``, default ``10000.0``.
            max_position: Manifest ``params.max_position``, ``int | None``, default ``None``.
            rotary_dim: Manifest ``params.rotary_dim``, ``int | None``, default ``None``,
                which rotates each call's full head dimension.
            validate_inputs: Check scale values and RoPE positions on the CPU.
                Synchronizes the device; enable only outside CUDA Graph capture.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.validate_inputs = validate_inputs
        self.max_seqlen_q = max_seqlen_q
        self.page_size = page_size
        self.is_causal = is_causal
        # None means the cache holds whatever element type forward is given.
        self.cache_dtype = cache_dtype
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.fuse_rope = fuse_rope
        self.rope_base = rope_base
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self._rope_cos_cache: Dict[
            tuple[torch.device, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def _resolved_cache_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Cache element type for an attention element type of *dtype*."""
        return dtype if self.cache_dtype is None else self.cache_dtype

    def _resolved_rotary_dim(self, dim: int) -> Optional[int]:
        """Rotated width for a head dimension of *dim*, or ``None`` without fused RoPE."""
        return _rope_rotary_dim(dim, self.rotary_dim) if self.fuse_rope else None

    def attention_call(
        self, q: torch.Tensor, k_new: torch.Tensor, block_table: torch.Tensor
    ) -> AttentionCall:
        """State what one paged prefill call is, for selection to filter against."""
        _, heads, dim = q.shape
        heads_kv = k_new.shape[1]
        batch, max_pages_per_req = block_table.shape
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            max_pages_per_req=max_pages_per_req,
            page_size=self.page_size,
            is_causal=self.is_causal,
            sm_scale=_attention_scale(dim, self.sm_scale),
            softcap=self.softcap,
            cache_dtype=self._resolved_cache_dtype(q.dtype),
            fuse_rope=self.fuse_rope,
            max_position=self.max_position,
            rotary_dim=self._resolved_rotary_dim(dim),
            device=q.device,
        )

    def _rope_tables(self, q: torch.Tensor):
        """Rotary tables for this call, or ``(None, None)`` when the op fuses no RoPE."""
        if not self.fuse_rope:
            return None, None
        return self._get_rope_cos_sin(q.device, q.dtype, self._resolved_rotary_dim(q.shape[2]))

    def _check_call_values(
        self,
        k_pages: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        """Refuse tensor contents the call is undefined for.

        An FP8 cache dequantizes by the scales, which must be finite and positive; fused
        RoPE indexes its table by position, which must stay below ``max_position``.
        """
        if k_pages.dtype == torch.float8_e4m3fn:
            for name, tensor in (("k_scale", k_scale), ("v_scale", v_scale)):
                if not torch.all(torch.isfinite(tensor) & (tensor > 0)).item():
                    raise ValueError(f"{name} must contain finite positive values")
        if self.fuse_rope:
            q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            max_total_len = int((cache_seqlens + q_lens).max().item())
            if max_total_len > self.max_position:
                raise ValueError(
                    "cache_seqlens + q_len exceeds RoPE max_position: "
                    f"max total length {max_total_len}, max_position {self.max_position}"
                )

    def _get_rope_cos_sin(
        self,
        device: torch.device,
        dtype: torch.dtype,
        rotary_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_position is None:
            raise ValueError("max_position is required when fuse_rope=True")
        key = (device, dtype, rotary_dim)
        cached = self._rope_cos_cache.get(key)
        if cached is None:
            cached = base_freqs(
                rotary_dim,
                self.max_position,
                base=self.rope_base,
                dtype=dtype,
                device=device,
            )
            self._rope_cos_cache[key] = cached
        return cached

    def forward(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Attend the packed chunk to each request's cache, appending its keys and values.

        Args:
            q: Queries of the new tokens, packed by request [total_q, heads, dim].
            k_new: Keys of the new tokens [total_q, heads_kv, dim].
            v_new: Values of the new tokens [total_q, heads_kv, dim].
            k_pages: Key page pool [physical_tokens, heads_kv, dim], written in place.
            v_pages: Value page pool [physical_tokens, heads_kv, dim], written in place.
            k_scale: Key dequantization scale of an FP8 pool [1].
            v_scale: Value dequantization scale of an FP8 pool [1].
            cu_seqlens_q: Request boundaries in the packed chunk [batch + 1].
            cache_seqlens: Each request's cache length before the append [batch].
            block_table: Physical page of each request's logical page [batch, pages].

        Returns:
            The attention output [total_q, heads, dim].

        Raises:
            ValueError: An FP8 pool's scales are not finite and positive, a fused-RoPE
                call reaches past ``max_position``, or no in-tree kernel serves the call.
        """
        return self._call_boundary(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )

    def _eager_forward(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        if self.validate_inputs:
            self._check_call_values(k_pages, k_scale, v_scale, cu_seqlens_q, cache_seqlens)
        q, k_new, v_new, k_scale, v_scale, cu_seqlens_q, cache_seqlens, block_table = (
            t.contiguous()
            for t in (q, k_new, v_new, k_scale, v_scale, cu_seqlens_q, cache_seqlens, block_table)
        )
        call = self.attention_call(q, k_new, block_table)
        cos_table, sin_table = self._rope_tables(q)
        inputs = (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )
        kernel = self.kernel_for("gqa_prefill_paged", call)
        return kernel(*inputs, self.max_seqlen_q, cos_table, sin_table)

    @property
    def total_flops(self) -> int:
        raise NotImplementedError(
            "total_flops is not defined for paged varlen ops; "
            "compute per-sample from cu_seqlens and cache_seqlens at call time."
        )

    @property
    def total_memory(self) -> int:
        raise NotImplementedError(
            "total_memory is not defined for paged varlen ops; "
            "compute per-sample from cu_seqlens and cache_seqlens at call time."
        )

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
