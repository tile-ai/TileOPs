from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    GQADecodePagedBs1Kernel,
    GQADecodePagedKernel,
    GQAPagedFwdKernel,
    GQAPagedFwdWSKernel,
)
from tileops.kernels.attention.call_spec import (
    AttentionCall,
    GQAPagedFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.attention.gqa.parameters import _attention_scale, _score_softcap
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["GQAPagedFwdOp"]


class GQAPagedFwdOp(Op):
    """Grouped-Query Attention (GQA) over a caller-owned paged KV cache.

    Packed Q and its cumulative sequence lengths cover both prefill and decode.
    ``page_table`` maps logical pages to physical entries in ``k_pages`` and
    ``v_pages``. This Op reads the cache only: allocation, append, and mutation
    remain runtime responsibilities. The in-tree kernels serve a call in which Q
    and the cache share a float16 or bfloat16 dtype, over any positive page size
    and any mix of per-request query lengths, a request with no query token
    included. RoPE rotates Q and cached K as they are read, in NeoX or interleaved
    layout, including a partial rotary width. The table must cover every request's
    cache length. FP8 calls still require an external target implementation.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_decode_paged_kernel": GQADecodePagedKernel,
        "gqa_decode_paged_bs1_kernel": GQADecodePagedBs1Kernel,
        "gqa_paged_varlen_kernel": GQAPagedFwdKernel,
        "gqa_paged_ws_kernel": GQAPagedFwdWSKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gqa_paged": GQAPagedFwdInterface}

    def roofline_inputs(self) -> "dict[str, int]":
        """The cached tokens this call's lengths name and the distinct pool rows it reads,
        which its flops and cache reads follow."""
        from tileops.perf.formulas import gqa_paged_cache_rows

        call = self.last_call
        return {
            "cached_tokens": sum(call.values("cache_seqlens")),
            "cache_rows": gqa_paged_cache_rows(call),
        }

    def __init__(
        self,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        out_dtype: Optional[torch.dtype] = None,
        pos_encoding_mode: str = "none",
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Configure paged GQA semantics without owning or mutating the cache.

        Args:
            is_causal: Apply a bottom-right-aligned causal mask per request.
            window_size_left: Visible keys to the left; ``-1`` is unlimited.
            window_size_right: Visible keys to the right; ``-1`` is unlimited.
            sm_scale: Score scale, or ``None`` for ``1 / sqrt(head_dim)``.
            softcap: Positive score cap; ``None`` or zero disables it.
            out_dtype: Output dtype, inferred from the input when omitted.
            pos_encoding_mode: ``"none"`` or ``"rope"``.
            rotary_dim: Even rotated width; ``None`` uses the full head dimension.
            rope_layout: ``"neox"`` or ``"interleaved"``.
            target: Backend target, or ``None`` to resolve from the input device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.out_dtype = out_dtype
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """Paged attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

    def paged_call(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        page_table: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
    ) -> AttentionCall:
        """State what one paged call is, for selection to filter against.

        Device offsets stay runtime data. Only a single request is provably uniform
        from shapes; total tokens / batch never establishes per-request lengths.
        """
        _, heads, dim = q.shape
        num_pages, page_size, heads_kv, _ = k_pages.shape
        batch, max_pages_per_req = page_table.shape
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            max_seqlen_q=q.shape[0],
            seqlen_kv=num_pages * page_size,
            page_size=page_size,
            max_pages_per_req=max_pages_per_req,
            is_causal=self.is_causal,
            sm_scale=_attention_scale(dim, self.sm_scale),
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=torch.float8_e4m3fn in (q.dtype, k_pages.dtype),
            is_uniform=batch == 1,
            cache_dtype=k_pages.dtype,
            fuse_rope=self.pos_encoding_mode == "rope",
            max_position=rope_cos.shape[0] if rope_cos is not None else None,
            rotary_dim=self.rotary_dim or dim,
            rope_layout=self.rope_layout,
            device=q.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run read-only paged GQA over packed Q and rank-4 KV pages.

        Args:
            q: Packed queries [total_q, heads, dim].
            k_pages: Key pages [num_pages, page_size, heads_kv, dim].
            v_pages: Value pages [num_pages, page_size, heads_kv, dim].
            page_table: Physical page of each request's logical page [batch, pages].
            cache_seqlens: Each request's cached length, its query tokens included [batch].
            cu_seqlens_q: Request boundaries in the packed queries [batch + 1].
            q_scale: Dequantization scale of an FP8 query [batch, heads_kv].
            k_scale: Dequantization scale of an FP8 key cache.
            v_scale: Dequantization scale of an FP8 value cache.
            rope_cos: Rotary cosine table when ``pos_encoding_mode='rope'``.
            rope_sin: Rotary sine table when ``pos_encoding_mode='rope'``.

        Returns:
            The attention output [total_q, heads, dim].
        """
        return self._call_boundary(
            q,
            k_pages,
            v_pages,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            q_scale,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
        )

    def _eager_forward(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder. The kernels read
        the pool as ``[num_pages * page_size, heads_kv, dim]``, a view of the pages.
        """
        q, k_pages, v_pages, page_table, cache_seqlens, cu_seqlens_q = (
            t.contiguous() for t in (q, k_pages, v_pages, page_table, cache_seqlens, cu_seqlens_q)
        )
        call = self.paged_call(q, k_pages, page_table, rope_cos)
        inputs = (
            q,
            k_pages.flatten(0, 1),
            v_pages.flatten(0, 1),
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            rope_cos.contiguous() if rope_cos is not None else None,
            rope_sin.contiguous() if rope_sin is not None else None,
        )
        return self.kernel_for("gqa_paged", call)(*inputs)
