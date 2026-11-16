from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.attention import (
    DSADecodeBasicKernel,
    DSADecodeKernel,
    DSADecodeWSKernel,
    DSAPagedClusterKernel,
    DSAPagedWSKernel,
)
from tileops._kernels.attention.call_spec import (
    DSADecodeCall,
    DSAPagedCall,
    DSAPagedFwdInterface,
    SparseMLADecodeFwdInterface,
)
from tileops._kernels.attention.dsa.paged_mma import DSAPagedMMAKernel
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["DSADenseFwdOp", "DSAPagedFwdOp"]


class DSADenseFwdOp(Op):
    """DeepSeek Sparse Attention (DSA) decode.

    This operation is part of a sparse attention mechanism, designed for use in decoding
    with key-value (KV) caching.

    The layout of the operation is BSHD.

    With ``is_causal`` false only the indices select keys, and -1 pads; ``attn_sink`` adds
    ``exp(attn_sink[h])`` to head ``h``'s softmax denominator.

    The in-tree kernels serve causal calls without ``attn_sink``, with a power-of-two value
    dimension and a zero or power-of-two tail dimension, subject to shared-memory limits.
    """

    # The WGMMA warp-specialized kernels serve SM90 -- the seesaw kernel value dim 512,
    # the older one the other widths it covers; the architecture-agnostic basic kernel
    # serves everywhere else. Selection reads the device when a call arrives.
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "dsa_decode_ws_kernel": DSADecodeWSKernel,
        "dsa_decode_kernel": DSADecodeKernel,
        "dsa_decode_basic_kernel": DSADecodeBasicKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "dsa_decode": SparseMLADecodeFwdInterface
    }

    def roofline_data_terms(self) -> "dict[str, int]":
        """The keys this call's selection makes it score, which its flops follow, and the
        distinct ``kv`` rows they reach, which its bytes follow."""
        from tileops._perf.formulas import dsa_distinct_kv_rows, dsa_selected_keys

        return {
            "selected_keys": dsa_selected_keys(self.last_call),
            "distinct_kv_rows": dsa_distinct_kv_rows(self.last_call),
        }

    def __init__(
        self,
        dim_tail: int,
        stride_kv: int,
        q_start_index_s: int,
        sm_scale: float | None = None,
        is_causal: bool = True,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            dim_tail (int): The dimension of the tail portion of the attention vectors.
            stride_kv (int): The stride for the key-value sequence.
            q_start_index_s (int): The start index for queries in the sequence.
            sm_scale (float | None, default=None): Scaling factor for the softmax
                function; ``None`` means ``(D + dim_tail) ** -0.5``, the query-key head
                dimension including the tail.
            is_causal (bool, default=True): Whether an index past the query's position
                selects nothing. ``False`` lets the indices alone select keys, as
                ``flash_mla_sparse_fwd`` does.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.dim_tail = dim_tail
        self.stride_kv = stride_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal

        cp0 = q_start_index_s == 0
        self.q_start_index_s = q_start_index_s

        self._cp0 = cp0
        super().__init__(target=target)

    def _dsa_decode_call(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        attn_sink: torch.Tensor | None,
    ) -> DSADecodeCall:
        """State what one call is, for selection to filter against."""
        batch, seq_len, heads, q_dim = q.shape
        _, seq_len_kv, heads_kv, _ = kv.shape
        return DSADecodeCall(
            batch=batch,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            heads=heads,
            dim=q_dim - self.dim_tail,
            tail_dim=self.dim_tail,
            dtype=q.dtype,
            topk=indices.shape[3],
            kv_stride=self.stride_kv,
            q_start_index_s=self.q_start_index_s,
            kv_group=heads_kv,
            sm_scale=self.sm_scale,
            is_causal=self.is_causal,
            cp0=self._cp0,
            has_attn_sink=attn_sink is not None,
            device=q.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        *,
        attn_sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the sparse attention operation.

        Args:
            q (torch.Tensor): The query tensor with shape
                        (batch, seq_len, heads, dim + dim_tail).
            kv (torch.Tensor): The key-value tensor with shape
                        (batch, seq_len_kv, heads_kv, dim + dim_tail).
            indices (torch.Tensor): Indices tensor for sparse attention.
            attn_sink (torch.Tensor | None): ``float32`` logits (heads,) that join each
                head's softmax denominator.

        Returns:
            torch.Tensor: The result of applying the sparse attention
                            operation on the input tensors.
        """
        kernel = self.kernel_for(
            "dsa_decode", lambda: self._dsa_decode_call(q, kv, indices, attn_sink)
        )
        return kernel(q, kv, indices, attn_sink)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class DSAPagedFwdOp(Op):
    """DeepSeek Sparse Attention (DSA) decode over a paged latent cache.

    Each query row attends to the cache positions ``indices`` names, resolved through the
    request's ``block_table`` page by page. A ``uint8`` cache row is FlashMLA's
    DeepSeek-V3.2 FP8 layout: 512 float8_e4m3fn latent values, one float32 scale per 128 of
    them, then 64 bfloat16 rope values. The key is the dequantized latent and the rope; the
    value is the dequantized latent. A ``bfloat16`` cache row holds the key, ``q``'s width:
    the value is its first ``head_dim_v`` columns. A position of -1, or at or past
    ``cache_seqlens``, is dead. ``attn_sink`` adds ``exp(attn_sink[h])`` to head ``h``'s
    softmax denominator and leaves ``lse`` unchanged.

    The kernel rounds the dequantized key and the softmax weights to bfloat16 before their
    matmuls, which accumulate in float32. ``lse`` is the natural-log sum of exponentials of
    the scaled scores; a row with no live position returns zeros and ``lse`` -inf.
    Serves the FP8 cache without ``attn_sink`` on SM89 and SM90, 64 or a multiple of 64
    heads, ``topk`` a multiple of 64, and a cache below 2^32 bytes.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "dsa_paged_mma_kernel": DSAPagedMMAKernel,
        "dsa_paged_cluster_kernel": DSAPagedClusterKernel,
        "dsa_paged_ws_kernel": DSAPagedWSKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"dsa_paged": DSAPagedFwdInterface}

    def __init__(
        self, sm_scale: float | None = None, head_dim_v: int = 512, *, target: Target = None
    ) -> None:
        """Build the op. Shapes are taken from each call.

        Args:
            sm_scale: Scale of the query-key scores; ``None`` means ``q``'s width to the power
                -0.5, 576 for the FP8 row.
            head_dim_v: The value width, the leading columns of the key; 512 for the FP8 row.
            target: Which set of kernels serves this op -- a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.sm_scale = sm_scale
        self.head_dim_v = head_dim_v
        super().__init__(target=target)

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        indices: torch.Tensor,
        *,
        attn_sink: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Attend each query row to the cache rows its positions resolve to.

        Args:
            q: ``bfloat16`` ``(batch, seq_len, heads, dk)``.
            kv_cache: ``uint8`` ``(num_pages, page_size, 656)`` FP8 latent rows, or
                ``bfloat16`` ``(num_pages, page_size, dk)`` rows.
            block_table: ``int32`` ``(batch, table_width)`` page of each page slot.
            cache_seqlens: ``int32`` ``(batch,)`` cache length of each request.
            indices: ``int32`` ``(batch, seq_len, topk)`` positions; -1 pads.
            attn_sink: ``float32`` ``(heads,)`` logits that join each head's softmax
                denominator.

        Returns:
            ``(o, lse)``: ``bfloat16`` ``(batch, seq_len, heads, head_dim_v)`` and ``float32``
            ``(batch, heads, seq_len)``, the head-major layout flash-attn returns.
        """
        batch, seq_len, heads, q_dim = q.shape
        return self.kernel_for(
            "dsa_paged",
            lambda: DSAPagedCall(
                batch=batch,
                seq_len=seq_len,
                heads=heads,
                topk=indices.shape[-1],
                page_size=kv_cache.shape[1],
                sm_scale=self.sm_scale,
                dim=self.head_dim_v,
                dim_tail=q_dim - self.head_dim_v,
                cache_dtype=kv_cache.dtype,
                has_attn_sink=attn_sink is not None,
                device=q.device,
            ),
        )(q, kv_cache, block_table, cache_seqlens, indices, attn_sink)

    def roof_key(self) -> str:
        """FLOPs are bfloat16 matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
