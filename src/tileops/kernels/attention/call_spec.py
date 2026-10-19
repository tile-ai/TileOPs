"""The attention call specs and kernel interfaces.

``AttentionCall``'s properties are the region predicates more than one kernel class
reads. See docs/design/ops-design.md § Kernel selection.
"""

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "ATTENTION_DTYPES",
    "AttentionCall",
    "DSADecodeCall",
    "DSAPagedCall",
    "DSAPagedFwdInterface",
    "GQABwdInterface",
    "GQADenseFwdInterface",
    "GQAPagedFwdInterface",
    "GQAPrefillPagedFwdInterface",
    "GQAPreprocessBwdInterface",
    "GQAVarlenFwdInterface",
    "MHAPagedDecodeFwdInterface",
    "MLADecodeCall",
    "MLADecodeFwdInterface",
    "MLAKVCacheWriteCall",
    "MLAKVCacheWriteFwdInterface",
    "MLAPagedCall",
    "MLAPagedFwdInterface",
    "MLAVarlenCall",
    "MLAVarlenFwdInterface",
    "MergeAttentionStatesCall",
    "MergeAttentionStatesFwdInterface",
    "NSACall",
    "NSACompressedFwdInterface",
    "NSAFwdInterface",
    "NSATopKFwdInterface",
    "PagedKVCacheWriteCall",
    "PagedKVCacheWriteFwdInterface",
    "SparseMLADecodeFwdInterface",
]

ATTENTION_DTYPES = (torch.float16, torch.bfloat16)


@dataclasses.dataclass(frozen=True)
class AttentionCall(CallSpec):
    """What one attention call is, as the op knows it.

    Assembled in ``forward`` from op state plus what only the call knows: the
    element type, whether packed ranges are provably uniform, whether the inputs are
    FP8. Mutable device metadata is consumed by kernels, never read to construct this spec. The device fields come from ``CallSpec``.
    """

    dtype: Optional[torch.dtype] = None
    batch: int = 0
    heads: int = 0
    heads_kv: int = 0
    dim: int = 0
    max_seqlen_q: int = 0
    seqlen_kv: int = 0
    page_size: int = 0
    max_pages_per_req: int = 0
    is_causal: bool = False
    sm_scale: Optional[float] = None
    softcap: float = 0.0
    window_size_left: int = -1
    window_size_right: int = -1
    is_fp8: bool = False
    is_uniform: bool = True
    # Every packed KV range is empty, so a TMA descriptor over K/V has no extent.
    empty_kv: bool = False
    has_sinks: bool = False
    cache_dtype: Optional[torch.dtype] = None
    # The query's element type: ``dtype``, or FP8 for a query computed in a 16-bit ``dtype``.
    query_dtype: Optional[torch.dtype] = None
    # An FP8 cache's scales are per request and KV head rather than one per tensor.
    per_request_kv_scale: bool = False
    fuse_rope: bool = False
    max_position: Optional[int] = None
    rotary_dim: Optional[int] = None
    rope_layout: str = "neox"

    @property
    def rope_args(self) -> dict:
        """The fused-RoPE construction arguments the contiguous kernels take."""
        return {
            "fuse_rope": self.fuse_rope,
            "max_position": self.max_position if self.max_position is not None else 1,
            "rotary_dim": self.rotary_dim if self.rotary_dim is not None else 0,
            "rope_layout": self.rope_layout,
        }

    @property
    def uses_sliding_window(self) -> bool:
        """Whether either window bound is set, which restricts what may serve the call."""
        return self.window_size_left != -1 or self.window_size_right != -1

    @property
    def dense_decode_region(self) -> bool:
        """The contiguous decode region: one query position, no window, not FP8."""
        return not self.is_fp8 and self.max_seqlen_q == 1 and not self.uses_sliding_window

    @property
    def decode_bs1_region(self) -> bool:
        """The batch-1 decode shape the contiguous and paged batch-1 kernels share."""
        if not (
            self.batch == 1
            and self.dtype == torch.float16
            and self.dim == 128
            and self.softcap == 0.0
        ):
            return False
        if self.heads_kv <= 0 or self.heads % self.heads_kv != 0:
            return False
        return 1 <= self.heads // self.heads_kv <= 64

    @property
    def paged_decode_refusal(self) -> Optional[str]:
        """Why the paged-decode kernels cannot serve this call, or ``None`` when they can.

        They serve one query length shared by every request against a 16-bit cache of
        the query's dtype, with no window, RoPE or FP8.
        """
        if self.max_seqlen_q < 1 or not self.is_uniform:
            return "requires the same query length for every request"
        if self.dtype not in ATTENTION_DTYPES:
            return "requires float16 or bfloat16 Q"
        if self.cache_dtype != self.dtype:
            return "requires Q and KV to share a dtype"
        if self.is_fp8:
            return "does not serve FP8"
        if self.uses_sliding_window:
            return "does not serve sliding windows"
        if self.has_sinks:
            return "does not serve attention sinks"
        if self.fuse_rope:
            return "does not serve RoPE"
        return None

    @property
    def tensor_core_dim_refusal(self) -> Optional[str]:
        """The contiguous prefill and windowed kernels step the head dimension by one MMA
        k-slice."""
        return None if self.dim % 16 == 0 else "requires head dimension a multiple of 16"


@dataclasses.dataclass(frozen=True)
class MLADecodeCall(CallSpec):
    """One Multi-Head Latent Attention (MLA) decode call: shapes and element type."""

    batch: int = 0
    heads: int = 0
    heads_kv: int = 0
    seqlen_kv: int = 0
    dim: int = 0
    pe_dim: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class MLAPagedCall(CallSpec):
    """One paged Multi-Head Latent Attention (MLA) decode call, as the op knows it.

    The batch, the page pool and the block table's width are absent: a serving step changes
    them, so the kernel reads them from the tensors it is handed and one build serves every
    step of one model.
    """

    s_q: int = 0
    heads: int = 0
    dim: int = 0
    dim_v: int = 0
    page_size: int = 0
    is_causal: bool = True
    sm_scale: Optional[float] = None
    dtype: Optional[torch.dtype] = None
    cache_dtype: Optional[torch.dtype] = None

    @property
    def rows(self) -> int:
        """Query rows of one request: ``s_q * heads``, row ``s * heads + h``."""
        return self.s_q * self.heads

    @property
    def quantized(self) -> bool:
        """Whether the cache stores FP8 E4M3 and the call passes its scale."""
        return self.cache_dtype == torch.float8_e4m3fn


@dataclasses.dataclass(frozen=True)
class DSADecodeCall(CallSpec):
    """One sparse MLA decode call: shapes, element type and the op's fixed params."""

    batch: int = 0
    seq_len: int = 0
    seq_len_kv: int = 0
    heads: int = 0
    dim: int = 0
    tail_dim: int = 0
    dtype: Optional[torch.dtype] = None
    topk: int = 0
    kv_stride: int = 0
    q_start_index_s: int = 0
    kv_group: int = 1
    sm_scale: Optional[float] = None
    is_causal: bool = True
    cp0: bool = True


@dataclasses.dataclass(frozen=True)
class DSAPagedCall(CallSpec):
    """One sparse Multi-head Latent Attention (MLA) decode call over a paged FP8 cache.

    The cache contents, the block table and the selected positions stay device data.
    """

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    topk: int = 0
    page_size: int = 0
    sm_scale: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class NSACall(CallSpec):
    """One Native Sparse Attention (NSA) call over a packed batch.

    ``batch`` requests hold ``c_seq_len`` tokens in all; ``chunk_num`` counts the compressed
    chunks and ``selected_blocks`` the blocks each token keeps. ``block_size``, ``scale`` and
    ``is_causal`` are the op's fixed params. Each NSA interface reads the fields it needs.
    """

    batch: int = 0
    c_seq_len: int = 0
    heads: int = 0
    heads_kv: int = 0
    dim: int = 0
    dim_v: int = 0
    chunk_num: int = 0
    selected_blocks: int = 0
    block_size: int = 0
    scale: float = 1.0
    is_causal: bool = True
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class MLAVarlenCall(CallSpec):
    """What one packed-varlen MLA prefill call is, as the op knows it.

    Packed totals are absent: the kernel reads them from the tensors it is
    handed, so one built kernel serves every packing of these head shapes.
    """

    batch: int = 0
    heads: int = 0
    dim_nope: int = 0
    dim_pe: int = 0
    dim_v: int = 0
    is_causal: bool = True
    sm_scale: Optional[float] = None
    dtype: Optional[torch.dtype] = None


class GQADenseFwdInterface(KernelInterface):
    """Grouped-Query Attention (GQA) over dense Q, K and V."""

    request = AttentionCall

    @abstractmethod
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
        """Attend each query row to the keys the call's mask and window admit.

        Every tensor is contiguous on ``call.device``, and nothing is written in place.
        Q, K and V are in ``call.dtype``, or ``float8_e4m3fn`` when ``call.is_fp8``.

        Args:
            q: ``(batch, max_seqlen_q, heads, dim)``.
            k: ``(batch, seqlen_kv, heads_kv, dim)``.
            v: ``(batch, seqlen_kv, heads_kv, dim)``.
            q_scale: ``float32`` ``(batch, heads_kv)``, passed exactly when ``call.is_fp8``.
            k_scale: The same, for ``k``.
            v_scale: The same, for ``v``.
            rope_cos: ``(max_position, rotary_dim / 2)`` in ``call.dtype``, passed exactly
                when ``call.fuse_rope``.
            rope_sin: The same layout, passed exactly when ``rope_cos`` is.

        Returns:
            A new ``(batch, max_seqlen_q, heads, dim)`` output in ``call.dtype``.
        """


class GQAVarlenFwdInterface(KernelInterface):
    """Grouped-Query Attention (GQA) over packed variable-length requests."""

    request = AttentionCall

    @abstractmethod
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
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend each request's queries to its own keys; nothing is written in place.

        Every tensor is contiguous on ``call.device``; Q, K and V are in ``call.dtype``.

        Args:
            q: ``(total_q, heads, dim)``, requests back to back.
            k: ``(total_kv, heads_kv, dim)``.
            v: ``(total_kv, heads_kv, dim)``.
            cu_seqlens_q: ``int32`` ``(batch + 1,)`` request boundaries in ``q``.
            cu_seqlens_kv: ``int32`` ``(batch + 1,)`` request boundaries in ``k`` and ``v``.
            q_scale: FP8 scale, passed exactly when ``call.is_fp8``; so are the other two.
            k_scale: The same, for ``k``.
            v_scale: The same, for ``v``.
            rope_cos: RoPE table, passed exactly when ``call.fuse_rope``.
            rope_sin: The same, passed exactly when ``rope_cos`` is.
            sinks: Optional float32 ``(heads,)`` logits adding denominator-only mass.

        Returns:
            A new ``(total_q, heads, dim)`` output in ``call.dtype``.
        """


class GQAPagedFwdInterface(KernelInterface):
    """Grouped-Query Attention (GQA) of packed queries over a paged KV pool it only reads."""

    request = AttentionCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend each request's queries, aligned to the end of its cache.

        Every tensor is contiguous on ``call.device``, and nothing is written in place.

        Args:
            q: ``(total_q, heads, dim)`` in ``call.query_dtype``, requests back to back.
            k_pool: ``(seqlen_kv, heads_kv, dim)`` in ``call.cache_dtype``, ``page_size`` rows
                a page.
            v_pool: The same layout, for the values.
            cache_seqlens: ``int32`` ``(batch,)``, each cache's length, its queries included.
            page_table: ``int32`` ``(batch, max_pages_per_req)`` pool page of each logical page.
            cu_seqlens_q: ``int32`` ``(batch + 1,)`` request boundaries in *q*. An
                implementation serving ``call.is_uniform`` only reads the lengths from
                ``call.max_seqlen_q`` instead.
            q_scale: float32 ``(batch, heads_kv)`` dequantization scale of an FP8 query,
                present exactly when the query is FP8.
            k_scale: float32 dequantization scale of an FP8 cache, ``(1,)`` per tensor or
                ``(batch, heads_kv)`` per request and KV head; present exactly when the cache
                is FP8.
            v_scale: The same, for the values.
            rope_cos: ``(max_position, rotary_dim / 2)`` in ``call.dtype``, present
                exactly when ``call.fuse_rope``. Query positions are cache length minus
                query length plus the query index; key positions are logical cache indices.
                RoPE rotates the dequantized query and keys.
            rope_sin: The same layout as *rope_cos*. Cached keys are never rewritten.
            sinks: Optional float32 ``(heads,)`` logits contributing only to the denominator.

        Returns:
            A new output shaped like *q*, in ``call.dtype``.
        """


class MHAPagedDecodeFwdInterface(KernelInterface):
    """Multi-Head Attention (MHA) of BSHD queries over a paged KV pool it only reads."""

    request = AttentionCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Attend each request's queries, aligned to the end of its cache.

        Every tensor is contiguous on ``call.device``, and nothing is written in place.

        Args:
            q: ``(batch, max_seqlen_q, heads, dim)`` in ``call.dtype``.
            k_pool: ``(seqlen_kv, heads, dim)`` in ``call.cache_dtype``, ``page_size`` rows a
                page.
            v_pool: The same layout, for the values.
            real_seqlen_kv: ``int32`` ``(batch,)``, each cache's length.
            block_table: ``int32`` ``(batch, max_pages_per_req)`` pool page of each logical page.

        Returns:
            A new output shaped like *q*, in ``call.dtype``.
        """


class GQAPrefillPagedFwdInterface(KernelInterface):
    """Packed GQA prefill that appends its keys and values to a paged cache."""

    request = AttentionCall

    @abstractmethod
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
        max_seqlen_q: int,
        cos_table: Optional[torch.Tensor] = None,
        sin_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append the new tokens to each request's pages, then attend the chunk to the cache.

        Every tensor is on ``call.device`` and all but the pools are contiguous. ``k_pages``
        and ``v_pages`` are written in place: the rows the block table names past each
        ``cache_seqlens`` entry receive the new keys and values (rotated keys under
        ``call.fuse_rope``).

        Args:
            q: ``(total_q, heads, dim)`` in ``call.dtype``, requests back to back.
            k_new: ``(total_q, heads_kv, dim)`` in ``call.dtype``.
            v_new: The same, for the values.
            k_pages: ``(pool_rows, heads_kv, dim)`` in ``call.cache_dtype``.
            v_pages: The same, for the values.
            k_scale: ``float32`` ``(1,)`` dequantization scale of an FP8 pool.
            v_scale: The same, for ``v_pages``.
            cu_seqlens_q: ``int32`` ``(batch + 1,)`` request boundaries in ``q``.
            cache_seqlens: ``int32`` ``(batch,)`` cache lengths before the append.
            block_table: ``int32`` ``(batch, max_pages_per_req)``.
            max_seqlen_q: The longest request the launch covers.
            cos_table: ``(max_position, rotary_dim / 2)`` in ``call.dtype``, passed exactly
                when ``call.fuse_rope``.
            sin_table: The same layout, passed exactly when ``cos_table`` is.

        Returns:
            A new ``(total_q, heads, dim)`` output in ``call.dtype``.
        """


class GQAPreprocessBwdInterface(KernelInterface):
    """The row statistics a grouped-query attention backward consumes."""

    request = AttentionCall

    @abstractmethod
    def forward(self, o: torch.Tensor, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``delta = rowsum(o * do)`` and a zeroed ``dq`` accumulator.

        *o* and *do* are ``(batch, max_seqlen_q, heads, dim)``, contiguous in ``call.dtype`` on
        ``call.device``; nothing is written in place.

        Returns:
            New ``float32`` ``(delta (batch, heads, max_seqlen_q), dq_accum)``, ``dq_accum``
            zero-filled with as many elements as *o*.
        """


class GQABwdInterface(KernelInterface):
    """Grouped-Query Attention (GQA) backward from the saved log-sum-exp."""

    request = AttentionCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
        delta: torch.Tensor,
        dq_accum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the gradients of *q*, *k* and *v*; ``dq_accum`` is written in place.

        Every tensor is on ``call.device``, and ``do`` is contiguous.

        Args:
            q: ``(batch, max_seqlen_q, heads, dim)`` in ``call.dtype``.
            k: ``(batch, max_seqlen_q, heads_kv, dim)`` in ``call.dtype``.
            v: The same, for the values.
            do: Shaped like *q*.
            lse: ``float32`` ``(batch, heads, max_seqlen_q)`` from the forward pass.
            delta: What ``GQAPreprocessBwdInterface`` returned first.
            dq_accum: What it returned second; this call accumulates into it, in its own order.

        Returns:
            New ``(dq, dk, dv)`` shaped like *q*, *k* and *v*, in ``call.dtype``.
        """


class MLAPagedFwdInterface(KernelInterface):
    """Multi-Head Latent Attention (MLA) decode over a paged latent cache, absorbed form."""

    request = MLAPagedCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each request's query rows to its cache; nothing is written in place.

        Row ``s * call.heads + h`` of request ``b`` is query token ``s`` of head ``h``,
        at position ``cache_seqlens[b] - call.s_q + s``. Its keys are the request's first
        ``cache_seqlens[b]`` cache rows, whole (``call.dim`` wide), and its values their
        first ``call.dim_v`` columns. With ``call.is_causal`` a row sees the keys at or
        before its position; otherwise every key of its request. A cache row past
        ``cache_seqlens[b]`` is never read into a result, whatever it holds.

        Args:
            q: ``(batch, call.s_q, call.heads, call.dim)`` in ``call.dtype``, contiguous.
            kv_cache: ``(pages, call.page_size, call.dim)`` in ``call.cache_dtype``,
                contiguous; key ``n`` of request ``b`` is row
                ``n % call.page_size`` of page ``block_table[b, n // call.page_size]``.
            block_table: ``int32`` ``(batch, width)``, contiguous.
            cache_seqlens: ``int32`` ``(batch,)``, each in ``[call.s_q, width *
                call.page_size]``.
            kv_scale: ``float32`` ``(1,)``, the cache's dequantization scale, present
                exactly when ``call.quantized``.

        Returns:
            ``o`` ``(batch, call.s_q, call.heads, call.dim_v)`` in ``call.dtype`` and
            ``lse`` ``(batch, call.s_q, call.heads)`` float32, the natural-log
            log-sum-exp of each row's scaled scores.
        """


class MLADecodeFwdInterface(KernelInterface):
    """Multi-Head Latent Attention (MLA) decode of one query position against a cache."""

    request = MLADecodeCall

    @abstractmethod
    def forward(
        self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Attend each query head to the cache; nothing is written in place.

        Every tensor is contiguous in ``call.dtype`` on ``call.device``.

        Args:
            q: ``(batch, heads, dim)``.
            q_pe: ``(batch, heads, pe_dim)`` positional part.
            k: ``(batch, seqlen_kv, heads_kv, dim)``, also the values.
            k_pe: ``(batch, seqlen_kv, heads_kv, pe_dim)``.

        Returns:
            A new ``(batch, heads, dim)`` output.
        """


class MLAVarlenFwdInterface(KernelInterface):
    """MLA prefill over packed requests, after the latent is decompressed."""

    request = MLAVarlenCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Attend each query row to the keys its own request's mask admits.

        The key of head ``h`` is ``k_nope[:, h]`` followed by ``k_pe``, which is
        one row per token and shared by every head. Queries and keys are the
        same tokens, so ``cu_seqlens`` describes both and the causal mask sits
        on the diagonal.

        Every tensor is contiguous on ``call.device``, and nothing is written in
        place.

        Args:
            q: ``(total_tokens, heads, dim_nope + dim_pe)``.
            k_nope: ``(total_tokens, heads, dim_nope)``.
            k_pe: ``(total_tokens, dim_pe)``.
            v: ``(total_tokens, heads, dim_v)``.
            cu_seqlens: ``(batch + 1)`` packed request offsets.

        Returns:
            A new ``(total_tokens, heads, dim_v)`` output in ``call.dtype``, and
            its float32 ``(total_tokens, heads)`` log-sum-exp.
        """


class SparseMLADecodeFwdInterface(KernelInterface):
    """Sparse MLA decode: each query attends to the ``call.topk`` cache rows it indexes."""

    request = DSADecodeCall

    @abstractmethod
    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Attend each query row to its selected rows; nothing is written in place.

        Every tensor is contiguous on ``call.device``; *q* and *kv* are in ``call.dtype``.

        Args:
            q: ``(batch, seq_len, heads, dim + tail_dim)``.
            kv: ``(batch, seq_len_kv, kv_group, dim + tail_dim)``, the first ``dim`` columns
                also the values.
            indices: ``int32`` ``(batch, seq_len, kv_group, topk)``; ``seq_len_kv`` pads.

        Returns:
            A new ``(batch, seq_len, heads, dim)`` output.
        """


class DSAPagedFwdInterface(KernelInterface):
    """Sparse MLA decode: each query row attends to the cache rows its positions resolve to."""

    request = DSAPagedCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        indices: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Attend each query row to its selected positions; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            q: ``bfloat16`` ``(batch, seq_len, heads, 576)``.
            kv_cache: ``uint8`` ``(num_pages, page_size, 656)`` FP8 latent rows.
            block_table: ``int32`` ``(batch, table_width)`` physical page of each page slot.
            cache_seqlens: ``int32`` ``(batch,)``; a position at or past it is dead.
            indices: ``int32`` ``(batch, seq_len, topk)`` positions; -1 is dead.

        Returns:
            A new ``(batch, seq_len, heads, 512)`` output and its float32
            ``(batch, seq_len, heads)`` natural-log sum of exponentials.
        """


class NSATopKFwdInterface(KernelInterface):
    """Native Sparse Attention (NSA) block selection with FLA's importance scores."""

    request = NSACall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Pick the blocks each token attends to; nothing is written in place.

        Every tensor is on ``call.device``; integer inputs are ``int32``.

        Args:
            q: ``(c_seq_len, heads, dim)`` in ``call.dtype``.
            k_cmp: ``(chunk_num, heads_kv, dim)`` compressed keys.
            offsets: ``(batch + 1,)`` request boundaries.
            chunk_offsets: ``(batch + 1,)`` each request's first chunk.
            token_indices: ``(c_seq_len, 2)`` request id and in-request position.

        Returns:
            New ``int32`` ``(c_seq_len, heads_kv, selected_blocks)`` block ids.
        """


class NSAFwdInterface(KernelInterface):
    """NSA attention of each token to the blocks selected for it."""

    request = NSACall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Attend each token to its selected blocks; nothing is written in place.

        Every tensor is contiguous on ``call.device``; integer inputs are ``int32``.

        Args:
            q: ``(c_seq_len, heads, dim)`` in ``call.dtype``.
            k: ``(c_seq_len, heads_kv, dim)``.
            v: The same, for the values.
            block_indices: ``(c_seq_len, heads_kv, selected_blocks)``.
            block_counts: ``(c_seq_len, heads_kv)`` valid entries of ``block_indices``.
            offsets: ``(batch + 1,)`` request boundaries.
            token_indices: ``(c_seq_len, 2)`` request id and in-request position.

        Returns:
            A new ``(c_seq_len, heads, dim)`` output in ``call.dtype``.
        """


class NSACompressedFwdInterface(KernelInterface):
    """NSA attention of each token to its request's compressed chunks."""

    request = NSACall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each token to the closed chunks before it; nothing is written in place.

        Every tensor is on ``call.device``; integer inputs are ``int32``.

        Args:
            q: ``(c_seq_len, heads, dim)`` in ``call.dtype``.
            k_cmp: ``(chunk_num, heads_kv, dim)``.
            v_cmp: ``(chunk_num, heads_kv, dim_v)``.
            offsets: ``(batch + 1,)`` request boundaries.
            chunk_offsets: ``(batch + 1,)`` each request's first chunk.
            token_indices: ``(c_seq_len, 2)`` request id and in-request position.

        Returns:
            New ``(o (c_seq_len, heads, dim_v), lse (c_seq_len, heads))`` in ``call.dtype``.
        """


@dataclasses.dataclass(frozen=True)
class MergeAttentionStatesCall(CallSpec):
    """Value width and dtype; the flattened token/head count stays dynamic."""

    dim: int = 0
    dtype: Optional[torch.dtype] = None


class MergeAttentionStatesFwdInterface(KernelInterface):
    """Merge two read-only states over disjoint key sets using natural-log LSE."""

    request = MergeAttentionStatesCall

    @abstractmethod
    def forward(self, v_a, s_a, v_b, s_b) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge tensors on call.device, without modifying inputs.

        Args:
            v_a: First normalized values [tokens, heads, dim], call.dtype.
            s_a: First natural-log LSE [tokens, heads], float32; -inf means empty.
            v_b: Second normalized values, same layout and dtype as v_a.
            s_b: Second natural-log LSE, same layout and dtype as s_a.

        Returns:
            New (v [tokens, heads, dim], s [tokens, heads]) in the input dtypes. Both
            empty yields v=0, s=-inf; an empty state's values are ignored.
        """


@dataclasses.dataclass(frozen=True)
class PagedKVCacheWriteCall(CallSpec):
    """The cache geometry and dtypes of one write; the token count arrives with the tensors.

    The token count is deliberately absent. A serving process writes one token per decode
    step, so keying a build on it would compile a kernel per step; the rows are a dynamic
    extent of the tensors the call passes instead.
    """

    heads_kv: int = 0
    dim: int = 0
    num_pages: int = 0
    page_size: int = 0
    dtype: Optional[torch.dtype] = None
    cache_dtype: Optional[torch.dtype] = None

    @property
    def row_width(self) -> int:
        """Elements in one token's K or V row: ``heads_kv * dim``."""
        return self.heads_kv * self.dim

    @property
    def physical_tokens(self) -> int:
        """Rows in the page pool: ``num_pages * page_size``."""
        return self.num_pages * self.page_size

    @property
    def quantized(self) -> bool:
        """Whether the cache stores FP8 E4M3 and the call passes the scales."""
        return self.cache_dtype == torch.float8_e4m3fn


class PagedKVCacheWriteFwdInterface(KernelInterface):
    """Scatter each token's K and V rows into the page-pool row its slot names."""

    request = PagedKVCacheWriteCall

    @abstractmethod
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
        """Write the rows of the tokens whose slot is not ``-1``; returns no tensor.

        ``slot_mapping[n]`` is ``page * call.page_size + offset``, an index into the page
        pool flattened to ``call.physical_tokens`` rows; a negative slot skips that token.
        The caller owes distinct non-negative slots. A slot outside the pool is the
        caller's error; the op does not check it. Every row the call names is written in
        place and nothing else is: a row no token names keeps whatever it held.

        Args:
            k: ``(tokens, call.heads_kv, call.dim)`` in ``call.dtype`` on ``call.device``,
                contiguous. The row count is dynamic; a call of no tokens writes nothing.
            v: The same rows of values.
            k_pages: ``(call.num_pages, call.page_size, call.heads_kv, call.dim)`` in
                ``call.cache_dtype``, contiguous, written in place. Of the tokens with a
                slot, it stores ``k / k_scale`` rounded to nearest, ties to even, with saturation
                when ``call.quantized``, and ``k`` unchanged otherwise.
            v_pages: The same page pool, holding ``v`` under ``v_scale``.
            slot_mapping: ``int64`` ``(tokens,)``, the flattened row each token writes.
            k_scale: ``float32`` ``(1,)`` divisor of the stored keys, present exactly when
                ``call.quantized``. A non-positive value is the caller's error.
            v_scale: The same divisor for the stored values.

        Returns:
            ``None``; the two page pools are mutated and no other tensor is touched.
        """


@dataclasses.dataclass(frozen=True)
class MLAKVCacheWriteCall(CallSpec):
    """The latent cache geometry, dtypes and RoPE form of one write; the token count is absent.

    A decode step writes one token per running request, so a build keyed on the token count
    would compile a kernel per step; the rows are a dynamic extent of the tensors instead.
    """

    latent_dim: int = 0
    rope_dim: int = 0
    num_pages: int = 0
    page_size: int = 0
    dtype: Optional[torch.dtype] = None
    cache_dtype: Optional[torch.dtype] = None
    fuse_rope: bool = False
    rope_layout: str = "neox"
    cos_sin_dtype: Optional[torch.dtype] = None
    max_position: int = 0

    @property
    def row_width(self) -> int:
        """Elements in one cache row: ``latent_dim + rope_dim``."""
        return self.latent_dim + self.rope_dim

    @property
    def physical_tokens(self) -> int:
        """Rows in the cache: ``num_pages * page_size``."""
        return self.num_pages * self.page_size

    @property
    def quantized(self) -> bool:
        """Whether the cache stores FP8 E4M3 and the call passes the scale."""
        return self.cache_dtype == torch.float8_e4m3fn


class MLAKVCacheWriteFwdInterface(KernelInterface):
    """Write each token's ``kv_c`` concatenated with its ``k_pe`` into the cache row its slot names."""

    request = MLAKVCacheWriteCall

    @abstractmethod
    def forward(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
    ) -> None:
        """Write the rows of the tokens whose slot is not ``-1``; returns no tensor.

        ``slot_mapping[n]`` is ``page * call.page_size + offset``, an index into the cache
        flattened to ``call.physical_tokens`` rows; a negative slot skips that token. The
        caller owes distinct non-negative slots inside the cache; the op does not check them.
        Every row the call names is written in place and nothing else is.

        Args:
            kv_c: ``(tokens, call.latent_dim)`` in ``call.dtype`` on ``call.device``,
                contiguous. The row count is dynamic; a call of no tokens writes nothing.
            k_pe: ``(tokens, call.rope_dim)`` in ``call.dtype``, contiguous, never modified.
            kv_cache: ``(call.num_pages, call.page_size, call.row_width)`` in
                ``call.cache_dtype``, contiguous, written in place. A named row holds
                ``kv_c`` followed by ``k_pe``, the latter rotated when ``call.fuse_rope``:
                each pair is rotated in float32 and rounded once to ``call.dtype``. With
                ``call.quantized`` every value is then divided by ``scale`` and cast to
                E4M3FN, rounding to nearest even and saturating finite overflow and
                infinities; NaN stays NaN.
            slot_mapping: ``int64`` ``(tokens,)``, the flattened row each token writes.
            scale: ``float32`` ``(1,)`` divisor of the stored values, present exactly when
                ``call.quantized``. A value that is not a positive normal float is the
                caller's error.
            positions: ``int64`` ``(tokens,)``, the ``cos_sin_cache`` row each token's
                ``k_pe`` rotates by, present exactly when ``call.fuse_rope``.
            cos_sin_cache: ``(call.max_position, call.rope_dim)`` in ``call.cos_sin_dtype``:
                the cosines of a row's ``rope_dim / 2`` angles, then their sines. Pair ``i``
                is ``(k_pe[i], k_pe[i + rope_dim / 2])`` for ``call.rope_layout == "neox"``
                and ``(k_pe[2i], k_pe[2i + 1])`` for ``"interleaved"``.

        Returns:
            ``None``; the cache is mutated and no other tensor is touched.
        """
