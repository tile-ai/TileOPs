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
    "GQABwdInterface",
    "GQADenseFwdInterface",
    "GQAPagedFwdInterface",
    "GQAPrefillPagedFwdInterface",
    "GQAPreprocessBwdInterface",
    "GQAVarlenFwdInterface",
    "MHAPagedDecodeFwdInterface",
    "MLADecodeCall",
    "MLADecodeFwdInterface",
    "MLAVarlenCall",
    "MLAVarlenFwdInterface",
    "NSACall",
    "NSACompressedFwdInterface",
    "NSAFwdInterface",
    "NSATopKFwdInterface",
    "SparseMLADecodeFwdInterface",
]

ATTENTION_DTYPES = (torch.float16, torch.bfloat16)


@dataclasses.dataclass(frozen=True)
class AttentionCall(CallSpec):
    """What one attention call is, as the op knows it.

    Assembled in ``forward`` from op state plus what only the call knows: the
    element type, whether the packed ranges are uniform, whether the inputs are
    FP8. The device fields come from ``CallSpec``.
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
    cache_dtype: Optional[torch.dtype] = None
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
    """Grouped-query attention over dense Q, K and V."""

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
    """Grouped-query attention over packed variable-length requests."""

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

        Returns:
            A new ``(total_q, heads, dim)`` output in ``call.dtype``.
        """


class GQAPagedFwdInterface(KernelInterface):
    """Grouped-query attention of packed queries over a paged KV pool it only reads."""

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
    ) -> torch.Tensor:
        """Attend each request's queries, aligned to the end of its cache.

        Every tensor is contiguous on ``call.device``, and nothing is written in place.

        Args:
            q: ``(total_q, heads, dim)`` in ``call.dtype``, requests back to back.
            k_pool: ``(seqlen_kv, heads_kv, dim)`` in ``call.cache_dtype``, ``page_size`` rows
                a page.
            v_pool: The same layout, for the values.
            cache_seqlens: ``int32`` ``(batch,)``, each cache's length, its queries included.
            page_table: ``int32`` ``(batch, max_pages_per_req)`` pool page of each logical page.
            cu_seqlens_q: ``int32`` ``(batch + 1,)`` request boundaries in *q*. An
                implementation serving ``call.is_uniform`` only reads the lengths from
                ``call.max_seqlen_q`` instead.

        Returns:
            A new output shaped like *q*, in ``call.dtype``.
        """


class MHAPagedDecodeFwdInterface(KernelInterface):
    """Multi-head attention of BSHD queries over a paged KV pool it only reads."""

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
    """Grouped-query attention backward from the saved log-sum-exp."""

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
