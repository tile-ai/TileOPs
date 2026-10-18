"""The facts of one Mamba-2 call that its in-tree kernels select and build on, and the kernel
interfaces their implementations inherit."""

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "CausalConv1dCall",
    "CausalConv1dFwdInterface",
    "CausalConv1dUpdateCall",
    "CausalConv1dUpdateFwdInterface",
    "SSDChunkCouplingCall",
    "SSDChunkCouplingFwdInterface",
    "SSDChunkCumsumCall",
    "SSDChunkCumsumFwdInterface",
    "SSDChunkScanCall",
    "SSDChunkScanFwdInterface",
    "SSDChunkStateCall",
    "SSDChunkStateFwdInterface",
    "SSDDecodeCall",
    "SSDDecodeFwdInterface",
    "SSDStatePassingCall",
    "SSDStatePassingFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class CausalConv1dCall(CallSpec):
    """One prefill convolution over packed sequences; their count and lengths are runtime data."""

    dim: int = 0
    width: int = 0
    dtype: Optional[torch.dtype] = None
    has_bias: bool = False
    has_init: bool = False
    silu: bool = True


@dataclasses.dataclass(frozen=True)
class CausalConv1dUpdateCall(CallSpec):
    """One decode step of the convolution, one token a sequence; the batch is runtime data."""

    dim: int = 0
    width: int = 0
    dtype: Optional[torch.dtype] = None
    has_bias: bool = False
    silu: bool = True


@dataclasses.dataclass(frozen=True)
class SSDChunkCouplingCall(CallSpec):
    """One CB producer call, as the op knows it after reading its inputs."""

    batch: int = 0
    seq_len: int = 0
    n_groups: int = 0
    d_state: int = 0
    chunk_len: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class SSDChunkCumsumCall(CallSpec):
    """One dA_cumsum call, with the dt transform the op fixed at construction."""

    batch: int = 0
    seq_len: int = 0
    n_heads: int = 0
    chunk_len: int = 0
    has_dt_bias: bool = False
    dt_softplus: bool = False
    dt_min: float = 0.0
    dt_max: float = float("inf")
    out_dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class SSDChunkScanCall(CallSpec):
    """One fused chunk-output call, as the op knows it after reading its inputs."""

    batch: int = 0
    num_chunks: int = 0
    chunk_len: int = 0
    n_heads: int = 0
    d_head: int = 0
    d_state: int = 0
    n_groups: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class SSDChunkStateCall(CallSpec):
    """One chunk-state call; ``dt`` may carry a dtype of its own."""

    batch: int = 0
    num_chunks: int = 0
    chunk_len: int = 0
    n_heads: int = 0
    d_head: int = 0
    d_state: int = 0
    n_groups: int = 0
    dtype: Optional[torch.dtype] = None
    dt_dtype: Optional[torch.dtype] = None
    has_seq_idx: bool = False


@dataclasses.dataclass(frozen=True)
class SSDDecodeCall(CallSpec):
    """One recurrent decode step, as the op knows it after reading its inputs."""

    batch: int = 0
    n_heads: int = 0
    d_head: int = 0
    d_state: int = 0
    n_groups: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class SSDStatePassingCall(CallSpec):
    """One inter-chunk scan, as the op knows it after reading its inputs."""

    batch: int = 0
    num_chunks: int = 0
    n_heads: int = 0
    d_state: int = 0
    has_initial_states: bool = False
    dtype: Optional[torch.dtype] = None


class CausalConv1dFwdInterface(KernelInterface):
    """The depthwise causal convolution of packed sequences, each carrying a convolution state."""

    request = CausalConv1dCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convolve each sequence along the token axis and leave its last W-1 tokens in its state.

        A sequence's first output reads the W-1 tokens before it from ``conv_state`` where
        ``has_initial_state`` is set, and zeros otherwise. Every tensor is on ``call.device``
        and contiguous; *conv_state* is written in place, no other input is.

        Args:
            x: ``(total, call.dim)`` in ``call.dtype``, the sequences packed token-major.
            weight: ``(call.dim, call.width)`` in ``call.dtype``.
            conv_state: ``(batch, call.dim, call.width - 1)`` in ``call.dtype``; overwritten
                with each sequence's last ``call.width - 1`` tokens.
            cu_seqlens: ``int32`` ``(batch + 1,)`` packed offsets, from 0 to ``total``.
            bias: ``(call.dim,)`` in ``call.dtype``, passed exactly when ``call.has_bias``.
            has_initial_state: ``bool`` ``(batch,)``, passed exactly when ``call.has_init``.

        Returns:
            A new ``(total, call.dim)`` output in ``call.dtype``, SiLU applied when ``call.silu``.
        """


class CausalConv1dUpdateFwdInterface(KernelInterface):
    """One decode token of the depthwise causal convolution, advancing each sequence's state."""

    request = CausalConv1dUpdateCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convolve each sequence's new token with its state, then shift the token into it.

        Every tensor is on ``call.device`` and contiguous; *conv_state* is written in place,
        no other input is.

        Args:
            x: ``(batch, call.dim)`` in ``call.dtype``, one token a sequence.
            weight: ``(call.dim, call.width)`` in ``call.dtype``.
            conv_state: ``(batch, call.dim, call.width - 1)`` in ``call.dtype``, the tokens
                before *x*; advanced by one token.
            bias: ``(call.dim,)`` in ``call.dtype``, passed exactly when ``call.has_bias``.

        Returns:
            A new ``(batch, call.dim)`` output in ``call.dtype``, SiLU applied when ``call.silu``.
        """


class SSDChunkCouplingFwdInterface(KernelInterface):
    """The causally masked per-chunk ``C @ B^T`` matrix of one Mamba-2 layer."""

    request = SSDChunkCouplingCall

    @abstractmethod
    def forward(self, C_mat: torch.Tensor, B_mat: torch.Tensor) -> torch.Tensor:
        """Contract each chunk's C rows with its B rows; nothing is written in place.

        Both tensors are contiguous on ``call.device`` in ``call.dtype``.

        Args:
            C_mat: ``(batch, seq_len, n_groups, d_state)``.
            B_mat: The same, for the keys.

        Returns:
            A new ``(batch, seq_len // chunk_len, n_groups, chunk_len, chunk_len)`` tensor in
            ``call.dtype``, zero above the diagonal of each chunk.
        """


class SSDChunkCumsumFwdInterface(KernelInterface):
    """The chunk-local inclusive prefix sum of ``dA = dt * A``, with the dt transform applied."""

    request = SSDChunkCumsumCall

    @abstractmethod
    def forward(
        self, dt: torch.Tensor, A: torch.Tensor, dt_bias: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform dt, then scan ``dt * A`` within each chunk; nothing is written in place.

        Every tensor is ``float32`` on ``call.device``; *dt* and *A* are contiguous, and the
        op passes *dt_bias* as the caller gave it.

        Args:
            dt: ``(batch, seq_len, n_heads)`` raw step sizes.
            A: ``(n_heads,)`` State Space Model (SSM) decay parameters.
            dt_bias: ``(n_heads,)`` per-head bias, passed exactly when ``call.has_dt_bias``.

        Returns:
            New ``(dt_out, dA_cumsum)``, both ``(batch, n_heads, num_chunks, chunk_len)``:
            ``dt_out`` in ``call.out_dtype``, ``dA_cumsum`` in ``float32`` and computed from
            the ``float32`` dt before the cast.
        """


class SSDChunkScanFwdInterface(KernelInterface):
    """State-Space Dual (SSD) chunk output: the history term and the intra-chunk term in one pass."""

    request = SSDChunkScanCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Add each chunk's history contribution to its causal decay; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: ``(batch, seq_len, n_heads, d_head)`` in ``call.dtype``.
            cb: ``(batch, num_chunks, n_groups, chunk_len, chunk_len)`` in ``call.dtype``.
            dA_cumsum: ``float32`` ``(batch, n_heads, num_chunks, chunk_len)``.
            C: ``(batch, seq_len, n_groups, d_state)`` in ``call.dtype``.
            prev_states: ``float32`` ``(batch, num_chunks, n_heads, d_head, d_state)``.
            dt: ``(batch, n_heads, num_chunks, chunk_len)`` in ``call.dtype``.

        Returns:
            A new ``float32`` ``(batch, seq_len, n_heads, d_head)`` output.
        """


class SSDChunkStateFwdInterface(KernelInterface):
    """The State-Space Dual (SSD) state each chunk ends in."""

    request = SSDChunkStateCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        seq_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate each chunk's decayed outer products; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: ``(batch, seq_len, n_heads, d_head)`` in ``call.dtype``.
            Bmat: ``(batch, seq_len, n_groups, d_state)`` in ``call.dtype``.
            dt: ``(batch, n_heads, num_chunks, chunk_len)`` in ``call.dt_dtype``.
            dA_cumsum: ``float32`` ``(batch, n_heads, num_chunks, chunk_len)``.
            seq_idx: ``int32`` ``(batch, seq_len)`` sequence ids. Read only when
                ``call.has_seq_idx``; otherwise it fills the slot and the built program
                has no branch reading it.

        Returns:
            A new ``float32`` ``(batch, num_chunks, n_heads, d_head, d_state)`` state.
        """


class SSDDecodeFwdInterface(KernelInterface):
    """One State-Space Dual (SSD) recurrent decode step over a caller-owned state."""

    request = SSDDecodeCall

    @abstractmethod
    def forward(
        self,
        A: torch.Tensor,
        dt: torch.Tensor,
        x: torch.Tensor,
        B_in: torch.Tensor,
        C_in: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the state by one token and read the output off it.

        Every tensor is contiguous on ``call.device``. *state* is written in place; no other
        input is.

        Args:
            A: ``float32`` ``(n_heads, d_head, d_state)`` decay parameters, at most zero.
            dt: ``float32`` ``(batch, n_heads, d_head)`` post-softplus step sizes.
            x: ``(batch, n_heads, d_head)`` in ``call.dtype``.
            B_in: ``(batch, n_groups, d_state)`` in ``call.dtype``.
            C_in: The same, for the output projection.
            state: ``float32`` ``(batch, n_heads, d_head, d_state)``, advanced in place.

        Returns:
            A new ``float32`` ``(batch, n_heads, d_head)`` output, without the skip
            connection or the output gate.
        """


class SSDStatePassingFwdInterface(KernelInterface):
    """The State-Space Dual (SSD) inter-chunk scan over chunk-end states."""

    request = SSDStatePassingCall

    @abstractmethod
    def forward(
        self,
        states: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        initial_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carry each chunk's state into the next; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            states: ``(batch, num_chunks, n_heads, d_state)`` in ``call.dtype``.
            dA_chunk_cumsum: ``float32`` ``(batch, n_heads, num_chunks)``.
            initial_states: ``float32`` ``(batch, n_heads, d_state)``. Read only when
                ``call.has_initial_states``; otherwise it fills the slot and the built
                program starts from zero.

        Returns:
            New ``(prev_states, final_states)`` in ``float32``: the state before each chunk,
            ``(batch, num_chunks, n_heads, d_state)``, and the state after the last one,
            ``(batch, n_heads, d_state)``.
        """
