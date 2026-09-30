"""The facts of one Mamba-2 call that its in-tree kernels select and build on, and the kernel
interfaces their implementations inherit."""

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "CBProducerCall",
    "CBProducerFwdInterface",
    "DaCumsumCall",
    "DaCumsumFwdInterface",
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
class CBProducerCall(CallSpec):
    """One CB producer call, as the op knows it after reading its inputs."""

    batch: int = 0
    seq_len: int = 0
    n_groups: int = 0
    d_state: int = 0
    chunk_len: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class DaCumsumCall(CallSpec):
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


class CBProducerFwdInterface(KernelInterface):
    """The causally masked per-chunk ``C @ B^T`` matrix of one Mamba-2 layer."""

    request = CBProducerCall

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


class DaCumsumFwdInterface(KernelInterface):
    """The chunk-local inclusive prefix sum of ``dA = dt * A``, with the dt transform applied."""

    request = DaCumsumCall

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
