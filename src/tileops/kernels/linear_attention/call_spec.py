"""The facts of one DeltaNet, Gated DeltaNet or Gated Linear Attention (GLA) call that its
in-tree kernels select and build on, and the kernel interfaces their implementations inherit.

The GLA inference contract lives in ``tileops.kernels.linear_attention.gla.call_spec``.
"""

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "DeltaNetBwdInterface",
    "DeltaNetChunkCall",
    "DeltaNetDecodeCall",
    "DeltaNetDecodeFwdInterface",
    "DeltaNetFwdInterface",
    "DeltaNetInferenceCall",
    "DeltaNetInferenceFwdInterface",
    "GLABwdInterface",
    "GLAChunkCall",
    "GLADecodeCall",
    "GLADecodeFwdInterface",
    "GLAFwdInterface",
    "GatedDeltaNetCall",
    "GatedDeltaNetFwdInterface",
    "KDACall",
    "KDAFwdInterface",
    "head_count_refusal",
]


def head_count_refusal(*counts: int) -> Optional[str]:
    """Why no in-tree linear-attention kernel serves these head counts, or ``None``.

    Every model that runs a gated delta rule, a delta rule or GLA splits its state into an
    even number of heads, or into a single one, so an odd count above one is declined
    rather than tuned for.
    """
    for count in counts:
        if count != 1 and count % 2:
            return f"requires an even head count or a single head, got {count}"
    return None


@dataclasses.dataclass(frozen=True)
class DeltaNetChunkCall(CallSpec):
    """One chunked DeltaNet training call, forward or backward."""

    batch: int = 0
    heads: int = 0
    seq_len: int = 0
    chunk_size: int = 0
    dim_k: int = 0
    dim_v: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class DeltaNetDecodeCall(CallSpec):
    """One DeltaNet decode step, as the op knows it after reading its inputs."""

    batch: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class DeltaNetInferenceCall(CallSpec):
    """One ungated DeltaNet inference call, with the recurrence semantics the op fixed."""

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    dtype: Optional[torch.dtype] = None
    scale: float = 0.0
    l2norm: bool = False
    varlen: bool = False
    has_initial_state: bool = False
    num_sequences: int = 0


@dataclasses.dataclass(frozen=True)
class GLAChunkCall(CallSpec):
    """One chunked GLA training call, forward or backward."""

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    chunk_size: int = 0
    scale: float = 0.0
    dtype: Optional[torch.dtype] = None
    has_initial_state: bool = False


@dataclasses.dataclass(frozen=True)
class GLADecodeCall(CallSpec):
    """One GLA decode step, as the op knows it after reading its inputs."""

    batch: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    scale: float = 0.0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class GatedDeltaNetCall(CallSpec):
    """One gated DeltaNet inference call, with the recurrence semantics the op fixed."""

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    value_heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    dtype: Optional[torch.dtype] = None
    scale: float = 0.0
    has_initial_state: bool = False
    varlen: bool = False
    state_v_first: bool = False
    l2norm: bool = False
    gate_in_kernel: bool = False
    beta_sigmoid: bool = False
    allow_neg_eigval: bool = False
    num_sequences: int = 0


@dataclasses.dataclass(frozen=True)
class KDACall(CallSpec):
    """One Kimi Delta Attention (KDA) call, with the recurrence semantics the op fixed.

    Kimi Delta Attention is the gated delta rule whose decay is one log-space
    value per key channel, so ``g`` is as wide as the state's key axis rather
    than one number per head.
    """

    batch: int = 0
    seq_len: int = 0
    # Independent recurrences the call carries: the batch size, or the packed
    # sequence count under `varlen`, which `cu_seqlens`'s own shape states.
    sequences: int = 0
    heads: int = 0
    value_heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    dtype: Optional[torch.dtype] = None
    scale: float = 0.0
    has_initial_state: bool = False
    varlen: bool = False
    state_v_first: bool = False
    l2norm: bool = False
    gate_in_kernel: bool = False
    beta_sigmoid: bool = False
    allow_neg_eigval: bool = False
    bounded_gate: bool = False

    @property
    def chunk_refusal(self) -> Optional[str]:
        """Why no in-tree Kimi Delta Attention program serves this call, or ``None``.

        The recurrence variants and state layouts the in-tree pair does not
        implement, whatever the sequence length. Both implementations ask this
        first, then state the lengths and widths they serve themselves.
        """
        unsupported = [
            name
            for name, present in (
                ("state_v_first=True", self.state_v_first),
                ("use_gate_in_kernel=True", self.gate_in_kernel),
                ("use_beta_sigmoid_in_kernel=True", self.beta_sigmoid),
                ("allow_neg_eigval=True", self.allow_neg_eigval),
                ("a lower_bound", self.bounded_gate),
            )
            if present
        ]
        if unsupported:
            return "does not support " + ", ".join(unsupported)
        heads = head_count_refusal(self.heads, self.value_heads)
        if heads is not None:
            return heads
        if self.value_heads % self.heads != 0:
            return f"requires HV a multiple of H, got {self.value_heads} and {self.heads}"
        # The chunk-local half stages the value tile in the buffers it sized for a
        # key tile, so the two widths have to agree, as they do in every model that
        # runs this recurrence.
        if self.dim_k != self.dim_v or self.dim_k not in (64, 128):
            return f"serves K equal to V at 64 or 128, got {self.dim_k} and {self.dim_v}"
        return None


class DeltaNetFwdInterface(KernelInterface):
    """Chunked ungated delta rule, forward, keeping what the backward reads."""

    request = DeltaNetChunkCall

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run the chunked recurrence; nothing is written in place.

        Every tensor is on ``call.device`` in ``call.dtype``, in BHSD layout; the op passes
        them as the caller gave them, so an implementation reads their strides.

        Args:
            q: ``(batch, heads, seq_len, dim_k)``.
            k: ``(batch, heads, seq_len, dim_k)``.
            v: ``(batch, heads, seq_len, dim_v)``.
            beta: ``(batch, heads, seq_len)`` delta-rule step sizes.

        Returns:
            New ``(o, S, Aw, Au, w, u)``: the output ``(batch, heads, seq_len, dim_v)``, the
            ``float32`` chunk boundary states ``(batch, heads, num_chunks + 1, dim_k, dim_v)``,
            the two inverse matrices ``(batch, heads, seq_len, chunk_size)``, and the WY
            vectors ``(batch, heads, seq_len, dim_k)`` and ``(batch, heads, seq_len, dim_v)``.
            Every one but ``S`` is in ``call.dtype``.
        """


class DeltaNetBwdInterface(KernelInterface):
    """Chunked ungated delta rule, backward, from what the forward kept."""

    request = DeltaNetChunkCall

    @abstractmethod
    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagate *do* back through the chunked recurrence; nothing is written in place.

        Every tensor is on ``call.device`` in BHSD layout, in ``call.dtype`` but the
        ``float32`` *S*; the last five are what the forward returned. The op passes them as
        the caller gave them, so an implementation reads their strides.

        Args:
            do: ``(batch, heads, seq_len, dim_v)`` gradient of the output.
            q: ``(batch, heads, seq_len, dim_k)``.
            k: ``(batch, heads, seq_len, dim_k)``.
            v: ``(batch, heads, seq_len, dim_v)``.
            beta: ``(batch, heads, seq_len)``.
            S: ``float32`` ``(batch, heads, num_chunks + 1, dim_k, dim_v)`` boundary states.
            Aw: ``(batch, heads, seq_len, chunk_size)``.
            Au: ``(batch, heads, seq_len, chunk_size)``.
            w: ``(batch, heads, seq_len, dim_k)``.
            u: ``(batch, heads, seq_len, dim_v)``.

        Returns:
            New ``(dq, dk, dv, dbeta)`` shaped like *q*, *k*, *v* and *beta*.
        """


class DeltaNetDecodeFwdInterface(KernelInterface):
    """One ungated delta-rule step over a caller-owned recurrent state."""

    request = DeltaNetDecodeCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the state by one token and read the output off it; nothing is written in place.

        Every tensor is on ``call.device`` in ``call.dtype``, in BHD layout; the op passes
        them as the caller gave them, so an implementation reads their strides.

        Args:
            q: ``(batch, heads, dim_k)``.
            k: ``(batch, heads, dim_k)``.
            v: ``(batch, heads, dim_v)``.
            beta: ``(batch, heads)`` delta-rule step size.
            state: ``(batch, heads, dim_k, dim_v)`` recurrent state.

        Returns:
            New ``(o, new_state)``: ``(batch, heads, dim_v)`` and a new state shaped like
            *state*, both in ``call.dtype`` with ``float32`` accumulation.
        """


class DeltaNetInferenceFwdInterface(KernelInterface):
    """Ungated delta rule for inference: one prefill or decode step over caller-owned state."""

    request = DeltaNetInferenceCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrence from *initial_state* over the call's sequence.

        Every tensor is contiguous on ``call.device`` but ``cu_seqlens_cpu``, and nothing is
        written in place. ``beta`` already carries the transformed update strength.

        Args:
            q: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            k: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            v: ``(batch, seq_len, heads, dim_v)`` in ``call.dtype``.
            beta: ``(batch, seq_len, heads)`` in ``call.dtype``.
            initial_state: ``float32`` ``(batch, heads, dim_k, dim_v)``, or ``None`` for zero.
            cu_seqlens: ``int64`` packed sequence offsets, passed exactly when ``call.varlen``.
            cu_seqlens_cpu: The same offsets on the CPU, or ``None``.

        Returns:
            New ``(o, final_state)``: ``o`` shaped like *v* in ``call.dtype``, and the
            ``float32`` ``(batch, heads, dim_k, dim_v)`` state after the last step.
        """


class GLAFwdInterface(KernelInterface):
    """Chunked Gated Linear Attention (GLA), forward, keeping the states the backward reads."""

    request = GLAChunkCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the chunked gated recurrence; nothing is written in place.

        Every tensor is on ``call.device`` in BTHD layout; the op passes them as the caller
        gave them, so an implementation reads their strides.

        Args:
            q: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            k: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            v: ``(batch, seq_len, heads, dim_v)`` in ``call.dtype``.
            g: ``(batch, seq_len, heads, dim_k)`` log-space forget gates in ``call.dtype``.
            initial_state: ``float32`` ``(batch, heads, dim_k, dim_v)``, passed exactly
                when ``call.has_initial_state``.

        Returns:
            New ``(o, final_state)``: the output shaped like *v* in ``call.dtype``, and the
            ``float32`` ``(batch, heads, dim_k, dim_v)`` state after the last chunk.
        """


class GLABwdInterface(KernelInterface):
    """Chunked Gated Linear Attention (GLA), backward, from the forward's hidden states."""

    request = GLAChunkCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        has_initial_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagate *do* and *dht* back through the recurrence; nothing is written in place.

        Every tensor is on ``call.device`` in BTHD layout; the op passes them as the caller
        gave them, so an implementation reads their strides.

        Args:
            q: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            k: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            v: ``(batch, seq_len, heads, dim_v)`` in ``call.dtype``.
            g: ``(batch, seq_len, heads, dim_k)`` log-space forget gates in ``call.dtype``.
            h: ``float32`` ``(batch, num_chunks + 1, heads, dim_k, dim_v)`` from the forward.
            do: ``(batch, seq_len, heads, dim_v)`` gradient of the output.
            dht: ``float32`` ``(batch, heads, dim_k, dim_v)`` gradient of the final state.
            has_initial_state: ``call.has_initial_state``, repeated at the launch because
                it decides whether the first chunk's gradient is kept, not what is built.

        Returns:
            New ``float32`` ``(dq, dk, dv, dg)``, shaped like *q*, *k*, *v* and *g*.
        """


class GLADecodeFwdInterface(KernelInterface):
    """One Gated Linear Attention (GLA) step over a caller-owned recurrent state."""

    request = GLADecodeCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the state by one token and read the output off it; nothing is written in place.

        Every tensor is on ``call.device`` in ``call.dtype``, in BHD layout; the op passes
        them as the caller gave them, so an implementation reads their strides.

        Args:
            q: ``(batch, heads, dim_k)``.
            k: ``(batch, heads, dim_k)``.
            v: ``(batch, heads, dim_v)``.
            gk: ``(batch, heads, dim_k)`` log-space key gate.
            state: ``(batch, heads, dim_k, dim_v)`` recurrent state.

        Returns:
            New ``(o, new_state)``: ``(batch, heads, dim_v)`` and a new state shaped like
            *state*, both in ``call.dtype`` with ``float32`` accumulation.
        """


class GatedDeltaNetFwdInterface(KernelInterface):
    """Gated delta rule for inference: one prefill or decode step over caller-owned state."""

    request = GatedDeltaNetCall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrence from *initial_state* over the call's sequence.

        Every tensor is contiguous on ``call.device`` but ``cu_seqlens_cpu``, and nothing is
        written in place. Unless ``call.gate_in_kernel``, *g* already carries the log-space
        decay; unless ``call.beta_sigmoid``, *beta* already carries the update strength.

        Args:
            q: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            k: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            v: ``(batch, seq_len, value_heads, dim_v)`` in ``call.dtype``.
            g: ``(batch, seq_len, value_heads)`` in ``call.dtype``.
            beta: ``(batch, seq_len, value_heads)`` in ``call.dtype``.
            initial_state: ``float32`` ``(n, value_heads, dim_k, dim_v)``, value-major
                ``(n, value_heads, dim_v, dim_k)`` when ``call.state_v_first``, or ``None``
                for zero. ``n`` is the batch size, or the packed sequence count under
                ``call.varlen``.
            cu_seqlens: ``int64`` packed sequence offsets, passed exactly when ``call.varlen``.
            cu_seqlens_cpu: The same offsets on the CPU, or ``None``.
            A_log: ``float32`` ``(value_heads,)``, passed exactly when ``call.gate_in_kernel``.
            dt_bias: ``float32`` ``(value_heads,)``, passed exactly when ``call.gate_in_kernel``.

        Returns:
            New ``(o, final_state)``: ``o`` shaped like *v* in ``call.dtype``, and the
            ``float32`` state after the last step, laid out like *initial_state*.
        """


class KDAFwdInterface(KernelInterface):
    """Kimi Delta Attention for inference: prefill or decode over caller-owned state."""

    request = KDACall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrence from *initial_state* over the call's sequence.

        Every tensor is contiguous on ``call.device`` but ``cu_seqlens_cpu``, and
        nothing is written in place. Unless ``call.gate_in_kernel``, *g* already
        carries the log-space decay of each key channel; unless
        ``call.beta_sigmoid``, *beta* already carries the update strength.

        Args:
            q: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            k: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            v: ``(batch, seq_len, value_heads, dim_v)`` in ``call.dtype``.
            g: ``(batch, seq_len, value_heads, dim_k)`` log-space per-channel
                decay in ``call.dtype``.
            beta: ``(batch, seq_len, value_heads)`` in ``call.dtype``.
            initial_state: ``float32`` ``(n, value_heads, dim_k, dim_v)``, or
                ``None`` for zero. ``n`` is the batch size, or the packed
                sequence count under ``call.varlen``.
            cu_seqlens: ``int64`` packed sequence offsets, passed exactly when
                ``call.varlen``.
            cu_seqlens_cpu: The same offsets on the CPU, or ``None``.
            A_log: ``float32`` ``(value_heads,)``, passed exactly when
                ``call.gate_in_kernel``.
            dt_bias: ``float32`` ``(value_heads * dim_k,)``, or ``None``.

        Returns:
            New ``(o, final_state)``: ``o`` shaped like *v* in ``call.dtype``, and
            the ``float32`` ``(n, value_heads, dim_k, dim_v)`` state after the
            last step.
        """
