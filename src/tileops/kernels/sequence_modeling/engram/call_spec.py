"""The facts of one Engram call that its in-tree kernels select and build on, and the kernel
interfaces their implementations inherit."""

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "EngramDecodeCall",
    "EngramDecodeFwdInterface",
    "EngramGateConvBwdInterface",
    "EngramGateConvCall",
    "EngramGateConvFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class EngramGateConvCall(CallSpec):
    """One GateConv call, with the extents and epsilon the op fixed at construction."""

    m: int = 0
    seq_len: int = 0
    d: int = 0
    eps: float = 0.0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class EngramDecodeCall(CallSpec):
    """One fused decode step, with the extents and conv geometry the op fixed."""

    batch: int = 0
    d_mem: int = 0
    d: int = 0
    max_conv_len: int = 0
    conv_kernel_size: int = 0
    dilation: int = 0
    eps: float = 0.0
    dtype: Optional[torch.dtype] = None


class EngramGateConvFwdInterface(KernelInterface):
    """Engram post-projection fusion: RMSNorm gating, causal depthwise conv, SiLU and residual."""

    request = EngramGateConvCall

    @abstractmethod
    def forward(
        self,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Gate, convolve and activate; nothing is written in place.

        Every tensor is contiguous on ``call.device`` in ``call.dtype``.

        Args:
            H: ``(m, seq_len, d)`` hidden states.
            k: ``(m, seq_len, d)`` key projection.
            v: ``(m, seq_len, d)`` value projection.
            rms_w_h: ``(d,)`` RMSNorm weight for *H* and *k*.
            rms_w_v: ``(d,)`` RMSNorm weight for the gated value.
            conv_w: ``(4, d)`` depthwise causal conv weights.

        Returns:
            New ``[Y, vhat, alpha, rrms_h, rrms_k, rrms_v]``: ``Y`` and ``vhat``
            ``(m, seq_len, d)`` in ``call.dtype``, then four ``float32`` ``(m, seq_len)``
            tensors the backward reads.
        """


class EngramGateConvBwdInterface(KernelInterface):
    """The gradients of the Engram GateConv fusion, from the forward's saved intermediates."""

    request = EngramGateConvCall

    @abstractmethod
    def forward(
        self,
        dY: torch.Tensor,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
        vhat: torch.Tensor,
        alpha: torch.Tensor,
        rrms_h: torch.Tensor,
        rrms_k: torch.Tensor,
        rrms_v: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Propagate *dY* back through the fusion; nothing is written in place.

        Every tensor is contiguous on ``call.device`` in ``call.dtype`` but the four
        ``float32`` per-row tensors, and the last five are what the forward returned.

        Args:
            dY: ``(m, seq_len, d)`` gradient of the output.
            H: ``(m, seq_len, d)`` forward input.
            k: ``(m, seq_len, d)`` forward input.
            v: ``(m, seq_len, d)`` forward input.
            rms_w_h: ``(d,)`` RMSNorm weight for *H* and *k*.
            rms_w_v: ``(d,)`` RMSNorm weight for ``vhat``.
            conv_w: ``(4, d)`` conv weights.
            vhat: ``(m, seq_len, d)`` saved from the forward.
            alpha: ``float32`` ``(m, seq_len)`` saved gate.
            rrms_h: ``float32`` ``(m, seq_len)`` saved reciprocal RMS of *H*.
            rrms_k: ``float32`` ``(m, seq_len)`` saved reciprocal RMS of *k*.
            rrms_v: ``float32`` ``(m, seq_len)`` saved reciprocal RMS of ``vhat``.

        Returns:
            New ``[dH, dk, dv, drms_w_h, drms_w_v, dconv_w]``: the first three
            ``(m, seq_len, d)`` in ``call.dtype``, then ``float32`` ``(d,)``, ``(d,)`` and
            ``(4, d)``.
        """


class EngramDecodeFwdInterface(KernelInterface):
    """One Engram decode token: projection, gating, dilated conv and SiLU in one launch."""

    request = EngramDecodeCall

    @abstractmethod
    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        conv_state: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run one token through the fusion; nothing is written in place.

        Every tensor is contiguous on ``call.device`` in ``call.dtype``.

        Args:
            e_t: ``(batch, d_mem)`` gathered N-gram embedding of the current token.
            h_t: ``(batch, d)`` hidden state of the current token.
            conv_state: ``(batch, length, d)`` conv history with ``length`` at most
                ``call.max_conv_len``, left-padded internally when it is shorter.
            W_K: ``(d_mem, d)`` key projection weight.
            W_V: ``(d_mem, d)`` value projection weight.
            rms_w_h: ``(d,)`` RMSNorm weight for *h_t* and the key projection.
            rms_w_v: ``(d,)`` RMSNorm weight for the gated value.
            conv_w: ``(call.conv_kernel_size, d)`` depthwise conv weights.

        Returns:
            New ``[y_t, new_conv_state]``: ``(batch, d)`` output to add as a residual, and
            ``(batch, call.max_conv_len, d)`` history for the next step.
        """
