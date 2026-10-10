"""The facts of one GLAFwdOp call that its in-tree kernels select and build on,
and the kernel interface its implementations inherit."""

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Entry, KernelInterface
from tileops.kernels.linear_attention.call_spec import head_count_refusal

__all__ = [
    "GLACall",
    "GLAFwdInterface",
    "build_entry",
    "dense_refusal",
    "extents_refusal",
]


@dataclasses.dataclass(frozen=True)
class GLACall(CallSpec):
    """One inference call, as the op knows it after reading its inputs."""

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    dtype: Optional[torch.dtype] = None
    scale: float = 0.0
    varlen: bool = False
    # Whether the call supplies initial_state, rather than starting the recurrence from zero.
    has_initial_state: bool = False
    num_sequences: int = 0


class GLAFwdInterface(KernelInterface):
    """Gated Linear Attention (GLA) for inference: one prefill or decode step over caller-owned state."""

    request = GLACall

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrence from *initial_state* over the call's sequence.

        Every tensor is contiguous on ``call.device`` but ``cu_seqlens_cpu``, and nothing
        is written in place.

        Args:
            q: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            k: ``(batch, seq_len, heads, dim_k)`` in ``call.dtype``.
            v: ``(batch, seq_len, heads, dim_v)`` in ``call.dtype``.
            g: ``(batch, seq_len, heads, dim_k)`` log-space gate in ``call.dtype``.
            initial_state: ``float32`` ``(batch, heads, dim_k, dim_v)``, or ``None`` for zero.
            cu_seqlens: ``int64`` packed sequence offsets, passed exactly when ``call.varlen``.
            cu_seqlens_cpu: The same offsets on the CPU, or ``None``.

        Returns:
            New ``(o, final_state)``: ``o`` shaped like *v* in ``call.dtype``, and the
            ``float32`` ``(batch, heads, dim_k, dim_v)`` state after the last step.
        """


def dense_refusal(call: GLACall) -> Optional[str]:
    """Why *call* is not a dense (not packed) call with K = V in {64, 128} in fp16 or bf16."""
    if call.varlen:
        return "serves a dense call, not a packed one"
    return extents_refusal(call)


def extents_refusal(call: GLACall) -> Optional[str]:
    """Why the in-tree GLA kernels do not compile *call*'s head count, widths and dtype."""
    reason = head_count_refusal(call.heads)
    if reason is not None:
        return reason
    if call.dim_k != call.dim_v or call.dim_k not in (64, 128):
        return f"requires K = V in (64, 128), got {call.dim_k} and {call.dim_v}"
    if call.dtype not in (torch.float16, torch.bfloat16):
        return f"requires float16 or bfloat16, got {call.dtype}"
    return None


def build_entry(cls: type, call: GLACall, **build_arguments: int | bool) -> Entry:
    """Build *cls* from the call's scale, dtype and device plus the *build_arguments* it compiles."""
    device_index = call.device.index if call.device is not None else None
    arguments = dict(build_arguments, scale=call.scale, dtype=call.dtype, device_index=device_index)
    return tuple(sorted(arguments.items(), key=lambda item: item[0])), lambda: cls(**arguments)
