"""Mamba-1 S6 selective scan (host-side Op layer).

Provides:
  - SelectiveScanFwdOp: the selective scan of a Mamba-1 block, with per-channel discretization,
    an optional skip and gate, a carried state and packed sequences
"""

from typing import ClassVar, Mapping, Optional

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.mamba import (
    SelectiveScanCall,
    SelectiveScanChunkKernel,
    SelectiveScanFwdInterface,
    SelectiveScanStepKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["SelectiveScanFwdOp"]


class SelectiveScanFwdOp(Op):
    """Mamba-1 S6 selective scan: a first-order recurrence per channel and state whose decay
    and input both follow the input.

    With ``dt = softplus(delta + delta_bias)`` (``delta + delta_bias`` without
    ``delta_softplus``) and channel ``d`` reading group ``g = d // (Di / G)`` of ``B`` and
    ``C``::

        h[t, n] = exp(dt[d, t] * A[d, n]) * h[t - 1, n] + dt[d, t] * u[d, t] * B[g, n, t]
        y[d, t] = (sum_n C[g, n, t] * h[t, n] + D[d] * u[d, t]) * silu(z[d, t])

    ``h`` starts from ``initial_states`` or zeros, and restarts from zeros at every step whose
    ``seq_idx`` differs from the step before; ``final_states`` is ``h`` after the last step.
    Every product and sum is float32 and ``y`` rounds once to the input dtype. The
    exponentials and divisions use the hardware's approximate instructions, each within two
    float32 units in the last place; the softplus is within a few. Where ``dt * A <= 0`` a
    run of steps decays the state by the product of their decays, and a product below
    float32's normal range is zero: ``h[t, n]`` is off by at most ``|h[s, n]| * 2**-126``
    for the state ``h[s, n]`` the product scales, and ``y[d, t]`` by at most
    ``sum_n |C[g, n, t]| * |h[s, n]| * 2**-126``, the gate's factor aside. A state that
    overflows float32 stays infinite or NaN, which of the two unspecified, until the next
    restart; every step after a restart is exact.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "selective_scan_chunk": SelectiveScanChunkKernel,
        "selective_scan_step": SelectiveScanStepKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "selective_scan": SelectiveScanFwdInterface
    }

    def __init__(self, delta_softplus: bool = True, *, target: Target = None) -> None:
        """Build the op. Shapes, presence of the optional inputs and dtype come from each call.

        Args:
            delta_softplus: Whether ``dt`` is the softplus of ``delta + delta_bias``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.delta_softplus = delta_softplus
        super().__init__(target=target)

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        delta_bias: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scan every channel's states along the sequence axis.

        Args:
            u: ``(Bsz, Di, S)`` float16, bfloat16 or float32, the scanned input.
            delta: ``(Bsz, Di, S)`` like *u*, the step before ``delta_bias`` and softplus.
            A: float32 ``(Di, N)``, the continuous-time rates.
            B: ``(Bsz, G, N, S)`` like *u*, the input gate of each group.
            C: ``(Bsz, G, N, S)`` like *u*, the output read of each group.
            D: Optional float32 ``(Di,)`` skip.
            z: Optional ``(Bsz, Di, S)`` gate like *u*.
            delta_bias: Optional float32 ``(Di,)``, added to *delta*.
            seq_idx: Optional int32 ``(Bsz, S)``; the state restarts where it changes.
            initial_states: Optional float32 ``(Bsz, Di, N)``, the state before the first step.

        Returns:
            New ``(y, final_states)``: ``y`` ``(Bsz, Di, S)`` in *u*'s dtype and
            ``final_states`` float32 ``(Bsz, Di, N)``.
        """
        kernel = self.kernel_for(
            "selective_scan",
            lambda: SelectiveScanCall(
                dim=u.shape[1],
                seqlen=u.shape[2],
                dstate=A.shape[1],
                groups=B.shape[1],
                dtype=u.dtype,
                has_skip=D is not None,
                has_gate=z is not None,
                has_bias=delta_bias is not None,
                has_seq_idx=seq_idx is not None,
                has_init=initial_states is not None,
                softplus=self.delta_softplus,
                device=u.device,
            ),
        )
        contiguous = lambda t: None if t is None else t.contiguous()  # noqa: E731
        return kernel(
            u.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            B.contiguous(),
            C.contiguous(),
            contiguous(D),
            contiguous(z),
            contiguous(delta_bias),
            contiguous(seq_idx),
            contiguous(initial_states),
        )
