"""The facts of one manifold hyper-connection call that its in-tree kernels select and build
on, and the kernel interfaces their implementations inherit."""

import dataclasses
from abc import abstractmethod

import torch

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import KernelInterface

__all__ = ["HyperConnectionPreCall", "HyperConnectionPreFwdInterface"]


@dataclasses.dataclass(frozen=True)
class HyperConnectionPreCall(CallSpec):
    """One pre step over ``tokens`` tokens of ``n`` streams of ``c`` channels, with the
    constants the op fixed at construction.

    ``broadcast`` states that every stream of a token is one stored stream, as a residual
    expanded from one stream with a stride-0 stream axis is.
    """

    tokens: int = 0
    n: int = 0
    c: int = 0
    broadcast: bool = False
    rms_eps: float = 0.0
    hc_pre_eps: float = 0.0
    hc_sinkhorn_eps: float = 0.0
    hc_post_mult_value: float = 0.0
    sinkhorn_repeat: int = 0


class HyperConnectionPreFwdInterface(KernelInterface):
    """The manifold hyper-connection pre step: the block input and the post and comb mixes."""

    request = HyperConnectionPreCall

    @abstractmethod
    def forward(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix the streams down to the block input; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            residual: bfloat16 ``(call.tokens, s, call.c)`` with ``s`` 1 when
                ``call.broadcast``, else ``call.n``.
            fn: float32 ``(call.n * call.n + 2 * call.n, call.n * call.c)``.
            hc_scale: float32 ``(3,)``.
            hc_base: float32 ``(call.n * call.n + 2 * call.n,)``.

        Returns:
            New float32 ``post_mix`` ``(call.tokens, call.n)``, float32 ``comb_mix``
            ``(call.tokens, call.n * call.n)`` and bfloat16 ``layer_input``
            ``(call.tokens, call.c)``.
        """
