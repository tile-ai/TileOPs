"""The facts of one manifold hyper-connection call that its in-tree kernels select and build
on, and the kernel interfaces their implementations inherit."""

import dataclasses
from abc import abstractmethod

import torch

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import KernelInterface

__all__ = [
    "HyperConnectionHeadCall",
    "HyperConnectionHeadFwdInterface",
    "HyperConnectionPostCall",
    "HyperConnectionPostFwdInterface",
    "HyperConnectionPreCall",
    "HyperConnectionPreFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class HyperConnectionPreCall(CallSpec):
    """One pre step over ``n`` streams of ``c`` channels a token, with the constants the op
    fixed at construction.

    ``broadcast`` states that the residual's stream axis has stride 0, as a residual expanded
    from one stream has. The token axes ``*S`` are not part of the call: one kernel serves
    any nonzero token count.
    """

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

        Every tensor is on ``call.device``.

        Args:
            residual: bfloat16 ``[*S, call.n, call.c]`` of any strides and at least one
                token; with ``call.broadcast`` its stream axis has stride 0 and only the first
                stream is read.
            fn: float32 ``(call.n * call.n + 2 * call.n, call.n * call.c)``, contiguous.
            hc_scale: float32 ``(3,)``, contiguous.
            hc_base: float32 ``(call.n * call.n + 2 * call.n,)``, contiguous.

        Returns:
            New contiguous float32 ``post_mix`` ``[*S, call.n, 1]``, float32 ``comb_mix``
            ``[*S, call.n, call.n]`` and bfloat16 ``layer_input`` ``[*S, call.c]``.
        """


@dataclasses.dataclass(frozen=True)
class HyperConnectionPostCall(CallSpec):
    """One manifold hyper-connection post mix of ``n`` streams of ``c`` columns a token.

    The token axes ``*S`` are not part of the call: one kernel serves any token count.
    """

    n: int = 0
    c: int = 0
    mix_dtype: torch.dtype = torch.float32


class HyperConnectionPostFwdInterface(KernelInterface):
    """The block output and the residual streams mixed into ``n`` new streams."""

    request = HyperConnectionPostCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Mix each token's block output into its residual streams.

        Stream ``j`` of the output is ``post_layer_mix[j] * x + sum_i comb_res_mix[i, j] *
        residual[i]``, summed in float32 and rounded once to bfloat16. Nothing is written in
        place, and the output aliases no input.

        Args:
            x: ``[*S, call.c]`` bfloat16, contiguous.
            residual: ``[*S, call.n, call.c]`` bfloat16, contiguous.
            post_layer_mix: ``[*S, call.n, 1]`` in ``call.mix_dtype``, contiguous.
            comb_res_mix: ``[*S, call.n, call.n]`` in ``call.mix_dtype``, contiguous.

        Returns:
            A new contiguous ``[*S, call.n, call.c]`` bfloat16 tensor on ``call.device``.
        """


@dataclasses.dataclass(frozen=True)
class HyperConnectionHeadCall(CallSpec):
    """One manifold hyper-connection head collapse of ``n`` streams of ``c`` columns a token.

    The token axes ``*S`` are not part of the call: one kernel serves any token count.
    """

    n: int = 0
    c: int = 0
    rms_norm_eps: float = 1e-6
    hc_eps: float = 1e-6


class HyperConnectionHeadFwdInterface(KernelInterface):
    """The ``n`` residual streams collapsed into one under weights computed from them."""

    request = HyperConnectionHeadCall

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        """Collapse each token's streams into one.

        Per token, with ``x`` the streams flattened to ``n * c`` float32 values,
        ``pre = sigmoid((x @ hc_fn^T) * rsqrt(mean(x^2) + call.rms_norm_eps) * hc_scale +
        hc_base) + call.hc_eps``, and the output is ``sum_i pre_i * hidden_states_i``,
        rounded once to bfloat16. Nothing is written in place, and the output aliases no
        input.

        Args:
            hidden_states: ``[*S, call.n, call.c]`` bfloat16, contiguous.
            hc_fn: ``[call.n, call.n * call.c]`` float32, contiguous.
            hc_scale: ``[1]`` float32.
            hc_base: ``[call.n]`` float32.

        Returns:
            A new contiguous ``[*S, call.c]`` bfloat16 tensor on ``call.device``.
        """
