"""Depthwise causal short convolution of a Mamba block (host-side Op layer).

Provides:
  - CausalConv1dFwdOp: prefill over packed sequences, carrying a convolution state
  - CausalConv1dUpdateFwdOp: one decode token per sequence, advancing that state
"""

from typing import ClassVar, Mapping, Optional

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.mamba import (
    CausalConv1dCall,
    CausalConv1dFwdInterface,
    CausalConv1dKernel,
    CausalConv1dUpdateCall,
    CausalConv1dUpdateFwdInterface,
    CausalConv1dUpdateKernel,
)
from tileops._ops._offsets import check_offsets
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["CausalConv1dFwdOp", "CausalConv1dUpdateFwdOp"]


class CausalConv1dFwdOp(Op):
    """Causal depthwise convolution (width 2 to 4) with SiLU; writes ``conv_state`` in place.

    ``x`` holds the sequences token-major, ``cu_seqlens`` delimiting them. For a sequence on
    tokens ``[a, e)`` and channel ``c``, with ``h`` the ``W - 1`` tokens before it
    (``conv_state[s]`` where ``has_initial_state[s]``, zeros otherwise)::

        xp = concat(h[c, :], x[a:e, c])
        output[a + i, c] = act(bias[c] + sum_k weight[c, k] * xp[i + k])

    and ``conv_state[s]`` is overwritten in place with the sequence's last ``W - 1`` tokens
    of ``xp``; an empty sequence keeps its state. The products accumulate in float32 and each output rounds once to the input
    dtype. SiLU uses the hardware's approximate exponential and reciprocal, each within two
    float32 units in the last place.

    The caller guarantees well-formed ``cu_seqlens``, or passes ``validate_inputs=True``, which
    checks it on the CPU on every call.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"causal_conv1d": CausalConv1dKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "causal_conv1d": CausalConv1dFwdInterface
    }

    def __init__(
        self,
        activation: Optional[str] = "silu",
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes, the sequence lengths and dtype are taken from each call.

        Args:
            activation: ``"silu"`` to apply SiLU to the convolution, or ``None`` for none.
            validate_inputs: Check on the CPU that ``cu_seqlens`` starts at 0, never decreases
                and ends at ``x``'s token count.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.activation = activation
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def _check_input_values(self, x: torch.Tensor, *rest: Optional[torch.Tensor]) -> None:
        """With ``validate_inputs``, check the offsets against the packed tokens."""
        if self.validate_inputs:
            check_offsets("cu_seqlens", rest[2], x.shape[0])

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convolve every packed sequence and leave its last ``W - 1`` tokens in its state.

        Args:
            x: ``(Tt, D)`` float16 or bfloat16, the sequences packed token-major.
            weight: ``(D, W)``, one filter a channel.
            conv_state: ``(B, D, W - 1)``, contiguous; read where ``has_initial_state`` is
                set and overwritten in place.
            cu_seqlens: int32 ``(B + 1,)`` packed offsets, from 0 to ``Tt``.
            bias: Optional ``(D,)`` bias.
            has_initial_state: Optional bool ``(B,)``; absent, every sequence starts from
                zeros.

        Returns:
            A new ``(Tt, D)`` tensor in ``x``'s dtype.
        """
        kernel = self.kernel_for(
            "causal_conv1d",
            lambda: CausalConv1dCall(
                dim=x.shape[1],
                width=weight.shape[1],
                dtype=x.dtype,
                has_bias=bias is not None,
                has_init=has_initial_state is not None,
                silu=self.activation is not None,
                device=x.device,
            ),
        )
        return kernel(
            x.contiguous(),
            weight.contiguous(),
            conv_state,
            cu_seqlens.contiguous(),
            None if bias is None else bias.contiguous(),
            None if has_initial_state is None else has_initial_state.contiguous(),
        )


class CausalConv1dUpdateFwdOp(Op):
    """One decode step of ``CausalConv1dFwdOp``; advances ``conv_state`` in place.

    For sequence ``b`` and channel ``c``, with ``s = conv_state[b, c]`` its last ``W - 1``
    tokens::

        output[b, c] = act(bias[c] + sum_{k < W-1} weight[c, k] * s[k] + weight[c, W-1] * x[b, c])
        conv_state[b, c] <- (s[1], ..., s[W-2], x[b, c])

    The products accumulate in float32 and each output rounds once to the input dtype.
    SiLU uses the hardware's approximate exponential and reciprocal, each within two float32
    units in the last place.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "causal_conv1d_update": CausalConv1dUpdateKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "causal_conv1d_update": CausalConv1dUpdateFwdInterface
    }

    def __init__(self, activation: Optional[str] = "silu", *, target: Target = None) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            activation: ``"silu"`` to apply SiLU to the convolution, or ``None`` for none.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.activation = activation
        super().__init__(target=target)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convolve each sequence's new token with its state and shift the token into it.

        Args:
            x: ``(B, D)`` float16 or bfloat16, one token a sequence.
            weight: ``(D, W)``, one filter a channel.
            conv_state: ``(B, D, W - 1)``, contiguous, advanced by one token in place.
            bias: Optional ``(D,)`` bias.

        Returns:
            A new ``(B, D)`` tensor in ``x``'s dtype.
        """
        kernel = self.kernel_for(
            "causal_conv1d_update",
            lambda: CausalConv1dUpdateCall(
                dim=x.shape[1],
                width=weight.shape[1],
                dtype=x.dtype,
                has_bias=bias is not None,
                silu=self.activation is not None,
                device=x.device,
            ),
        )
        return kernel(
            x.contiguous(),
            weight.contiguous(),
            conv_state,
            None if bias is None else bias.contiguous(),
        )
