"""Depthwise causal short convolution of a Mamba block (host-side Op layer).

Provides:
  - CausalConv1dFwdOp: prefill over packed sequences, carrying a convolution state
  - CausalConv1dUpdateFwdOp: one decode token per sequence, advancing that state
"""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.mamba import (
    CausalConv1dCall,
    CausalConv1dFwdInterface,
    CausalConv1dKernel,
    CausalConv1dUpdateCall,
    CausalConv1dUpdateFwdInterface,
    CausalConv1dUpdateKernel,
    CausalConv1dUpdateRollbackKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["CausalConv1dFwdOp", "CausalConv1dUpdateFwdOp"]


class CausalConv1dFwdOp(Op):
    """Causal depthwise convolution (width 2 to 4) with SiLU; writes ``conv_state`` in place.

    ``x`` holds the sequences token-major, ``cu_seqlens`` delimiting them. For a sequence on
    tokens ``[a, e)`` and channel ``c``, with ``d`` the dilation and ``h`` the
    ``(W - 1) * d`` tokens before it (the first ``(W - 1) * d`` columns of ``conv_state[s]``
    where ``has_initial_state[s]``, zeros otherwise)::

        xp = concat(h[c, :], x[a:e, c])
        output[a + i, c] = act(bias[c] + sum_k weight[c, k] * xp[i + k * d])

    and those columns of ``conv_state[s]`` are overwritten in place with the sequence's last
    ``(W - 1) * d`` tokens of ``xp``; an empty sequence keeps its state. With
    ``cache_indices``, sequence ``s``'s state is ``conv_state[cache_indices[s]]``. The
    products accumulate in float32, a float16 or bfloat16 weight and bias read as float32,
    and each output rounds once to the input dtype. SiLU uses the hardware's approximate
    exponential and reciprocal, each within two float32 units in the last place.

    The caller guarantees well-formed ``cu_seqlens``, or passes ``validate_inputs=True``, which
    checks it on the CPU on every call.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"causal_conv1d": CausalConv1dKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "causal_conv1d": CausalConv1dFwdInterface
    }

    def __init__(
        self,
        activation: str | None = "silu",
        *,
        dilation: int = 1,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes, the sequence lengths and dtype are taken from each call.

        Args:
            activation: ``"silu"`` to apply SiLU to the convolution, or ``None`` for none.
            dilation: Tokens between adjacent filter taps.
            validate_inputs: Check on the CPU that ``cu_seqlens`` starts at 0, never decreases
                and ends at ``x``'s token count, and that ``cache_indices`` lies in its
                manifest range.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.activation = activation
        self.dilation = dilation
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bias: torch.Tensor | None,
        has_initial_state: torch.Tensor | None = None,
        *,
        cache_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convolve every packed sequence and leave its last tokens in its state.

        Args:
            x: ``(Tt, D)`` float16 or bfloat16, the sequences packed token-major.
            weight: ``(D, W)``, one filter a channel, in ``x``'s dtype or float32.
            conv_state: ``(B, D, L)``, or ``(P, D, L)`` with ``cache_indices``, contiguous,
                ``L >= (W - 1) * dilation``; read where ``has_initial_state`` is set and
                overwritten in place.
            cu_seqlens: int32 ``(B + 1,)`` packed offsets, from 0 to ``Tt``.
            bias: ``(D,)`` bias in ``weight``'s dtype, or ``None``.
            has_initial_state: Optional bool ``(B,)``; absent, every sequence starts from
                zeros.
            cache_indices: Optional int32 ``(B,)``, each sequence's distinct row of
                ``conv_state``.

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
                weight_dtype=weight.dtype,
                state_len=conv_state.shape[2],
                dilation=self.dilation,
                pooled=cache_indices is not None,
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
            None if cache_indices is None else cache_indices.contiguous(),
        )


class CausalConv1dUpdateFwdOp(Op):
    """Decode steps of ``CausalConv1dFwdOp``; advances ``conv_state`` in place.

    For sequence ``b`` and channel ``c``, with ``s = conv_state[b, c]`` its last ``W - 1``
    tokens::

        output[b, c] = act(bias[c] + sum_{k < W-1} weight[c, k] * s[k] + weight[c, W-1] * x[b, c])
        conv_state[b, c] <- (s[1], ..., s[W-2], x[b, c])

    In general, with ``d`` the dilation, a state is the first ``(W - 1) * d`` of its ``L``
    columns and the taps are ``d`` tokens apart, as in ``CausalConv1dFwdOp``.
    ``conv_state_indices`` maps sequence ``b`` to state ``conv_state[conv_state_indices[b]]``;
    ``query_start_loc`` packs several tokens a sequence in ``x``, each convolved in turn.
    With ``num_accepted_tokens``, as vLLM's ``causal_conv1d_update`` takes it, sequence ``b``
    with ``a = num_accepted_tokens[b]`` and ``q`` tokens reads its history from column
    ``a - 1``, and its first ``(W - 1) * d + q - 1`` columns become its last that many
    tokens, so the next call resumes after whichever draft was accepted. A sequence of more
    than ``L - (W - 1) * d + 1`` tokens breaks the call's contract, which no check enforces;
    the in-tree kernel then leaves the outputs of its tokens past that count unwritten.

    The products accumulate in float32 and each output rounds once to the input dtype.
    SiLU uses the hardware's approximate exponential and reciprocal, each within two float32
    units in the last place.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "causal_conv1d_update": CausalConv1dUpdateKernel,
        "causal_conv1d_update_rollback": CausalConv1dUpdateRollbackKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "causal_conv1d_update": CausalConv1dUpdateFwdInterface
    }

    def __init__(
        self,
        activation: str | None = "silu",
        *,
        dilation: int = 1,
        max_query_len: int = -1,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            activation: ``"silu"`` to apply SiLU to the convolution, or ``None`` for none.
            dilation: Tokens between adjacent filter taps.
            max_query_len: The most tokens a sequence holds, with ``query_start_loc``.
            validate_inputs: Check on the CPU that ``query_start_loc`` starts at 0, never
                decreases and ends at ``x``'s token count, and that ``conv_state_indices`` and
                ``num_accepted_tokens`` lie in their manifest ranges.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.activation = activation
        self.dilation = dilation
        self.max_query_len = max_query_len
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        conv_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convolve each sequence's new tokens with its state and shift them into it.

        Args:
            x: ``(B, D)`` float16 or bfloat16, one token a sequence; ``(Tt, D)``, the
                sequences packed, with ``query_start_loc``.
            weight: ``(D, W)``, one filter a channel, in ``x``'s dtype or float32.
            conv_state: ``(B, D, L)``, or ``(P, D, L)`` with ``conv_state_indices``,
                contiguous, advanced in place.
            bias: Optional ``(D,)`` bias, in ``weight``'s dtype.
            conv_state_indices: Optional int32 ``(B,)``, each sequence's distinct row of
                ``conv_state``.
            num_accepted_tokens: Optional int32 ``(B,)``, the drafts of the previous call
                each sequence accepted, at least one.
            query_start_loc: Optional int32 ``(B + 1,)`` packed offsets, from 0 to ``Tt``.

        Returns:
            A new tensor shaped like ``x``, in ``x``'s dtype.
        """
        kernel = self.kernel_for(
            "causal_conv1d_update",
            lambda: CausalConv1dUpdateCall(
                dim=x.shape[1],
                width=weight.shape[1],
                dtype=x.dtype,
                has_bias=bias is not None,
                silu=self.activation is not None,
                weight_dtype=weight.dtype,
                state_len=conv_state.shape[2],
                dilation=self.dilation,
                pooled=conv_state_indices is not None,
                varlen=query_start_loc is not None,
                rollback=num_accepted_tokens is not None,
                device=x.device,
            ),
        )
        return kernel(
            x.contiguous(),
            weight.contiguous(),
            conv_state,
            None if bias is None else bias.contiguous(),
            None if conv_state_indices is None else conv_state_indices.contiguous(),
            None if num_accepted_tokens is None else num_accepted_tokens.contiguous(),
            None if query_start_loc is None else query_start_loc.contiguous(),
        )
