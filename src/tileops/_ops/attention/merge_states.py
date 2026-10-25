"""Combine partial attention outputs over disjoint key sets."""

from typing import ClassVar, Mapping

import torch

from tileops._kernels.attention.call_spec import (
    MergeAttentionStatesCall,
    MergeAttentionStatesFwdInterface,
)
from tileops._kernels.attention.merge_states import MergeAttentionStatesKernel
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["MergeAttentionStatesFwdOp"]


class MergeAttentionStatesFwdOp(Op):
    """Merge two attention states using their log-sum-exp weights.

    The LSE base is natural by default; ``lse_base="2"`` reads and writes log2 LSE, as
    FlashInfer's ``merge_state`` does. A state with LSE ``-inf`` contributes nothing. If
    both states are empty, the result is a zero value and LSE ``-inf``, where
    ``merge_state`` returns NaN. Inputs are read-only.

    The LSE keeps ``merge_state``'s token-major ``[tokens, heads]`` layout. The packed
    attention forwards return a head-major ``[heads, tokens]`` LSE, as flash-attn does;
    pass its transpose.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "merge_attention_states_kernel": MergeAttentionStatesKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "merge_attention_states": MergeAttentionStatesFwdInterface
    }

    def __init__(self, lse_base: str = "e", *, target: Target = None) -> None:
        """Configure the LSE base and implementation target; dimensions are resolved per call.

        Args:
            lse_base: Logarithm base of the input and output LSE, ``"e"`` or ``"2"``.
            target: Backend target, or None to resolve from the input device.
        """
        self.lse_base = lse_base
        super().__init__(target=target)

    def forward(
        self, v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine normalized values and log partition sums.

        Args:
            v_a: First partial value [tokens, heads, dim], FP16 or BF16.
            s_a: First log-sum-exp in base ``lse_base`` [tokens, heads], FP32.
            v_b: Second partial value, with the same shape and dtype as v_a.
            s_b: Second log-sum-exp, with the same shape, dtype and base as s_a.

        Returns:
            New (value, LSE) tensors with the respective input shapes and dtypes.
        """
        return self.kernel_for(
            "merge_attention_states",
            lambda: MergeAttentionStatesCall(
                dim=v_a.shape[-1], dtype=v_a.dtype, lse_base=self.lse_base, device=v_a.device
            ),
        )(v_a, s_a, v_b, s_b)
