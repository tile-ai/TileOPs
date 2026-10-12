"""Combine partial attention outputs over disjoint key sets."""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.attention.call_spec import (
    MergeAttentionStatesCall,
    MergeAttentionStatesFwdInterface,
)
from tileops.kernels.attention.merge_states import MergeAttentionStatesKernel
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op

__all__ = ["MergeAttentionStatesFwdOp"]


class MergeAttentionStatesFwdOp(Op):
    """Merge two attention states using their natural-log-sum-exp weights.

    A state with LSE ``-inf`` contributes nothing. If both states are empty, the
    result is a zero value and LSE ``-inf``. Inputs are read-only. FlashInfer's
    ``merge_state`` instead uses log2 LSE and returns NaN for two empty states.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "merge_attention_states_kernel": MergeAttentionStatesKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "merge_attention_states": MergeAttentionStatesFwdInterface
    }

    def __init__(self, *, target: Target = None) -> None:
        """Configure the implementation target; dimensions are resolved per call.

        Args:
            target: Backend target, or None to resolve from the input device.
        """
        super().__init__(target=target)

    def forward(
        self, v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine normalized values and log partition sums.

        Args:
            v_a: First partial value [tokens, heads, dim], FP16 or BF16.
            s_a: First natural-log-sum-exp [tokens, heads], FP32.
            v_b: Second partial value, with the same shape and dtype as v_a.
            s_b: Second natural-log-sum-exp, with the same shape and dtype as s_a.

        Returns:
            New (value, LSE) tensors with the respective input shapes and dtypes.
        """
        call = MergeAttentionStatesCall(dim=v_a.shape[-1], dtype=v_a.dtype, device=v_a.device)
        return self.kernel_for("merge_attention_states", call)(v_a, s_a, v_b, s_b)
