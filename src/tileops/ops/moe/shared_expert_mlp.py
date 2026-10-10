"""The shared expert of an MoE layer: a dense gated MLP."""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.moe import (
    SharedExpertMLPCall,
    SharedExpertMLPFwdInterface,
    SharedExpertMLPKernel,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["SharedExpertMLPFwdOp"]


class SharedExpertMLPFwdOp(Op):
    """Dense gated MLP: ``down(silu(hidden @ gate.T) * (hidden @ up.T))``.

    ``w_gate_up`` stacks the gate rows over the up rows. Both GEMMs accumulate in float32;
    the gated activation is rounded to the input dtype before the down projection.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "shared_expert_mlp": SharedExpertMLPKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "shared_expert_mlp": SharedExpertMLPFwdInterface
    }

    def __init__(
        self,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["D"])

    def forward(
        self, hidden_states: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
    ) -> torch.Tensor:
        """Apply the shared expert to every token.

        Args:
            hidden_states: $[T \\times H]$, ``float16`` or ``bfloat16``.
            w_gate_up: $[2S \\times H]$, the gate rows then the up rows.
            w_down: $[H \\times S]$.

        Returns:
            $[T \\times H]$ in the dtype of ``hidden_states``.
        """
        tokens, hidden = hidden_states.shape
        tensors = (hidden_states, w_gate_up, w_down)
        call = SharedExpertMLPCall(
            num_tokens=tokens,
            hidden_size=hidden,
            ffn_size=w_down.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return self.kernel_for("shared_expert_mlp", call)(*tensors)
