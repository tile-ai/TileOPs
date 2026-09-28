"""The shared expert of an MoE layer: a dense gated MLP."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.moe import SharedExpertMLPKernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["SharedExpertMLPFwdOp"]


class SharedExpertMLPFwdOp(Op):
    """Dense gated MLP: ``down(silu(hidden @ gate.T) * (hidden @ up.T))``.

    ``w_gate_up`` stacks the gate rows over the up rows. Both GEMMs accumulate in float32;
    the gated activation is rounded to the input dtype before the down projection.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "shared_expert_mlp": SharedExpertMLPKernel
    }

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["D"])

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per token count, width, expert width and dtype."""
        tokens, hidden, ffn, dtype = call
        return call, lambda: self.kernel_map[role](
            num_tokens=tokens, hidden_size=hidden, ffn_size=ffn, dtype=dtype, tune=self.tune
        )

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
        return self._call_boundary(hidden_states, w_gate_up, w_down)

    def _eager_forward(
        self, hidden_states: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        tokens, hidden = hidden_states.shape
        tensors = (hidden_states, w_gate_up, w_down)
        call = (tokens, hidden, w_down.shape[1], hidden_states.dtype)
        return self.kernel_for("shared_expert_mlp", tensors, call)(*tensors)
