"""Where op: out = condition ? input : other (with broadcasting)."""

from typing import ClassVar, Mapping

import torch

from tileops._kernels.elementwise import WhereFwdKernel
from tileops._kernels.elementwise.call_spec import ElementwiseCall, WhereFwdInterface
from tileops._kernels.kernel_base import KernelInterface
from tileops._ops.elementwise._base import ELEMENTWISE
from tileops._ops.op_base import Op
from tileops.backend import Target


class WhereFwdOp(Op):
    """Where: out = condition ? input : other (with full PyTorch broadcasting).

    Conforms to ``torch.where(condition, input, other)``: ``condition`` is a
    bool tensor and ``input`` / ``other`` may broadcast with each other and
    with ``condition`` to produce the output. The three operand shapes arrive with
    the tensors; broadcasting them is the kernel's business.
    """

    kernel_types = {"where": WhereFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: WhereFwdInterface}

    def __init__(
        self,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self,
        condition: torch.Tensor,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on ``condition``, ``input`` and ``other``."""
        n_total = torch.broadcast_shapes(condition.shape, input.shape, other.shape).numel()
        condition = condition.contiguous()
        input = input.contiguous()
        other = other.contiguous()
        kernel = self.kernel_for(
            ELEMENTWISE,
            lambda: ElementwiseCall(device=input.device, n_total=n_total, dtype=input.dtype),
        )
        return kernel(condition, input, other)
