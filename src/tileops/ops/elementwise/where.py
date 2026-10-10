"""Where op: out = condition ? input : other (with broadcasting)."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import WhereFwdKernel
from tileops.kernels.elementwise.call_spec import ElementwiseCall, WhereFwdInterface
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.elementwise._base import ELEMENTWISE
from tileops.ops.op_base import Op


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
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional dispatch override mapping kernel keys to
                ``Kernel`` subclasses. Falls back to ``kernel_types``.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

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
        call = ElementwiseCall(device=input.device, n_total=n_total, dtype=input.dtype)
        kernel = self.kernel_for(ELEMENTWISE, call)
        return kernel(condition, input, other)
