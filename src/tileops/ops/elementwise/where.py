"""Where op: out = condition ? input : other (with broadcasting)."""

from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import WhereFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _PerDtypeKernels


class WhereFwdOp(_PerDtypeKernels, Op):
    """Where: out = condition ? input : other (with full PyTorch broadcasting).

    Conforms to ``torch.where(condition, input, other)``: ``condition`` is a
    bool tensor and ``input`` / ``other`` may broadcast with each other and
    with ``condition`` to produce the output. The three operand shapes arrive with
    the tensors; broadcasting them is the kernel's business.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"where": WhereFwdKernel}

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
                ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int):
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(n_total, ctor_dtype, tune=self.tune)

    def _eager_forward(
        self,
        condition: torch.Tensor,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        n_total = torch.broadcast_shapes(condition.shape, input.shape, other.shape).numel()
        condition = condition.contiguous()
        input = input.contiguous()
        other = other.contiguous()
        return self._kernel((condition, input, other), input.dtype, n_total)(
            condition, input, other
        )

    def forward(
        self,
        condition: torch.Tensor,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on ``condition``, ``input`` and ``other``."""
        return self._call_boundary(condition, input, other)
