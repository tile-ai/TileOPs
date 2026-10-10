"""Clamp ops: Tensor-bound bounds, and the scalar-bound form."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import ClampFwdKernel, ClampTensorFwdKernel
from tileops.kernels.elementwise.call_spec import (
    BoundedUnaryFwdInterface,
    BoundsCall,
    ClampTensorCall,
    ClampTensorFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.elementwise._base import ELEMENTWISE, UnaryOp
from tileops.ops.op_base import Op


class ClampTensorFwdOp(Op):
    """Clamp with Tensor lower and/or upper bounds (broadcasting).

    Conforms to ``torch.clamp(input, min, max)`` where ``min`` and ``max``
    are each either a Tensor or ``None``. At least one of the two bounds
    must be a Tensor. All Tensor operands broadcast together. A single bound
    is ``torch.clamp_min`` / ``torch.clamp_max``.

    Which bounds this call carries is read off the call, not settled at
    construction, and it reaches the kernel's cache key because it changes what
    gets built. One instance therefore serves ``clamp``, ``clamp_min`` and
    ``clamp_max``, one specialization each.
    """

    kernel_types = {"clamp_tensor": ClampTensorFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: ClampTensorFwdInterface
    }

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
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _eager_forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shapes = [t.shape for t in (input, min, max) if t is not None]
        n_total = torch.broadcast_shapes(*shapes).numel()
        input = input.contiguous()
        min = None if min is None else min.contiguous()
        max = None if max is None else max.contiguous()
        call = ClampTensorCall(
            device=input.device,
            n_total=n_total,
            dtype=input.dtype,
            has_min=min is not None,
            has_max=max is not None,
        )
        return self.kernel_for(ELEMENTWISE, call)(input, min, max)

    def forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the op on ``input`` and whichever bounds the call passes."""
        return self._call_boundary(input, min, max)


class ClampScalarFwdOp(UnaryOp):
    """Scalar-bound clamp (``torch.clamp(input, min: Number|None, max: Number|None)``)."""

    kernel_types = {"clamp": ClampFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BoundedUnaryFwdInterface
    }

    def __init__(
        self,
        *,
        min: Optional[float] = None,
        max: Optional[float] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            min: Lower bound (Number or None).
            max: Upper bound (Number or None); at least one bound is given.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.min = min
        self.max = max
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _call_spec(self, input: torch.Tensor) -> BoundsCall:
        return BoundsCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            min_val=self.min,
            max_val=self.max,
        )
