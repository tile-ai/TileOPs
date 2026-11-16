"""Clamp op: Number or Tensor bounds."""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.elementwise import ClampFwdKernel, ClampTensorFwdKernel
from tileops._kernels.elementwise.call_spec import (
    BoundedUnaryFwdInterface,
    BoundsCall,
    ClampTensorCall,
    ClampTensorFwdInterface,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

# The interface of each form of the bounds.
_NUMBER_BOUNDS, _TENSOR_BOUNDS = "number_bounds", "tensor_bounds"


class ClampFwdOp(Op):
    """Clamp with Number or Tensor bounds (``torch.clamp(input, min, max)``).

    Each bound is a Number, a Tensor broadcasting with ``input``, or ``None``; at least one is
    given, and a Number bound is never paired with a Tensor bound, as in ``torch.clamp``'s two
    overloads. A single bound is ``torch.clamp_min`` / ``torch.clamp_max``.

    Which form and which bounds a call passes is read off the call, so one instance serves
    every form, one specialization each. A Number bound's value reaches the kernel's cache key,
    since the kernel is built for it.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "clamp": ClampFwdKernel,
        "clamp_tensor": ClampTensorFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        _NUMBER_BOUNDS: BoundedUnaryFwdInterface,
        _TENSOR_BOUNDS: ClampTensorFwdInterface,
    }

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
        input: torch.Tensor,
        min: float | torch.Tensor | None = None,
        max: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the op on ``input`` and whichever bounds the call passes.

        Args:
            input: Tensor to clamp.
            min: Lower bound: a Number, a Tensor broadcastable with ``input``, or ``None``
                for no lower bound.
            max: Upper bound, the same way. At least one bound is given, and both given
                bounds are Numbers or both Tensors.

        Returns:
            A new tensor of the broadcast shape of ``input`` and the Tensor bounds.
        """
        if isinstance(min, torch.Tensor) or isinstance(max, torch.Tensor):
            shapes = [t.shape for t in (input, min, max) if t is not None]
            n_total = torch.broadcast_shapes(*shapes).numel()
            input = input.contiguous()
            min = None if min is None else min.contiguous()
            max = None if max is None else max.contiguous()
            return self.kernel_for(
                _TENSOR_BOUNDS,
                lambda: ClampTensorCall(
                    device=input.device,
                    n_total=n_total,
                    dtype=input.dtype,
                    has_min=min is not None,
                    has_max=max is not None,
                ),
            )(input, min, max)
        input = input.contiguous()
        return self.kernel_for(
            _NUMBER_BOUNDS,
            lambda: BoundsCall(
                device=input.device,
                n_total=input.numel(),
                dtype=input.dtype,
                min_val=min,
                max_val=max,
            ),
        )(input)
