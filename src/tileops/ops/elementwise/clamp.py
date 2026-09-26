"""Clamp ops: Tensor-bound bounds, and the scalar-bound form."""

from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import ClampFwdKernel, ClampTensorFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _PerDtypeKernels, _validate_scalar_param_repr


class ClampFwdOp(_PerDtypeKernels, Op):
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

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"clamp_tensor": ClampTensorFwdKernel}

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

    def _build(self, dtype: torch.dtype, n_total: int, has_min: bool, has_max: bool):
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(
            n_total,
            ctor_dtype,
            has_min=has_min,
            has_max=has_max,
            tune=self.tune,
        )

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
        kernel = self._kernel(
            (input, min, max),
            input.dtype,
            n_total,
            min is not None,
            max is not None,
        )
        return kernel(input, min, max)

    def forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the op on ``input`` and whichever bounds the call passes."""
        return self._call_boundary(input, min, max)


class ClampScalarFwdOp(_PerDtypeKernels, Op):
    """Scalar-bound clamp (``torch.clamp(input, min: Number|None, max: Number|None)``)."""

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"clamp": ClampFwdKernel}

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

    def _build(self, dtype: torch.dtype, n_total: int):
        """The bounds are baked into the kernel, so they are checked per dtype."""
        if self.min is not None:
            _validate_scalar_param_repr("min", self.min, dtype, self._slot)
        if self.max is not None:
            _validate_scalar_param_repr("max", self.max, dtype, self._slot)
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(
            n_total,
            ctor_dtype,
            min_val=self.min,
            max_val=self.max,
            tune=self.tune,
        )

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        return self._kernel((input,), input.dtype, input.numel())(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)
