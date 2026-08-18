"""Clamp ops: Tensor-bound bounds, and the scalar-bound form."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import ClampFwdKernel, ClampTensorFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import (
    KernelEntry,
    _ClampTensorBase,
    _PerDtypeKernels,
    _validate_scalar_param_repr,
    resolve_output_dtype,
)


class ClampFwdOp(_PerDtypeKernels, _ClampTensorBase):
    """Clamp with Tensor lower and/or upper bounds (broadcasting).

    Conforms to ``torch.clamp(input, min, max)`` where ``min`` and ``max``
    are each either a Tensor or ``None``. At least one of the two bounds
    must be a Tensor. All Tensor operands broadcast together. A single bound
    is ``torch.clamp_min`` / ``torch.clamp_max``.

    The bounds given here decide which call this instance serves: the
    broadcast output shape is settled at construction, and it depends on
    which bounds are present. ``forward`` rejects a call whose bounds differ
    from them, so a different combination needs a different instance — the
    same as a different input shape does.

    Args:
        input: Shape of the input tensor.
        min: Shape of the lower-bound tensor, or ``None`` for no lower bound.
        max: Shape of the upper-bound tensor, or ``None`` for no upper bound.

    Raises:
        ValueError: If both ``min`` and ``max`` are ``None``.
    """

    _op_name = "clamp"
    _wrapped = None

    def __init__(
        self,
        input: tuple,
        min: Optional[tuple] = None,
        max: Optional[tuple] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if min is None and max is None:
            raise ValueError(
                "ClampFwdOp requires at least one of `min` or `max` to be a "
                "Tensor shape; both None is not a valid clamp."
            )
        self.input_shape = tuple(input)
        self.min_shape = None if min is None else tuple(min)
        self.max_shape = None if max is None else tuple(max)
        broadcast_args = [self.input_shape]
        if self.min_shape is not None:
            broadcast_args.append(self.min_shape)
        if self.max_shape is not None:
            broadcast_args.append(self.max_shape)
        self.out_shape = tuple(torch.broadcast_shapes(*broadcast_args))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _build_entry(self, dtype: torch.dtype, *shape: int) -> KernelEntry:
        impl, ctor_dtype = self._selected_kernel_cls("clamp_tensor").specialize(dtype)
        kernel = impl(
            self.N_total, ctor_dtype,
            has_min=self.min_shape is not None,
            has_max=self.max_shape is not None,
            tune=self.tune,
        )

        return KernelEntry(
            kernel=kernel,
            compute_dtype=ctor_dtype,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )

    @property
    def default_kernel_map(self):
        return {"clamp_tensor": ClampTensorFwdKernel}

    def _eager_forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Broadcast all operands to ``out_shape`` and dispatch the
        # TileLang Tensor-bound clamp kernel. The kernel branches on
        # ``has_min`` / ``has_max`` at build time, so this single Op
        # class also covers the mixed Tensor/None cases.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        lo_flat = None if min is None else self._expand_flat(min, out_shape)
        hi_flat = None if max is None else self._expand_flat(max, out_shape)
        result = self._entry(x_flat.dtype).kernel(x_flat, lo_flat, hi_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Validate that the runtime None / Tensor pattern matches what
        # __init__ was configured for — the broadcast shape and the
        # presence of each bound is baked in at construction.
        if (min is None) != (self.min_shape is None):
            raise ValueError(
                f"min was {'None' if self.min_shape is None else 'a Tensor shape'} at "
                f"__init__ but {'None' if min is None else 'a Tensor'} at forward()"
            )
        if (max is None) != (self.max_shape is None):
            raise ValueError(
                f"max was {'None' if self.max_shape is None else 'a Tensor shape'} at "
                f"__init__ but {'None' if max is None else 'a Tensor'} at forward()"
            )
        tensors = [("input", input, self.input_shape)]
        if min is not None:
            tensors.append(("min", min, self.min_shape))
        if max is not None:
            tensors.append(("max", max, self.max_shape))
        for _, t, _ in tensors:
            if not t.is_cuda:
                raise ValueError("Inputs must be CUDA tensors")
        for name, t, expected in tensors:
            if t.dtype != input.dtype:
                raise ValueError(f"Expected {name}.dtype {input.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, min, max, self._instance_key)
        return self._eager_forward(input, min, max)


class ClampScalarFwdOp(_PerDtypeKernels, Op):
    """Scalar-bound clamp (``torch.clamp(input, min: Number|None, max: Number|None)``).

    Args:
        input: Shape of the input tensor.
        min: Lower bound (Number or None).
        max: Upper bound (Number or None).
    """

    _op_name = "clamp"
    _wrapped = None

    def __init__(
        self,
        input: tuple,
        min: Optional[float] = None,
        max: Optional[float] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if min is None and max is None:
            raise ValueError(
                "ClampScalarFwdOp requires at least one of `min` or `max` to be a "
                "Number; both None is not a valid clamp."
            )
        self.input_shape = tuple(input)
        self.N_total = prod(self.input_shape) if self.input_shape else 1
        self.min = min
        self.max = max
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _build_entry(self, dtype: torch.dtype, *shape: int) -> KernelEntry:
        """The bounds are baked into the kernel, so they are checked per dtype."""
        if self.min is not None:
            _validate_scalar_param_repr("min", self.min, dtype, self._op_name)
        if self.max is not None:
            _validate_scalar_param_repr("max", self.max, dtype, self._op_name)
        impl, ctor_dtype = self._selected_kernel_cls("clamp").specialize(dtype)
        kernel = impl(
            self.N_total, ctor_dtype, min_val=self.min, max_val=self.max,
            tune=self.tune,
        )

        return KernelEntry(
            kernel=kernel,
            compute_dtype=ctor_dtype,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )

    @property
    def default_kernel_map(self):
        return {"clamp": ClampFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        return self._entry(input.dtype).kernel(
            input.contiguous().reshape(-1)
        ).reshape(orig_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        self._validate_dtypes(input)
        if tuple(input.shape) != self.input_shape:
            raise ValueError(
                f"Expected input.shape {self.input_shape}, got {tuple(input.shape)}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)
