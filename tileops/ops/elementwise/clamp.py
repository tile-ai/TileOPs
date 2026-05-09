"""Clamp ops (Tensor-bound and scalar-bound variants)."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import ClampFwdKernel, ClampTensorFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import (
    _OP_REGISTRY,
    _apply_fp8_post_cast,
    _ClampTensorBase,
    _validate_scalar_param_repr,
)


class ClampFwdOp(_ClampTensorBase):
    """Clamp with Tensor lower and/or upper bounds (broadcasting).

    Conforms to ``torch.clamp(input, min, max)`` where ``min`` and ``max``
    are each either a Tensor or ``None``. At least one of the two bounds
    must be a Tensor. All Tensor operands broadcast together. The
    primary spec entry in ``tileops/manifest/`` covers the both-Tensor
    form; the mixed Tensor/``None`` cases are runtime-equivalent to
    ``ClampMinFwdOp`` / ``ClampMaxFwdOp`` and are accepted here so callers
    can mirror PyTorch's ``torch.clamp`` API directly.

    Args:
        input: Shape of the input tensor.
        min: Shape of the lower-bound tensor, or ``None`` for no lower bound.
        max: Shape of the upper-bound tensor, or ``None`` for no upper bound.
        dtype: Torch dtype for all operands.

    Raises:
        ValueError: If both ``min`` and ``max`` are ``None``.
    """

    _op_name = "clamp"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002 — manifest-aligned PyTorch param name
        min: Optional[tuple] = None,  # noqa: A002 — manifest-aligned PyTorch param name
        max: Optional[tuple] = None,  # noqa: A002 — manifest-aligned PyTorch param name
        dtype: torch.dtype = torch.float32,
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
        self.dtype = dtype
        broadcast_args = [self.input_shape]
        if self.min_shape is not None:
            broadcast_args.append(self.min_shape)
        if self.max_shape is not None:
            broadcast_args.append(self.max_shape)
        self.out_shape = tuple(torch.broadcast_shapes(*broadcast_args))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp_tensor"](
            self.N_total, dtype,
            has_min=self.min_shape is not None,
            has_max=self.max_shape is not None,
            tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"clamp_tensor": ClampTensorFwdKernel}

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002
        min: Optional[torch.Tensor] = None,  # noqa: A002
        max: Optional[torch.Tensor] = None,  # noqa: A002
    ) -> torch.Tensor:
        # Broadcast all operands to ``out_shape`` and dispatch the
        # TileLang Tensor-bound clamp kernel. The kernel branches on
        # ``has_min`` / ``has_max`` at build time, so this single Op
        # class also covers the mixed Tensor/None cases.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        lo_flat = None if min is None else self._expand_flat(min, out_shape)
        hi_flat = None if max is None else self._expand_flat(max, out_shape)
        result = self.kernel(x_flat, lo_flat, hi_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        min: Optional[torch.Tensor] = None,  # noqa: A002
        max: Optional[torch.Tensor] = None,  # noqa: A002
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
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, min, max, self._instance_key)
        return self._eager_forward(input, min, max)


class ClampMinFwdOp(_ClampTensorBase):
    """Single-bound Tensor lower clamp (``torch.clamp_min``).

    Args:
        input: Shape of the input tensor.
        min: Shape of the lower-bound tensor.
        dtype: Torch dtype.
    """

    _op_name = "clamp_min"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        min: tuple,    # noqa: A002
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.input_shape = tuple(input)
        self.min_shape = tuple(min)
        self.dtype = dtype
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.min_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp_tensor"](
            self.N_total, dtype, has_min=True, has_max=False, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"clamp_tensor": ClampTensorFwdKernel}

    def _eager_forward(
        self, input: torch.Tensor, min: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        # Broadcast input/min to out_shape and dispatch the TileLang
        # min-only Tensor-bound clamp kernel.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        lo_flat = self._expand_flat(min, out_shape)
        result = self.kernel(x_flat, lo_flat, None)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self, input: torch.Tensor, min: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        if not (input.is_cuda and min.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("min", min, self.min_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, min, self._instance_key)
        return self._eager_forward(input, min)


class ClampMaxFwdOp(_ClampTensorBase):
    """Single-bound Tensor upper clamp (``torch.clamp_max``).

    Args:
        input: Shape of the input tensor.
        max: Shape of the upper-bound tensor.
        dtype: Torch dtype.
    """

    _op_name = "clamp_max"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        max: tuple,    # noqa: A002
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.input_shape = tuple(input)
        self.max_shape = tuple(max)
        self.dtype = dtype
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.max_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp_tensor"](
            self.N_total, dtype, has_min=False, has_max=True, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"clamp_tensor": ClampTensorFwdKernel}

    def _eager_forward(
        self, input: torch.Tensor, max: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        # Broadcast input/max to out_shape and dispatch the TileLang
        # max-only Tensor-bound clamp kernel.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        hi_flat = self._expand_flat(max, out_shape)
        result = self.kernel(x_flat, None, hi_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self, input: torch.Tensor, max: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        if not (input.is_cuda and max.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("max", max, self.max_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, max, self._instance_key)
        return self._eager_forward(input, max)


class ClampScalarFwdOp(Op):
    """Scalar-bound clamp (``torch.clamp(input, min: Number|None, max: Number|None)``).

    Args:
        input: Shape of the input tensor.
        min: Lower bound (Number or None).
        max: Upper bound (Number or None).
        dtype: Torch dtype.
    """

    _op_name = "clamp"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        min: Optional[float] = None,  # noqa: A002
        max: Optional[float] = None,  # noqa: A002
        dtype: torch.dtype = torch.float32,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if min is None and max is None:
            raise ValueError(
                "ClampScalarFwdOp requires at least one of `min` or `max` to be a "
                "Number; both None is not a valid clamp."
            )
        if min is not None:
            _validate_scalar_param_repr("min", min, dtype, self._op_name)
        if max is not None:
            _validate_scalar_param_repr("max", max, dtype, self._op_name)
        self.input_shape = tuple(input)
        self.N_total = prod(self.input_shape) if self.input_shape else 1
        self.dtype = dtype
        self.min = min
        self.max = max
        # Backwards-compat aliases for legacy callers.
        self.min_val = min
        self.max_val = max
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp"](
            self.N_total, dtype, min_val=min, max_val=max, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"clamp": ClampFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        orig_shape = input.shape
        result = self.kernel(input.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if tuple(input.shape) != self.input_shape:
            raise ValueError(
                f"Expected input.shape {self.input_shape}, got {tuple(input.shape)}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)
