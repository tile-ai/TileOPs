"""MaskedFill ops (Tensor-value and scalar-value variants)."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import (
    MaskedFillFwdKernel,
    MaskedFillTensorValueFwdKernel,
)
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _OP_REGISTRY, _apply_fp8_post_cast, _validate_scalar_param_repr


class MaskedFillFwdOp(Op):
    """MaskedFill with 0-dim Tensor value (``torch.Tensor.masked_fill(mask, value: Tensor)``).

    Output shape is the bidirectional broadcast of ``input`` and ``mask``;
    ``value`` must be a 0-dim Tensor. The Op expands ``input`` and ``mask``
    to the broadcast shape and dispatches the existing flat scalar kernel
    using ``value.item()`` as the fill literal — this keeps the
    fast vectorized kernel path while satisfying the manifest's Tensor-value
    contract (the kernel reads ``value`` once at forward time, which is
    consistent with the 0-dim semantics).

    Args:
        input: Shape of the input tensor.
        mask: Shape of the mask tensor (bool).
        value: Shape of the value tensor (must be ``()`` per the manifest).
        dtype: Torch dtype for ``input`` / ``value``.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "masked_fill"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        mask: tuple,
        value: tuple,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if tuple(value) != ():
            raise ValueError(
                f"MaskedFillFwdOp requires a 0-dim value Tensor; got shape {tuple(value)}"
            )
        # Bool routes through uint8 at the Op layer (TileLang bool codegen is unvectorized);
        # all other dtypes go straight to the kernel.
        kernel_supported = MaskedFillTensorValueFwdKernel.SUPPORTED_DTYPES
        if (
            dtype != torch.bool
            and kernel_supported is not None
            and dtype not in kernel_supported
        ):
            names = ", ".join(str(dt) for dt in (torch.bool, *kernel_supported))
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.input_shape = tuple(input)
        self.mask_shape = tuple(mask)
        self.value_shape = tuple(value)
        self.dtype = dtype
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.mask_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        # Bool input path: the kernel runs in uint8 storage.
        self._bool_storage = dtype == torch.bool
        kernel_dtype = torch.uint8 if self._bool_storage else dtype
        self.kernel = self.kernel_map["masked_fill_tensor_value"](self.N_total, kernel_dtype)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"masked_fill_tensor_value": MaskedFillTensorValueFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self, input: torch.Tensor, mask: torch.Tensor, value: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        # Broadcast input/mask to out_shape, pack mask as uint8, reshape
        # the 0-dim value to (1,), and dispatch the TileLang kernel.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        if self._bool_storage:
            x_flat = x_flat.view(torch.uint8)
        mask_b = mask if mask.dtype == torch.bool else mask.bool()
        mask_flat = self._expand_flat(mask_b, out_shape).view(torch.uint8)
        value_1d = value.contiguous().view(1)
        if self._bool_storage:
            value_1d = value_1d.view(torch.uint8)
        result = self.kernel(x_flat, mask_flat, value_1d)
        if self._bool_storage:
            result = result.view(torch.bool)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor, value: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        if not (input.is_cuda and mask.is_cuda and value.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if mask.dtype != torch.bool:
            raise ValueError(f"Expected mask.dtype torch.bool, got {mask.dtype}")
        if value.dtype != self.dtype:
            raise ValueError(f"Expected value.dtype {self.dtype}, got {value.dtype}")
        if tuple(input.shape) != self.input_shape:
            raise ValueError(
                f"Expected input.shape {self.input_shape}, got {tuple(input.shape)}"
            )
        if tuple(mask.shape) != self.mask_shape:
            raise ValueError(
                f"Expected mask.shape {self.mask_shape}, got {tuple(mask.shape)}"
            )
        if tuple(value.shape) != ():
            raise ValueError(f"Expected value.shape (), got {tuple(value.shape)}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, mask, value, self._instance_key)
        return self._eager_forward(input, mask, value)


class MaskedFillScalarFwdOp(Op):
    """MaskedFill with Number (scalar) value.

    Conforms to ``torch.Tensor.masked_fill(mask, value: Number)``. Output
    shape follows the bidirectional broadcast of ``input`` and ``mask``.

    The manifest declares the PyTorch dtype union (``bool | uint8 |
    int8 | int16 | int32 | int64 | float16 | bfloat16 | float32``); every
    union member dispatches to a real kernel. The bool dtype path views
    ``input`` as ``uint8`` before dispatch and the result as ``bool`` on
    return, so the kernel itself only handles the integer and floating-
    point storage dtypes.

    Args:
        input: Shape of the input tensor.
        mask: Shape of the mask tensor (bool).
        value: Scalar fill value (bool / int / float). Range-validated
            against ``dtype`` with PyTorch ``Tensor.masked_fill``
            coercion: bool reduces non-zero to ``True``; integer dtypes
            range-check the real value against ``torch.iinfo`` and
            truncate floats toward zero (``1.5 -> 1``); ``torch.uint8``
            additionally wraps Python ints in ``[-255, 0)`` via two's
            complement.
        dtype: Torch dtype. Must be a manifest-declared dtype.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "masked_fill"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        mask: tuple,
        value: bool | int | float = 0,
        dtype: torch.dtype = torch.float32,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        # Bool routes through uint8 at the Op layer (TileLang bool codegen is unvectorized);
        # all other dtypes go straight to the kernel.
        kernel_supported = MaskedFillFwdKernel.SUPPORTED_DTYPES
        if (
            dtype != torch.bool
            and kernel_supported is not None
            and dtype not in kernel_supported
        ):
            names = ", ".join(str(dt) for dt in (torch.bool, *kernel_supported))
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.input_shape = tuple(input)
        self.mask_shape = tuple(mask)
        self.dtype = dtype
        self.value = value
        # Backwards-compat alias.
        self.fill_value = value
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.mask_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        _validate_scalar_param_repr(
            "value", value, dtype, self._op_name, allow_nonfinite_float=True,
        )
        self.dispatch_kernel(kernel_map)
        # Bool input path: the kernel runs in uint8 storage.
        self._bool_storage = dtype == torch.bool
        kernel_dtype = torch.uint8 if self._bool_storage else dtype
        kernel_value = (1 if bool(value) else 0) if self._bool_storage else value
        self.kernel = self.kernel_map["masked_fill"](
            self.N_total, kernel_dtype, kernel_value,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"masked_fill": MaskedFillFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: A002
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        if self._bool_storage:
            x_flat = x_flat.view(torch.uint8)
        mask_b = mask if mask.dtype == torch.bool else mask.bool()
        mask_flat = self._expand_flat(mask_b, out_shape).view(torch.uint8)
        result = self.kernel(x_flat, mask_flat)
        if self._bool_storage:
            result = result.view(torch.bool)
        result = result.view(self.out_shape if self.out_shape else ())
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if tuple(input.shape) != self.input_shape:
            raise ValueError(
                f"Expected input.shape {self.input_shape}, got {tuple(input.shape)}"
            )
        if not mask.is_cuda:
            raise ValueError("Mask must be a CUDA tensor")
        if mask.dtype != torch.bool:
            raise ValueError(f"Expected mask.dtype torch.bool, got {mask.dtype}")
        if tuple(mask.shape) != self.mask_shape:
            raise ValueError(
                f"Expected mask.shape {self.mask_shape}, got {tuple(mask.shape)}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, mask, self._instance_key)
        return self._eager_forward(input, mask)
