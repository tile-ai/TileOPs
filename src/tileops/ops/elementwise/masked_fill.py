"""MaskedFill ops (Tensor-value and scalar-value variants)."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import (
    MaskedFillFwdKernel,
    MaskedFillTensorValueFwdKernel,
)
from tileops.kernels.kernel_base import Kernel

from ..compile_boundary import get_instance
from ..op_base import Op
from ._base import (
    _PerDtypeKernels,
    _require_one_device,
    _require_shape_inference,
    _validate_scalar_param_repr,
    broadcast_or_raise,
    resolve_output_dtype,
)


class MaskedFillFwdOp(_PerDtypeKernels, Op):
    """MaskedFill with 0-dim Tensor value (``torch.Tensor.masked_fill(mask, value: Tensor)``).

    Output shape is the bidirectional broadcast of ``input`` and ``mask``;
    ``value`` must be a 0-dim Tensor. The kernel reads ``value`` at forward time,
    which is consistent with the 0-dim semantics.

    Args:
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
            the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "masked_fill"
    _wrapped = None

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.target = target
        self.input_shape: Optional[tuple] = None
        self.mask_shape: Optional[tuple] = None
        self.value_shape: Optional[tuple] = None
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int):
        """The kernel names the implementation and storage for this dtype."""
        impl, compute = self._selected_kernel_cls().specialize(dtype)
        supported = impl.SUPPORTED_DTYPES
        if supported is not None and compute not in supported:
            names = ", ".join(str(dt) for dt in (torch.bool, *supported))
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. Supported: [{names}]"
            )
        return impl(n_total, compute)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"masked_fill_tensor_value": MaskedFillTensorValueFwdKernel}

    def _infer_output_shapes(
        self,
        input_shape: tuple,
        mask_shape: tuple,
        value_shape: tuple,
    ) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: the broadcast of ``input`` and ``mask``."""
        return {"output": broadcast_or_raise("MaskedFillFwdOp", input=input_shape, mask=mask_shape)}

    @property
    def out_shape(self) -> tuple:
        """Broadcast output shape of the most recent forward."""
        if self.input_shape is None:
            raise RuntimeError(
                "MaskedFillFwdOp needs a prior forward() call: the operand shapes "
                "arrive with the tensors"
            )
        return self._infer_output_shapes(self.input_shape, self.mask_shape, self.value_shape)[
            "output"
        ]

    @property
    def N_total(self) -> int:
        """Output element count of the most recent forward."""
        return prod(self.out_shape)

    def _eager_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        _require_one_device("MaskedFillFwdOp", input=input, mask=mask, value=value)
        self._validate_dtypes(input, mask, value)
        if value.ndim != 0:
            raise ValueError(f"Expected a 0-dim value Tensor, got shape {tuple(value.shape)}")
        if value.dtype != input.dtype:
            raise ValueError(f"Expected value.dtype {input.dtype}, got {value.dtype}")
        shapes = dict(
            input_shape=tuple(input.shape),
            mask_shape=tuple(mask.shape),
            value_shape=tuple(value.shape),
        )
        n_total = prod(self._infer_output_shapes(*shapes.values())["output"])
        input = input.contiguous()
        mask = mask.contiguous()
        value = value.contiguous()
        result = self._kernel((input, mask, value), input.dtype, n_total)(input, mask, value)
        self._note_call(input.dtype, **shapes)
        return result

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return type(self)._wrapped(input, mask, value, self._instance_key)


class MaskedFillScalarFwdOp(_PerDtypeKernels, Op):
    """MaskedFill with Number (scalar) value.

    Conforms to ``torch.Tensor.masked_fill(mask, value: Number)``. Output
    shape follows the bidirectional broadcast of ``input`` and ``mask``.

    The manifest declares the PyTorch dtype union (``bool | uint8 |
    int8 | int16 | int32 | int64 | float16 | bfloat16 | float32``); every
    union member dispatches to a real kernel. A bool operand is served by
    whatever storage the selected kernel requires; the op passes and receives
    semantic bool either way.

    Args:
        value: Scalar fill value (bool / int / float). Range-validated
            against the element type of the call with PyTorch
            ``Tensor.masked_fill`` coercion: bool reduces non-zero to
            ``True``; integer dtypes range-check the real value against
            ``torch.iinfo`` and truncate floats toward zero (``1.5 -> 1``);
            ``torch.uint8`` additionally wraps Python ints in ``[-255, 0)``
            via two's complement.
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
            the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "masked_fill"
    _wrapped = None

    def __init__(
        self,
        *,
        value: bool | int | float = 0,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.value = value
        self.target = target
        self.input_shape: Optional[tuple] = None
        self.mask_shape: Optional[tuple] = None
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int):
        """The fill value is baked in, so it is checked against each dtype."""
        impl, compute = self._selected_kernel_cls().specialize(dtype)
        supported = impl.SUPPORTED_DTYPES
        if supported is not None and compute not in supported:
            names = ", ".join(str(dt) for dt in (torch.bool, *supported))
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. Supported: [{names}]"
            )
        _validate_scalar_param_repr(
            "value",
            self.value,
            dtype,
            self._op_name,
            allow_nonfinite_float=True,
        )
        # The scalar is baked in, so it is normalized to the semantic dtype's
        # value set — bool takes 0 or 1 whatever storage the kernel picked.
        value = (1 if bool(self.value) else 0) if dtype == torch.bool else self.value
        return impl(n_total, compute, value)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"masked_fill": MaskedFillFwdKernel}

    def _infer_output_shapes(self, input_shape: tuple, mask_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: the broadcast of ``input`` and ``mask``."""
        return {
            "output": broadcast_or_raise(
                "MaskedFillScalarFwdOp", input=input_shape, mask=mask_shape
            )
        }

    @property
    def out_shape(self) -> tuple:
        """Broadcast output shape of the most recent forward."""
        if self.input_shape is None:
            raise RuntimeError(
                "MaskedFillScalarFwdOp needs a prior forward() call: the operand "
                "shapes arrive with the tensors"
            )
        return self._infer_output_shapes(self.input_shape, self.mask_shape)["output"]

    @property
    def N_total(self) -> int:
        """Output element count of the most recent forward."""
        return prod(self.out_shape)

    def _eager_forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _require_one_device("MaskedFillScalarFwdOp", input=input, mask=mask)
        self._validate_dtypes(input, mask)
        shapes = dict(input_shape=tuple(input.shape), mask_shape=tuple(mask.shape))
        n_total = prod(self._infer_output_shapes(*shapes.values())["output"])
        input = input.contiguous()
        mask = mask.contiguous()
        result = self._kernel((input, mask), input.dtype, n_total)(input, mask)
        self._note_call(input.dtype, **shapes)
        return result

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return type(self)._wrapped(input, mask, self._instance_key)


# The compile boundary. Module-level because registration happens once per qualified name
# at import time and the schema is read off the annotations, so ``self`` cannot appear; the
# instance comes back from the string key. See src/tileops/ops/compile_boundary.py.

# Two variants, two operators: the Tensor-value one takes a third tensor, so it cannot
# share a schema with the scalar one.
_require_shape_inference(MaskedFillFwdOp)
_require_shape_inference(MaskedFillScalarFwdOp)


@torch.library.custom_op("top::elementwise_masked_fill_tensor_value", mutates_args=())
def _masked_fill_tensor_value_fwd(
    input: torch.Tensor,
    mask: torch.Tensor,
    value: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(input, mask, value)


@_masked_fill_tensor_value_fwd.register_fake
def _masked_fill_tensor_value_fwd_fake(
    input: torch.Tensor,
    mask: torch.Tensor,
    value: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(input.shape), tuple(mask.shape), tuple(value.shape))
    return input.new_empty(
        shapes["output"], dtype=resolve_output_dtype("MaskedFillFwdOp", input.dtype)
    )


@torch.library.custom_op("top::elementwise_masked_fill", mutates_args=())
def _masked_fill_fwd(
    input: torch.Tensor,
    mask: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(input, mask)


@_masked_fill_fwd.register_fake
def _masked_fill_fwd_fake(
    input: torch.Tensor,
    mask: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(input.shape), tuple(mask.shape))
    return input.new_empty(
        shapes["output"], dtype=resolve_output_dtype("MaskedFillScalarFwdOp", input.dtype)
    )


MaskedFillFwdOp._wrapped = _masked_fill_tensor_value_fwd
MaskedFillFwdOp.compile_op_names = ("top::elementwise_masked_fill_tensor_value",)
MaskedFillScalarFwdOp._wrapped = _masked_fill_fwd
MaskedFillScalarFwdOp.compile_op_names = ("top::elementwise_masked_fill",)
