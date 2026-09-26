"""MaskedFill ops (Tensor-value and scalar-value variants)."""

from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import (
    MaskedFillFwdKernel,
    MaskedFillTensorValueFwdKernel,
)
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _PerDtypeKernels, _validate_scalar_param_repr


class MaskedFillFwdOp(_PerDtypeKernels, Op):
    """MaskedFill with 0-dim Tensor value (``torch.Tensor.masked_fill(mask, value: Tensor)``).

    Output shape is the bidirectional broadcast of ``input`` and ``mask``;
    ``value`` is a 0-dim Tensor, which the kernel reads at forward time.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"masked_fill_tensor_value": MaskedFillTensorValueFwdKernel}

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
        """The kernel names the implementation and storage for this dtype."""
        impl, compute = self._selected_kernel_cls().specialize(dtype)
        self._check_kernel_dtype(impl, dtype, compute)
        return impl(n_total, compute, tune=self.tune)

    def _eager_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        n_total = torch.broadcast_shapes(input.shape, mask.shape).numel()
        input = input.contiguous()
        mask = mask.contiguous()
        value = value.contiguous()
        return self._kernel((input, mask, value), input.dtype, n_total)(input, mask, value)

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on ``input``, ``mask`` and ``value``."""
        return self._call_boundary(input, mask, value)


class MaskedFillScalarFwdOp(_PerDtypeKernels, Op):
    """MaskedFill with Number (scalar) value.

    Conforms to ``torch.Tensor.masked_fill(mask, value: Number)``. Output
    shape follows the bidirectional broadcast of ``input`` and ``mask``.
    Every dtype of the manifest union dispatches to a real kernel. A bool operand
    is served by whatever storage the selected kernel requires; the op passes and
    receives semantic bool either way.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"masked_fill": MaskedFillFwdKernel}

    def __init__(
        self,
        *,
        value: bool | int | float = 0.0,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

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
            tune: Whether to autotune.
        """
        self.value = value
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int):
        """The fill value is baked in, so it is checked against each dtype."""
        impl, compute = self._selected_kernel_cls().specialize(dtype)
        self._check_kernel_dtype(impl, dtype, compute)
        _validate_scalar_param_repr(
            "value",
            self.value,
            dtype,
            self._slot,
            allow_nonfinite_float=True,
        )
        # The scalar is baked in, so it is normalized to the semantic dtype's
        # value set — bool takes 0 or 1 whatever storage the kernel picked.
        value = (1 if bool(self.value) else 0) if dtype == torch.bool else self.value
        return impl(n_total, compute, value, tune=self.tune)

    def _eager_forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_total = torch.broadcast_shapes(input.shape, mask.shape).numel()
        input = input.contiguous()
        mask = mask.contiguous()
        return self._kernel((input, mask), input.dtype, n_total)(input, mask)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``mask``."""
        return self._call_boundary(input, mask)
