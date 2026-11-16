"""MaskedFill op: a Number, bool or 0-dim Tensor fill value."""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.elementwise import (
    MaskedFillFwdKernel,
    MaskedFillTensorValueFwdKernel,
)
from tileops._kernels.elementwise.call_spec import (
    ElementwiseCall,
    MaskedFillCall,
    MaskedFillFwdInterface,
    MaskedFillTensorValueFwdInterface,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

# The interface of each form of the fill value.
_NUMBER_VALUE, _TENSOR_VALUE = "number_value", "tensor_value"


class MaskedFillFwdOp(Op):
    """MaskedFill (``torch.Tensor.masked_fill(mask, value)``) with a scalar or 0-dim Tensor value.

    Output shape is the bidirectional broadcast of ``input`` and ``mask``. A Number value
    follows PyTorch's coercion to the element type: bool reduces non-zero to ``True``;
    integer dtypes range-check the real value against ``torch.iinfo`` and truncate floats
    toward zero (``1.5 -> 1``); ``torch.uint8`` additionally wraps Python ints in
    ``[-255, 0)`` via two's complement. A Tensor value is read by the kernel at forward time.
    Every dtype of the manifest union dispatches to a real kernel; a bool operand is served
    by whatever storage the selected kernel requires.

    Which form a call passes is read off the call, so one instance serves both. A Number
    value reaches the kernel's cache key, since the kernel is built for it.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "masked_fill": MaskedFillFwdKernel,
        "masked_fill_tensor_value": MaskedFillTensorValueFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        _NUMBER_VALUE: MaskedFillFwdInterface,
        _TENSOR_VALUE: MaskedFillTensorValueFwdInterface,
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
        mask: torch.Tensor,
        value: bool | int | float | torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on ``input``, ``mask`` and ``value``.

        Args:
            input: Tensor to fill.
            mask: ``torch.bool`` mask, broadcastable with ``input``; ``True`` marks the
                elements to fill.
            value: Fill value: a Number, a bool, or a 0-dim tensor in the dtype of ``input``.

        Returns:
            A new tensor of the broadcast shape of ``input`` and ``mask``.
        """
        n_total = torch.broadcast_shapes(input.shape, mask.shape).numel()
        input = input.contiguous()
        mask = mask.contiguous()
        if isinstance(value, torch.Tensor):
            value = value.contiguous()
            return self.kernel_for(
                _TENSOR_VALUE,
                lambda: ElementwiseCall(device=input.device, n_total=n_total, dtype=input.dtype),
            )(input, mask, value)
        return self.kernel_for(
            _NUMBER_VALUE,
            lambda: MaskedFillCall(
                device=input.device, n_total=n_total, dtype=input.dtype, value=value
            ),
        )(input, mask)
