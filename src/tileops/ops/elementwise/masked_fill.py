"""MaskedFill ops (Tensor-value and scalar-value variants)."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import (
    MaskedFillFwdKernel,
    MaskedFillTensorValueFwdKernel,
)
from tileops.kernels.elementwise.call_spec import (
    ElementwiseCall,
    MaskedFillCall,
    MaskedFillFwdInterface,
    MaskedFillTensorValueFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.elementwise._base import ELEMENTWISE
from tileops.ops.op_base import Op


class MaskedFillTensorFwdOp(Op):
    """MaskedFill with 0-dim Tensor value (``torch.Tensor.masked_fill(mask, value: Tensor)``).

    Output shape is the bidirectional broadcast of ``input`` and ``mask``;
    ``value`` is a 0-dim Tensor, which the kernel reads at forward time.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"masked_fill_tensor_value": MaskedFillTensorValueFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: MaskedFillTensorValueFwdInterface
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
            kernel_map: Optional dispatch override mapping kernel keys to
                ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

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
        call = ElementwiseCall(device=input.device, n_total=n_total, dtype=input.dtype)
        return self.kernel_for(ELEMENTWISE, call)(input, mask, value)

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on ``input``, ``mask`` and ``value``."""
        return self._call_boundary(input, mask, value)


class MaskedFillScalarFwdOp(Op):
    """MaskedFill with Number (scalar) value.

    Conforms to ``torch.Tensor.masked_fill(mask, value: Number)``. Output
    shape follows the bidirectional broadcast of ``input`` and ``mask``.
    Every dtype of the manifest union dispatches to a real kernel. A bool operand
    is served by whatever storage the selected kernel requires; the op passes and
    receives semantic bool either way.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"masked_fill": MaskedFillFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: MaskedFillFwdInterface
    }

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

    def _eager_forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_total = torch.broadcast_shapes(input.shape, mask.shape).numel()
        input = input.contiguous()
        mask = mask.contiguous()
        call = MaskedFillCall(
            device=input.device, n_total=n_total, dtype=input.dtype, value=self.value
        )
        return self.kernel_for(ELEMENTWISE, call)(input, mask)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``mask``."""
        return self._call_boundary(input, mask)
