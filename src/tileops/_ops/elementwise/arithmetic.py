"""Binary arithmetic elementwise ops with broadcasting."""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.elementwise import (
    AddFwdKernel,
    DivFwdKernel,
    DivTruncFwdKernel,
    FloorDivideFwdKernel,
    LerpFwdKernel,
    LerpTensorFwdKernel,
    MaximumFwdKernel,
    MinimumFwdKernel,
    MulFwdKernel,
    PowFwdKernel,
    RemainderFwdKernel,
    SubFwdKernel,
)
from tileops._kernels.elementwise.call_spec import (
    AlphaScaledBinaryFwdInterface,
    AlphaScaledCall,
    DivCall,
    ElementwiseCall,
    LerpCall,
    LerpFwdInterface,
    LerpTensorFwdInterface,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.elementwise._base import ELEMENTWISE, BinaryOp
from tileops._ops.op_base import Op
from tileops.backend import Target

# The interface of each form of lerp's weight.
_NUMBER_WEIGHT, _TENSOR_WEIGHT = "number_weight", "tensor_weight"


class _AlphaScaledBinaryOp(BinaryOp):
    """Shared base for ops that scale ``other`` by a scalar ``alpha`` before the binary op;
    one kernel specialization per ``alpha``."""

    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: AlphaScaledBinaryFwdInterface
    }

    def __init__(
        self,
        *,
        alpha: int | float = 1,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            alpha: Multiplier on ``other`` (default 1).
            target: Backend target to serve this op, or ``None`` to decide from the input device.
        """
        self.alpha = alpha
        super().__init__(target=target)

    def _call_spec(self, input: torch.Tensor, other: torch.Tensor) -> AlphaScaledCall:
        return AlphaScaledCall(
            device=input.device,
            a_shape=tuple(input.shape),
            b_shape=tuple(other.shape),
            dtype=input.dtype,
            alpha=self.alpha,
        )


class AddFwdOp(_AlphaScaledBinaryOp):
    """Element-wise addition with broadcast: y = input + alpha * other, as ``torch.add``."""

    kernel_types = {"add": AddFwdKernel}


class SubFwdOp(_AlphaScaledBinaryOp):
    """Element-wise subtraction with broadcast: y = input - alpha * other, as ``torch.sub``."""

    kernel_types = {"sub": SubFwdKernel}


class MulFwdOp(BinaryOp):
    """Element-wise multiplication with broadcast: y = input * other."""

    kernel_types = {"mul": MulFwdKernel}


class DivFwdOp(BinaryOp):
    """Element-wise division with broadcast: y = input / other, as ``torch.div``.

    ``rounding_mode`` is ``None`` (true division), ``"trunc"`` or ``"floor"``.
    """

    kernel_types = {
        "div": DivFwdKernel,
        "div_trunc": DivTruncFwdKernel,
        "floor_divide": FloorDivideFwdKernel,
    }

    def __init__(
        self,
        *,
        rounding_mode: str | None = None,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            rounding_mode: ``None``, ``"trunc"`` or ``"floor"``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
        """
        self.rounding_mode = rounding_mode
        super().__init__(target=target)

    def _call_spec(self, input: torch.Tensor, other: torch.Tensor) -> DivCall:
        return DivCall(
            device=input.device,
            a_shape=tuple(input.shape),
            b_shape=tuple(other.shape),
            dtype=input.dtype,
            rounding_mode=self.rounding_mode,
        )


class RemainderFwdOp(BinaryOp):
    """Element-wise remainder with broadcast: y = a % b."""

    kernel_types = {"remainder": RemainderFwdKernel}


class PowFwdOp(BinaryOp):
    """Element-wise power with broadcast: y = input ** exponent, as ``torch.pow``.

    Accuracy differs from ``torch.pow``: the power is computed as
    ``exp2(exponent * log2|input|)``, whose error scales with
    ``|exponent * log2(input)|`` rather than staying flat. Measured against a
    float64 reference, in units of a float32 ulp:

    | ``input`` | ``exponent`` | here | ``torch.pow`` |
    | --- | --- | --- | --- |
    | 0.9 to 1.1 | -1 to 1 | 1.2 | 0.5 |
    | 0.5 to 2 | -2 to 2 | 1.8 | 0.6 |
    | 0.25 to 4 | -3 to 3 | 3.4 | 0.6 |
    | 0.01 to 100 | -3 to 3 | 10.6 | 0.6 |
    | 1e-6 to 1e6 | -2 to 2 | 22.5 | 0.6 |
    """

    kernel_types = {"pow": PowFwdKernel}

    def forward(self, input: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``exponent``.

        Args:
            input: Base.
            exponent: Exponent, broadcastable with ``input``.

        Returns:
            A new tensor of the broadcast shape of ``input`` and ``exponent``.
        """
        return super().forward(input, exponent)


class FloorDivideFwdOp(BinaryOp):
    """Element-wise floor division with broadcast: y = a // b, as ``torch.floor_divide``."""

    kernel_types = {"floor_divide": FloorDivideFwdKernel}

    def _call_spec(self, input: torch.Tensor, other: torch.Tensor) -> DivCall:
        # The floor_divide implementation is shared with DivFwdOp and reads rounding_mode.
        return DivCall(
            device=input.device,
            a_shape=tuple(input.shape),
            b_shape=tuple(other.shape),
            dtype=input.dtype,
            rounding_mode="floor",
        )


class MaximumFwdOp(BinaryOp):
    """Element-wise maximum with broadcast: y = max(a, b).

    A NaN operand gives a NaN result, canonical rather than the operand's own
    payload and sign bit; `torch.maximum` returns the operand's bit pattern.
    """

    kernel_types = {"maximum": MaximumFwdKernel}


class MinimumFwdOp(BinaryOp):
    """Element-wise minimum with broadcast: y = min(a, b).

    A NaN operand gives a NaN result, canonical rather than the operand's own
    payload and sign bit; `torch.minimum` returns the operand's bit pattern.
    """

    kernel_types = {"minimum": MinimumFwdKernel}


class LerpFwdOp(Op):
    """Linear interpolation ``out = input + weight * (end - input)`` (``torch.lerp``).

    ``weight`` is a Number or a Tensor that broadcasts with ``input`` and ``end``; one
    instance serves both. A Number weight is baked into its kernel.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "lerp": LerpFwdKernel,
        "lerp_tensor": LerpTensorFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        _NUMBER_WEIGHT: LerpFwdInterface,
        _TENSOR_WEIGHT: LerpTensorFwdInterface,
    }

    def __init__(
        self,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op.
        """
        super().__init__(target=target)

    def forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
        weight: float | torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on ``input``, ``end`` and ``weight``.

        Args:
            input: Start point.
            end: End point, broadcastable with ``input``.
            weight: Interpolation weight: a Number, or a Tensor broadcastable with ``input``
                and ``end``.

        Returns:
            A new tensor of the broadcast shape of the inputs.
        """
        input = input.contiguous()
        end = end.contiguous()
        if not isinstance(weight, torch.Tensor):
            return self.kernel_for(
                _NUMBER_WEIGHT,
                lambda: LerpCall(
                    device=input.device,
                    a_shape=tuple(input.shape),
                    b_shape=tuple(end.shape),
                    dtype=input.dtype,
                    weight=weight,
                ),
            )(input, end)
        n_total = torch.broadcast_shapes(input.shape, end.shape, weight.shape).numel()
        weight = weight.contiguous()
        kernel = self.kernel_for(
            _TENSOR_WEIGHT,
            lambda: ElementwiseCall(device=input.device, n_total=n_total, dtype=input.dtype),
        )
        return kernel(input, end, weight)
