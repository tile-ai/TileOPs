"""Binary arithmetic elementwise ops with broadcasting."""

from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import (
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
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import BinaryOp, _AlphaScaledBinaryOp, _PerDtypeKernels


class AddFwdOp(_AlphaScaledBinaryOp):
    """Element-wise addition with broadcast: y = input + alpha * other.

    Conforms to ``torch.add(input, other, *, alpha=1)``. ``alpha`` is baked
    into the kernel, so non-default ``alpha`` runs through the same fast
    kernel as the default.
    """

    kernel_types = {"add": AddFwdKernel}


class SubFwdOp(_AlphaScaledBinaryOp):
    """Element-wise subtraction with broadcast: y = input - alpha * other.

    Conforms to ``torch.sub(input, other, *, alpha=1)``. ``alpha`` is baked
    into the kernel, so non-default ``alpha`` runs through the same fast
    kernel as the default.
    """

    kernel_types = {"sub": SubFwdKernel}


class MulFwdOp(BinaryOp):
    """Element-wise multiplication with broadcast: y = input * other."""

    kernel_types = {"mul": MulFwdKernel}


_DIV_KEY_BY_ROUNDING_MODE = {None: "div", "trunc": "div_trunc", "floor": "floor_divide"}


class DivFwdOp(BinaryOp):
    """Element-wise division with broadcast: y = input / other.

    Conforms to ``torch.div(input, other, *, rounding_mode=None)``.
    ``rounding_mode`` accepts ``None`` (true division), ``"trunc"``
    (truncation toward zero), or ``"floor"`` (floor division); each
    value selects a dedicated kernel. It is fixed for the instance, which is
    why it is not part of the memory key.
    """

    kernel_types = {
        "div": DivFwdKernel,
        "div_trunc": DivTruncFwdKernel,
        "floor_divide": FloorDivideFwdKernel,
    }

    def __init__(
        self,
        *,
        rounding_mode: Optional[str] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            rounding_mode: ``None``, ``"trunc"`` or ``"floor"``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.rounding_mode = rounding_mode
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        key = _DIV_KEY_BY_ROUNDING_MODE[self.rounding_mode]
        return {key: self.kernel_types[key]}


class RemainderFwdOp(BinaryOp):
    """Element-wise remainder with broadcast: y = a % b."""

    kernel_types = {"remainder": RemainderFwdKernel}


class PowFwdOp(BinaryOp):
    """Element-wise power with broadcast: y = input ** exponent.

    Conforms to ``torch.pow(input, exponent)``: the second operand carries
    the manifest-declared name ``exponent`` rather than the generic
    ``other`` so the L1 signature check matches the manifest.

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

    The bound is the table above, not a constant. Callers raising a base far
    from one to a large exponent, or comparing results for equality, need to
    account for it.
    """

    kernel_types = {"pow": PowFwdKernel}

    def forward(self, input: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``exponent``."""
        return self._call_boundary(input, exponent)


class FloorDivideFwdOp(BinaryOp):
    """Element-wise floor division with broadcast: y = floor(a / b)."""

    kernel_types = {"floor_divide": FloorDivideFwdKernel}


class LerpFwdOp(BinaryOp):
    """Element-wise lerp with broadcast: y = a + weight * (b - a).

    Unlike ``torch.lerp(a, b, weight)`` where weight is a runtime parameter,
    here weight is a **construction-time constant** baked into the compiled
    kernel. This enables compile-time folding but means a new Op instance is
    needed for each distinct weight value.

    """

    kernel_types = {"lerp": LerpFwdKernel}

    def __init__(
        self,
        *,
        weight: float = 0.5,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            weight: Scalar interpolation weight, fixed at construction (manifest
                ``params.weight``, default 0.5).
            target: Which set of kernels serves this op.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.weight = weight
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _build_kernel_instance(self, tune, dtype, impl, a_shape, b_shape):
        return impl(a_shape, b_shape, dtype, tune=tune, weight=self.weight)

    def forward(self, input: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``end``."""
        return self._call_boundary(input, end)


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


class LerpTensorFwdOp(_PerDtypeKernels, Op):
    """Tensor-weight lerp: out = input + weight * (end - input).

    Conforms to the Tensor-weight overload of ``torch.lerp`` —
    ``torch.lerp(input, end, weight: Tensor)`` where ``weight`` is a Tensor that
    broadcasts together with ``input`` and ``end`` to the output shape. The scalar-weight
    overload is handled separately by ``LerpFwdOp``.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"lerp_tensor": LerpTensorFwdKernel}

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int):
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(n_total, ctor_dtype, tune=self.tune)

    def _eager_forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        n_total = torch.broadcast_shapes(input.shape, end.shape, weight.shape).numel()
        input = input.contiguous()
        end = end.contiguous()
        weight = weight.contiguous()
        return self._kernel((input, end, weight), input.dtype, n_total)(input, end, weight)

    def forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on ``input``, ``end`` and ``weight``."""
        return self._call_boundary(input, end, weight)
