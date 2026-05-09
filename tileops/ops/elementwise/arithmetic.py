"""Binary arithmetic elementwise ops with broadcasting."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import (
    AddFwdKernel,
    DivFwdKernel,
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
from ._base import (
    _OP_REGISTRY,
    BinaryOp,
    _AlphaScaledBinaryOp,
    coalesce_broadcast_dims,
)


class AddFwdOp(_AlphaScaledBinaryOp):
    """Element-wise addition with broadcast: y = input + alpha * other.

    Conforms to ``torch.add(input, other, *, alpha=1)``. ``alpha`` is baked
    into the kernel at construction time, so non-default ``alpha`` runs
    through the same fast kernel as the default.
    """

    _op_name = "add"
    kernel_cls = AddFwdKernel


class SubFwdOp(_AlphaScaledBinaryOp):
    """Element-wise subtraction with broadcast: y = input - alpha * other.

    Conforms to ``torch.sub(input, other, *, alpha=1)``. ``alpha`` is baked
    into the kernel at construction time, so non-default ``alpha`` runs
    through the same fast kernel as the default.
    """

    _op_name = "sub"
    kernel_cls = SubFwdKernel


class MulFwdOp(BinaryOp):
    """Element-wise multiplication with broadcast: y = input * other."""

    _op_name = "mul"
    kernel_cls = MulFwdKernel


class DivFwdOp(BinaryOp):
    """Element-wise division with broadcast: y = input / other.

    Conforms to ``torch.div(input, other, *, rounding_mode=None)``.
    ``rounding_mode`` accepts ``None`` (true division), ``"trunc"``
    (truncation toward zero), or ``"floor"`` (floor division). Only
    ``rounding_mode is None`` dispatches to the kernel; the trunc /
    floor variants raise ``NotImplementedError`` until a rounded-divide
    kernel lands (tracked in a follow-up issue). The leading ``*``
    makes ``rounding_mode`` and the existing ``strategy`` /
    ``kernel_map`` / ``tune`` parameters keyword-only; only the
    positional triplet ``(a_shape, b_shape, dtype)`` is shared with
    ``BinaryOp``.
    """

    _op_name = "div"
    kernel_cls = DivFwdKernel

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        *,
        rounding_mode: Optional[str] = None,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if rounding_mode is not None and rounding_mode not in ("trunc", "floor"):
            raise ValueError(
                f"DivFwdOp received rounding_mode={rounding_mode!r}; "
                "manifest allows None, 'trunc', or 'floor'"
            )
        super().__init__(
            a_shape, b_shape, dtype, strategy=strategy,
            kernel_map=kernel_map, tune=tune,
        )
        self.rounding_mode = rounding_mode

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        other: torch.Tensor,
    ) -> torch.Tensor:
        if self.rounding_mode is not None:
            raise NotImplementedError(
                f"DivFwdOp(rounding_mode={self.rounding_mode!r}) is not yet "
                "implemented; the current kernel only honors rounding_mode is "
                "None. A follow-up issue tracks the kernel work."
            )
        return super().forward(input, other)


class RemainderFwdOp(BinaryOp):
    """Element-wise remainder with broadcast: y = a % b."""

    _op_name = "remainder"
    kernel_cls = RemainderFwdKernel


class PowFwdOp(BinaryOp):
    """Element-wise power with broadcast: y = input ** exponent.

    Conforms to ``torch.pow(input, exponent)``: the second operand carries
    the manifest-declared name ``exponent`` rather than the generic
    ``other`` so the L1 signature check matches the manifest.
    """

    _op_name = "pow"
    kernel_cls = PowFwdKernel
    _other_name = "exponent"


class FloorDivideFwdOp(BinaryOp):
    """Element-wise floor division with broadcast: y = floor(a / b)."""

    _op_name = "floor_divide"
    kernel_cls = FloorDivideFwdKernel


class LerpFwdOp(BinaryOp):
    """Element-wise lerp with broadcast: y = a + weight * (b - a).

    Unlike ``torch.lerp(a, b, weight)`` where weight is a runtime parameter,
    here weight is a **construction-time constant** baked into the compiled
    kernel. This enables compile-time folding but means a new Op instance is
    needed for each distinct weight value.

    Args:
        a_shape: Shape of input a.
        b_shape: Shape of input b.
        dtype: Torch dtype.
        weight: Scalar interpolation weight, fixed at construction (default 0.5).
        strategy: Kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "lerp"
    kernel_cls = LerpFwdKernel
    _other_name = "end"

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        weight: float = 0.5,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        supported = self.kernel_cls.SUPPORTED_DTYPES
        if supported is not None and dtype not in supported:
            names = ", ".join(str(dt) for dt in supported)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.dtype = dtype
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        self.strategy = strategy
        self._weight = weight
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
            a_shape, b_shape,
        )
        self.out_shape = out_shape
        self._out_shape_list = list(out_shape)  # cached for custom_op hot path
        self.N_total = prod(out_shape)
        self.a_numel = prod(a_shape)
        self.b_numel = prod(b_shape)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            self.N_total, dtype, coalesced_shape, a_strides, b_strides,
            self.a_numel, self.b_numel, strategy=strategy, tune=tune,
            weight=weight,
        )
        # Register in global registry for torch.compile dispatch
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self


class MaximumFwdOp(BinaryOp):
    """Element-wise maximum with broadcast: y = max(a, b)."""

    _op_name = "maximum"
    kernel_cls = MaximumFwdKernel


class MinimumFwdOp(BinaryOp):
    """Element-wise minimum with broadcast: y = min(a, b)."""

    _op_name = "minimum"
    kernel_cls = MinimumFwdKernel


class LerpTensorFwdOp(Op):
    """Tensor-weight lerp: out = input + weight * (end - input).

    Conforms to the Tensor-weight overload of ``torch.lerp`` —
    ``torch.lerp(input, end, weight: Tensor)`` where ``weight`` is a
    Tensor that broadcasts together with ``input`` and ``end`` to the
    output shape. The Op layer expands the three inputs to the broadcast
    shape and dispatches the flat ``LerpTensorFwdKernel`` on
    ``N_total = product(broadcast_shape)`` elements. The scalar-weight
    overload is handled separately by ``LerpFwdOp``.

    Args:
        input: Shape of the start tensor.
        end: Shape of the end tensor.
        weight: Shape of the per-element weight tensor.
        dtype: Torch dtype for all three operands.
    """

    _op_name = "lerp_tensor"
    _wrapped = None

    # Manifest declares all three operands as ``float16 | bfloat16 | float32``;
    # fp8 dtypes are rejected at the op-layer signature so the impl matches
    # the manifest contract (the kernel also rejects fp8 independently).
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        *,
        input: tuple,  # noqa: A002 — manifest-aligned PyTorch param name
        end: tuple,
        weight: tuple,
        dtype: torch.dtype,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"LerpTensorFwdOp does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.input_shape = tuple(input)
        self.end_shape = tuple(end)
        self.weight_shape = tuple(weight)
        self.dtype = dtype
        self.strategy = strategy
        self.out_shape = tuple(
            torch.broadcast_shapes(
                self.input_shape, self.end_shape, self.weight_shape,
            )
        )
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            self.N_total, dtype, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"lerp_tensor": LerpTensorFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        out_shape = self.out_shape if self.out_shape else (1,)
        a_flat = self._expand_flat(input, out_shape)
        b_flat = self._expand_flat(end, out_shape)
        w_flat = self._expand_flat(weight, out_shape)
        result = self.kernel(a_flat, b_flat, w_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if not (input.is_cuda and end.is_cuda and weight.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("end", end, self.end_shape),
            ("weight", weight, self.weight_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(
                    f"Expected {name}.dtype {self.dtype}, got {t.dtype}"
                )
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, end, weight, self._instance_key)
        return self._eager_forward(input, end, weight)
