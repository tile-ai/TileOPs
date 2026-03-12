"""Elementwise op templates and broadcast utility.

Three Op template base classes:
- UnaryOp: wraps UnaryKernel with reshape/flatten
- BinaryOp: wraps BinaryKernel with broadcast coalescing
- FusedGatedOp: wraps FusedGatedKernel with (M, 2N) layout

Utility:
- coalesce_broadcast_dims: reduces N-dim broadcast to minimal effective dims
"""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import (
    AbsKernel,
    AddKernel,
    BitwiseAndKernel,
    BitwiseNotKernel,
    BitwiseOrKernel,
    BitwiseXorKernel,
    CeilKernel,
    CosKernel,
    DivKernel,
    EqKernel,
    ErfKernel,
    ExpKernel,
    Expm1Kernel,
    FloorDivideKernel,
    FloorKernel,
    GeKernel,
    GeluAndMulKernel,
    GeluKernel,
    GeluTanhAndMulKernel,
    GtKernel,
    HardsigmoidKernel,
    HardswishKernel,
    IsfiniteKernel,
    IsinfKernel,
    IsnanKernel,
    LeKernel,
    LerpKernel,
    Log1pKernel,
    LogicalAndKernel,
    LogicalNotKernel,
    LogicalOrKernel,
    LogKernel,
    LtKernel,
    MaximumKernel,
    MinimumKernel,
    MishKernel,
    MulKernel,
    NegKernel,
    NeKernel,
    PowKernel,
    ReciprocalKernel,
    ReluKernel,
    RemainderKernel,
    RoundKernel,
    RsqrtKernel,
    SeluKernel,
    SigmoidKernel,
    SignKernel,
    SiluAndMulKernel,
    SiluKernel,
    SinKernel,
    SqrtKernel,
    SubKernel,
    TanhKernel,
    TruncKernel,
)
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = [
    "coalesce_broadcast_dims",
    "UnaryOp",
    "BinaryOp",
    "FusedGatedOp",
    # Unary
    "ReluOp",
    # Binary arithmetic
    "AddOp",
    "SubOp",
    "MulOp",
    "DivOp",
    "RemainderOp",
    "PowOp",
    "FloorDivideOp",
    "LerpOp",
    "MaximumOp",
    "MinimumOp",
    # Comparison (output bool)
    "EqOp",
    "NeOp",
    "GtOp",
    "LtOp",
    "GeOp",
    "LeOp",
    # Logical (output bool)
    "LogicalAndOp",
    "LogicalOrOp",
    # Bitwise
    "BitwiseAndOp",
    "BitwiseOrOp",
    "BitwiseXorOp",
    # Fused gated
    "SiluAndMulOp",
    "GeluAndMulOp",
    "GeluTanhAndMulOp",
    # --- math (17) ---
    "AbsOp",
    "CeilOp",
    "CosOp",
    "ErfOp",
    "ExpOp",
    "Expm1Op",
    "FloorOp",
    "Log1pOp",
    "LogOp",
    "NegOp",
    "ReciprocalOp",
    "RoundOp",
    "RsqrtOp",
    "SignOp",
    "SinOp",
    "SqrtOp",
    "TruncOp",
    # --- activations (8) ---
    "GeluOp",
    "HardsigmoidOp",
    "HardswishOp",
    "MishOp",
    "SeluOp",
    "SigmoidOp",
    "SiluOp",
    "TanhOp",
    # --- logical (1) ---
    "LogicalNotOp",
    # --- bitwise (1) ---
    "BitwiseNotOp",
    # --- special predicates (3) ---
    "IsfiniteOp",
    "IsinfOp",
    "IsnanOp",
]


def coalesce_broadcast_dims(a_shape, b_shape):
    """Coalesce N-dim broadcast into minimal effective dimensions.

    Merges adjacent dimensions that have the same broadcast behaviour
    (both real or both broadcast) to minimise the number of divmod
    operations inside the kernel loop.

    Args:
        a_shape: Shape tuple of input a.
        b_shape: Shape tuple of input b.

    Returns:
        Tuple of (out_shape, coalesced_shape, a_strides, b_strides) where
        strides use 0 for broadcast dimensions.
    """
    # Normalise scalar (0-dim) inputs to 1-dim with size 1
    if len(a_shape) == 0:
        a_shape = (1,)
    if len(b_shape) == 0:
        b_shape = (1,)

    out_shape = torch.broadcast_shapes(a_shape, b_shape)
    ndim = len(out_shape)
    a_pad = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b_pad = (1,) * (ndim - len(b_shape)) + tuple(b_shape)

    def _make_strides(padded_shape):
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * padded_shape[i + 1]
        # Only zero strides for genuinely broadcast dims (size-1 expanded to >1)
        return [
            0 if padded_shape[i] == 1 and out_shape[i] > 1 else strides[i]
            for i in range(ndim)
        ]

    a_raw = _make_strides(a_pad)
    b_raw = _make_strides(b_pad)

    # Coalesce adjacent dims with compatible broadcast patterns
    groups = [(out_shape[0], a_raw[0], b_raw[0])]
    for i in range(1, ndim):
        prev_out, prev_as, prev_bs = groups[-1]
        a_can = (a_raw[i] == 0 and prev_as == 0) or (
            a_raw[i] != 0 and prev_as == a_raw[i] * out_shape[i]
        )
        b_can = (b_raw[i] == 0 and prev_bs == 0) or (
            b_raw[i] != 0 and prev_bs == b_raw[i] * out_shape[i]
        )
        if a_can and b_can:
            groups[-1] = (prev_out * out_shape[i], a_raw[i], b_raw[i])
        else:
            groups.append((out_shape[i], a_raw[i], b_raw[i]))

    # Remove trivial size-1 groups (unless all trivial)
    groups = [g for g in groups if g[0] > 1] or [(1, 0, 0)]
    coalesced_shape = tuple(g[0] for g in groups)
    a_strides = tuple(g[1] for g in groups)
    b_strides = tuple(g[2] for g in groups)
    return out_shape, coalesced_shape, a_strides, b_strides


class UnaryOp(Op):
    """Template base class for unary elementwise ops.

    Subclass must set ``kernel_cls`` and ``_op_name``.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        strategy: Kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    kernel_cls: type
    _op_name: str

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N_total = N_total
        self.dtype = dtype
        self.strategy = strategy
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            N_total, dtype, strategy=strategy, tune=tune,
        )
        self.output_dtype = getattr(self.kernel, "output_dtype", dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        return self.N_total * (self.dtype.itemsize + self.output_dtype.itemsize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(
                f"Expected {self.N_total} elements, got {x.numel()}"
            )
        orig_shape = x.shape
        x = x.contiguous().reshape(-1)
        return self.kernel(x).reshape(orig_shape)


class BinaryOp(Op):
    """Template base class for binary elementwise ops with broadcast.

    Subclass must set ``kernel_cls`` and ``_op_name``.

    Args:
        a_shape: Shape of input a.
        b_shape: Shape of input b.
        dtype: Torch dtype.
        strategy: Kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    kernel_cls: type
    _op_name: str

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.dtype = dtype
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        self.strategy = strategy
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
            a_shape, b_shape,
        )
        self.out_shape = out_shape
        self.N_total = prod(out_shape)
        self.a_numel = prod(a_shape)
        self.b_numel = prod(b_shape)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            self.N_total, dtype, coalesced_shape, a_strides, b_strides,
            self.a_numel, self.b_numel, strategy=strategy, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read a + read b + write y."""
        elem = self.dtype.itemsize
        return (self.a_numel + self.b_numel + self.N_total) * elem

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        if a.numel() != self.a_numel:
            raise ValueError(
                f"Expected a to have {self.a_numel} elements, got {a.numel()}"
            )
        if b.numel() != self.b_numel:
            raise ValueError(
                f"Expected b to have {self.b_numel} elements, got {b.numel()}"
            )
        return self.kernel(
            a.contiguous().view(-1), b.contiguous().view(-1),
        ).reshape(self.out_shape)


class FusedGatedOp(Op):
    """Template base class for fused gated elementwise ops.

    Input: x of shape (M, 2*N). gate = x[:, :N], value = x[:, N:].
    Output: y = activation(gate) * value, shape (M, N).

    Subclass must set ``kernel_cls`` and ``_op_name``.

    Args:
        M: Number of rows.
        N: Half column dim (output width).
        dtype: Torch dtype.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    kernel_cls: type
    _op_name: str

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](M, N, dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x (M*2N) + write y (M*N)."""
        elem = self.dtype.itemsize
        return (self.M * 2 * self.N + self.M * self.N) * elem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.shape != (self.M, 2 * self.N):
            raise ValueError(
                f"Expected shape ({self.M}, {2 * self.N}), got {tuple(x.shape)}"
            )
        x = x.contiguous()
        return self.kernel(x)


# ---------------------------------------------------------------------------
# Concrete op subclasses
# ---------------------------------------------------------------------------


class ReluOp(UnaryOp):
    """ReLU activation: y = max(x, 0)."""

    _op_name = "relu"
    kernel_cls = ReluKernel


class AddOp(BinaryOp):
    """Element-wise addition with broadcast: y = a + b."""

    _op_name = "add"
    kernel_cls = AddKernel


class SubOp(BinaryOp):
    """Element-wise subtraction with broadcast: y = a - b."""

    _op_name = "sub"
    kernel_cls = SubKernel


class MulOp(BinaryOp):
    """Element-wise multiplication with broadcast: y = a * b."""

    _op_name = "mul"
    kernel_cls = MulKernel


class DivOp(BinaryOp):
    """Element-wise division with broadcast: y = a / b."""

    _op_name = "div"
    kernel_cls = DivKernel


class RemainderOp(BinaryOp):
    """Element-wise remainder with broadcast: y = a % b."""

    _op_name = "remainder"
    kernel_cls = RemainderKernel


class PowOp(BinaryOp):
    """Element-wise power with broadcast: y = a ** b."""

    _op_name = "pow"
    kernel_cls = PowKernel


class FloorDivideOp(BinaryOp):
    """Element-wise floor division with broadcast: y = floor(a / b)."""

    _op_name = "floor_divide"
    kernel_cls = FloorDivideKernel


class LerpOp(BinaryOp):
    """Element-wise lerp with broadcast: y = a + weight * (b - a).

    PyTorch lerp is ternary; weight is a compile-time scalar here.

    Args:
        a_shape: Shape of input a.
        b_shape: Shape of input b.
        dtype: Torch dtype.
        weight: Scalar interpolation weight (default 0.5).
        strategy: Kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "lerp"
    kernel_cls = LerpKernel

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
        self.dtype = dtype
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        self.strategy = strategy
        self._weight = weight
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
            a_shape, b_shape,
        )
        self.out_shape = out_shape
        self.N_total = prod(out_shape)
        self.a_numel = prod(a_shape)
        self.b_numel = prod(b_shape)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            self.N_total, dtype, coalesced_shape, a_strides, b_strides,
            self.a_numel, self.b_numel, strategy=strategy, tune=tune,
            weight=weight,
        )


class MaximumOp(BinaryOp):
    """Element-wise maximum with broadcast: y = max(a, b)."""

    _op_name = "maximum"
    kernel_cls = MaximumKernel


class MinimumOp(BinaryOp):
    """Element-wise minimum with broadcast: y = min(a, b)."""

    _op_name = "minimum"
    kernel_cls = MinimumKernel


# ---------------------------------------------------------------------------
# Comparison op subclasses (output bool)
# ---------------------------------------------------------------------------
#
# Kernels produce int8 (1/0) because TileLang cannot vectorize bool.
# The Op forward() casts to torch.bool after the kernel call.


class _BoolOutputBinaryOp(BinaryOp):
    """Mixin that casts kernel int8 output to torch.bool."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = super().forward(a, b)
        return result.to(torch.bool)


class EqOp(_BoolOutputBinaryOp):
    """Element-wise equality with broadcast: y = (a == b)."""

    _op_name = "eq"
    kernel_cls = EqKernel


class NeOp(_BoolOutputBinaryOp):
    """Element-wise not-equal with broadcast: y = (a != b)."""

    _op_name = "ne"
    kernel_cls = NeKernel


class GtOp(_BoolOutputBinaryOp):
    """Element-wise greater-than with broadcast: y = (a > b)."""

    _op_name = "gt"
    kernel_cls = GtKernel


class LtOp(_BoolOutputBinaryOp):
    """Element-wise less-than with broadcast: y = (a < b)."""

    _op_name = "lt"
    kernel_cls = LtKernel


class GeOp(_BoolOutputBinaryOp):
    """Element-wise greater-equal with broadcast: y = (a >= b)."""

    _op_name = "ge"
    kernel_cls = GeKernel


class LeOp(_BoolOutputBinaryOp):
    """Element-wise less-equal with broadcast: y = (a <= b)."""

    _op_name = "le"
    kernel_cls = LeKernel


# ---------------------------------------------------------------------------
# Logical op subclasses (output bool)
# ---------------------------------------------------------------------------


class LogicalAndOp(_BoolOutputBinaryOp):
    """Element-wise logical AND with broadcast: y = a && b."""

    _op_name = "logical_and"
    kernel_cls = LogicalAndKernel


class LogicalOrOp(_BoolOutputBinaryOp):
    """Element-wise logical OR with broadcast: y = a || b."""

    _op_name = "logical_or"
    kernel_cls = LogicalOrKernel


# ---------------------------------------------------------------------------
# Bitwise op subclasses
# ---------------------------------------------------------------------------


class BitwiseAndOp(BinaryOp):
    """Element-wise bitwise AND with broadcast: y = a & b."""

    _op_name = "bitwise_and"
    kernel_cls = BitwiseAndKernel


class BitwiseOrOp(BinaryOp):
    """Element-wise bitwise OR with broadcast: y = a | b."""

    _op_name = "bitwise_or"
    kernel_cls = BitwiseOrKernel


class BitwiseXorOp(BinaryOp):
    """Element-wise bitwise XOR with broadcast: y = a ^ b."""

    _op_name = "bitwise_xor"
    kernel_cls = BitwiseXorKernel


# ---------------------------------------------------------------------------
# Fused gated op subclasses
# ---------------------------------------------------------------------------


class SiluAndMulOp(FusedGatedOp):
    """SiLU-and-Mul: y = silu(gate) * value."""

    _op_name = "silu_and_mul"
    kernel_cls = SiluAndMulKernel


class GeluAndMulOp(FusedGatedOp):
    """GELU-and-Mul: y = gelu(gate) * value (exact GELU)."""

    _op_name = "gelu_and_mul"
    kernel_cls = GeluAndMulKernel


class GeluTanhAndMulOp(FusedGatedOp):
    """GELU-Tanh-and-Mul: y = gelu_tanh(gate) * value (tanh approximation)."""

    _op_name = "gelu_tanh_and_mul"
    kernel_cls = GeluTanhAndMulKernel


# ---------------------------------------------------------------------------
# Unary math ops (17)
# ---------------------------------------------------------------------------


class ExpOp(UnaryOp):
    """Element-wise exp(x)."""

    _op_name = "exp"
    kernel_cls = ExpKernel


class LogOp(UnaryOp):
    """Element-wise log(x)."""

    _op_name = "log"
    kernel_cls = LogKernel


class SqrtOp(UnaryOp):
    """Element-wise sqrt(x)."""

    _op_name = "sqrt"
    kernel_cls = SqrtKernel


class RsqrtOp(UnaryOp):
    """Element-wise 1/sqrt(x)."""

    _op_name = "rsqrt"
    kernel_cls = RsqrtKernel


class AbsOp(UnaryOp):
    """Element-wise |x|."""

    _op_name = "abs"
    kernel_cls = AbsKernel


class NegOp(UnaryOp):
    """Element-wise -x."""

    _op_name = "neg"
    kernel_cls = NegKernel


class ReciprocalOp(UnaryOp):
    """Element-wise 1/x."""

    _op_name = "reciprocal"
    kernel_cls = ReciprocalKernel


class SignOp(UnaryOp):
    """Element-wise sign(x): -1, 0, or +1."""

    _op_name = "sign"
    kernel_cls = SignKernel


class SinOp(UnaryOp):
    """Element-wise sin(x)."""

    _op_name = "sin"
    kernel_cls = SinKernel


class CosOp(UnaryOp):
    """Element-wise cos(x)."""

    _op_name = "cos"
    kernel_cls = CosKernel


class FloorOp(UnaryOp):
    """Element-wise floor(x)."""

    _op_name = "floor"
    kernel_cls = FloorKernel


class CeilOp(UnaryOp):
    """Element-wise ceil(x)."""

    _op_name = "ceil"
    kernel_cls = CeilKernel


class RoundOp(UnaryOp):
    """Element-wise round(x)."""

    _op_name = "round"
    kernel_cls = RoundKernel


class TruncOp(UnaryOp):
    """Element-wise trunc(x)."""

    _op_name = "trunc"
    kernel_cls = TruncKernel


class ErfOp(UnaryOp):
    """Element-wise erf(x)."""

    _op_name = "erf"
    kernel_cls = ErfKernel


class Log1pOp(UnaryOp):
    """Element-wise log(1 + x)."""

    _op_name = "log1p"
    kernel_cls = Log1pKernel


class Expm1Op(UnaryOp):
    """Element-wise exp(x) - 1."""

    _op_name = "expm1"
    kernel_cls = Expm1Kernel


# ---------------------------------------------------------------------------
# Activation ops (8)
# ---------------------------------------------------------------------------


class GeluOp(UnaryOp):
    """Element-wise GELU using the standard erf formulation."""

    _op_name = "gelu"
    kernel_cls = GeluKernel


class SiluOp(UnaryOp):
    """Element-wise SiLU (Swish): x * sigmoid(x)."""

    _op_name = "silu"
    kernel_cls = SiluKernel


class SigmoidOp(UnaryOp):
    """Element-wise sigmoid(x)."""

    _op_name = "sigmoid"
    kernel_cls = SigmoidKernel


class TanhOp(UnaryOp):
    """Element-wise tanh(x)."""

    _op_name = "tanh"
    kernel_cls = TanhKernel


class HardswishOp(UnaryOp):
    """Element-wise HardSwish: x * clamp(x + 3, 0, 6) / 6."""

    _op_name = "hardswish"
    kernel_cls = HardswishKernel


class HardsigmoidOp(UnaryOp):
    """Element-wise HardSigmoid: clamp(x + 3, 0, 6) / 6."""

    _op_name = "hardsigmoid"
    kernel_cls = HardsigmoidKernel


class MishOp(UnaryOp):
    """Element-wise Mish: x * tanh(softplus(x))."""

    _op_name = "mish"
    kernel_cls = MishKernel


class SeluOp(UnaryOp):
    """Element-wise SELU."""

    _op_name = "selu"
    kernel_cls = SeluKernel


# ---------------------------------------------------------------------------
# Logical op (1)
# ---------------------------------------------------------------------------


class LogicalNotOp(UnaryOp):
    """Element-wise logical NOT with bool output."""

    _op_name = "logical_not"
    kernel_cls = LogicalNotKernel


# ---------------------------------------------------------------------------
# Bitwise op (1)
# ---------------------------------------------------------------------------


class BitwiseNotOp(UnaryOp):
    """Element-wise bitwise NOT (~x) for bool/integer inputs."""

    _op_name = "bitwise_not"
    kernel_cls = BitwiseNotKernel


# ---------------------------------------------------------------------------
# Special predicate ops (3)
# ---------------------------------------------------------------------------


class IsnanOp(UnaryOp):
    """Element-wise isnan with bool output."""

    _op_name = "isnan"
    kernel_cls = IsnanKernel


class IsinfOp(UnaryOp):
    """Element-wise isinf with bool output."""

    _op_name = "isinf"
    kernel_cls = IsinfKernel


class IsfiniteOp(UnaryOp):
    """Element-wise isfinite with bool output."""

    _op_name = "isfinite"
    kernel_cls = IsfiniteKernel
