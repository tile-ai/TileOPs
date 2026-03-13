"""Elementwise op templates and broadcast utility.

Three Op template base classes:
- UnaryOp: wraps UnaryKernel with reshape/flatten
- BinaryOp: wraps BinaryKernel with broadcast coalescing
- FusedGatedOp: wraps FusedGatedKernel with (M, 2N) layout

torch.compile support:
- All 55 concrete ops are registered via @torch.library.custom_op at module load time
- Three factory functions (_register_unary_custom_op, _register_binary_custom_op,
  _register_fused_gated_custom_op) register every op; instances are looked up at
  runtime via _OP_REGISTRY keyed by id(instance)

Utility:
- coalesce_broadcast_dims: reduces N-dim broadcast to minimal effective dims
"""

import weakref
from math import prod
from typing import Dict, List, Optional

import torch

from tileops.kernels.elementwise import (
    AbsKernel,
    AddKernel,
    AlibiKernel,
    BitwiseAndKernel,
    BitwiseNotKernel,
    BitwiseOrKernel,
    BitwiseXorKernel,
    CeilKernel,
    ClampKernel,
    CosKernel,
    DivKernel,
    EluKernel,
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
    HardtanhKernel,
    IsfiniteKernel,
    IsinfKernel,
    IsnanKernel,
    LeakyReluKernel,
    LeKernel,
    LerpKernel,
    Log1pKernel,
    LogicalAndKernel,
    LogicalNotKernel,
    LogicalOrKernel,
    LogKernel,
    LtKernel,
    MaskedFillKernel,
    MaximumKernel,
    MinimumKernel,
    MishKernel,
    MulKernel,
    NanToNumKernel,
    NegKernel,
    NeKernel,
    PowKernel,
    PreluKernel,
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
    SinusoidalKernel,
    SoftplusKernel,
    SqrtKernel,
    SubKernel,
    TanhKernel,
    TruncKernel,
    WhereKernel,
)
from tileops.kernels.kernel import Kernel

from .op import Op

# ---------------------------------------------------------------------------
# torch.compile registration factories
#
# Each factory creates a @torch.library.custom_op + register_fake pair.
# Instances register themselves in _OP_REGISTRY keyed by integer id.
# The custom_op receives this key and looks up the instance to call the
# pre-built tilelang kernel.  The key is a plain int so dynamo can trace
# through forward() without hitting unsupported Python side-effects.
# ---------------------------------------------------------------------------

_OP_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _register_unary_custom_op(op_cls, output_dtype_override=None):
    """Register a unary elementwise op for torch.compile.

    Args:
        op_cls: The Op subclass to register (must have ``_op_name``).
        output_dtype_override: If set, the output dtype (e.g. torch.bool for predicates).
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_unary_{op_name}", mutates_args=())
    def _wrapped(x: torch.Tensor, instance_key: int) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, instance_key: int) -> torch.Tensor:
        out_dtype = output_dtype_override if output_dtype_override is not None else x.dtype
        return torch.empty_like(x, dtype=out_dtype)

    op_cls._wrapped = _wrapped


def _register_binary_custom_op(op_cls, output_bool: bool = False):
    """Register a binary elementwise op for torch.compile.

    Args:
        op_cls: The Op subclass to register.
        output_bool: If True, output dtype is torch.bool (for comparison/logical ops).
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_binary_{op_name}", mutates_args=())
    def _wrapped(
        a: torch.Tensor,
        b: torch.Tensor,
        out_shape: List[int],
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(a, b)

    @_wrapped.register_fake
    def _(
        a: torch.Tensor,
        b: torch.Tensor,
        out_shape: List[int],
        instance_key: int,
    ) -> torch.Tensor:
        out_dtype = torch.bool if output_bool else a.dtype
        return a.new_empty(out_shape, dtype=out_dtype)

    op_cls._wrapped = _wrapped


def _register_fused_gated_custom_op(op_cls):
    """Register a fused gated elementwise op for torch.compile.

    Args:
        op_cls: The Op subclass to register.
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_fused_gated_{op_name}", mutates_args=())
    def _wrapped(
        x: torch.Tensor,
        M: int,
        N: int,
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(
        x: torch.Tensor,
        M: int,
        N: int,
        instance_key: int,
    ) -> torch.Tensor:
        return x.new_empty((M, N), dtype=x.dtype)

    op_cls._wrapped = _wrapped


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
    # --- independent (custom-signature, 11) ---
    "LeakyReluOp",
    "EluOp",
    "HardtanhOp",
    "SoftplusOp",
    "PreluOp",
    "WhereOp",
    "ClampOp",
    "MaskedFillOp",
    "NanToNumOp",
    "AlibiOp",
    "SinusoidalOp",
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
    Subclass should also set ``_wrapped`` via ``_register_unary_custom_op``
    to enable torch.compile support.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        strategy: Kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    kernel_cls: type
    _op_name: str
    _wrapped = None  # Set by _register_unary_custom_op at class definition

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
        # Register in global registry for torch.compile dispatch
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        return self.N_total * (self.dtype.itemsize + self.output_dtype.itemsize)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        orig_shape = x.shape
        x = x.contiguous().reshape(-1)
        return self.kernel(x).reshape(orig_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(
                f"Expected {self.N_total} elements, got {x.numel()}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self._instance_key)
        return self._eager_forward(x)


class BinaryOp(Op):
    """Template base class for binary elementwise ops with broadcast.

    Subclass must set ``kernel_cls`` and ``_op_name``.
    Subclass should also set ``_wrapped`` via ``_register_binary_custom_op``
    to enable torch.compile support.

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
    _wrapped = None  # Set by _register_binary_custom_op at class definition

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
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
        )
        # Register in global registry for torch.compile dispatch
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read a + read b + write y."""
        elem = self.dtype.itemsize
        return (self.a_numel + self.b_numel + self.N_total) * elem

    def _eager_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        return self.kernel(
            a.contiguous().view(-1), b.contiguous().view(-1),
        ).reshape(self.out_shape)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        if a.dtype != self.dtype:
            raise ValueError(f"Expected a.dtype {self.dtype}, got {a.dtype}")
        if b.dtype != self.dtype:
            raise ValueError(f"Expected b.dtype {self.dtype}, got {b.dtype}")
        if a.numel() != self.a_numel:
            raise ValueError(
                f"Expected a to have {self.a_numel} elements, got {a.numel()}"
            )
        if b.numel() != self.b_numel:
            raise ValueError(
                f"Expected b to have {self.b_numel} elements, got {b.numel()}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(a, b, self._out_shape_list, self._instance_key)
        return self._eager_forward(a, b)


class FusedGatedOp(Op):
    """Template base class for fused gated elementwise ops.

    Input: x of shape (M, 2*N). gate = x[:, :N], value = x[:, N:].
    Output: y = activation(gate) * value, shape (M, N).

    Subclass must set ``kernel_cls`` and ``_op_name``.
    Subclass should also set ``_wrapped`` via ``_register_fused_gated_custom_op``
    to enable torch.compile support.

    Args:
        M: Number of rows.
        N: Half column dim (output width).
        dtype: Torch dtype.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    kernel_cls: type
    _op_name: str
    _wrapped = None  # Set by _register_fused_gated_custom_op at class definition

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
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
        self.M = M
        self.N = N
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](M, N, dtype, tune=tune)
        # Register in global registry for torch.compile dispatch
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x (M*2N) + write y (M*N)."""
        elem = self.dtype.itemsize
        return (self.M * 2 * self.N + self.M * self.N) * elem

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        x = x.contiguous()
        return self.kernel(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.shape != (self.M, 2 * self.N):
            raise ValueError(
                f"Expected shape ({self.M}, {2 * self.N}), got {tuple(x.shape)}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self.M, self.N, self._instance_key)
        return self._eager_forward(x)


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
    """Mixin that casts kernel int8 output to torch.bool.

    _eager_forward casts the int8 kernel output to bool.  In the torch.compile
    path, register_fake already declares torch.bool as the output dtype, and
    the actual execution goes through _eager_forward which handles the cast.
    No forward override is needed because BinaryOp.forward delegates to
    _eager_forward (eager) or _wrapped (compile, where register_fake is correct).
    """

    def _eager_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = super()._eager_forward(a, b)
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
    """Element-wise logical AND with broadcast using non-zero truthiness."""

    _op_name = "logical_and"
    kernel_cls = LogicalAndKernel


class LogicalOrOp(_BoolOutputBinaryOp):
    """Element-wise logical OR with broadcast using non-zero truthiness."""

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


# ---------------------------------------------------------------------------
# Independent (custom-signature) op classes (11)
# ---------------------------------------------------------------------------


class LeakyReluOp(Op):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        negative_slope: Slope for negative inputs (default 0.01).
    """

    def __init__(self, N_total: int, dtype: torch.dtype, negative_slope: float = 0.01):
        self.N_total = N_total
        self.dtype = dtype
        self.negative_slope = negative_slope
        self.kernel = LeakyReluKernel(N_total, dtype, negative_slope=negative_slope)

    @property
    def default_kernel_map(self):
        return {"leaky_relu": LeakyReluKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        return self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)


class EluOp(Op):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        alpha: Scale for the negative part (default 1.0).
    """

    def __init__(self, N_total: int, dtype: torch.dtype, alpha: float = 1.0):
        self.N_total = N_total
        self.dtype = dtype
        self.alpha = alpha
        self.kernel = EluKernel(N_total, dtype, alpha=alpha)

    @property
    def default_kernel_map(self):
        return {"elu": EluKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        return self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)


class HardtanhOp(Op):
    """Hardtanh: y = clamp(x, min_val, max_val).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        min_val: Lower bound (default -1.0).
        max_val: Upper bound (default 1.0).
    """

    def __init__(self, N_total: int, dtype: torch.dtype,
                 min_val: float = -1.0, max_val: float = 1.0):
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.kernel = HardtanhKernel(N_total, dtype, min_val=min_val, max_val=max_val)

    @property
    def default_kernel_map(self):
        return {"hardtanh": HardtanhKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        return self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)


class SoftplusOp(Op):
    """Softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        beta: Scaling factor (default 1.0).
        threshold: Linear regime threshold (default 20.0).
    """

    def __init__(self, N_total: int, dtype: torch.dtype,
                 beta: float = 1.0, threshold: float = 20.0):
        self.N_total = N_total
        self.dtype = dtype
        self.beta = beta
        self.threshold = threshold
        self.kernel = SoftplusKernel(N_total, dtype, beta=beta, threshold=threshold)

    @property
    def default_kernel_map(self):
        return {"softplus": SoftplusKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        return self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)


class PreluOp(Op):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel dimension follows PyTorch convention: dimension 1 for inputs
    with ndim >= 2, dimension 0 for 1-D inputs.

    Args:
        shape: Shape of the input tensor (must have a channel dimension).
        dtype: Torch dtype.
        num_channels: Number of channels (weight length).
    """

    def __init__(self, shape: tuple, dtype: torch.dtype, num_channels: int):
        self.shape = shape
        self.dtype = dtype
        self.num_channels = num_channels
        N_total = prod(shape)
        self.N_total = N_total
        # PyTorch PReLU: channel dim is 1 for ndim>=2, else 0
        inner_size = (prod(shape[2:]) if len(shape) > 2 else 1) if len(shape) >= 2 else 1
        self.inner_size = inner_size
        self.kernel = PreluKernel(N_total, num_channels, inner_size, dtype)

    @property
    def default_kernel_map(self):
        return {"prelu": PreluKernel}

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        return self.kernel(x.contiguous().reshape(-1), weight.contiguous().reshape(-1)).reshape(orig_shape)


class WhereOp(Op):
    """Where: out = cond ? x : y.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype for x and y.
    """

    def __init__(self, N_total: int, dtype: torch.dtype):
        self.N_total = N_total
        self.dtype = dtype
        self.kernel = WhereKernel(N_total, dtype)

    @property
    def default_kernel_map(self):
        return {"where": WhereKernel}

    def forward(self, cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        cond_flat = cond.contiguous().reshape(-1).to(torch.int8)
        return self.kernel(
            cond_flat,
            x.contiguous().reshape(-1),
            y.contiguous().reshape(-1),
        ).reshape(orig_shape)


class ClampOp(Op):
    """Clamp: y = clamp(x, min, max) with optional bounds.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        min_val: Lower bound (None = no lower bound).
        max_val: Upper bound (None = no upper bound).
    """

    def __init__(self, N_total: int, dtype: torch.dtype,
                 min_val: float = None, max_val: float = None):
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.kernel = ClampKernel(N_total, dtype, min_val=min_val, max_val=max_val)

    @property
    def default_kernel_map(self):
        return {"clamp": ClampKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        return self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)


class MaskedFillOp(Op):
    """MaskedFill: out = mask ? fill_value : x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        fill_value: Scalar value to fill where mask is True.
    """

    def __init__(self, N_total: int, dtype: torch.dtype, fill_value: float):
        self.N_total = N_total
        self.dtype = dtype
        self.fill_value = fill_value
        self.kernel = MaskedFillKernel(N_total, dtype, fill_value)

    @property
    def default_kernel_map(self):
        return {"masked_fill": MaskedFillKernel}

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        mask_flat = mask.contiguous().reshape(-1).to(torch.int8)
        return self.kernel(
            x.contiguous().reshape(-1),
            mask_flat,
        ).reshape(orig_shape)


class NanToNumOp(Op):
    """NanToNum: replace NaN, +Inf, -Inf with specified values.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        nan_val: Replacement for NaN (default 0.0).
        posinf_val: Replacement for +Inf (default 1e4).
        neginf_val: Replacement for -Inf (default -1e4).
    """

    def __init__(self, N_total: int, dtype: torch.dtype,
                 nan_val: float = 0.0, posinf_val: float = 1e4, neginf_val: float = -1e4):
        self.N_total = N_total
        self.dtype = dtype
        self.nan_val = nan_val
        self.posinf_val = posinf_val
        self.neginf_val = neginf_val
        self.kernel = NanToNumKernel(
            N_total, dtype, nan_val=nan_val, posinf_val=posinf_val, neginf_val=neginf_val,
        )

    @property
    def default_kernel_map(self):
        return {"nan_to_num": NanToNumKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        orig_shape = x.shape
        return self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)


class AlibiOp(Op):
    """ALiBi position encoding: bias[h, i, j] = -slope_h * |i - j|.

    Generates the full (num_heads, seq_len, seq_len) bias tensor.

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        dtype: Torch dtype.
    """

    def __init__(self, seq_len: int, num_heads: int, dtype: torch.dtype):
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dtype = dtype
        self.kernel = AlibiKernel(seq_len, num_heads, dtype)

    @property
    def default_kernel_map(self):
        return {"alibi": AlibiKernel}

    def forward(self) -> torch.Tensor:
        out = self.kernel()
        return out.reshape(self.num_heads, self.seq_len, self.seq_len)


class SinusoidalOp(Op):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Generates the full (seq_len, d_model) encoding tensor.

    Args:
        seq_len: Sequence length.
        d_model: Model dimension.
        dtype: Torch dtype.
    """

    def __init__(self, seq_len: int, d_model: int, dtype: torch.dtype):
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        self.kernel = SinusoidalKernel(seq_len, d_model, dtype)

    @property
    def default_kernel_map(self):
        return {"sinusoidal": SinusoidalKernel}

    def forward(self) -> torch.Tensor:
        out = self.kernel()
        return out.reshape(self.seq_len, self.d_model)


# ---------------------------------------------------------------------------
# torch.compile registration for all 55 concrete ops
# ---------------------------------------------------------------------------

# --- Unary ops: float-preserving output (1 + 17 + 8 + 1 = 27 ops) ---
for _cls in [
    ReluOp,
    # math (17)
    ExpOp, LogOp, SqrtOp, RsqrtOp, AbsOp, NegOp, ReciprocalOp, SignOp,
    SinOp, CosOp, FloorOp, CeilOp, RoundOp, TruncOp, ErfOp, Log1pOp, Expm1Op,
    # activations (8)
    GeluOp, SiluOp, SigmoidOp, TanhOp, HardswishOp, HardsigmoidOp, MishOp, SeluOp,
    # bitwise (1) -- output same dtype as input
    BitwiseNotOp,
]:
    _register_unary_custom_op(_cls)

# --- Unary ops: bool output (4 ops) ---
for _cls in [LogicalNotOp, IsnanOp, IsinfOp, IsfiniteOp]:
    _register_unary_custom_op(_cls, output_dtype_override=torch.bool)

# --- Binary ops: same-dtype output (10 + 3 = 13 ops) ---
for _cls in [
    # arithmetic (10)
    AddOp, SubOp, MulOp, DivOp, RemainderOp, PowOp, FloorDivideOp,
    LerpOp, MaximumOp, MinimumOp,
    # bitwise (3)
    BitwiseAndOp, BitwiseOrOp, BitwiseXorOp,
]:
    _register_binary_custom_op(_cls)

# --- Binary ops: bool output (comparison 6 + logical 2 = 8 ops) ---
for _cls in [
    EqOp, NeOp, GtOp, LtOp, GeOp, LeOp,
    LogicalAndOp, LogicalOrOp,
]:
    _register_binary_custom_op(_cls, output_bool=True)

# --- Fused gated ops (3 ops) ---
for _cls in [SiluAndMulOp, GeluAndMulOp, GeluTanhAndMulOp]:
    _register_fused_gated_custom_op(_cls)

# Clean up loop variable
del _cls
