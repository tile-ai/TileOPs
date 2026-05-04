"""Elementwise op templates and broadcast utility.

Three Op template base classes:
- UnaryOp: wraps UnaryKernel with reshape/flatten
- BinaryOp: wraps BinaryKernel with broadcast coalescing
- FusedGatedOp: wraps FusedGatedKernel with (M, 2N) layout

torch.compile support:
- All 66 concrete ops are registered via @torch.library.custom_op at module load time
- Three factory functions (_register_unary_custom_op, _register_binary_custom_op,
  _register_fused_gated_custom_op) register every op; instances are looked up at
  runtime via _OP_REGISTRY keyed by id(instance)

Utility:
- coalesce_broadcast_dims: reduces N-dim broadcast to minimal effective dims
"""

import math
import weakref
from math import prod
from typing import Dict, List, Optional

import torch

from tileops.kernels.elementwise import (
    AbsFwdKernel,
    AddFwdKernel,
    AlibiFwdKernel,
    BitwiseAndFwdKernel,
    BitwiseNotFwdKernel,
    BitwiseOrFwdKernel,
    BitwiseXorFwdKernel,
    CeilFwdKernel,
    ClampFwdKernel,
    ClampTensorFwdKernel,
    CosFwdKernel,
    DivFwdKernel,
    EluFwdKernel,
    EqFwdKernel,
    ErfFwdKernel,
    ExpFwdKernel,
    Expm1FwdKernel,
    FloorDivideFwdKernel,
    FloorFwdKernel,
    GeFwdKernel,
    GeluAndMulFwdKernel,
    GeluFwdKernel,
    GeluTanhAndMulFwdKernel,
    GtFwdKernel,
    HardsigmoidFwdKernel,
    HardswishFwdKernel,
    HardtanhFwdKernel,
    IsfiniteFwdKernel,
    IsinfFwdKernel,
    IsnanFwdKernel,
    LeakyReluFwdKernel,
    LeFwdKernel,
    LerpFwdKernel,
    Log1pFwdKernel,
    LogFwdKernel,
    LogicalAndFwdKernel,
    LogicalNotFwdKernel,
    LogicalOrFwdKernel,
    LtFwdKernel,
    MaskedFillFwdKernel,
    MaskedFillTensorValueFwdKernel,
    MaximumFwdKernel,
    MinimumFwdKernel,
    MishFwdKernel,
    MulFwdKernel,
    NanToNumFwdKernel,
    NeFwdKernel,
    NegFwdKernel,
    PowFwdKernel,
    PreluFwdKernel,
    ReciprocalFwdKernel,
    ReluFwdKernel,
    RemainderFwdKernel,
    RoundFwdKernel,
    RsqrtFwdKernel,
    SeluFwdKernel,
    SigmoidFwdKernel,
    SignFwdKernel,
    SiluAndMulFwdKernel,
    SiluFwdKernel,
    SinFwdKernel,
    SinusoidalFwdKernel,
    SoftplusFwdKernel,
    SqrtFwdKernel,
    SubFwdKernel,
    TanhFwdKernel,
    TruncFwdKernel,
    WhereFwdKernel,
)
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

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

_FP8_NONSAT_OUTPUT_DTYPES = {
    torch.float8_e5m2: torch.float16,
}


def _effective_scalar_kernel_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the dtype used when scalar literals are materialized in kernels."""
    return _FP8_NONSAT_OUTPUT_DTYPES.get(dtype, dtype)


def _validate_scalar_param_repr(
    param_name: str, value: float, dtype: torch.dtype, op_name: str,
) -> None:
    """Reject scalar params that cannot be represented in the kernel dtype."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{op_name} expected scalar {param_name} to be int/float, got {type(value)}")

    kernel_dtype = _effective_scalar_kernel_dtype(dtype)
    finfo = torch.finfo(kernel_dtype)
    value_f64 = float(value)
    if math.isnan(value_f64):
        return
    if math.isinf(value_f64):
        raise ValueError(
            f"{op_name} received {param_name}={value!r}, but {param_name} must be finite and "
            f"representable in effective kernel dtype {kernel_dtype}"
        )
    if not (finfo.min <= value_f64 <= finfo.max):
        raise ValueError(
            f"{op_name} received {param_name}={value!r}, which is not representable in "
            f"effective kernel dtype {kernel_dtype} (valid finite range: "
            f"[{finfo.min}, {finfo.max}])"
        )


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


def _register_prelu_custom_op(op_cls):
    """Register a PReLU-style op (x, weight -> y) for torch.compile."""
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        x: torch.Tensor,
        weight: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(x, weight)

    @_wrapped.register_fake
    def _(
        x: torch.Tensor,
        weight: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        return torch.empty_like(x)

    op_cls._wrapped = _wrapped


def _register_where_custom_op(op_cls):
    """Register a where-style op (cond, x, y -> out) for torch.compile.

    The fake function computes the broadcast output shape from
    ``cond`` / ``x`` / ``y`` so that ``torch.compile(fullgraph=True)``
    works for both same-shape and broadcasting inputs.
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        cond: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(cond, x, y)

    @_wrapped.register_fake
    def _(
        cond: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        out_shape = torch.broadcast_shapes(cond.shape, x.shape, y.shape)
        return x.new_empty(out_shape)

    op_cls._wrapped = _wrapped


def _register_masked_fill_custom_op(op_cls):
    """Register a masked-fill-style op (x, mask -> y) for torch.compile.

    The fake function computes the bidirectional broadcast output shape
    of ``x`` and ``mask`` so ``torch.compile(fullgraph=True)`` works for
    both same-shape and broadcasting inputs.
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        x: torch.Tensor,
        mask: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(x, mask)

    @_wrapped.register_fake
    def _(
        x: torch.Tensor,
        mask: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        out_shape = torch.broadcast_shapes(x.shape, mask.shape)
        return x.new_empty(out_shape)

    op_cls._wrapped = _wrapped


def _register_masked_fill_tensor_value_custom_op(op_cls):
    """Register a masked-fill (Tensor value) op (input, mask, value -> out).

    The fake function computes the broadcast output shape of ``input`` and
    ``mask`` (``value`` is a 0-dim Tensor). Registered under a distinct
    namespace from the scalar masked_fill variant to avoid collision.
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(
        f"top::elementwise_{op_name}_tensor_value", mutates_args=(),
    )
    def _wrapped(
        input: torch.Tensor,  # noqa: A002
        mask: torch.Tensor,
        value: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(input, mask, value)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,  # noqa: A002
        mask: torch.Tensor,
        value: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        out_shape = torch.broadcast_shapes(input.shape, mask.shape)
        return input.new_empty(out_shape)

    op_cls._wrapped = _wrapped


def _register_clamp_tensor_custom_op(op_cls):
    """Register a Tensor-bound clamp op (input, min?, max? -> out).

    ``min`` and ``max`` are each ``Optional[Tensor]``; the schema is
    inferred by ``torch.library.custom_op`` from the ``Optional[torch.Tensor]``
    annotations, producing ``Tensor? min, Tensor? max`` in the underlying
    custom-op schema. The fake function computes the broadcast output
    shape of all non-``None`` operands so ``torch.compile(fullgraph=True)``
    works for both same-shape and broadcasting inputs. Registered under
    a distinct ``_tensor`` namespace from the scalar-bound clamp variant.
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(
        f"top::elementwise_{op_name}_tensor", mutates_args=(),
    )
    def _wrapped(
        input: torch.Tensor,  # noqa: A002
        min: Optional[torch.Tensor],  # noqa: A002
        max: Optional[torch.Tensor],  # noqa: A002
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(input, min, max)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,  # noqa: A002
        min: Optional[torch.Tensor],  # noqa: A002
        max: Optional[torch.Tensor],  # noqa: A002
        instance_key: int,
    ) -> torch.Tensor:
        shapes = [input.shape]
        if min is not None:
            shapes.append(min.shape)
        if max is not None:
            shapes.append(max.shape)
        out_shape = torch.broadcast_shapes(*shapes)
        return input.new_empty(out_shape)

    op_cls._wrapped = _wrapped


def _register_clamp_min_custom_op(op_cls):
    """Register single-bound Tensor lower-clamp (input, min -> out)."""
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        input: torch.Tensor,  # noqa: A002
        min: torch.Tensor,    # noqa: A002
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(input, min)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,  # noqa: A002
        min: torch.Tensor,    # noqa: A002
        instance_key: int,
    ) -> torch.Tensor:
        out_shape = torch.broadcast_shapes(input.shape, min.shape)
        return input.new_empty(out_shape)

    op_cls._wrapped = _wrapped


def _register_clamp_max_custom_op(op_cls):
    """Register single-bound Tensor upper-clamp (input, max -> out)."""
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        input: torch.Tensor,  # noqa: A002
        max: torch.Tensor,    # noqa: A002
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(input, max)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,  # noqa: A002
        max: torch.Tensor,    # noqa: A002
        instance_key: int,
    ) -> torch.Tensor:
        out_shape = torch.broadcast_shapes(input.shape, max.shape)
        return input.new_empty(out_shape)

    op_cls._wrapped = _wrapped


def _register_generative_custom_op(op_cls, out_shape_fn):
    """Register a generative op (no tensor input -> out) for torch.compile.

    A scalar ``device_carrier`` tensor is passed so that ``register_fake``
    can derive the correct device and dtype from a real tensor reference,
    which is required by the torch.compile tracing infrastructure.

    Args:
        op_cls: The Op subclass to register.
        out_shape_fn: Callable(carrier, num_a, num_b) -> Tensor returning
            the output metadata so register_fake can produce the right shape.
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        device_carrier: torch.Tensor,
        num_a: int,
        num_b: int,
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward()

    @_wrapped.register_fake
    def _(
        device_carrier: torch.Tensor,
        num_a: int,
        num_b: int,
        instance_key: int,
    ) -> torch.Tensor:
        return out_shape_fn(device_carrier, num_a, num_b)

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
    "ReluFwdOp",
    # Binary arithmetic
    "AddFwdOp",
    "SubFwdOp",
    "MulFwdOp",
    "DivFwdOp",
    "RemainderFwdOp",
    "PowFwdOp",
    "FloorDivideFwdOp",
    "LerpFwdOp",
    "MaximumFwdOp",
    "MinimumFwdOp",
    # Comparison (output bool)
    "EqFwdOp",
    "NeFwdOp",
    "GtFwdOp",
    "LtFwdOp",
    "GeFwdOp",
    "LeFwdOp",
    # Logical (output bool)
    "LogicalAndFwdOp",
    "LogicalOrFwdOp",
    # Bitwise
    "BitwiseAndFwdOp",
    "BitwiseOrFwdOp",
    "BitwiseXorFwdOp",
    # Fused gated
    "SiluAndMulFwdOp",
    "GeluAndMulFwdOp",
    "GeluTanhAndMulFwdOp",
    # --- math (17) ---
    "AbsFwdOp",
    "CeilFwdOp",
    "CosFwdOp",
    "ErfFwdOp",
    "ExpFwdOp",
    "Expm1FwdOp",
    "FloorFwdOp",
    "Log1pFwdOp",
    "LogFwdOp",
    "NegFwdOp",
    "ReciprocalFwdOp",
    "RoundFwdOp",
    "RsqrtFwdOp",
    "SignFwdOp",
    "SinFwdOp",
    "SqrtFwdOp",
    "TruncFwdOp",
    # --- activations (8) ---
    "GeluFwdOp",
    "HardsigmoidFwdOp",
    "HardswishFwdOp",
    "MishFwdOp",
    "SeluFwdOp",
    "SigmoidFwdOp",
    "SiluFwdOp",
    "TanhFwdOp",
    # --- logical (1) ---
    "LogicalNotFwdOp",
    # --- bitwise (1) ---
    "BitwiseNotFwdOp",
    # --- special predicates (3) ---
    "IsfiniteFwdOp",
    "IsinfFwdOp",
    "IsnanFwdOp",
    # --- independent (custom-signature, 11) ---
    "LeakyReluFwdOp",
    "EluFwdOp",
    "HardtanhFwdOp",
    "SoftplusFwdOp",
    "PreluFwdOp",
    "WhereFwdOp",
    "ClampFwdOp",
    "ClampScalarFwdOp",
    "ClampMinFwdOp",
    "ClampMaxFwdOp",
    "MaskedFillFwdOp",
    "MaskedFillScalarFwdOp",
    "NanToNumFwdOp",
    "AlibiFwdOp",
    "SinusoidalFwdOp",
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


def _apply_fp8_post_cast(result: torch.Tensor, kernel) -> torch.Tensor:
    """Apply fp8 output cast if the kernel requires it.

    For e5m2 dtypes the kernel produces fp16 output to preserve Inf/NaN;
    this helper performs the final non-saturating cast via PyTorch.
    """
    fp8_out = getattr(kernel, "_fp8_output_dtype", None)
    if fp8_out is not None:
        return result.to(fp8_out)
    return result


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_fp8(dtype: torch.dtype) -> bool:
    """Return True iff ``dtype`` is one of the supported fp8 dtypes."""
    return dtype in _FP8_DTYPES


def _fp8_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the compute dtype used to emulate fp8 elementwise fallbacks.

    PyTorch's CUDA backend does not implement ``clamp``/``maximum``/
    ``minimum``/``masked_fill_`` on Float8 tensors (raises NotImplementedError
    on ``clamp_cuda`` / ``max_elementwise_cuda`` / ``min_elementwise_cuda`` /
    ``masked_fill_``). Both e4m3fn (finite range ±448) and e5m2 (finite range
    ±57344) fit in fp16, so we upcast to fp16, run the op, and cast back. The
    final cast preserves Inf/NaN for e5m2 (PyTorch's fp16->e5m2 conversion is
    non-saturating) and saturates for e4m3fn (matching PyTorch's default
    fp16->e4m3fn behaviour).
    """
    if not _is_fp8(dtype):
        raise ValueError(f"_fp8_compute_dtype expects an fp8 dtype, got {dtype}")
    return torch.float16


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
        # Use _fp8_output_dtype (the final dtype after Op-layer post-cast)
        # rather than kernel.output_dtype (which is fp16 for e5m2).
        fp8_out = getattr(self.kernel, "_fp8_output_dtype", None)
        self.output_dtype = fp8_out or getattr(self.kernel, "output_dtype", dtype)
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

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this unary elementwise op instance.

        Mirrors the elementwise_unary_math manifest roofline:
        ``flops = N`` and ``bytes = N * input_elem_bytes + N * output_elem_bytes``.
        For ops whose output dtype matches the input (e.g. ``neg``, ``abs``),
        this collapses to ``2 * N * elem_bytes``; for ops with a smaller output
        dtype (e.g. ``isnan`` / ``isinf`` / ``isfinite`` / ``logical_not`` →
        bool), ``self.output_dtype.itemsize`` already captures the difference.
        """
        return self.N_total, int(self.total_memory)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Direct kernel call for use inside custom_op implementation."""
        orig_shape = input.shape
        flat = input.contiguous().reshape(-1)
        result = self.kernel(flat).reshape(orig_shape)
        # For e5m2: kernel produces fp16 to preserve Inf/NaN;
        # cast to e5m2 here using PyTorch's non-saturating conversion.
        return _apply_fp8_post_cast(result, self.kernel)

    def _validate_input(self, input: torch.Tensor) -> None:  # noqa: A002
        """Validate input tensor against the op's dtype / numel contract."""
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(
                f"Expected input.dtype {self.dtype}, got {input.dtype}"
            )
        if input.numel() != self.N_total:
            raise ValueError(
                f"Expected {self.N_total} elements, got {input.numel()}"
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        self._validate_input(input)
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)


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
        in_elem = self.dtype.itemsize
        fp8_out = getattr(self.kernel, "_fp8_output_dtype", None)
        out_elem = fp8_out.itemsize if fp8_out is not None else in_elem
        return (self.a_numel + self.b_numel) * in_elem + self.N_total * out_elem

    def _eager_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        result = self.kernel(
            a.contiguous().view(-1), b.contiguous().view(-1),
        ).reshape(self.out_shape)
        return _apply_fp8_post_cast(result, self.kernel)

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
        strategy: Kernel strategy override.
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
        self.M = M
        self.N = N
        self.dtype = dtype
        self.strategy = strategy
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            M, N, dtype, strategy=strategy, tune=tune,
        )
        # Register in global registry for torch.compile dispatch
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x (M*2N) + write y (M*N)."""
        in_elem = self.dtype.itemsize
        fp8_out = getattr(self.kernel, "_fp8_output_dtype", None)
        out_elem = fp8_out.itemsize if fp8_out is not None else in_elem
        return self.M * 2 * self.N * in_elem + self.M * self.N * out_elem

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        x = x.contiguous()
        result = self.kernel(x)
        return _apply_fp8_post_cast(result, self.kernel)

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


class ReluFwdOp(UnaryOp):
    """ReLU activation: y = max(x, 0)."""

    _op_name = "relu"
    kernel_cls = ReluFwdKernel


class AddFwdOp(BinaryOp):
    """Element-wise addition with broadcast: y = a + b."""

    _op_name = "add"
    kernel_cls = AddFwdKernel


class SubFwdOp(BinaryOp):
    """Element-wise subtraction with broadcast: y = a - b."""

    _op_name = "sub"
    kernel_cls = SubFwdKernel


class MulFwdOp(BinaryOp):
    """Element-wise multiplication with broadcast: y = a * b."""

    _op_name = "mul"
    kernel_cls = MulFwdKernel


class DivFwdOp(BinaryOp):
    """Element-wise division with broadcast: y = a / b."""

    _op_name = "div"
    kernel_cls = DivFwdKernel


class RemainderFwdOp(BinaryOp):
    """Element-wise remainder with broadcast: y = a % b."""

    _op_name = "remainder"
    kernel_cls = RemainderFwdKernel


class PowFwdOp(BinaryOp):
    """Element-wise power with broadcast: y = a ** b."""

    _op_name = "pow"
    kernel_cls = PowFwdKernel


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


class EqFwdOp(_BoolOutputBinaryOp):
    """Element-wise equality with broadcast: y = (a == b)."""

    _op_name = "eq"
    kernel_cls = EqFwdKernel


class NeFwdOp(_BoolOutputBinaryOp):
    """Element-wise not-equal with broadcast: y = (a != b)."""

    _op_name = "ne"
    kernel_cls = NeFwdKernel


class GtFwdOp(_BoolOutputBinaryOp):
    """Element-wise greater-than with broadcast: y = (a > b)."""

    _op_name = "gt"
    kernel_cls = GtFwdKernel


class LtFwdOp(_BoolOutputBinaryOp):
    """Element-wise less-than with broadcast: y = (a < b)."""

    _op_name = "lt"
    kernel_cls = LtFwdKernel


class GeFwdOp(_BoolOutputBinaryOp):
    """Element-wise greater-equal with broadcast: y = (a >= b)."""

    _op_name = "ge"
    kernel_cls = GeFwdKernel


class LeFwdOp(_BoolOutputBinaryOp):
    """Element-wise less-equal with broadcast: y = (a <= b)."""

    _op_name = "le"
    kernel_cls = LeFwdKernel


# ---------------------------------------------------------------------------
# Logical op subclasses (output bool)
# ---------------------------------------------------------------------------


class LogicalAndFwdOp(_BoolOutputBinaryOp):
    """Element-wise logical AND with broadcast using non-zero truthiness."""

    _op_name = "logical_and"
    kernel_cls = LogicalAndFwdKernel


class LogicalOrFwdOp(_BoolOutputBinaryOp):
    """Element-wise logical OR with broadcast using non-zero truthiness."""

    _op_name = "logical_or"
    kernel_cls = LogicalOrFwdKernel


# ---------------------------------------------------------------------------
# Bitwise op subclasses
# ---------------------------------------------------------------------------


class BitwiseAndFwdOp(BinaryOp):
    """Element-wise bitwise AND with broadcast: y = a & b."""

    _op_name = "bitwise_and"
    kernel_cls = BitwiseAndFwdKernel


class BitwiseOrFwdOp(BinaryOp):
    """Element-wise bitwise OR with broadcast: y = a | b."""

    _op_name = "bitwise_or"
    kernel_cls = BitwiseOrFwdKernel


class BitwiseXorFwdOp(BinaryOp):
    """Element-wise bitwise XOR with broadcast: y = a ^ b."""

    _op_name = "bitwise_xor"
    kernel_cls = BitwiseXorFwdKernel


# ---------------------------------------------------------------------------
# Fused gated op subclasses
# ---------------------------------------------------------------------------


class SiluAndMulFwdOp(FusedGatedOp):
    """SiLU-and-Mul: y = silu(gate) * value."""

    _op_name = "silu_and_mul"
    kernel_cls = SiluAndMulFwdKernel


class GeluAndMulFwdOp(FusedGatedOp):
    """GELU-and-Mul: y = gelu(gate) * value (exact GELU)."""

    _op_name = "gelu_and_mul"
    kernel_cls = GeluAndMulFwdKernel


class GeluTanhAndMulFwdOp(FusedGatedOp):
    """GELU-Tanh-and-Mul: y = gelu_tanh(gate) * value (tanh approximation)."""

    _op_name = "gelu_tanh_and_mul"
    kernel_cls = GeluTanhAndMulFwdKernel


# ---------------------------------------------------------------------------
# Unary math ops (17)
# ---------------------------------------------------------------------------


class ExpFwdOp(UnaryOp):
    """Element-wise exp(x)."""

    _op_name = "exp"
    kernel_cls = ExpFwdKernel


class LogFwdOp(UnaryOp):
    """Element-wise log(x)."""

    _op_name = "log"
    kernel_cls = LogFwdKernel


class SqrtFwdOp(UnaryOp):
    """Element-wise sqrt(x)."""

    _op_name = "sqrt"
    kernel_cls = SqrtFwdKernel


class RsqrtFwdOp(UnaryOp):
    """Element-wise 1/sqrt(x)."""

    _op_name = "rsqrt"
    kernel_cls = RsqrtFwdKernel


class AbsFwdOp(UnaryOp):
    """Element-wise |x|."""

    _op_name = "abs"
    kernel_cls = AbsFwdKernel


class NegFwdOp(UnaryOp):
    """Element-wise -x."""

    _op_name = "neg"
    kernel_cls = NegFwdKernel


class ReciprocalFwdOp(UnaryOp):
    """Element-wise 1/x."""

    _op_name = "reciprocal"
    kernel_cls = ReciprocalFwdKernel


class SignFwdOp(UnaryOp):
    """Element-wise sign(x): -1, 0, or +1."""

    _op_name = "sign"
    kernel_cls = SignFwdKernel


class SinFwdOp(UnaryOp):
    """Element-wise sin(x)."""

    _op_name = "sin"
    kernel_cls = SinFwdKernel


class CosFwdOp(UnaryOp):
    """Element-wise cos(x)."""

    _op_name = "cos"
    kernel_cls = CosFwdKernel


class FloorFwdOp(UnaryOp):
    """Element-wise floor(x)."""

    _op_name = "floor"
    kernel_cls = FloorFwdKernel


class CeilFwdOp(UnaryOp):
    """Element-wise ceil(x)."""

    _op_name = "ceil"
    kernel_cls = CeilFwdKernel


class RoundFwdOp(UnaryOp):
    """Element-wise round(x) to ``decimals`` decimal places.

    The underlying kernel performs banker's round-to-nearest-integer, matching
    ``torch.round`` for ``decimals=0``. Non-zero ``decimals`` is supported at
    the op layer via the standard decomposition:
    ``round(x, decimals=k) == round(x * 10**k) / 10**k``.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        strategy: Kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "round"
    kernel_cls = RoundFwdKernel

    def forward(  # noqa: A002
        self, input: torch.Tensor, decimals: int = 0,
    ) -> torch.Tensor:
        if decimals == 0:
            return super().forward(input)
        # Non-zero decimals path still owes the same input contract as the
        # ``decimals=0`` fast path (UnaryOp.forward). Run the shared validator
        # before any fp32 arithmetic so a CPU tensor / wrong dtype / wrong
        # numel cannot silently bypass the checks.
        self._validate_input(input)
        # Non-zero decimals: scale, round-to-integer, unscale at the op layer.
        # The whole decimals path runs in fp32 so that low-precision inputs
        # (fp16/bf16) do not overflow when multiplied by ``10**decimals`` —
        # e.g. ``100 * 10**4 = 1e6`` exceeds fp16 max (~65504). Down-casting
        # only happens once at the end, matching ``torch.round(x.float(),
        # decimals=k).to(orig_dtype)``. ``torch.round`` is used directly here
        # because the kernel signature is fixed to ``self.dtype``; this is op-
        # layer composition and the manifest's ``kernel_map`` continues to
        # describe the round-to-nearest-integer kernel that handles the
        # ``decimals=0`` fast path above.
        scale = 10.0 ** decimals
        scaled_fp32 = input.float() * scale
        rounded_fp32 = torch.round(scaled_fp32)
        return (rounded_fp32 / scale).to(self.dtype)


class TruncFwdOp(UnaryOp):
    """Element-wise trunc(x)."""

    _op_name = "trunc"
    kernel_cls = TruncFwdKernel


class ErfFwdOp(UnaryOp):
    """Element-wise erf(x)."""

    _op_name = "erf"
    kernel_cls = ErfFwdKernel


class Log1pFwdOp(UnaryOp):
    """Element-wise log(1 + x)."""

    _op_name = "log1p"
    kernel_cls = Log1pFwdKernel


class Expm1FwdOp(UnaryOp):
    """Element-wise exp(x) - 1."""

    _op_name = "expm1"
    kernel_cls = Expm1FwdKernel


# ---------------------------------------------------------------------------
# Activation ops (8)
# ---------------------------------------------------------------------------


class GeluFwdOp(UnaryOp):
    """Element-wise GELU using the standard erf formulation."""

    _op_name = "gelu"
    kernel_cls = GeluFwdKernel


class SiluFwdOp(UnaryOp):
    """Element-wise SiLU (Swish): x * sigmoid(x)."""

    _op_name = "silu"
    kernel_cls = SiluFwdKernel


class SigmoidFwdOp(UnaryOp):
    """Element-wise sigmoid(x)."""

    _op_name = "sigmoid"
    kernel_cls = SigmoidFwdKernel


class TanhFwdOp(UnaryOp):
    """Element-wise tanh(x)."""

    _op_name = "tanh"
    kernel_cls = TanhFwdKernel


class HardswishFwdOp(UnaryOp):
    """Element-wise HardSwish: x * clamp(x + 3, 0, 6) / 6."""

    _op_name = "hardswish"
    kernel_cls = HardswishFwdKernel


class HardsigmoidFwdOp(UnaryOp):
    """Element-wise HardSigmoid: clamp(x + 3, 0, 6) / 6."""

    _op_name = "hardsigmoid"
    kernel_cls = HardsigmoidFwdKernel


class MishFwdOp(UnaryOp):
    """Element-wise Mish: x * tanh(softplus(x))."""

    _op_name = "mish"
    kernel_cls = MishFwdKernel


class SeluFwdOp(UnaryOp):
    """Element-wise SELU."""

    _op_name = "selu"
    kernel_cls = SeluFwdKernel


# ---------------------------------------------------------------------------
# Logical op (1)
# ---------------------------------------------------------------------------


class LogicalNotFwdOp(UnaryOp):
    """Element-wise logical NOT with bool output."""

    _op_name = "logical_not"
    kernel_cls = LogicalNotFwdKernel


# ---------------------------------------------------------------------------
# Bitwise op (1)
# ---------------------------------------------------------------------------


class BitwiseNotFwdOp(UnaryOp):
    """Element-wise bitwise NOT (~x) for bool/integer inputs."""

    _op_name = "bitwise_not"
    kernel_cls = BitwiseNotFwdKernel


# ---------------------------------------------------------------------------
# Special predicate ops (3)
# ---------------------------------------------------------------------------


class IsnanFwdOp(UnaryOp):
    """Element-wise isnan with bool output."""

    _op_name = "isnan"
    kernel_cls = IsnanFwdKernel


class IsinfFwdOp(UnaryOp):
    """Element-wise isinf with bool output."""

    _op_name = "isinf"
    kernel_cls = IsinfFwdKernel


class IsfiniteFwdOp(UnaryOp):
    """Element-wise isfinite with bool output."""

    _op_name = "isfinite"
    kernel_cls = IsfiniteFwdKernel


# ---------------------------------------------------------------------------
# Independent (custom-signature) op classes (11)
# ---------------------------------------------------------------------------


class LeakyReluFwdOp(Op):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        negative_slope: Slope for negative inputs (default 0.01).
    """

    _op_name = "leaky_relu"
    _wrapped = None

    def __init__(self, N_total: int, dtype: torch.dtype, negative_slope: float = 0.01):
        _validate_scalar_param_repr("negative_slope", negative_slope, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.negative_slope = negative_slope
        self.kernel = LeakyReluFwdKernel(N_total, dtype, negative_slope=negative_slope)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"leaky_relu": LeakyReluFwdKernel}

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        result = self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {x.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self._instance_key)
        return self._eager_forward(x)


class EluFwdOp(Op):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        alpha: Scale for the negative part (default 1.0).
    """

    _op_name = "elu"
    _wrapped = None

    def __init__(self, N_total: int, dtype: torch.dtype, alpha: float = 1.0):
        _validate_scalar_param_repr("alpha", alpha, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.alpha = alpha
        self.kernel = EluFwdKernel(N_total, dtype, alpha=alpha)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"elu": EluFwdKernel}

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        result = self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {x.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self._instance_key)
        return self._eager_forward(x)


class HardtanhFwdOp(Op):
    """Hardtanh: y = clamp(x, min_val, max_val).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        min_val: Lower bound (default -1.0).
        max_val: Upper bound (default 1.0).
    """

    _op_name = "hardtanh"
    _wrapped = None

    def __init__(self, N_total: int, dtype: torch.dtype,
                 min_val: float = -1.0, max_val: float = 1.0):
        _validate_scalar_param_repr("min_val", min_val, dtype, self._op_name)
        _validate_scalar_param_repr("max_val", max_val, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.kernel = HardtanhFwdKernel(N_total, dtype, min_val=min_val, max_val=max_val)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"hardtanh": HardtanhFwdKernel}

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        result = self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {x.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self._instance_key)
        return self._eager_forward(x)


class SoftplusFwdOp(Op):
    """Softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        beta: Scaling factor (default 1.0).
        threshold: Linear regime threshold (default 20.0).
    """

    _op_name = "softplus"
    _wrapped = None

    def __init__(self, N_total: int, dtype: torch.dtype,
                 beta: float = 1.0, threshold: float = 20.0):
        _validate_scalar_param_repr("beta", beta, dtype, self._op_name)
        _validate_scalar_param_repr("threshold", threshold, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.beta = beta
        self.threshold = threshold
        self.kernel = SoftplusFwdKernel(N_total, dtype, beta=beta, threshold=threshold)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"softplus": SoftplusFwdKernel}

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        result = self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {x.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self._instance_key)
        return self._eager_forward(x)


class PreluFwdOp(Op):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel dimension follows PyTorch convention: dimension 1 for inputs
    with ndim >= 2, dimension 0 for 1-D inputs.

    Args:
        shape: Shape of the input tensor (must have a channel dimension).
        dtype: Torch dtype.
        num_channels: Number of channels (weight length).
    """

    _op_name = "prelu"
    _wrapped = None

    def __init__(self, shape: tuple, dtype: torch.dtype, num_channels: int):
        self.shape = shape
        self.dtype = dtype
        self.num_channels = num_channels
        N_total = prod(shape)
        self.N_total = N_total
        # PyTorch PReLU: channel dim is 1 for ndim>=2, else 0
        inner_size = (prod(shape[2:]) if len(shape) > 2 else 1) if len(shape) >= 2 else 1
        self.inner_size = inner_size
        self.kernel = PreluFwdKernel(N_total, num_channels, inner_size, dtype)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"prelu": PreluFwdKernel}

    def _eager_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        result = self.kernel(
            x.contiguous().reshape(-1), weight.contiguous().reshape(-1),
        ).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {x.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, weight, self._instance_key)
        return self._eager_forward(x, weight)


class WhereFwdOp(Op):
    """Where: out = condition ? input : other (with full PyTorch broadcasting).

    Conforms to ``torch.where(condition, input, other)``: ``condition`` is a
    bool tensor and ``input`` / ``other`` may broadcast with each other and
    with ``condition`` to produce the output. The Op layer expands all
    three inputs to the broadcast shape and dispatches the existing flat
    where kernel on ``N_total = product(broadcast_shape)`` elements.

    Args:
        condition: Shape of the condition tensor (any shape broadcastable
            with ``input`` / ``other``).
        input: Shape of the value-when-true tensor.
        other: Shape of the value-when-false tensor.
        dtype: Torch dtype for ``input`` / ``other``.
    """

    _op_name = "where"
    _wrapped = None

    def __init__(
        self,
        condition: tuple,
        input: tuple,  # noqa: A002 — manifest-aligned PyTorch param name
        other: tuple,
        dtype: torch.dtype,
    ):
        self.condition_shape = tuple(condition)
        self.input_shape = tuple(input)
        self.other_shape = tuple(other)
        self.dtype = dtype
        self.out_shape = tuple(
            torch.broadcast_shapes(self.condition_shape, self.input_shape, self.other_shape)
        )
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.kernel = WhereFwdKernel(self.N_total, dtype)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"where": WhereFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        out_shape = self.out_shape if self.out_shape else (1,)
        cond_b = condition if condition.dtype == torch.bool else condition.bool()
        cond_flat = self._expand_flat(cond_b, out_shape).view(torch.uint8)
        x_flat = self._expand_flat(input, out_shape)
        y_flat = self._expand_flat(other, out_shape)
        result = self.kernel(cond_flat, x_flat, y_flat).view(out_shape if self.out_shape else ())
        return result

    def forward(
        self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        if not (condition.is_cuda and input.is_cuda and other.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        if condition.dtype != torch.bool:
            raise ValueError(
                f"Expected condition.dtype torch.bool, got {condition.dtype}"
            )
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if other.dtype != self.dtype:
            raise ValueError(f"Expected other.dtype {self.dtype}, got {other.dtype}")
        if tuple(condition.shape) != self.condition_shape:
            raise ValueError(
                f"Expected condition.shape {self.condition_shape}, got {tuple(condition.shape)}"
            )
        if tuple(input.shape) != self.input_shape:
            raise ValueError(
                f"Expected input.shape {self.input_shape}, got {tuple(input.shape)}"
            )
        if tuple(other.shape) != self.other_shape:
            raise ValueError(
                f"Expected other.shape {self.other_shape}, got {tuple(other.shape)}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(condition, input, other, self._instance_key)
        return self._eager_forward(condition, input, other)


class _ClampTensorBase(Op):
    """Shared infrastructure for Tensor-bound clamp variants (broadcasting)."""

    _wrapped = None

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)


class ClampFwdOp(_ClampTensorBase):
    """Clamp with Tensor lower and/or upper bounds (broadcasting).

    Conforms to ``torch.clamp(input, min, max)`` where ``min`` and ``max``
    are each either a Tensor or ``None``. At least one of the two bounds
    must be a Tensor. All Tensor operands broadcast together. The
    primary spec entry in ``tileops/manifest/`` covers the both-Tensor
    form; the mixed Tensor/``None`` cases are runtime-equivalent to
    ``ClampMinFwdOp`` / ``ClampMaxFwdOp`` and are accepted here so callers
    can mirror PyTorch's ``torch.clamp`` API directly.

    Args:
        input: Shape of the input tensor.
        min: Shape of the lower-bound tensor, or ``None`` for no lower bound.
        max: Shape of the upper-bound tensor, or ``None`` for no upper bound.
        dtype: Torch dtype for all operands.

    Raises:
        ValueError: If both ``min`` and ``max`` are ``None``.
    """

    _op_name = "clamp"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002 — manifest-aligned PyTorch param name
        min: Optional[tuple] = None,  # noqa: A002 — manifest-aligned PyTorch param name
        max: Optional[tuple] = None,  # noqa: A002 — manifest-aligned PyTorch param name
        dtype: torch.dtype = torch.float32,
    ):
        if min is None and max is None:
            raise ValueError(
                "ClampFwdOp requires at least one of `min` or `max` to be a "
                "Tensor shape; both None is not a valid clamp."
            )
        self.input_shape = tuple(input)
        self.min_shape = None if min is None else tuple(min)
        self.max_shape = None if max is None else tuple(max)
        self.dtype = dtype
        broadcast_args = [self.input_shape]
        if self.min_shape is not None:
            broadcast_args.append(self.min_shape)
        if self.max_shape is not None:
            broadcast_args.append(self.max_shape)
        self.out_shape = tuple(torch.broadcast_shapes(*broadcast_args))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.kernel = ClampTensorFwdKernel(
            self.N_total, dtype,
            has_min=self.min_shape is not None,
            has_max=self.max_shape is not None,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"clamp_tensor": ClampTensorFwdKernel}

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002
        min: Optional[torch.Tensor] = None,  # noqa: A002
        max: Optional[torch.Tensor] = None,  # noqa: A002
    ) -> torch.Tensor:
        # Broadcast all operands to ``out_shape`` and dispatch the
        # TileLang Tensor-bound clamp kernel. The kernel branches on
        # ``has_min`` / ``has_max`` at build time, so this single Op
        # class also covers the mixed Tensor/None cases.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        lo_flat = None if min is None else self._expand_flat(min, out_shape)
        hi_flat = None if max is None else self._expand_flat(max, out_shape)
        result = self.kernel(x_flat, lo_flat, hi_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        min: Optional[torch.Tensor] = None,  # noqa: A002
        max: Optional[torch.Tensor] = None,  # noqa: A002
    ) -> torch.Tensor:
        # Validate that the runtime None / Tensor pattern matches what
        # __init__ was configured for — the broadcast shape and the
        # presence of each bound is baked in at construction.
        if (min is None) != (self.min_shape is None):
            raise ValueError(
                f"min was {'None' if self.min_shape is None else 'a Tensor shape'} at "
                f"__init__ but {'None' if min is None else 'a Tensor'} at forward()"
            )
        if (max is None) != (self.max_shape is None):
            raise ValueError(
                f"max was {'None' if self.max_shape is None else 'a Tensor shape'} at "
                f"__init__ but {'None' if max is None else 'a Tensor'} at forward()"
            )
        tensors = [("input", input, self.input_shape)]
        if min is not None:
            tensors.append(("min", min, self.min_shape))
        if max is not None:
            tensors.append(("max", max, self.max_shape))
        for _, t, _ in tensors:
            if not t.is_cuda:
                raise ValueError("Inputs must be CUDA tensors")
        for name, t, expected in tensors:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, min, max, self._instance_key)
        return self._eager_forward(input, min, max)


class ClampMinFwdOp(_ClampTensorBase):
    """Single-bound Tensor lower clamp (``torch.clamp_min``).

    Args:
        input: Shape of the input tensor.
        min: Shape of the lower-bound tensor.
        dtype: Torch dtype.
    """

    _op_name = "clamp_min"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        min: tuple,    # noqa: A002
        dtype: torch.dtype,
    ):
        self.input_shape = tuple(input)
        self.min_shape = tuple(min)
        self.dtype = dtype
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.min_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.kernel = ClampTensorFwdKernel(
            self.N_total, dtype, has_min=True, has_max=False,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"clamp_tensor": ClampTensorFwdKernel}

    def _eager_forward(
        self, input: torch.Tensor, min: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        # Broadcast input/min to out_shape and dispatch the TileLang
        # min-only Tensor-bound clamp kernel.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        lo_flat = self._expand_flat(min, out_shape)
        result = self.kernel(x_flat, lo_flat, None)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self, input: torch.Tensor, min: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        if not (input.is_cuda and min.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("min", min, self.min_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, min, self._instance_key)
        return self._eager_forward(input, min)


class ClampMaxFwdOp(_ClampTensorBase):
    """Single-bound Tensor upper clamp (``torch.clamp_max``).

    Args:
        input: Shape of the input tensor.
        max: Shape of the upper-bound tensor.
        dtype: Torch dtype.
    """

    _op_name = "clamp_max"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        max: tuple,    # noqa: A002
        dtype: torch.dtype,
    ):
        self.input_shape = tuple(input)
        self.max_shape = tuple(max)
        self.dtype = dtype
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.max_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.kernel = ClampTensorFwdKernel(
            self.N_total, dtype, has_min=False, has_max=True,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"clamp_tensor": ClampTensorFwdKernel}

    def _eager_forward(
        self, input: torch.Tensor, max: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        # Broadcast input/max to out_shape and dispatch the TileLang
        # max-only Tensor-bound clamp kernel.
        out_shape = self.out_shape if self.out_shape else (1,)
        x_flat = self._expand_flat(input, out_shape)
        hi_flat = self._expand_flat(max, out_shape)
        result = self.kernel(x_flat, None, hi_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self, input: torch.Tensor, max: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        if not (input.is_cuda and max.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("max", max, self.max_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, max, self._instance_key)
        return self._eager_forward(input, max)


class ClampScalarFwdOp(Op):
    """Scalar-bound clamp (``torch.clamp(input, min: Number|None, max: Number|None)``).

    Args:
        input: Shape of the input tensor.
        min: Lower bound (Number or None).
        max: Upper bound (Number or None).
        dtype: Torch dtype.
    """

    _op_name = "clamp"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        min: Optional[float] = None,  # noqa: A002
        max: Optional[float] = None,  # noqa: A002
        dtype: torch.dtype = torch.float32,
    ):
        if min is None and max is None:
            raise ValueError(
                "ClampScalarFwdOp requires at least one of `min` or `max` to be a "
                "Number; both None is not a valid clamp."
            )
        if min is not None:
            _validate_scalar_param_repr("min", min, dtype, self._op_name)
        if max is not None:
            _validate_scalar_param_repr("max", max, dtype, self._op_name)
        self.input_shape = tuple(input)
        self.N_total = prod(self.input_shape) if self.input_shape else 1
        self.dtype = dtype
        self.min = min
        self.max = max
        # Backwards-compat aliases for legacy callers.
        self.min_val = min
        self.max_val = max
        self.kernel = ClampFwdKernel(self.N_total, dtype, min_val=min, max_val=max)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"clamp": ClampFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        orig_shape = input.shape
        result = self.kernel(input.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if tuple(input.shape) != self.input_shape:
            raise ValueError(
                f"Expected input.shape {self.input_shape}, got {tuple(input.shape)}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)


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
    """

    _op_name = "masked_fill"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        mask: tuple,
        value: tuple,
        dtype: torch.dtype,
    ):
        if tuple(value) != ():
            raise ValueError(
                f"MaskedFillFwdOp requires a 0-dim value Tensor; got shape {tuple(value)}"
            )
        self.input_shape = tuple(input)
        self.mask_shape = tuple(mask)
        self.value_shape = tuple(value)
        self.dtype = dtype
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.mask_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.kernel = MaskedFillTensorValueFwdKernel(self.N_total, dtype)
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
        mask_b = mask if mask.dtype == torch.bool else mask.bool()
        mask_flat = self._expand_flat(mask_b, out_shape).view(torch.uint8)
        value_1d = value.contiguous().view(1)
        result = self.kernel(x_flat, mask_flat, value_1d)
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

    Args:
        input: Shape of the input tensor.
        mask: Shape of the mask tensor (bool).
        value: Scalar fill value.
        dtype: Torch dtype.
    """

    _op_name = "masked_fill"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        mask: tuple,
        value: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        _validate_scalar_param_repr("value", value, dtype, self._op_name)
        self.input_shape = tuple(input)
        self.mask_shape = tuple(mask)
        self.dtype = dtype
        self.value = value
        # Backwards-compat alias.
        self.fill_value = value
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.mask_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        # The kernel is always built on the broadcast (output) flat size.
        # When input/mask already match out_shape, this is a no-op expand;
        # otherwise the Op layer broadcasts both before dispatch.
        self._needs_broadcast = (
            self.input_shape != self.out_shape or self.mask_shape != self.out_shape
        )
        self.kernel = MaskedFillFwdKernel(self.N_total, dtype, value)
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
        mask_b = mask if mask.dtype == torch.bool else mask.bool()
        mask_flat = self._expand_flat(mask_b, out_shape).view(torch.uint8)
        result = self.kernel(x_flat, mask_flat).view(self.out_shape if self.out_shape else ())
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


class NanToNumFwdOp(Op):
    """NanToNum: replace NaN, +Inf, -Inf with specified values.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        nan_val: Replacement for NaN (default 0.0).
        posinf_val: Replacement for +Inf (default 1e4).
        neginf_val: Replacement for -Inf (default -1e4).
    """

    _op_name = "nan_to_num"
    _wrapped = None

    def __init__(self, N_total: int, dtype: torch.dtype,
                 nan_val: float = 0.0, posinf_val: float = 1e4, neginf_val: float = -1e4):
        _validate_scalar_param_repr("nan_val", nan_val, dtype, self._op_name)
        _validate_scalar_param_repr("posinf_val", posinf_val, dtype, self._op_name)
        _validate_scalar_param_repr("neginf_val", neginf_val, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.nan_val = nan_val
        self.posinf_val = posinf_val
        self.neginf_val = neginf_val
        self.kernel = NanToNumFwdKernel(
            N_total, dtype, nan_val=nan_val, posinf_val=posinf_val, neginf_val=neginf_val,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"nan_to_num": NanToNumFwdKernel}

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        result = self.kernel(x.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {x.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self._instance_key)
        return self._eager_forward(x)


class AlibiFwdOp(Op):
    """ALiBi position encoding: bias[h, i, j] = -slope_h * |i - j|.

    Generates the full (num_heads, seq_len, seq_len) bias tensor.

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        dtype: Torch dtype.
    """

    _op_name = "alibi"
    _wrapped = None

    def __init__(self, seq_len: int, num_heads: int, dtype: torch.dtype):
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dtype = dtype
        self.kernel = AlibiFwdKernel(seq_len, num_heads, dtype)
        # Scalar tensor used as device/dtype carrier for torch.compile tracing
        self._device_carrier = torch.empty((), dtype=dtype, device="cuda")
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"alibi": AlibiFwdKernel}

    def _eager_forward(self) -> torch.Tensor:
        out = self.kernel()
        result = out.reshape(self.num_heads, self.seq_len, self.seq_len)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self) -> torch.Tensor:
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(
                self._device_carrier,
                self.num_heads, self.seq_len,
                self._instance_key,
            )
        return self._eager_forward()


class SinusoidalFwdOp(Op):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Generates the full (seq_len, d_model) encoding tensor.

    Args:
        seq_len: Sequence length.
        d_model: Model dimension.
        dtype: Torch dtype.
    """

    _op_name = "sinusoidal"
    _wrapped = None

    def __init__(self, seq_len: int, d_model: int, dtype: torch.dtype):
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        self.kernel = SinusoidalFwdKernel(seq_len, d_model, dtype)
        # Scalar tensor used as device/dtype carrier for torch.compile tracing
        self._device_carrier = torch.empty((), dtype=dtype, device="cuda")
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"sinusoidal": SinusoidalFwdKernel}

    def _eager_forward(self) -> torch.Tensor:
        out = self.kernel()
        result = out.reshape(self.seq_len, self.d_model)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self) -> torch.Tensor:
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(
                self._device_carrier,
                self.seq_len, self.d_model,
                self._instance_key,
            )
        return self._eager_forward()


# ---------------------------------------------------------------------------
# torch.compile registration for all 66 concrete ops
# ---------------------------------------------------------------------------

# --- Unary ops: float-preserving output (1 + 17 + 8 + 1 = 27 ops) ---
for _cls in [
    ReluFwdOp,
    # math (17)
    ExpFwdOp, LogFwdOp, SqrtFwdOp, RsqrtFwdOp, AbsFwdOp, NegFwdOp, ReciprocalFwdOp, SignFwdOp,
    SinFwdOp, CosFwdOp, FloorFwdOp, CeilFwdOp, RoundFwdOp, TruncFwdOp, ErfFwdOp, Log1pFwdOp, Expm1FwdOp,
    # activations (8)
    GeluFwdOp, SiluFwdOp, SigmoidFwdOp, TanhFwdOp, HardswishFwdOp, HardsigmoidFwdOp, MishFwdOp, SeluFwdOp,
    # bitwise (1) -- output same dtype as input
    BitwiseNotFwdOp,
]:
    _register_unary_custom_op(_cls)

# --- Unary ops: bool output (4 ops) ---
for _cls in [LogicalNotFwdOp, IsnanFwdOp, IsinfFwdOp, IsfiniteFwdOp]:
    _register_unary_custom_op(_cls, output_dtype_override=torch.bool)

# --- Binary ops: same-dtype output (10 + 3 = 13 ops) ---
for _cls in [
    # arithmetic (10)
    AddFwdOp, SubFwdOp, MulFwdOp, DivFwdOp, RemainderFwdOp, PowFwdOp, FloorDivideFwdOp,
    LerpFwdOp, MaximumFwdOp, MinimumFwdOp,
    # bitwise (3)
    BitwiseAndFwdOp, BitwiseOrFwdOp, BitwiseXorFwdOp,
]:
    _register_binary_custom_op(_cls)

# --- Binary ops: bool output (comparison 6 + logical 2 = 8 ops) ---
for _cls in [
    EqFwdOp, NeFwdOp, GtFwdOp, LtFwdOp, GeFwdOp, LeFwdOp,
    LogicalAndFwdOp, LogicalOrFwdOp,
]:
    _register_binary_custom_op(_cls, output_bool=True)

# --- Fused gated ops (3 ops) ---
for _cls in [SiluAndMulFwdOp, GeluAndMulFwdOp, GeluTanhAndMulFwdOp]:
    _register_fused_gated_custom_op(_cls)

# --- Independent unary-like ops (6 ops: x -> y with baked params) ---
# ClampScalarFwdOp is the scalar-bound clamp (single-tensor input + min/max
# baked into __init__). The Tensor-bound ClampFwdOp / ClampMinFwdOp /
# ClampMaxFwdOp variants register their own multi-input custom_ops below.
for _cls in [
    LeakyReluFwdOp, EluFwdOp, HardtanhFwdOp, SoftplusFwdOp, ClampScalarFwdOp,
    NanToNumFwdOp,
]:
    _register_unary_custom_op(_cls)

# --- PReLU op (1 op: x, weight -> y) ---
_register_prelu_custom_op(PreluFwdOp)

# --- Tensor-bound clamp variants (3 ops: multi-tensor inputs -> out) ---
# Registered under distinct custom_op namespaces from ClampScalarFwdOp:
# ``top::elementwise_clamp_tensor`` (Optional Tensor min/max),
# ``top::elementwise_clamp_min`` and ``top::elementwise_clamp_max`` for the
# single-bound variants. register_fake is broadcast-aware so
# torch.compile(fullgraph=True) traces correctly for both same-shape and
# broadcasting inputs.
_register_clamp_tensor_custom_op(ClampFwdOp)
_register_clamp_min_custom_op(ClampMinFwdOp)
_register_clamp_max_custom_op(ClampMaxFwdOp)

# --- MaskedFill variants (input, mask[, value] -> out) ---
# Both register broadcast-aware fake functions so torch.compile(fullgraph=True)
# works for same-shape and broadcasting inputs. The Tensor-value variant is
# registered under a distinct ``_tensor_value`` namespace to avoid colliding
# with the scalar variant's ``top::elementwise_masked_fill``.
_register_masked_fill_custom_op(MaskedFillScalarFwdOp)
_register_masked_fill_tensor_value_custom_op(MaskedFillFwdOp)

# --- Where op (1 op: cond, x, y -> out) ---
# The fake function is broadcast-aware so torch.compile(fullgraph=True)
# traces correctly for both same-shape and broadcasting inputs.
_register_where_custom_op(WhereFwdOp)

# --- Generative ops (2 ops: no tensor input -> out) ---
_register_generative_custom_op(
    AlibiFwdOp,
    out_shape_fn=lambda carrier, num_heads, seq_len: carrier.new_empty(
        (num_heads, seq_len, seq_len),
    ),
)
_register_generative_custom_op(
    SinusoidalFwdOp,
    out_shape_fn=lambda carrier, seq_len, d_model: carrier.new_empty(
        (seq_len, d_model),
    ),
)

# Clean up loop variable
del _cls
