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
from typing import Callable, Dict, List, Optional

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

# Manifest-declared dtypes for binary arithmetic / comparison ops that
# the underlying float-only TileLang kernels cannot compile. The op
# layer routes these through a torch eager fallback (see
# ``BinaryOp._TORCH_FALLBACK_DTYPES``) so the manifest contract is
# honored end-to-end. ``_BINARY_FALLBACK_INT_DTYPES`` includes
# ``torch.bool`` for ops where the manifest signature accepts bool
# input (e.g. ``torch.add``); the ``_no_bool`` variant excludes it for
# ``torch.sub``, which mirrors PyTorch's rejection of bool arithmetic
# for sub.
_BINARY_FALLBACK_INT_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_BINARY_FALLBACK_INT_DTYPES_NO_BOOL = _BINARY_FALLBACK_INT_DTYPES[1:]


def _effective_scalar_kernel_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the dtype used when scalar literals are materialized in kernels."""
    return _FP8_NONSAT_OUTPUT_DTYPES.get(dtype, dtype)


def _validate_scalar_param_repr(
    param_name: str, value: float, dtype: torch.dtype, op_name: str,
) -> None:
    """Reject scalar params that cannot be represented in the user dtype.

    Validation targets the *user-facing* ``dtype`` rather than the
    intermediate ``_effective_scalar_kernel_dtype(dtype)``.  For fp8
    dtypes the kernel runs in fp16 to preserve Inf/NaN, but a value that
    only fits in fp16 would surface as ``+/-Inf`` after the final fp8
    post-cast. Validating against the user dtype keeps explicit
    replacements finite end-to-end.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{op_name} expected scalar {param_name} to be int/float, got {type(value)}")

    finfo = torch.finfo(dtype)
    value_f64 = float(value)
    if math.isnan(value_f64):
        return
    if math.isinf(value_f64):
        raise ValueError(
            f"{op_name} received {param_name}={value!r}, but {param_name} must be finite and "
            f"representable in dtype {dtype}"
        )
    if not (finfo.min <= value_f64 <= finfo.max):
        raise ValueError(
            f"{op_name} received {param_name}={value!r}, which is not representable in "
            f"dtype {dtype} (valid finite range: "
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
    # Per-element FLOP count, matching the manifest's ``roofline.flops``
    # coefficient on ``N``. Subclasses override when the op is more than one
    # arithmetic op per element (e.g. ``sigmoid`` ≈ 4, ``tanh`` ≈ 5). The
    # base class default of 1 covers the common ``flops: "N"`` entries.
    FLOPS_PER_ELEM: int = 1

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
        ``flops = FLOPS_PER_ELEM * N`` and
        ``bytes = N * input_elem_bytes + N * output_elem_bytes``. Subclasses
        whose manifest entry uses a higher coefficient (e.g. ``sigmoid`` →
        ``4 * N``, ``tanh`` → ``5 * N``) override ``FLOPS_PER_ELEM``. For ops
        whose output dtype matches the input (e.g. ``neg``, ``abs``), bytes
        collapse to ``2 * N * elem_bytes``; for ops with a smaller output
        dtype (e.g. ``isnan`` / ``isinf`` / ``isfinite`` / ``logical_not`` →
        bool), ``self.output_dtype.itemsize`` already captures the difference.
        """
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)

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
    # Subclasses set this to a torch reference (e.g. ``torch.add``) when the
    # manifest dtype union extends beyond the underlying kernel's runtime
    # support. ``None`` means "kernel-only"; passing a dtype in
    # ``_TORCH_FALLBACK_DTYPES`` then raises.
    _torch_fallback_fn: Optional[Callable] = None
    # Manifest-declared dtypes that the kernel cannot compile and that
    # must route through ``_torch_fallback_fn`` instead. The subclass
    # owns this list because the kernel's ``SUPPORTED_DTYPES`` is
    # frequently ``None`` (i.e. "no static check") even when the
    # underlying TileLang codegen still rejects bool / integer
    # arithmetic. Empty tuple means "every manifest dtype goes through
    # the kernel path".
    _TORCH_FALLBACK_DTYPES: tuple = ()
    # Default for subclasses (e.g. ``LerpFwdOp``) that override
    # ``__init__`` and do not exercise the dtype-fallback branch. Such
    # subclasses leave ``_torch_fallback_fn = None`` and rely on the
    # base class dtype check inline; the instance-level attribute may
    # never be assigned, so ``forward`` consults the class-level default.
    _use_torch_fallback = False

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        # Two-stage dtype gate:
        #   1. The kernel may declare its own ``SUPPORTED_DTYPES``; honor
        #      that as the kernel-path filter.
        #   2. The op subclass owns ``_TORCH_FALLBACK_DTYPES`` -- dtypes
        #      that are inside the manifest contract but outside what the
        #      kernel can compile (e.g. bool / integer for the
        #      float-only Add / Mul / Maximum kernels). For those, route
        #      through ``_torch_fallback_fn`` instead.
        kernel_supported = self.kernel_cls.SUPPORTED_DTYPES
        is_fallback_dtype = (
            self._torch_fallback_fn is not None
            and dtype in self._TORCH_FALLBACK_DTYPES
        )
        kernel_supports = (
            kernel_supported is None or dtype in kernel_supported
        )
        if not kernel_supports and not is_fallback_dtype:
            names = ", ".join(str(dt) for dt in kernel_supported)
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
        # When the manifest dtype is outside kernel support, take the torch
        # eager fallback path: skip kernel construction (it would raise) and
        # mark the instance so forward() routes through ``_torch_fallback_fn``.
        self._use_torch_fallback = is_fallback_dtype
        if self._use_torch_fallback:
            self.kernel_map = self.default_kernel_map
            self.kernel = None
        else:
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

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        other: torch.Tensor,
    ) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        if self._use_torch_fallback:
            # Manifest-declared dtype outside kernel support: defer to the
            # registered torch reference so the manifest contract is honored
            # without a kernel rewrite. Broadcasting is handled by torch.
            return type(self)._torch_fallback_fn(input, other)
        result = self.kernel(
            input.contiguous().view(-1), other.contiguous().view(-1),
        ).reshape(self.out_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def _validate_binary_inputs(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        other: torch.Tensor,
    ) -> None:
        """Validate device / dtype / numel against the op's declared contract.

        Shared by the kernel path and any subclass torch fallback (e.g.
        ``torch.add(..., alpha=2)``) so that error types and messages stay
        consistent regardless of which branch ``forward`` takes.
        """
        if not input.is_cuda or not other.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if other.dtype != self.dtype:
            raise ValueError(f"Expected other.dtype {self.dtype}, got {other.dtype}")
        if input.numel() != self.a_numel:
            raise ValueError(
                f"Expected input to have {self.a_numel} elements, got {input.numel()}"
            )
        if other.numel() != self.b_numel:
            raise ValueError(
                f"Expected other to have {self.b_numel} elements, got {other.numel()}"
            )

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        other: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_binary_inputs(input, other)
        wrapped = type(self)._wrapped
        # ``register_fake`` declares the kernel output dtype, so the
        # ``_wrapped`` path cannot model the torch fallback faithfully.
        # Route fallback dtypes through eager so the result dtype matches
        # the manifest contract.
        if wrapped is not None and not self._use_torch_fallback:
            return wrapped(input, other, self._out_shape_list, self._instance_key)
        return self._eager_forward(input, other)


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


class _InplaceMixin:
    """Implement out-of-kernel ``inplace=True`` for activation ops.

    Used by activation ops whose manifest entry declares ``inplace:
    bool = False`` (e.g. ReLU, SiLU, HardSwish, HardSigmoid, Mish, SELU,
    LeakyReLU, ELU, Hardtanh). Activations whose manifest does not
    declare ``inplace`` (e.g. GELU, Softplus) intentionally do not mix
    this in.

    The underlying kernels always write to a fresh output buffer, but
    PyTorch's activation contract for ``inplace=True`` requires the
    *user-visible* tensor identity to be preserved (``y is x``).  We
    honor the manifest contract by computing into a fresh buffer and
    then copying the result back into the input tensor when
    ``inplace=True``; the returned tensor is the original input. The
    op-level ``inplace`` mutation is intentionally only honored on the
    eager ``forward()`` path: the ``torch.compile`` ``_wrapped`` route
    is registered with ``mutates_args=()`` and would mis-trace under
    aliasing, so we route inplace=True around the custom-op dispatch.
    """

    @staticmethod
    def _apply_inplace(
        inplace: bool, input_tensor: torch.Tensor, output: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize ``output`` back into ``input_tensor`` when inplace=True.

        Returns ``input_tensor`` (preserving identity) if ``inplace`` is
        True, otherwise returns ``output`` unchanged.

        The kernel path frequently flattens the input before launch, so
        ``output`` may differ from ``input_tensor`` in *layout* while
        carrying the same number of elements. A ``reshape`` followed by
        ``copy_`` recovers the caller's layout without broadcasting; any
        true element-count mismatch is rejected up front with a
        ``ValueError`` rather than letting ``reshape`` raise a downstream
        ``RuntimeError``.

        Raises:
            ValueError: if ``output.dtype`` does not match
                ``input_tensor.dtype`` (an inplace write must not change
                the storage dtype), or if ``output.numel()`` does not
                equal ``input_tensor.numel()`` (broadcasting is not
                supported here).
        """
        if not inplace:
            return output
        if output.dtype != input_tensor.dtype:
            raise ValueError(
                f"inplace=True requires output.dtype ({output.dtype}) to "
                f"match input.dtype ({input_tensor.dtype}); they differ"
            )
        if output.numel() != input_tensor.numel():
            raise ValueError(
                f"inplace=True requires output.numel() ({output.numel()}) "
                f"to match input.numel() ({input_tensor.numel()}); "
                f"broadcasting is not supported on the inplace path"
            )
        # Reshape to the input layout to handle the contiguous/flatten
        # round-trip used by the kernel path without forcing the caller
        # to think about it. The numel guard above ensures reshape never
        # raises here.
        input_tensor.copy_(output.reshape(input_tensor.shape))
        return input_tensor

    def _inplace_aware_forward(
        self, input: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        """Run validate + dispatch + ``_apply_inplace`` for unary activations.

        Routes ``inplace=True`` around the ``_wrapped`` custom-op so the
        functional torch.compile path never sees a mutation it cannot
        model (the custom op is registered with ``mutates_args=()``).
        """
        self._validate_input(input)
        wrapped = type(self)._wrapped
        if wrapped is not None and not self.inplace:
            return wrapped(input, self._instance_key)
        result = self._eager_forward(input)
        return self._apply_inplace(self.inplace, input, result)


class ReluFwdOp(UnaryOp, _InplaceMixin):
    """ReLU activation: y = max(x, 0)."""

    _op_name = "relu"
    kernel_cls = ReluFwdKernel
    # Manifest: flops = "2 * N" (compare + select per element).
    FLOPS_PER_ELEM = 2

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self._inplace_aware_forward(input)


class _AlphaScaledBinaryOp(BinaryOp):
    """Shared base for ops that take a scalar ``alpha`` multiplier on ``other``.

    PyTorch ``torch.add(input, other, alpha=1)`` and ``torch.sub(input,
    other, alpha=1)`` scale ``other`` by ``alpha`` before the binary op.
    The kernel does not bake ``alpha`` in: only the manifest-declared
    default (``alpha=1``) is wired through the fast kernel path.
    Non-default ``alpha`` values are routed through a ``torch.add`` /
    ``torch.sub`` eager fallback after the same input validation as the
    kernel path runs, so the manifest contract is honored without a
    kernel rewrite. The leading ``*`` makes ``alpha`` and the existing
    ``strategy`` / ``kernel_map`` / ``tune`` parameters keyword-only;
    only the positional triplet ``(a_shape, b_shape, dtype)`` is shared
    with ``BinaryOp``.
    """

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        *,
        alpha: int | float = 1,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            a_shape, b_shape, dtype, strategy=strategy,
            kernel_map=kernel_map, tune=tune,
        )
        self.alpha = alpha


class AddFwdOp(_AlphaScaledBinaryOp):
    """Element-wise addition with broadcast: y = input + alpha * other.

    Conforms to ``torch.add(input, other, *, alpha=1)``. ``alpha`` defaults
    to ``1``; non-default values run through a ``torch.add`` eager
    fallback after the standard ``BinaryOp`` device / dtype / numel
    checks so the manifest contract is honored end-to-end.
    """

    _op_name = "add"
    kernel_cls = AddFwdKernel
    _torch_fallback_fn = staticmethod(torch.add)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        other: torch.Tensor,
    ) -> torch.Tensor:
        if self.alpha != 1:
            # Fallback path bypasses ``BinaryOp.forward`` so we run the
            # shared validator here. The ``alpha == 1`` path delegates to
            # ``super().forward()``, which already validates.
            self._validate_binary_inputs(input, other)
            return torch.add(input, other, alpha=self.alpha)
        return super().forward(input, other)


class SubFwdOp(_AlphaScaledBinaryOp):
    """Element-wise subtraction with broadcast: y = input - alpha * other.

    Conforms to ``torch.sub(input, other, *, alpha=1)``. Non-default
    ``alpha`` values run through a ``torch.sub`` eager fallback after the
    standard ``BinaryOp`` device / dtype / numel checks.
    """

    _op_name = "sub"
    kernel_cls = SubFwdKernel
    _torch_fallback_fn = staticmethod(torch.sub)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES_NO_BOOL

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        other: torch.Tensor,
    ) -> torch.Tensor:
        if self.alpha != 1:
            # Fallback path bypasses ``BinaryOp.forward``; validate here.
            # The ``alpha == 1`` path delegates to ``super().forward()``.
            self._validate_binary_inputs(input, other)
            return torch.sub(input, other, alpha=self.alpha)
        return super().forward(input, other)


class MulFwdOp(BinaryOp):
    """Element-wise multiplication with broadcast: y = input * other."""

    _op_name = "mul"
    kernel_cls = MulFwdKernel
    _torch_fallback_fn = staticmethod(torch.mul)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


class DivFwdOp(BinaryOp):
    """Element-wise division with broadcast: y = input / other.

    Conforms to ``torch.div(input, other, *, rounding_mode=None)``.
    ``rounding_mode`` accepts ``None`` (true division), ``"trunc"``
    (truncation toward zero), or ``"floor"`` (floor division). Non-None
    rounding modes run through a ``torch.div`` eager fallback after the
    standard ``BinaryOp`` device / dtype / numel checks. The leading ``*``
    makes ``rounding_mode`` and the existing ``strategy`` / ``kernel_map``
    / ``tune`` parameters keyword-only; only the positional triplet
    ``(a_shape, b_shape, dtype)`` is shared with ``BinaryOp``.
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
            # Fallback path bypasses ``BinaryOp.forward``; validate here.
            # The default path delegates to ``super().forward()``.
            self._validate_binary_inputs(input, other)
            return torch.div(input, other, rounding_mode=self.rounding_mode)
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

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002
        exponent: torch.Tensor,
    ) -> torch.Tensor:
        return super()._eager_forward(input, exponent)

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        exponent: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(input, exponent)


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

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002
        end: torch.Tensor,
    ) -> torch.Tensor:
        return super()._eager_forward(input, end)

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        end: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(input, end)

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
    _torch_fallback_fn = staticmethod(torch.maximum)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


class MinimumFwdOp(BinaryOp):
    """Element-wise minimum with broadcast: y = min(a, b)."""

    _op_name = "minimum"
    kernel_cls = MinimumFwdKernel
    _torch_fallback_fn = staticmethod(torch.minimum)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


# ---------------------------------------------------------------------------
# Comparison op subclasses (output bool)
# ---------------------------------------------------------------------------
#
# Kernels produce int8 (1/0) because TileLang cannot vectorize bool.
# The Op forward() casts to torch.bool after the kernel call.


class _BoolOutputBinaryOp(BinaryOp):
    """Mixin that casts kernel int8 output to torch.bool.

    ``_eager_forward`` casts the int8 kernel output to bool.  In the
    ``torch.compile`` path, ``register_fake`` already declares
    ``torch.bool`` as the output dtype, and the actual execution goes
    through ``_eager_forward`` which handles the cast.  When the manifest
    dtype falls outside the underlying float kernel's support set, the
    base ``BinaryOp._eager_forward`` returns the torch reference's
    output directly (already ``torch.bool``), so the post-cast in this
    branch is a no-op rather than a double-cast.
    """

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        other: torch.Tensor,
    ) -> torch.Tensor:
        result = super()._eager_forward(input, other)
        return result.to(torch.bool)


class EqFwdOp(_BoolOutputBinaryOp):
    """Element-wise equality with broadcast: y = (a == b)."""

    _op_name = "eq"
    kernel_cls = EqFwdKernel
    _torch_fallback_fn = staticmethod(torch.eq)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


class NeFwdOp(_BoolOutputBinaryOp):
    """Element-wise not-equal with broadcast: y = (a != b)."""

    _op_name = "ne"
    kernel_cls = NeFwdKernel
    _torch_fallback_fn = staticmethod(torch.ne)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


class GtFwdOp(_BoolOutputBinaryOp):
    """Element-wise greater-than with broadcast: y = (a > b)."""

    _op_name = "gt"
    kernel_cls = GtFwdKernel
    _torch_fallback_fn = staticmethod(torch.gt)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


class LtFwdOp(_BoolOutputBinaryOp):
    """Element-wise less-than with broadcast: y = (a < b)."""

    _op_name = "lt"
    kernel_cls = LtFwdKernel
    _torch_fallback_fn = staticmethod(torch.lt)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


class GeFwdOp(_BoolOutputBinaryOp):
    """Element-wise greater-equal with broadcast: y = (a >= b)."""

    _op_name = "ge"
    kernel_cls = GeFwdKernel
    _torch_fallback_fn = staticmethod(torch.ge)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


class LeFwdOp(_BoolOutputBinaryOp):
    """Element-wise less-equal with broadcast: y = (a <= b)."""

    _op_name = "le"
    kernel_cls = LeFwdKernel
    _torch_fallback_fn = staticmethod(torch.le)
    _TORCH_FALLBACK_DTYPES = _BINARY_FALLBACK_INT_DTYPES


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


_MANIFEST_INT_DTYPES = (
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
)


def _int_identity(input: torch.Tensor) -> torch.Tensor:
    return input.clone()


class _IntIdentityUnaryOp(UnaryOp):
    """Base for unary ops whose manifest declares integer dtypes but whose
    kernel is float-only.

    Several manifest entries (floor / ceil / round / trunc, abs / neg / sign,
    isnan / isinf / isfinite) declare both integer and floating-point input
    dtypes, while the underlying ``*FwdKernel`` classes are float-only
    (``FloatUnaryKernel``). For integer inputs we short-circuit at the op
    layer: skip kernel construction in ``__init__`` and route through
    ``_int_handler`` in ``_eager_forward``.

    Subclasses override ``_int_handler`` (default = identity = ``input.clone()``)
    and ``_int_output_dtype`` (default = same as input) to express the
    appropriate integer semantics — e.g. ``torch.abs`` for ``AbsFwdOp``,
    constant-False ``torch.bool`` for ``IsnanFwdOp``.

    The short-circuit is restricted to the integer dtypes declared in the
    manifest. Other non-float dtypes (bool, complex) are not in the
    contract and fall through to ``UnaryOp.__init__``, which raises via the
    kernel's dtype check.
    """

    _int_handler: Callable[[torch.Tensor], torch.Tensor] = staticmethod(
        _int_identity)
    _int_output_dtype: Optional[torch.dtype] = None
    # Subclasses may extend the fallback dtype set when the manifest
    # signature includes additional non-float dtypes (e.g. torch.bool for
    # the is{nan,inf,finite} predicates).
    _fallback_dtypes: tuple = _MANIFEST_INT_DTYPES

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if dtype in type(self)._fallback_dtypes:
            self.N_total = N_total
            self.dtype = dtype
            self.strategy = strategy
            # Skip dispatch_kernel: the float-only kernel cannot be
            # instantiated for an integer dtype. Expose a kernel_map shape
            # consistent with the float path for any introspection that
            # iterates it, but leave the kernel itself unconstructed.
            self.kernel_map = kernel_map or self.default_kernel_map
            self.kernel = None
            self.output_dtype = (
                type(self)._int_output_dtype
                if type(self)._int_output_dtype is not None
                else dtype
            )
            self._instance_key = id(self)
            _OP_REGISTRY[self._instance_key] = self
            return
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if self.kernel is None:
            return type(self)._int_handler(input)
        return super()._eager_forward(input)


class AbsFwdOp(_IntIdentityUnaryOp):
    """Element-wise |x|."""

    _op_name = "abs"
    kernel_cls = AbsFwdKernel
    _int_handler = staticmethod(torch.abs)


class NegFwdOp(_IntIdentityUnaryOp):
    """Element-wise -x."""

    _op_name = "neg"
    kernel_cls = NegFwdKernel
    _int_handler = staticmethod(torch.neg)


class ReciprocalFwdOp(UnaryOp):
    """Element-wise 1/x.

    The manifest's int-dtype declaration is a known gap: ``torch.reciprocal``
    promotes int input to float32, which does not match the manifest's
    ``same_as(input)`` output rule. The op layer here only supports the
    float dtypes the kernel implements; int input falls through to the
    kernel's dtype check and raises.
    """

    _op_name = "reciprocal"
    kernel_cls = ReciprocalFwdKernel


class SignFwdOp(_IntIdentityUnaryOp):
    """Element-wise sign(x): -1, 0, or +1."""

    _op_name = "sign"
    kernel_cls = SignFwdKernel
    # Manifest: flops = "2 * N" (two compares + selects per element).
    FLOPS_PER_ELEM = 2
    _int_handler = staticmethod(torch.sign)


class SinFwdOp(UnaryOp):
    """Element-wise sin(x)."""

    _op_name = "sin"
    kernel_cls = SinFwdKernel


class CosFwdOp(UnaryOp):
    """Element-wise cos(x)."""

    _op_name = "cos"
    kernel_cls = CosFwdKernel


class FloorFwdOp(_IntIdentityUnaryOp):
    """Element-wise floor(x)."""

    _op_name = "floor"
    kernel_cls = FloorFwdKernel


class CeilFwdOp(_IntIdentityUnaryOp):
    """Element-wise ceil(x)."""

    _op_name = "ceil"
    kernel_cls = CeilFwdKernel


class RoundFwdOp(_IntIdentityUnaryOp):
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
        # Integer dtypes are no-ops regardless of decimals (rounding an int
        # produces the same int). Match the float-path identity contract.
        if self.dtype in _MANIFEST_INT_DTYPES:
            return input.clone()
        # Run through fp32 so low-precision inputs (fp16/bf16) cannot overflow
        # when ``torch.round`` internally scales by ``10**decimals`` — e.g.
        # ``100 * 10**4 = 1e6`` exceeds fp16 max (~65504). The single down-cast
        # at the end restores the op's contract dtype. The manifest's
        # ``kernel_map`` continues to describe the round-to-nearest-integer
        # kernel that handles the ``decimals=0`` fast path above.
        return torch.round(input.float(), decimals=decimals).to(self.dtype)


class TruncFwdOp(_IntIdentityUnaryOp):
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
    # Manifest: flops = "2 * N" (1 add + 1 log).
    FLOPS_PER_ELEM = 2


class Expm1FwdOp(UnaryOp):
    """Element-wise exp(x) - 1."""

    _op_name = "expm1"
    kernel_cls = Expm1FwdKernel
    # Manifest: flops = "2 * N" (1 exp + 1 sub).
    FLOPS_PER_ELEM = 2


# ---------------------------------------------------------------------------
# Activation ops (8)
# ---------------------------------------------------------------------------


class GeluFwdOp(UnaryOp):
    """Element-wise GELU honoring the manifest ``approximate`` contract.

    Args:
        N_total: Number of elements (flattened input).
        dtype: Torch dtype.
        approximate: Approximation mode. ``'none'`` (default) routes to the
            erf-based ``GeluFwdKernel``; ``'tanh'`` routes through a torch
            reference path (``torch.nn.functional.gelu(..., approximate='tanh')``)
            because no fused tanh-GELU kernel is exposed for the unary op
            yet. Both values are honored end-to-end.
        strategy: Optional kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "gelu"
    kernel_cls = GeluFwdKernel
    # Manifest: flops = "8 * N" (erf-based: mul + erf + add + mul + mul ≈ 8).
    FLOPS_PER_ELEM = 8

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        approximate: str = "none",
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if approximate not in ("none", "tanh"):
            raise ValueError(
                f"GeluFwdOp: approximate must be 'none' or 'tanh', got {approximate!r}"
            )
        # Apply the same dtype gate both branches share so the tanh
        # fallback rejects unsupported dtypes (e.g. integer / fp8) up
        # front instead of deferring the failure to forward(). The
        # ``approximate='none'`` branch enforces this implicitly via
        # GeluFwdKernel's SUPPORTED_DTYPES check inside super().__init__;
        # mirror that contract here so both branches behave consistently.
        supported = self.kernel_cls.SUPPORTED_DTYPES
        if supported is not None and dtype not in supported:
            names = ", ".join(str(dt) for dt in supported)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.approximate = approximate
        if approximate == "tanh":
            # No fused tanh-GELU unary kernel is exposed to the op layer
            # yet; honor the manifest contract by routing through torch
            # so callers passing the manifest-allowed value still get a
            # correct result rather than an error.
            self.N_total = N_total
            self.dtype = dtype
            self.strategy = strategy
            self.kernel_map = kernel_map or self.default_kernel_map
            self.kernel = None
            self.output_dtype = dtype
            self._instance_key = id(self)
            _OP_REGISTRY[self._instance_key] = self
            return
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if self.kernel is None:
            # approximate='tanh' fallback path — see __init__.
            orig_shape = input.shape
            flat = input.contiguous().reshape(-1)
            out = torch.nn.functional.gelu(flat, approximate="tanh")
            return out.reshape(orig_shape)
        return super()._eager_forward(input)


class SiluFwdOp(UnaryOp, _InplaceMixin):
    """Element-wise SiLU (Swish): y = x * sigmoid(x)."""

    _op_name = "silu"
    kernel_cls = SiluFwdKernel
    # Manifest: flops = "4 * N" (sigmoid + multiply).
    FLOPS_PER_ELEM = 4

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self._inplace_aware_forward(input)


class SigmoidFwdOp(UnaryOp):
    """Element-wise sigmoid(x)."""

    _op_name = "sigmoid"
    kernel_cls = SigmoidFwdKernel
    # Manifest: flops = "4 * N" (sigmoid(x) = 1 / (1 + exp(-x)) ≈ 4 ops/elem).
    FLOPS_PER_ELEM = 4


class TanhFwdOp(UnaryOp):
    """Element-wise tanh(x)."""

    _op_name = "tanh"
    kernel_cls = TanhFwdKernel
    # Manifest: flops = "5 * N" (tanh(x) = 2 * sigmoid(2x) - 1 ≈ 5 ops/elem).
    FLOPS_PER_ELEM = 5


class HardswishFwdOp(UnaryOp, _InplaceMixin):
    """Element-wise HardSwish: y = x * clamp(x + 3, 0, 6) / 6."""

    _op_name = "hardswish"
    kernel_cls = HardswishFwdKernel
    # Manifest: flops = "7 * N" (add + clamp(2 cmp+2 sel) + mul + div).
    FLOPS_PER_ELEM = 7

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self._inplace_aware_forward(input)


class HardsigmoidFwdOp(UnaryOp, _InplaceMixin):
    """Element-wise HardSigmoid: y = clamp(x + 3, 0, 6) / 6."""

    _op_name = "hardsigmoid"
    kernel_cls = HardsigmoidFwdKernel
    # Manifest: flops = "6 * N" (add + clamp(2 cmp+2 sel) + div).
    FLOPS_PER_ELEM = 6

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self._inplace_aware_forward(input)


class MishFwdOp(UnaryOp, _InplaceMixin):
    """Element-wise Mish: y = x * tanh(softplus(x))."""

    _op_name = "mish"
    kernel_cls = MishFwdKernel
    # Manifest: flops = "7 * N" (softplus + tanh + mul).
    FLOPS_PER_ELEM = 7

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self._inplace_aware_forward(input)


class SeluFwdOp(UnaryOp, _InplaceMixin):
    """Element-wise SELU activation."""

    _op_name = "selu"
    kernel_cls = SeluFwdKernel
    # Manifest: flops = "5 * N" (branch + exp/sub/mul + lambda mul).
    FLOPS_PER_ELEM = 5

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        *,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self._inplace_aware_forward(input)


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


def _int_all_false(input: torch.Tensor) -> torch.Tensor:
    return torch.zeros(input.shape, dtype=torch.bool, device=input.device)


def _int_all_true(input: torch.Tensor) -> torch.Tensor:
    return torch.ones(input.shape, dtype=torch.bool, device=input.device)


_PREDICATE_FALLBACK_DTYPES = _MANIFEST_INT_DTYPES + (torch.bool,)


class IsnanFwdOp(_IntIdentityUnaryOp):
    """Element-wise isnan with bool output.

    Always False on integer / bool input (no NaN representation in those
    dtypes).
    """

    _op_name = "isnan"
    kernel_cls = IsnanFwdKernel
    _int_handler = staticmethod(_int_all_false)
    _int_output_dtype = torch.bool
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES


class IsinfFwdOp(_IntIdentityUnaryOp):
    """Element-wise isinf with bool output.

    Always False on integer / bool input (no Inf representation in those
    dtypes).
    """

    _op_name = "isinf"
    kernel_cls = IsinfFwdKernel
    _int_handler = staticmethod(_int_all_false)
    _int_output_dtype = torch.bool
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES


class IsfiniteFwdOp(_IntIdentityUnaryOp):
    """Element-wise isfinite with bool output.

    Always True on integer / bool input (every value in those dtypes is
    finite).
    """

    _op_name = "isfinite"
    kernel_cls = IsfiniteFwdKernel
    _int_handler = staticmethod(_int_all_true)
    _int_output_dtype = torch.bool
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES


# ---------------------------------------------------------------------------
# Independent (custom-signature) op classes (11)
# ---------------------------------------------------------------------------


class LeakyReluFwdOp(Op, _InplaceMixin):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        negative_slope: Slope for negative inputs (default 0.01).
        inplace: When True, copy the result back into ``input`` and
            return ``input`` (preserving tensor identity). The kernel
            still computes into a fresh buffer; only the user-visible
            tensor is mutated, mirroring ``torch.nn.functional.leaky_relu``.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "leaky_relu"
    _wrapped = None
    # Manifest: flops = "3 * N" (compare + mul + select).
    FLOPS_PER_ELEM = 3

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        negative_slope: float = 0.01,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        _validate_scalar_param_repr("negative_slope", negative_slope, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["leaky_relu"](
            N_total, dtype, negative_slope=negative_slope, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"leaky_relu": LeakyReluFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        orig_shape = input.shape
        result = self.kernel(input.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
        wrapped = type(self)._wrapped
        # inplace=True bypasses the torch.compile route; the custom_op
        # is registered with mutates_args=() and would mis-trace.
        if wrapped is not None and not self.inplace:
            return wrapped(input, self._instance_key)
        result = self._eager_forward(input)
        return self._apply_inplace(self.inplace, input, result)


class EluFwdOp(Op, _InplaceMixin):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        alpha: Scale for the negative part (default 1.0).
        inplace: When True, copy the result back into ``input`` and
            return ``input`` (preserving tensor identity).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "elu"
    _wrapped = None
    # Manifest: flops = "5 * N" (compare + (exp + sub + mul) + branch select).
    FLOPS_PER_ELEM = 5

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        alpha: float = 1.0,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        _validate_scalar_param_repr("alpha", alpha, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.alpha = alpha
        self.inplace = inplace
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["elu"](N_total, dtype, alpha=alpha, tune=tune)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"elu": EluFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        orig_shape = input.shape
        result = self.kernel(input.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None and not self.inplace:
            return wrapped(input, self._instance_key)
        result = self._eager_forward(input)
        return self._apply_inplace(self.inplace, input, result)


class HardtanhFwdOp(Op, _InplaceMixin):
    """Hardtanh: y = clamp(x, min_val, max_val).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        min_val: Lower bound (default -1.0).
        max_val: Upper bound (default 1.0).
        inplace: When True, copy the result back into ``input`` and
            return ``input`` (preserving tensor identity).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "hardtanh"
    _wrapped = None
    # Manifest: flops = "4 * N" (2 compares + 2 selects per element).
    FLOPS_PER_ELEM = 4

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        min_val: float = -1.0,
        max_val: float = 1.0,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        _validate_scalar_param_repr("min_val", min_val, dtype, self._op_name)
        _validate_scalar_param_repr("max_val", max_val, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["hardtanh"](
            N_total, dtype, min_val=min_val, max_val=max_val, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"hardtanh": HardtanhFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        orig_shape = input.shape
        result = self.kernel(input.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None and not self.inplace:
            return wrapped(input, self._instance_key)
        result = self._eager_forward(input)
        return self._apply_inplace(self.inplace, input, result)


class SoftplusFwdOp(Op):
    """Softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        beta: Scaling factor (default 1.0).
        threshold: Linear regime threshold (default 20.0).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "softplus"
    _wrapped = None

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        beta: float = 1.0,
        threshold: float = 20.0,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        _validate_scalar_param_repr("beta", beta, dtype, self._op_name)
        _validate_scalar_param_repr("threshold", threshold, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.beta = beta
        self.threshold = threshold
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["softplus"](
            N_total, dtype, beta=beta, threshold=threshold, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"softplus": SoftplusFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        orig_shape = input.shape
        result = self.kernel(input.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)


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

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        weight: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = input.shape
        result = self.kernel(
            input.contiguous().reshape(-1), weight.contiguous().reshape(-1),
        ).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
        # ``weight`` is part of the manifest contract; validate device,
        # dtype, and length so a malformed weight fails fast at the op
        # boundary instead of corrupting the kernel.
        if not weight.is_cuda:
            raise ValueError("Weight must be a CUDA tensor")
        if weight.dtype != self.dtype:
            raise ValueError(
                f"Expected weight.dtype {self.dtype}, got {weight.dtype}"
            )
        if weight.numel() != self.num_channels:
            raise ValueError(
                f"Expected weight to have {self.num_channels} elements, "
                f"got {weight.numel()}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, weight, self._instance_key)
        return self._eager_forward(input, weight)


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
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp_tensor"](
            self.N_total, dtype,
            has_min=self.min_shape is not None,
            has_max=self.max_shape is not None,
            tune=tune,
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
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.input_shape = tuple(input)
        self.min_shape = tuple(min)
        self.dtype = dtype
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.min_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp_tensor"](
            self.N_total, dtype, has_min=True, has_max=False, tune=tune,
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
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.input_shape = tuple(input)
        self.max_shape = tuple(max)
        self.dtype = dtype
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.max_shape))
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp_tensor"](
            self.N_total, dtype, has_min=False, has_max=True, tune=tune,
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
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["clamp"](
            self.N_total, dtype, min_val=min, max_val=max, tune=tune,
        )
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


_MASKED_FILL_INT_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


class MaskedFillScalarFwdOp(Op):
    """MaskedFill with Number (scalar) value.

    Conforms to ``torch.Tensor.masked_fill(mask, value: Number)``. Output
    shape follows the bidirectional broadcast of ``input`` and ``mask``.

    The manifest declares the full PyTorch dtype union (``bool | uint8 |
    int8 | int16 | int32 | int64 | float16 | bfloat16 | float32``). The
    underlying TileLang kernel only supports float dtypes; integer and
    bool dtypes route through a torch eager fallback at the op layer so
    the manifest contract is honored end-to-end without a kernel rewrite.

    Args:
        input: Shape of the input tensor.
        mask: Shape of the mask tensor (bool).
        value: Scalar fill value (bool / int / float; range-validated
            against ``dtype`` for the float vectorized path; passed
            through verbatim on the integer / bool fallback path).
        dtype: Torch dtype.
    """

    _op_name = "masked_fill"
    _wrapped = None

    def __init__(
        self,
        input: tuple,  # noqa: A002
        mask: tuple,
        value: bool | int | float = 0,
        dtype: torch.dtype = torch.float32,
    ):
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
        # Integer / bool dtypes route through a torch fallback because the
        # underlying float-only kernel cannot run them. Float dtypes go
        # through the fast vectorized kernel as before, with the scalar
        # representability validator gating against silent overflow into
        # the user-facing dtype.
        self._use_torch_fallback = dtype in _MASKED_FILL_INT_DTYPES
        if self._use_torch_fallback:
            self.kernel = None
        else:
            _validate_scalar_param_repr("value", value, dtype, self._op_name)
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
        if self._use_torch_fallback:
            return input.masked_fill(mask, self.value)
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
        if wrapped is not None and not self._use_torch_fallback:
            return wrapped(input, mask, self._instance_key)
        return self._eager_forward(input, mask)


_NAN_TO_NUM_SENTINEL = object()


class NanToNumFwdOp(Op):
    """NanToNum: replace NaN, +Inf, -Inf with specified values.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        nan: Replacement for NaN (default 0.0). Manifest-aligned name.
        posinf: Replacement for +Inf. Manifest default ``None`` resolves
            to the largest finite value representable in the user-facing
            ``dtype`` (matches ``torch.nan_to_num``). Explicit values
            must also be representable in ``dtype`` end-to-end; values
            that fit only in the kernel's intermediate dtype (e.g. fp16
            for fp8_e5m2) are rejected so the post-cast cannot resurface
            them as Inf.
        neginf: Replacement for -Inf. Manifest default ``None`` resolves
            to the smallest (most negative) finite value representable
            in the user-facing ``dtype``.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.

    Note:
        ``nan_val`` / ``posinf_val`` / ``neginf_val`` remain accepted as
        legacy aliases for ``nan`` / ``posinf`` / ``neginf`` to keep
        existing call sites working during the manifest-alignment
        migration.
    """

    _op_name = "nan_to_num"
    _wrapped = None

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        # Legacy aliases kept for back-compat with existing tests / benches.
        nan_val: object = _NAN_TO_NUM_SENTINEL,
        posinf_val: object = _NAN_TO_NUM_SENTINEL,
        neginf_val: object = _NAN_TO_NUM_SENTINEL,
    ):
        if nan_val is not _NAN_TO_NUM_SENTINEL:
            nan = nan_val  # type: ignore[assignment]
        if posinf_val is not _NAN_TO_NUM_SENTINEL:
            posinf = posinf_val  # type: ignore[assignment]
        if neginf_val is not _NAN_TO_NUM_SENTINEL:
            neginf = neginf_val  # type: ignore[assignment]
        # User-supplied scalars must be representable in the user-facing
        # dtype, so they go through the standard validation path below.
        # The manifest default ``None`` resolves to the *final*
        # user-facing dtype's max / min, not ``+/-inf``: the kernel runs
        # in ``output_dtype`` (fp16 for e5m2 to preserve Inf/NaN) and
        # _clamp_to_dtype_range targets that intermediate, so forwarding
        # ``+inf`` would resolve to fp16's 65504.0 and then surface as
        # ``+Inf`` after the e5m2 post-cast (e5m2 max is 57344.0).
        # Picking ``torch.finfo(dtype).max`` here keeps the replacement
        # value finite end-to-end and matches ``torch.nan_to_num``
        # semantics (replace Inf with the dtype's max finite value).
        _validate_scalar_param_repr("nan", nan, dtype, self._op_name)
        if posinf is None:
            kernel_posinf = torch.finfo(dtype).max
        else:
            _validate_scalar_param_repr("posinf", posinf, dtype, self._op_name)
            kernel_posinf = posinf
        if neginf is None:
            kernel_neginf = torch.finfo(dtype).min
        else:
            _validate_scalar_param_repr("neginf", neginf, dtype, self._op_name)
            kernel_neginf = neginf
        self.N_total = N_total
        self.dtype = dtype
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        # Legacy attribute aliases retained so callers reading the
        # original names keep working.
        self.nan_val = nan
        self.posinf_val = kernel_posinf
        self.neginf_val = kernel_neginf
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["nan_to_num"](
            N_total, dtype, nan_val=nan, posinf_val=kernel_posinf,
            neginf_val=kernel_neginf, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"nan_to_num": NanToNumFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        orig_shape = input.shape
        result = self.kernel(input.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)


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
