"""Elementwise op infrastructure: umbrella bases, helpers, registration factories.

Three umbrella Op base classes:
- UnaryOp: wraps UnaryKernel with reshape/flatten
- BinaryOp: wraps BinaryKernel with broadcast coalescing
- FusedGatedOp: wraps FusedGatedKernel with (M, 2N) layout

torch.compile support:
- Concrete ops are registered via @torch.library.custom_op at package load time
- Three factory functions (_register_unary_custom_op, _register_binary_custom_op,
  _register_fused_gated_custom_op) register every op; instances are looked up at
  runtime via _OP_REGISTRY keyed by id(instance)

Utility:
- coalesce_broadcast_dims: reduces N-dim broadcast to minimal effective dims
"""

import inspect
import math
import weakref
from math import prod
from typing import Callable, Dict, List, Optional

import torch

from tileops.kernels.kernel_base import Kernel

from ..op_base import Op

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


def _register_unary_inplace_custom_op(op_cls):
    """Register the ``inplace=True`` companion for a unary activation op.

    The kernel writes into a fresh buffer; this wrapper copies the result
    back into ``x`` and returns ``x`` so the caller sees ``y is x`` and
    ``x`` carries the activation output. The custom op is registered with
    ``mutates_args=("x",)`` so ``torch.compile`` traces the mutation
    correctly. Sets ``op_cls._wrapped_inplace`` for ``forward()`` to
    dispatch through.
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(
        f"top::elementwise_unary_{op_name}_inplace", mutates_args=("x",),
    )
    def _wrapped_inplace(x: torch.Tensor, instance_key: int) -> None:
        instance = _OP_REGISTRY[instance_key]
        result = instance._eager_forward(x)
        x.copy_(result.reshape(x.shape))

    op_cls._wrapped_inplace = _wrapped_inplace


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


def _register_lerp_tensor_custom_op(op_cls):
    """Register a Tensor-weight lerp op (input, end, weight -> out).

    The fake function computes the broadcast output shape from ``input`` /
    ``end`` / ``weight`` so that ``torch.compile(fullgraph=True)`` works
    for both same-shape and broadcasting inputs. Registered under a
    distinct ``_tensor`` namespace to avoid colliding with the scalar
    ``LerpFwdOp`` (which bakes ``weight`` at construction time and uses
    the binary registration path).
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        end: torch.Tensor,
        weight: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(input, end, weight)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,  # noqa: A002
        end: torch.Tensor,
        weight: torch.Tensor,
        instance_key: int,
    ) -> torch.Tensor:
        out_shape = torch.broadcast_shapes(input.shape, end.shape, weight.shape)
        return input.new_empty(out_shape)

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
    # Subclasses may set ``_other_name`` to a manifest-aligned parameter
    # name (e.g. ``"exponent"`` for ``PowFwdOp``, ``"end"`` for
    # ``LerpFwdOp``); the L1 signature check sees the renamed parameter
    # via ``__init_subclass__`` rebinding ``forward.__signature__``.
    _other_name: str = "other"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        other_name = cls.__dict__.get("_other_name")
        if other_name is None or other_name == "other":
            return
        base_forward = cls.forward
        try:
            sig = inspect.signature(base_forward)
        except (ValueError, TypeError):
            return
        new_params = [
            p.replace(name=other_name) if p.name == "other" else p
            for p in sig.parameters.values()
        ]
        new_sig = sig.replace(parameters=new_params)

        def forward(self, *args, **kwargs):
            if other_name in kwargs:
                kwargs["other"] = kwargs.pop(other_name)
            return base_forward(self, *args, **kwargs)

        forward.__signature__ = new_sig
        forward.__name__ = "forward"
        forward.__qualname__ = f"{cls.__qualname__}.forward"
        cls.forward = forward

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        kernel_supported = self.kernel_cls.SUPPORTED_DTYPES
        if kernel_supported is not None and dtype not in kernel_supported:
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
        self.dispatch_kernel(kernel_map)
        self.kernel = self._build_kernel_instance(
            coalesced_shape, a_strides, b_strides, strategy, tune,
        )
        # Register in global registry for torch.compile dispatch
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    def _build_kernel_instance(
        self, coalesced_shape, a_strides, b_strides, strategy, tune,
    ):
        """Construct the kernel. Subclasses override to inject extra kwargs."""
        return self.kernel_map[self._op_name](
            self.N_total, self.dtype, coalesced_shape, a_strides, b_strides,
            self.a_numel, self.b_numel, strategy=strategy, tune=tune,
        )

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
        result = self.kernel(
            input.contiguous().view(-1), other.contiguous().view(-1),
        ).reshape(self.out_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        other: torch.Tensor,
    ) -> torch.Tensor:
        a_name = getattr(self, "_input_name", "input")
        b_name = getattr(self, "_other_name", "other")
        if not input.is_cuda or not other.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected {a_name}.dtype {self.dtype}, got {input.dtype}")
        if other.dtype != self.dtype:
            raise ValueError(f"Expected {b_name}.dtype {self.dtype}, got {other.dtype}")
        if input.numel() != self.a_numel:
            raise ValueError(
                f"Expected {a_name} to have {self.a_numel} elements, got {input.numel()}"
            )
        if other.numel() != self.b_numel:
            raise ValueError(
                f"Expected {b_name} to have {self.b_numel} elements, got {other.numel()}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
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
# Intermediate (private) base classes shared by leaf op modules
# ---------------------------------------------------------------------------


class _UnaryActivationMixin:
    """Shared ``forward`` / inplace dispatch for unary activation Ops.

    The ten unary activation Ops (six param-free: ReLU, SiLU, HardSwish,
    HardSigmoid, Mish, SELU; four parametric: LeakyReLU, ELU, Hardtanh,
    Softplus) share an identical ``forward`` template:

    1. validate ``input`` against the op's ``dtype`` / ``N_total`` contract,
    2. when ``self.inplace`` is true, dispatch through ``_wrapped_inplace``
       (registered with ``mutates_args=("x",)`` so ``torch.compile`` traces
       the mutation correctly) and return the original ``input`` so callers
       see ``y is x``,
    3. otherwise dispatch through the standard ``_wrapped`` custom op or
       fall back to ``_eager_forward``.

    Concrete classes provide ``_validate_input`` and ``_eager_forward``
    (both inherited from ``UnaryOp``) plus ``self.inplace`` /
    ``self._instance_key`` state. Leaves that do not expose ``inplace``
    in their signature (e.g. Softplus) simply default ``self.inplace`` to
    ``False`` via ``_finalize_init``.
    """

    # Set by ``_register_unary_inplace_custom_op`` for leaves that
    # declare ``inplace`` in their manifest signature. Stays ``None``
    # when the leaf does not support inplace (e.g. Softplus, or a
    # test-only subclass that skipped registration).
    _wrapped_inplace = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        self._validate_input(input)
        if self.inplace:
            wrapped_inplace = type(self)._wrapped_inplace
            if wrapped_inplace is not None:
                wrapped_inplace(input, self._instance_key)
                return input
            # No inplace custom op registered (e.g. test-only subclass);
            # fall back to direct mutation via the eager path.
            result = self._eager_forward(input)
            input.copy_(result.reshape(input.shape))
            return input
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)


class _ParamFreeActivationOp(_UnaryActivationMixin, UnaryOp):
    """Shared base for the param-free activation Op group.

    Centralizes the canonical constructor used by activations whose only
    manifest-declared parameter is ``inplace`` (ReLU, SiLU, HardSwish,
    HardSigmoid, Mish, SELU). Each leaf only declares its op-specific
    class fields (``_op_name``, ``kernel_cls``, ``FLOPS_PER_ELEM``,
    docstring); ``forward``/``_eager_forward`` come from
    ``_UnaryActivationMixin`` / ``UnaryOp``.
    """

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


class _ParametricActivationOp(_UnaryActivationMixin, UnaryOp):
    """Shared base for the parametric activation Op group.

    Used by activations that take one or more scalar construction-time
    parameters (LeakyReLU, ELU, Hardtanh, Softplus). Leaves own their
    ``__init__`` (scalar parameter names and defaults vary per leaf):
    each leaf validates its scalars, populates ``self.<param>`` for
    introspection, instantiates ``self.kernel`` with typed kwargs, and
    registers itself with ``_OP_REGISTRY`` via the
    ``_finalize_init`` helper. ``UnaryOp.__init__`` is intentionally
    bypassed; ``_finalize_init`` performs the equivalent state setup.

    Leaves that declare ``inplace`` in the manifest signature accept it
    in ``__init__`` and pass it to ``_finalize_init``. ``forward`` and
    ``_eager_forward`` are inherited from the mixin and ``UnaryOp``.
    """

    def _finalize_init(
        self,
        N_total: int,
        dtype: torch.dtype,
        kernel: Kernel,
        *,
        inplace: bool = False,
    ) -> None:
        """Record the leaf-built kernel and wire shared base state.

        The leaf has already called ``self.dispatch_kernel(kernel_map)``
        and instantiated its kernel directly with typed kwargs. This
        helper records the kernel on ``self`` and runs the
        ``_OP_REGISTRY`` registration shared by every parametric leaf.
        """
        self.N_total = N_total
        self.dtype = dtype
        self.inplace = inplace
        self.kernel = kernel
        # Mirror ``UnaryOp.__init__``: surface ``output_dtype`` so callers
        # and ``total_memory`` can reason about FP8 post-casts. Parametric
        # activations do not currently declare an FP8 path, so the common
        # branch returns ``self.dtype``; the lookup is kept for parity.
        fp8_out = getattr(self.kernel, "_fp8_output_dtype", None)
        self.output_dtype = fp8_out or getattr(self.kernel, "output_dtype", dtype)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self


class _AlphaScaledBinaryOp(BinaryOp):
    """Shared base for ops that take a scalar ``alpha`` multiplier on ``other``.

    PyTorch ``torch.add(input, other, alpha=1)`` and ``torch.sub(input,
    other, alpha=1)`` scale ``other`` by ``alpha`` before the binary op.
    ``alpha`` is baked into the kernel at construction time (one
    specialization per distinct alpha value), so non-default alpha runs
    through the same fast kernel as the default.

    The leading ``*`` makes ``alpha`` and the existing
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
        self.alpha = alpha
        super().__init__(
            a_shape, b_shape, dtype,
            strategy=strategy, kernel_map=kernel_map, tune=tune,
        )

    def _build_kernel_instance(
        self, coalesced_shape, a_strides, b_strides, strategy, tune,
    ):
        return self.kernel_map[self._op_name](
            self.N_total, self.dtype, coalesced_shape, a_strides, b_strides,
            self.a_numel, self.b_numel, strategy=strategy, tune=tune,
            alpha=self.alpha,
        )


class _BoolOutputBinaryOp(BinaryOp):
    """Binary op base whose kernel emits int8 (1/0) and whose Op output is bool.

    TileLang cannot vectorize bool, so the kernel produces int8. The Op
    casts to ``torch.bool`` after the kernel call. ``register_fake``
    already declares ``torch.bool`` as the output dtype, so the
    ``torch.compile`` path stays consistent.
    """

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        other: torch.Tensor,
    ) -> torch.Tensor:
        result = super()._eager_forward(input, other)
        return result.to(torch.bool)


_MANIFEST_INT_DTYPES = (
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
)


def _int_identity(input: torch.Tensor) -> torch.Tensor:  # noqa: A002
    return input.clone()


def _int_all_false(input: torch.Tensor) -> torch.Tensor:  # noqa: A002
    return torch.zeros(input.shape, dtype=torch.bool, device=input.device)


def _int_all_true(input: torch.Tensor) -> torch.Tensor:  # noqa: A002
    return torch.ones(input.shape, dtype=torch.bool, device=input.device)


_PREDICATE_FALLBACK_DTYPES = _MANIFEST_INT_DTYPES + (torch.bool,)


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
            # The float-only kernel cannot be instantiated for an integer
            # dtype, so the kernel itself stays unconstructed. The kernel_map
            # is still installed through the shared validate-and-install path
            # so a user-supplied override is arch-checked identically to the
            # auto-discovered map on the float path.
            self._install_kernel_map(kernel_map)
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


class _GeluApproximateBase(UnaryOp):
    """Intermediate base that resolves the manifest ``approximate`` field.

    Validates the ``approximate`` argument against the manifest's allowed
    values (``'none'`` / ``'tanh'``), records it on ``self.approximate``
    for introspection, and then delegates to ``UnaryOp.__init__``. The
    ``default_kernel_map`` of the leaf op picks the kernel implementation
    from ``self.approximate``.
    """

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
                f"{type(self).__name__}: approximate must be 'none' or "
                f"'tanh', got {approximate!r}"
            )
        self.approximate = approximate
        super().__init__(
            N_total, dtype, strategy=strategy, kernel_map=kernel_map, tune=tune,
        )


class _ClampTensorBase(Op):
    """Shared infrastructure for Tensor-bound clamp variants (broadcasting)."""

    _wrapped = None

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)
