"""Elementwise op infrastructure: umbrella bases, helpers, registration factories.

Three umbrella Op base classes:
- UnaryOp: wraps UnaryKernel with reshape/flatten
- BinaryOp: wraps BinaryKernel with broadcast coalescing
- FusedGatedOp: wraps FusedGatedKernel with (M, 2N) layout

torch.compile support:
- Concrete ops are registered via @torch.library.custom_op at package load time
- Three factory functions (_register_unary_custom_op, _register_binary_custom_op,
  _register_fused_gated_custom_op) register every op; instances are looked up at
  runtime via the shared instance registry in tileops.ops.compile_boundary

Utility:
- broadcast_out_shape: PyTorch broadcast output shape of two operand shapes

The broadcast *lowering* decision (dim coalescing and stride synthesis) belongs
to the kernel layer; this module only passes the two operand shapes down.
"""

import functools
import inspect
import math
from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, List, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.manifest import load_manifest
from tileops.manifest.dtype_rules import promote_int_to_float_ref, same_as_ref

from ..compile_boundary import get_instance
from ..op_base import Op

# torch.compile registration factories (see module docstring). The registry
# key is a plain int so dynamo can trace through forward() without hitting
# unsupported Python side-effects.


_MANIFEST_INT_SCALAR_DTYPES = (
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
)


def _validate_scalar_param_repr(
    param_name: str, value, dtype: torch.dtype, op_name: str,
    *, allow_nonfinite_float: bool = False,
) -> None:
    """Reject scalar params that cannot be represented in the user dtype.

    Validates against the *user-facing* ``dtype``, not the kernel's fp16
    intermediate: an fp8 kernel computes in fp16, so a value that only fits in
    fp16 would surface as ``+/-Inf`` after the fp8 post-cast.

    Integer and bool mirror PyTorch ``Tensor.masked_fill`` coercion:

    - bool: any int/float, reduced to ``{0, 1}``.
    - Signed int: any value in ``[iinfo.min, iinfo.max]``, truncated toward
      zero. NaN/Inf and out-of-range raise.
    - ``uint8``: ints in ``[-255, 255]``, negatives wrapping via ``& 0xFF``;
      float scalars must be in ``[0, 255]``.

    Floats always accept ``NaN`` and require finite values in ``finfo`` range.
    ``+/-Inf`` passes only under ``allow_nonfinite_float`` — used by
    ``MaskedFillScalarFwdOp``, which writes the scalar into tensor storage.
    """
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; treat explicitly so the int
        # range checks below operate on the integer/float branch.
        return
    if not isinstance(value, (int, float)):
        raise TypeError(f"{op_name} expected scalar {param_name} to be int/float, got {type(value)}")

    if dtype == torch.bool:
        return

    if dtype in _MANIFEST_INT_SCALAR_DTYPES:
        iinfo = torch.iinfo(dtype)
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(
                    f"{op_name} received {param_name}={value!r}, but {param_name} must be finite "
                    f"and representable in dtype {dtype}"
                )
            # PyTorch range-checks the real float value, then truncates
            # toward zero. Negative float scalars never wrap into uint8
            # (``uint8.masked_fill(mask, -1.0)`` raises in PyTorch).
            if not (iinfo.min <= value <= iinfo.max):
                raise ValueError(
                    f"{op_name} received {param_name}={value!r}, which is not representable in "
                    f"dtype {dtype} (valid finite range: [{iinfo.min}, {iinfo.max}])"
                )
            return
        # Python int branch. uint8 wraps negatives in [-255, 255] via
        # two's complement, matching PyTorch.
        if dtype == torch.uint8 and value < 0:
            if value < -255:
                raise ValueError(
                    f"{op_name} received {param_name}={value!r}, which is not representable in "
                    f"dtype {dtype} (valid integer range: [-255, 255] with wraparound, "
                    f"or [0, 255] direct)"
                )
            return
        if not (iinfo.min <= value <= iinfo.max):
            raise ValueError(
                f"{op_name} received {param_name}={value!r}, which is not representable in "
                f"dtype {dtype} (valid integer range: [{iinfo.min}, {iinfo.max}])"
            )
        return

    finfo = torch.finfo(dtype)
    value_f64 = float(value)
    if math.isnan(value_f64):
        return
    if math.isinf(value_f64):
        # PyTorch preserves +/-Inf for float tensor scalars. Ops needing a
        # finite scalar (elu alpha, softplus beta) reject here; masked_fill
        # writes the scalar into storage and opts in.
        if allow_nonfinite_float:
            return
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


def _register_unary_custom_op(op_cls):
    """Register a unary elementwise op for torch.compile.

    Args:
        op_cls: The Op subclass to register (must have ``_op_name``).
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_unary_{op_name}", mutates_args=())
    def _wrapped(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        # Manifest-driven: covers a predicate's bool output and an integer
        # input promoted to float32 alike, without a per-registration override.
        # ``new_empty``, not ``empty_like``: the real path flattens to
        # contiguous storage, so a non-contiguous input's strides must not
        # survive into the fake or the compiled graph asserts on the mismatch.
        return x.new_empty(
            x.shape, dtype=resolve_output_dtype(op_cls.__name__, x.dtype),
        )

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
    def _wrapped_inplace(x: torch.Tensor, instance_key: str) -> None:
        instance = get_instance(instance_key)
        result = instance._eager_forward(x)
        x.copy_(result.reshape(x.shape))

    op_cls._wrapped_inplace = _wrapped_inplace


def _register_binary_custom_op(op_cls):
    """Register a binary elementwise op for torch.compile.

    Args:
        op_cls: The Op subclass to register.
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_binary_{op_name}", mutates_args=())
    def _wrapped(
        a: torch.Tensor,
        b: torch.Tensor,
        out_shape: List[int],
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(a, b)

    @_wrapped.register_fake
    def _(
        a: torch.Tensor,
        b: torch.Tensor,
        out_shape: List[int],
        instance_key: str,
    ) -> torch.Tensor:
        return a.new_empty(
            out_shape, dtype=resolve_output_dtype(op_cls.__name__, a.dtype)
        )

    op_cls._wrapped = _wrapped


def _register_prelu_custom_op(op_cls):
    """Register a PReLU-style op (x, weight -> y) for torch.compile."""
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        x: torch.Tensor,
        weight: torch.Tensor,
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(x, weight)

    @_wrapped.register_fake
    def _(
        x: torch.Tensor,
        weight: torch.Tensor,
        instance_key: str,
    ) -> torch.Tensor:
        return x.new_empty(x.shape, dtype=x.dtype)

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
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(cond, x, y)

    @_wrapped.register_fake
    def _(
        cond: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        instance_key: str,
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
    ``LerpFwdOp`` (which takes ``weight`` as a constructor argument and uses
    the binary registration path).
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        input: torch.Tensor,
        end: torch.Tensor,
        weight: torch.Tensor,
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(input, end, weight)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,
        end: torch.Tensor,
        weight: torch.Tensor,
        instance_key: str,
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
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(x, mask)

    @_wrapped.register_fake
    def _(
        x: torch.Tensor,
        mask: torch.Tensor,
        instance_key: str,
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
        input: torch.Tensor,
        mask: torch.Tensor,
        value: torch.Tensor,
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(input, mask, value)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,
        mask: torch.Tensor,
        value: torch.Tensor,
        instance_key: str,
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
        input: torch.Tensor,
        min: Optional[torch.Tensor],
        max: Optional[torch.Tensor],
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(input, min, max)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,
        min: Optional[torch.Tensor],
        max: Optional[torch.Tensor],
        instance_key: str,
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
        input: torch.Tensor,
        min: torch.Tensor,
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(input, min)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,
        min: torch.Tensor,
        instance_key: str,
    ) -> torch.Tensor:
        out_shape = torch.broadcast_shapes(input.shape, min.shape)
        return input.new_empty(out_shape)

    op_cls._wrapped = _wrapped


def _register_clamp_max_custom_op(op_cls):
    """Register single-bound Tensor upper-clamp (input, max -> out)."""
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::elementwise_{op_name}", mutates_args=())
    def _wrapped(
        input: torch.Tensor,
        max: torch.Tensor,
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(input, max)

    @_wrapped.register_fake
    def _(
        input: torch.Tensor,
        max: torch.Tensor,
        instance_key: str,
    ) -> torch.Tensor:
        out_shape = torch.broadcast_shapes(input.shape, max.shape)
        return input.new_empty(out_shape)

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
        instance_key: str,
    ) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(
        x: torch.Tensor,
        M: int,
        N: int,
        instance_key: str,
    ) -> torch.Tensor:
        return x.new_empty((M, N), dtype=x.dtype)

    op_cls._wrapped = _wrapped


def broadcast_out_shape(a_shape, b_shape) -> torch.Size:
    """Return the PyTorch broadcast output shape of two operand shapes.

    0-dim operands are normalised to a single size-1 dimension, so a scalar
    paired with a scalar yields ``(1,)`` rather than ``()``.

    Args:
        a_shape: Shape tuple of input a.
        b_shape: Shape tuple of input b.

    Returns:
        The broadcast output shape.
    """
    return torch.broadcast_shapes(tuple(a_shape) or (1,), tuple(b_shape) or (1,))


# Target dtype for integral inputs under ``promote_int_to_float``, matching
# PyTorch's int-input promotion (e.g. ``torch.reciprocal``).
_PROMOTED_FLOAT_DTYPE = torch.float32


@functools.lru_cache(maxsize=None)
def _manifest_output_dtype_expr(op_class_name: str) -> str:
    """Return the manifest ``signature.outputs`` dtype expression for an op.

    Args:
        op_class_name: Op class name, which is the manifest entry key.

    Returns:
        The declared dtype expression, e.g. ``"same_as(input)"`` or ``"bool"``.

    Raises:
        KeyError: If the manifest has no entry for *op_class_name*.
        ValueError: If the entry declares other than exactly one output.
    """
    entry = load_manifest().get(op_class_name)
    if entry is None:
        raise KeyError(
            f"{op_class_name} has no manifest entry; the output dtype is "
            "declared under signature.outputs"
        )
    outputs = entry["signature"]["outputs"]
    if len(outputs) != 1:
        raise ValueError(
            f"{op_class_name} declares {len(outputs)} outputs; the elementwise "
            "bases resolve a single output dtype"
        )
    return next(iter(outputs.values()))["dtype"]


def resolve_output_dtype(op_class_name: str, input_dtype: torch.dtype) -> torch.dtype:
    """Resolve an op's output dtype from its manifest declaration.

    Args:
        op_class_name: Op class name, which is the manifest entry key.
        input_dtype: Declared input dtype of the op instance.

    Returns:
        The output dtype. ``same_as(...)`` and dtype unions follow the input;
        ``promote_int_to_float(...)`` promotes integral inputs to float32; a
        bare dtype name resolves to that dtype.

    Raises:
        ValueError: If the declared expression names an unknown dtype.
    """
    expr = _manifest_output_dtype_expr(op_class_name)
    if same_as_ref(expr) is not None or "|" in expr:
        return input_dtype
    if promote_int_to_float_ref(expr) is not None:
        if input_dtype.is_floating_point:
            return input_dtype
        return _PROMOTED_FLOAT_DTYPE
    resolved = getattr(torch, expr, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(
            f"{op_class_name}: manifest output dtype {expr!r} is not a torch dtype"
        )
    return resolved


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_fp8(dtype: torch.dtype) -> bool:
    """Return True iff ``dtype`` is one of the supported fp8 dtypes."""
    return dtype in _FP8_DTYPES


@dataclass(frozen=True)
class KernelEntry:
    """One element type's specialization, resolved together.

    An op instance now serves whatever dtype its caller passes, so nothing
    derived from the element type may live in a slot on the instance: a later
    call with a different dtype would read the earlier call's value. Each
    entry is built once and carries everything a call needs.

    The fields state semantics only. How a backend represents those semantics
    — reinterpreting bool storage as uint8, say — belongs to the kernel, not
    here: an op that branches on a backend's representation cannot be served by
    a second backend that represents it differently.

    Attributes:
        kernel: built for ``compute_dtype``.
        compute_dtype: what the kernel was specialized for. Differs from the
            key when the semantic dtype cannot be computed in directly — an
            integer input computing in float32, a bool operand on a uint8
            kernel.
        output_dtype: resolved from the manifest for the *semantic* dtype.
    """

    kernel: Optional[Kernel]
    compute_dtype: torch.dtype
    output_dtype: torch.dtype


class _PerDtypeKernels:
    """The family's one way to reach a kernel: ``self._entry(dtype)``.

    Every elementwise op specializes on the element type of its inputs, so
    every one of them needs the same three things — build once per dtype,
    keep whatever else that specialization implies together with the kernel,
    and record which dtype the most recent call used.

    A subclass supplies ``_build_entry(dtype)``. It may return extra state in
    the entry (bool operands running on a uint8 kernel, an integer input
    computing in float32); an op with no such split simply gets
    ``compute_dtype == dtype``.

    ``self.dtype`` is metadata for ``eval_roofline`` and ``total_memory``, and
    execution must never read it: by the time a second dtype arrives it no
    longer describes the call in flight.

    ``_note_call`` records it wherever a call commits to an element type, which
    is every execution path: the eager one, the one ``torch.compile`` enters
    behind the custom op, and the ones that answer without a kernel at all.

    The record therefore describes the most recent call that *selected a
    specialization*, not the most recent one that succeeded — a call that fails
    afterwards leaves its dtype behind. Narrowing it to successful calls needs
    the invocation context to reach ``eval_roofline`` instead of living in a
    mutable slot, which is an ``Op``-wide contract change. A snapshot-and-restore
    transaction here is not that fix: it cannot see the paths that bypass
    ``__call__``, and two calls sharing an instance can erase each other's
    published value.
    """

    def _note_call(self, dtype: torch.dtype) -> None:
        """Record the element type this call committed to."""
        self.dtype = dtype

    def _selected_kernel_cls(self, slot: Optional[str] = None):
        """The kernel class that will run, honoring a ``kernel_map`` override.

        Capability questions must go to this class, never to the family default:
        an override that supports a different dtype set is the whole point of
        supplying one.
        """
        return self.kernel_map[slot if slot is not None else self._op_name]

    def _entry(self, dtype: torch.dtype, *shape: int) -> "KernelEntry":
        """Return the specialization for *dtype*, building it on first use.

        ``shape`` is empty for ops whose shape is fixed at construction; ops
        that learn it from the tensor pass it so each shape keys its own entry.
        """
        key = (dtype, *shape) if shape else dtype
        entry = self._entries.get(key)
        if entry is None:
            entry = self._build_entry(dtype, *shape)
            self._entries[key] = entry
        self._note_call(dtype)
        return entry

    def _init_entries(self) -> None:
        """Subclass constructors call this instead of building a kernel."""
        self._entries: Dict[object, KernelEntry] = {}

    def _build_entry(self, dtype: torch.dtype, *shape: int) -> "KernelEntry":
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_entry")


class UnaryOp(_PerDtypeKernels, Op):
    """Template base class for unary elementwise ops.

    Subclass must set ``kernel_cls`` and ``_op_name``.
    Subclass should also set ``_wrapped`` via ``_register_unary_custom_op``
    to enable torch.compile support.

    Args:
        N_total: Total number of elements (flattened).
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
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N_total = N_total
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._init_entries()

    def _build_entry(self, dtype: torch.dtype) -> KernelEntry:
        """Build one specialization for the semantic *dtype*."""
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return KernelEntry(
            kernel=self._build_kernel_instance(
                N_total=self.N_total, dtype=ctor_dtype, tune=self.tune, impl=impl,
            ),
            compute_dtype=ctor_dtype,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )

    def _build_kernel_instance(
        self,
        *,
        N_total: int,
        dtype: torch.dtype,
        tune: bool,
        impl: type,
    ):
        """Construct the kernel. Subclasses override to specialize construction."""
        return impl(N_total, dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + write y, for the element type of the most recent forward."""
        if self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.total_memory requires a prior forward() "
                "call to bind the element type")
        out = self._entry(self.dtype).output_dtype
        return self.N_total * (self.dtype.itemsize + out.itemsize)

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
        bool), the entry's ``output_dtype.itemsize`` already captures it.
        """
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        orig_shape = input.shape
        flat = input.contiguous().reshape(-1)
        return self._entry(input.dtype).kernel(flat).reshape(orig_shape)

    def _validate_input(self, input: torch.Tensor) -> None:
        """Validate the input against the manifest dtype union and the numel."""
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        self._validate_dtypes(input)
        if input.numel() != self.N_total:
            raise ValueError(
                f"Expected {self.N_total} elements, got {input.numel()}"
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate_input(input)
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)


class BinaryOp(_PerDtypeKernels, Op):
    """Template base class for binary elementwise ops with broadcast.

    Subclass must set ``kernel_cls`` and ``_op_name``.
    Subclass should also set ``_wrapped`` via ``_register_binary_custom_op``
    to enable torch.compile support.

    Args:
        a_shape: Shape of input a.
        b_shape: Shape of input b.
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
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        out_shape = broadcast_out_shape(self.a_shape, self.b_shape)
        self.out_shape = out_shape
        self._out_shape_list = list(out_shape)  # cached for custom_op hot path
        self.N_total = prod(out_shape)
        self.a_numel = prod(a_shape)
        self.b_numel = prod(b_shape)
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._init_entries()

    def _build_entry(self, dtype: torch.dtype) -> KernelEntry:
        """Build one specialization for the semantic *dtype*."""
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        supported = impl.SUPPORTED_DTYPES
        if supported is not None and ctor_dtype not in supported:
            names = ", ".join(str(dt) for dt in supported)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        return KernelEntry(
            kernel=self._build_kernel_instance(self.tune, ctor_dtype, impl=impl),
            compute_dtype=ctor_dtype,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )

    def _build_kernel_instance(self, tune, dtype, impl):
        """Construct the kernel. Subclasses override to inject extra kwargs."""
        return impl(self.a_shape, self.b_shape, dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read a + read b + write y, for the most recent forward's dtype."""
        if self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.total_memory requires a prior forward() "
                "call to bind the element type")
        out_elem = self._entry(self.dtype).output_dtype.itemsize
        return ((self.a_numel + self.b_numel) * self.dtype.itemsize
                + self.N_total * out_elem)

    def _eager_forward(
        self,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        return self._entry(input.dtype).kernel(
            input.contiguous().view(-1), other.contiguous().view(-1),
        ).reshape(self.out_shape)

    def forward(
        self,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        a_name = getattr(self, "_input_name", "input")
        b_name = getattr(self, "_other_name", "other")
        if not input.is_cuda or not other.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        self._validate_dtypes(input, other)
        if other.dtype != input.dtype:
            raise ValueError(
                f"Expected {b_name}.dtype {input.dtype}, got {other.dtype}")
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


class FusedGatedOp(_PerDtypeKernels, Op):
    """Template base class for fused gated elementwise ops.

    Input: x of shape (M, 2*N). gate = x[:, :N], value = x[:, N:].
    Output: y = activation(gate) * value, shape (M, N).

    Subclass must set ``kernel_cls`` and ``_op_name``.
    Subclass should also set ``_wrapped`` via ``_register_fused_gated_custom_op``
    to enable torch.compile support.

    Args:
        M: Optional number of rows. Inferred from ``x`` when omitted.
        N: Optional half column dim (output width). Inferred from ``x`` when
            omitted.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    kernel_cls: type
    _op_name: str
    _wrapped = None  # Set by _register_fused_gated_custom_op at class definition
    FLOPS_PER_ELEM: int = 6

    def __init__(
        self,
        M: Optional[int] = None,
        N: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if (M is None) != (N is None):
            raise ValueError("M and N must be provided together")
        self._explicit_shape = M is not None and N is not None
        self.M = M
        self.N = N
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._init_entries()

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x (M*2N) + write y (M*N)."""
        if self.M is None or self.N is None or self.dtype is None:
            raise RuntimeError(
                "Fused gated dimensions are available after first forward"
            )
        out_elem = self._entry(self.dtype, self.M, self.N).output_dtype.itemsize
        return (self.M * 2 * self.N * self.dtype.itemsize
                + self.M * self.N * out_elem)

    def eval_roofline(self) -> tuple[int, int]:
        if self.M is None or self.N is None or self.dtype is None:
            raise RuntimeError(
                "Fused gated roofline is available after first forward"
            )
        flops = self.FLOPS_PER_ELEM * self.M * self.N
        return flops, int(self.total_memory)

    def _validate_dtype(self, dtype: torch.dtype) -> None:
        supported = self._selected_kernel_cls().SUPPORTED_DTYPES
        if supported is not None and dtype not in supported:
            names = ", ".join(str(dt) for dt in supported)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )

    def _build_entry(self, dtype: torch.dtype, *shape: int) -> KernelEntry:
        M, N = shape
        self._validate_dtype(dtype)
        return KernelEntry(
            kernel=self.kernel_map[self._op_name](M, N, dtype, tune=self.tune),
            compute_dtype=dtype,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )

    def _validate_runtime_input(self, x: torch.Tensor) -> tuple[int, int]:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got {x.ndim}D")
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Expected x.shape[1] to be even, got {x.shape[1]}")
        M = x.shape[0]
        N = x.shape[1] // 2
        if self._explicit_shape and (M, N) != (self.M, self.N):
            raise ValueError(
                f"Expected shape ({self.M}, {2 * self.N}), got {tuple(x.shape)}"
            )
        return M, N

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation."""
        M, N = self._validate_runtime_input(x)
        entry = self._entry(x.dtype, M, N)  # may reject the dtype; commit after
        self.M, self.N = M, N
        return entry.kernel(x.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the locally derived M/N rather than building the kernel to
        # populate self.M/self.N: a traced forward must not enter the TileLang
        # builder, and _eager_forward builds on the far side of the boundary.
        M, N = self._validate_runtime_input(x)
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, M, N, self._instance_key)
        return self._eager_forward(x)


# Intermediate (private) base classes shared by leaf op modules


class _UnaryActivationMixin:
    """Shared ``forward`` / inplace dispatch for unary activation Ops.

    The inplace path dispatches through ``_wrapped_inplace`` (registered
    ``mutates_args=("x",)`` so ``torch.compile`` traces the mutation) and
    returns the original ``input``, so callers see ``y is x``.

    Concrete classes supply ``_validate_input`` / ``_eager_forward`` from
    ``UnaryOp`` plus ``self.inplace`` and ``self._instance_key``. Leaves
    without ``inplace`` in their signature default it to ``False``.
    """

    # Set by ``_register_unary_inplace_custom_op`` for leaves that
    # declare ``inplace`` in their manifest signature. Stays ``None``
    # when the leaf does not support inplace (e.g. Softplus, or a
    # test-only subclass that skipped registration).
    _wrapped_inplace = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
        inplace: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(N_total, kernel_map=kernel_map, tune=tune)
        self.inplace = inplace


class _ParametricActivationOp(_UnaryActivationMixin, UnaryOp):
    """Shared base for the parametric activation Op group.

    Used by activations that take one or more scalar construction-time
    parameters (LeakyReLU, ELU, Hardtanh, Softplus). Leaves own their
    ``__init__`` because scalar names and defaults vary per leaf: each records
    its scalars on ``self``, calls ``dispatch_kernel``, then ``_finalize_init``
    for the shared state ``UnaryOp.__init__`` would otherwise set.

    Leaves that declare ``inplace`` in the manifest signature accept it
    in ``__init__`` and pass it to ``_finalize_init``. ``forward`` and
    ``_eager_forward`` are inherited from the mixin and ``UnaryOp``.
    """

    #: Names of the scalar parameters baked into the kernel; each names both the
    #: attribute on ``self`` and the kernel kwarg. The entry builder validates
    #: them against the element type before baking, which is why the check
    #: cannot live in ``__init__``: it needs a dtype, and none exists until a
    #: tensor arrives.
    _scalar_params: tuple[str, ...] = ()

    def _finalize_init(self, N_total: int, *, inplace: bool = False) -> None:
        """Wire shared base state for a leaf that owns its ``__init__``.

        The leaf has recorded its scalars and called ``dispatch_kernel``;
        kernels are built per element type by ``_build_entry``.
        """
        self.N_total = N_total
        self.inplace = inplace
        self._init_entries()

    def _build_entry(self, dtype: torch.dtype) -> KernelEntry:
        kwargs = {}
        for name in type(self)._scalar_params:
            value = getattr(self, name)
            _validate_scalar_param_repr(name, value, dtype, self._op_name)
            kwargs[name] = value
        return KernelEntry(
            kernel=self.kernel_map[self._op_name](
                self.N_total, dtype, tune=self.tune, **kwargs,
            ),
            compute_dtype=dtype,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )


class _AlphaScaledBinaryOp(BinaryOp):
    """Shared base for ops that take a scalar ``alpha`` multiplier on ``other``.

    PyTorch ``torch.add(input, other, alpha=1)`` and ``torch.sub(input,
    other, alpha=1)`` scale ``other`` by ``alpha`` before the binary op.
    ``alpha`` is baked into the kernel — one specialization per
    ``(alpha, element type)`` pair, built on first use — so non-default alpha
    runs through the same fast kernel as the default.

    The leading ``*`` makes ``alpha`` and the existing
    ``kernel_map`` / ``tune`` parameters keyword-only; only the
    positional pair ``(a_shape, b_shape)`` is shared with ``BinaryOp``.
    """

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        *,
        alpha: int | float = 1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.alpha = alpha
        super().__init__(a_shape, b_shape, kernel_map=kernel_map, tune=tune)

    def _build_kernel_instance(self, tune, dtype, impl):
        return impl(self.a_shape, self.b_shape, dtype, tune=tune, alpha=self.alpha)


class _BoolOutputBinaryOp(BinaryOp):
    """Binary op base whose public output dtype is bool.

    A bool *operand* needs no special handling here: ``Kernel.specialize`` names
    whichever implementation this backend uses for it.
    """

    def _eager_forward(
        self,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        result = super()._eager_forward(input, other)
        if result.dtype is not torch.bool:
            return result.to(torch.bool)
        return result


_MANIFEST_INT_DTYPES = (
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
)


def _int_identity(input: torch.Tensor) -> torch.Tensor:
    return input.clone()


def _int_all_false(input: torch.Tensor) -> torch.Tensor:
    return torch.zeros(input.shape, dtype=torch.bool, device=input.device)


def _int_all_true(input: torch.Tensor) -> torch.Tensor:
    return torch.ones(input.shape, dtype=torch.bool, device=input.device)


_PREDICATE_FALLBACK_DTYPES = _MANIFEST_INT_DTYPES + (torch.bool,)


class _IntIdentityUnaryOp(UnaryOp):
    """Base for unary ops whose manifest declares integer dtypes but whose
    kernel is float-only.

    Integer inputs short-circuit at the op layer: no kernel is constructed and
    ``_eager_forward`` routes through ``_int_handler``. Subclasses override
    ``_int_handler`` (default ``input.clone()``) and ``_int_output_dtype``
    (default: same as input) for the op's integer semantics.

    Only the integer dtypes declared in the manifest short-circuit. Other
    non-float dtypes fall through to ``UnaryOp.__init__``, which raises via the
    kernel's dtype check.
    """

    _int_handler: Callable[[torch.Tensor], torch.Tensor] = staticmethod(
        _int_identity)
    _int_output_dtype: Optional[torch.dtype] = None
    # Subclasses may extend the fallback dtype set when the manifest
    # signature includes additional non-float dtypes (e.g. torch.bool for
    # the is{nan,inf,finite} predicates).
    _fallback_dtypes: tuple = _MANIFEST_INT_DTYPES

    def _build_entry(self, dtype: torch.dtype) -> KernelEntry:
        """An integer dtype gets an entry with no kernel — it never reaches one."""
        if dtype in type(self)._fallback_dtypes:
            return KernelEntry(
                kernel=None,
                compute_dtype=dtype,
                output_dtype=(
                    type(self)._int_output_dtype
                    if type(self)._int_output_dtype is not None
                    else dtype
                ),
            )
        return super()._build_entry(dtype)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._entry(input.dtype).kernel is None:
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
        *,
        approximate: str = "none",
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if approximate not in ("none", "tanh"):
            raise ValueError(
                f"{type(self).__name__}: approximate must be 'none' or "
                f"'tanh', got {approximate!r}"
            )
        self.approximate = approximate
        super().__init__(N_total, kernel_map=kernel_map, tune=tune)


class _ClampTensorBase(Op):
    """Shared infrastructure for Tensor-bound clamp variants (broadcasting)."""

    _wrapped = None

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)
