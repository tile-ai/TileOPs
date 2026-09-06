"""Elementwise op infrastructure: umbrella bases, helpers, registration factories.

Three umbrella Op base classes, one per shape the family's kernels take:

- ``UnaryOp`` — one tensor in, the same shape out
- ``BinaryOp`` — two tensors broadcast against each other
- ``FusedGatedOp`` — one ``(M, 2N)`` tensor split into gate and value

What they share is where each thing happens. An op validates the call, normalizes
contiguity, and hands the *manifest-declared* shapes to its kernel; flattening,
broadcasting and restoring the output shape are the kernel's own business, so both
sides of the boundary speak the shapes the manifest declares. Element type is not
a construction parameter either: an instance serves whichever dtype its caller
passes, one specialization per element type, built on first use.

torch.compile support: each concrete op is registered as one opaque operator at
package load time, and publishes its name through ``compile_op_names`` so a test can
assert the traced graph holds nothing else. The factories below do that for the ops
that share a shape — unary, binary, fused gated, and the in-place companions; an op
whose signature is its own registers in its own file, next to the class. Either way
the fake reads its output shape from the op's ``_infer_output_shapes`` and its dtype
from the manifest, never from a kernel class, which would make the compiled graph
depend on which target served the op. Instances are recovered inside the operator by
key, via the registry in ``tileops.ops.compile_boundary``.
"""

import functools
import inspect
import math
from math import prod
from typing import Callable, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel

from .._output_dtype import resolve_output_dtype
from ..compile_boundary import get_instance
from ..op_base import Op

_MANIFEST_INT_SCALAR_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _validate_scalar_param_repr(
    param_name: str,
    value,
    dtype: torch.dtype,
    op_name: str,
    *,
    allow_nonfinite_float: bool = False,
) -> None:
    """Reject scalar params that cannot be represented in the user dtype.

    Validated against the dtype the caller passes, which is the dtype the result
    is stored in. A kernel widening an operand for the arithmetic does not widen
    what the scalar has to fit in.

    Integer and bool mirror PyTorch ``Tensor.masked_fill`` coercion:

    - bool: any int/float, reduced to ``{0, 1}``.
    - Signed int: any value in ``[iinfo.min, iinfo.max]``, truncated toward
      zero. NaN/Inf and out-of-range raise.
    - ``uint8``: ints in ``[-255, 255]``, negatives wrapping via ``& 0xFF``;
      float scalars must be in $[0 \\times 255]$.

    Floats always accept ``NaN`` and require finite values in ``finfo`` range.
    ``+/-Inf`` passes only under ``allow_nonfinite_float`` — used by
    ``MaskedFillScalarFwdOp``, which writes the scalar into tensor storage.
    """
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; treat explicitly so the int
        # range checks below operate on the integer/float branch.
        return
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{op_name} expected scalar {param_name} to be int/float, got {type(value)}"
        )

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


def _require_shape_inference(op_cls) -> None:
    """Refuse to register a boundary for a class with no ``_infer_output_shapes``.

    The registered fake is all the compiler learns about the node, and it takes the
    output shape from that method.
    """
    owner = next(b for b in op_cls.__mro__ if "_infer_output_shapes" in b.__dict__)
    if owner is Op:
        raise TypeError(
            f"{op_cls.__name__} registers a compile boundary but implements no "
            "_infer_output_shapes; its fake has no output shape to give"
        )


def _register_unary_custom_op(op_cls):
    """Register a unary elementwise op for torch.compile.

    Args:
        op_cls: The Op subclass to register (must have ``_op_name``).
    """
    _require_shape_inference(op_cls)
    op_name = f"tileops::elementwise_unary_{op_cls._op_name}"

    @torch.library.custom_op(op_name, mutates_args=())
    def _wrapped(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        # Shape from the op, dtype from the manifest: one rule covers a predicate's
        # bool output and an integer input promoted to float32. ``new_empty``, not
        # ``empty_like`` — the real path writes fresh contiguous storage, and a
        # non-contiguous input's strides in the fake fail the graph's assertion.
        op = get_instance(instance_key)
        shapes = op._infer_output_shapes(tuple(x.shape))
        return x.new_empty(
            shapes["output"],
            dtype=resolve_output_dtype(op_cls.__name__, x.dtype),
        )

    op_cls._wrapped = _wrapped
    op_cls.compile_op_names = (op_name,)


def _register_unary_inplace_custom_op(op_cls):
    """Register the ``inplace=True`` companion for a unary activation op.

    The kernel writes into a fresh buffer; this wrapper copies the result
    back into ``x`` and returns ``x`` so the caller sees ``y is x`` and
    ``x`` carries the activation output. The custom op is registered with
    ``mutates_args=("x",)`` so ``torch.compile`` traces the mutation
    correctly. Sets ``op_cls._wrapped_inplace`` for ``forward()`` to
    dispatch through.
    """
    op_name = f"tileops::elementwise_unary_{op_cls._op_name}_inplace"

    @torch.library.custom_op(op_name, mutates_args=("x",))
    def _wrapped_inplace(x: torch.Tensor, instance_key: str) -> None:
        instance = get_instance(instance_key)
        result = instance._eager_forward(x)
        x.copy_(result.reshape(x.shape))

    op_cls._wrapped_inplace = _wrapped_inplace
    # Two registrations, so two names: which one runs is decided per call by
    # ``inplace``, while registration happens once per class.
    op_cls.compile_op_names = tuple(op_cls.compile_op_names) + (op_name,)


def _register_binary_custom_op(op_cls):
    """Register a binary elementwise op for torch.compile.

    Args:
        op_cls: The Op subclass to register.
    """
    _require_shape_inference(op_cls)
    op_name = f"tileops::elementwise_binary_{op_cls._op_name}"

    @torch.library.custom_op(op_name, mutates_args=())
    def _wrapped(a: torch.Tensor, b: torch.Tensor, instance_key: str) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(a, b)

    @_wrapped.register_fake
    def _(a: torch.Tensor, b: torch.Tensor, instance_key: str) -> torch.Tensor:
        op = get_instance(instance_key)
        shapes = op._infer_output_shapes(tuple(a.shape), tuple(b.shape))
        return a.new_empty(shapes["output"], dtype=resolve_output_dtype(op_cls.__name__, a.dtype))

    op_cls._wrapped = _wrapped
    op_cls.compile_op_names = (op_name,)


def _register_fused_gated_custom_op(op_cls):
    """Register a fused gated elementwise op for torch.compile.

    Args:
        op_cls: The Op subclass to register.
    """
    _require_shape_inference(op_cls)
    op_name = f"tileops::elementwise_fused_gated_{op_cls._op_name}"

    @torch.library.custom_op(op_name, mutates_args=())
    def _wrapped(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        instance = get_instance(instance_key)
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        op = get_instance(instance_key)
        shapes = op._infer_output_shapes(tuple(x.shape))
        return x.new_empty(shapes["output"], dtype=resolve_output_dtype(op_cls.__name__, x.dtype))

    op_cls._wrapped = _wrapped
    op_cls.compile_op_names = (op_name,)


def broadcast_or_raise(op_name: str, **shapes: Optional[tuple]) -> tuple:
    """The shape these operands broadcast to, or a ``ValueError`` naming the ones that
    do not fit.

    ``()`` when every operand is 0-dim, which is what the manifest's shape rule says.
    Reads only its arguments; the registered fake calls it too.

    Args:
        op_name: Named in the error.
        shapes: Operand shapes by their manifest names; ``None`` for an optional input
            this call did not pass.
    """
    present = {name: tuple(shape) for name, shape in shapes.items() if shape is not None}
    try:
        return tuple(torch.broadcast_shapes(*present.values()))
    except RuntimeError as exc:
        listed = ", ".join(f"{name}={shape}" for name, shape in present.items())
        raise ValueError(f"{op_name} cannot broadcast {listed}") from exc


# Target dtype for integral inputs under ``promote_int_to_float``, matching
# PyTorch's int-input promotion (e.g. ``torch.reciprocal``).
_PROMOTED_FLOAT_DTYPE = torch.float32


@functools.lru_cache(maxsize=None)
def _require_one_device(op_name: str, **tensors: Optional[torch.Tensor]) -> None:
    """Refuse a call whose tensors are not all on one device.

    Which device that is, and whether any kernel runs there, the kernel answers.

    Args:
        op_name: Named in the error, so a caller sees which op refused.
        tensors: The call's tensors by their manifest names; ``None`` for an optional
            input this call did not pass.
    """
    device = None
    first = ""
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        if device is None:
            device, first = tensor.device, name
        elif tensor.device != device:
            raise ValueError(
                f"{op_name} needs every input on one device; got {first} on {device} "
                f"and {name} on {tensor.device}"
            )


class _PerDtypeKernels:
    """The family's one way to reach a kernel: ``self._kernel(inputs, dtype, *dims)``.

    A subclass supplies ``_build(dtype, *dims)`` for one specialization. What comes
    back is called with the manifest-declared tensors, so the in-tree path and a
    target's path hand back the same kind of thing.

    ``self.dtype`` and the recorded input shapes describe the most recent call that
    completed. ``eval_roofline`` and ``total_memory`` read them; no execution path may.
    """

    def _note_call(self, dtype: torch.dtype, **shapes: Optional[tuple]) -> None:
        """Record the element type and input shapes of the call that just returned.

        Written after the launch, so a call that raised leaves the previous one's
        account in place rather than half of its own.
        """
        for name, shape in shapes.items():
            setattr(self, name, shape)
        self.dtype = dtype

    @property
    def _slot(self) -> str:
        """The one dispatch key this op's ``kernel_map`` declares; also its memory role."""
        ((slot, _),) = self.kernel_map.items()
        return slot

    def _selected_kernel_cls(self):
        """The kernel class that will run, honoring a ``kernel_map`` override.

        Capability questions must go to this class, never to the family default:
        an override that supports a different dtype set is the whole point of
        supplying one.
        """
        ((_, kernel_cls),) = self.kernel_map.items()
        return kernel_cls

    def _kernel(self, inputs: tuple, dtype: torch.dtype, *dims):
        """Return what serves this call, building it once per specialization.

        Args:
            inputs: The tensors the kernel will be handed, one slot per
                ``signature.inputs`` entry, in that order; an optional input this call
                did not pass keeps its slot as ``None``.
            dtype: This call's element type.
            dims: What else the *in-tree* kernel is compiled for — the dimensions it
                bakes in, plus any presence that changes what gets built. A target's
                kernel is keyed on the input signature instead, by the base class.
        """
        return self.get_or_build_kernel(
            self._slot,
            inputs,
            key=(dtype, *dims),
            build=lambda: self._build(dtype, *dims),
        )

    def _build(self, dtype: torch.dtype, *dims):
        """Construct the in-tree kernel for one specialization."""
        raise NotImplementedError(f"{type(self).__name__} must implement _build")


class UnaryOp(_PerDtypeKernels, Op):
    """Template base class for unary elementwise ops.

    Subclass must set ``kernel_cls`` and ``_op_name``. The element count arrives with
    the tensor, so nothing about shape is a construction parameter.

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
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.input_shape: Optional[tuple] = None
        self.dispatch_kernel(kernel_map)

    def _infer_output_shapes(self, input_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == input.shape``."""
        return {"output": tuple(input_shape)}

    def _build(self, dtype: torch.dtype, n_total: int):
        """Build one specialization for the semantic *dtype*."""
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return self._build_kernel_instance(
            N_total=n_total,
            dtype=ctor_dtype,
            tune=self.tune,
            impl=impl,
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
    def N_total(self) -> int:
        """Element count of the most recent forward — the roofline's ``N``."""
        if self.input_shape is None:
            raise RuntimeError(
                f"{type(self).__name__}.N_total requires a prior forward() call: the "
                "element count arrives with the tensor"
            )
        return prod(self.input_shape)

    @property
    def total_memory(self) -> float:
        """Read x + write y, for the element type of the most recent forward."""
        if self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.total_memory requires a prior forward() "
                "call to bind the element type"
            )
        out = resolve_output_dtype(type(self).__name__, self.dtype)
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
        bool), the manifest's output dtype already captures it.
        """
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)

    def _validate_input(self, input: torch.Tensor) -> None:
        """Validate the input against the manifest dtype union."""
        self._validate_dtypes(input)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Validate, normalize, resolve the kernel and launch, inside the operator."""
        self._validate_input(input)
        input = input.contiguous()
        result = self._kernel((input,), input.dtype, input.numel())(input)
        self._note_call(input.dtype, input_shape=tuple(input.shape))
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return type(self)._wrapped(input, self._instance_key)


class BinaryOp(_PerDtypeKernels, Op):
    """Template base class for binary elementwise ops with broadcast.

    Subclass must set ``kernel_cls`` and ``_op_name``. Both operand shapes arrive with
    the tensors; the broadcast *lowering* — dim coalescing and stride synthesis — is the
    kernel's, so this class only hands the two shapes down.

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
            p.replace(name=other_name) if p.name == "other" else p for p in sig.parameters.values()
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
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.input_shape: Optional[tuple] = None
        self.other_shape: Optional[tuple] = None
        self.dispatch_kernel(kernel_map)

    def _infer_output_shapes(self, input_shape: tuple, other_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == broadcast_shapes(...)``."""
        b_name = type(self)._other_name
        return {
            "output": broadcast_or_raise(
                type(self).__name__, **{"input": input_shape, b_name: other_shape}
            )
        }

    def _build(self, dtype: torch.dtype, a_shape: tuple, b_shape: tuple):
        """Build one specialization for the semantic *dtype* and this broadcast."""
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        supported = impl.SUPPORTED_DTYPES
        if supported is not None and ctor_dtype not in supported:
            names = ", ".join(str(dt) for dt in supported)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. Supported: [{names}]"
            )
        return self._build_kernel_instance(self.tune, ctor_dtype, impl, a_shape, b_shape)

    def _build_kernel_instance(self, tune, dtype, impl, a_shape, b_shape):
        """Construct the kernel. Subclasses override to inject extra kwargs."""
        return impl(a_shape, b_shape, dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def out_shape(self) -> tuple:
        """Broadcast output shape of the most recent forward."""
        return self._infer_output_shapes(self._operand_shapes()[0], self._operand_shapes()[1])[
            "output"
        ]

    def _operand_shapes(self) -> tuple:
        if self.input_shape is None or self.other_shape is None:
            raise RuntimeError(
                f"{type(self).__name__} needs a prior forward() call: both operand "
                "shapes arrive with the tensors"
            )
        return self.input_shape, self.other_shape

    @property
    def N_total(self) -> int:
        """Output element count of the most recent forward."""
        return prod(self.out_shape)

    @property
    def a_numel(self) -> int:
        """Elements actually read from the first operand — the roofline reads this."""
        return prod(self._operand_shapes()[0])

    @property
    def b_numel(self) -> int:
        """Elements actually read from the second operand."""
        return prod(self._operand_shapes()[1])

    @property
    def total_memory(self) -> float:
        """Read a + read b + write y, for the most recent forward's dtype."""
        if self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.total_memory requires a prior forward() "
                "call to bind the element type"
            )
        a_shape, b_shape = self._operand_shapes()
        out_elem = resolve_output_dtype(type(self).__name__, self.dtype).itemsize
        reads = (prod(a_shape) + prod(b_shape)) * self.dtype.itemsize
        return reads + self.N_total * out_elem

    def _validate_operands(self, input: torch.Tensor, other: torch.Tensor) -> None:
        """Manifest dtype union, the shared element type, and one device."""
        b_name = type(self)._other_name
        _require_one_device(type(self).__name__, input=input, **{b_name: other})
        self._validate_dtypes(input, other)
        if other.dtype != input.dtype:
            raise ValueError(f"Expected {b_name}.dtype {input.dtype}, got {other.dtype}")

    def _eager_forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Validate, normalize, resolve the kernel and launch, inside the operator."""
        self._validate_operands(input, other)
        input = input.contiguous()
        other = other.contiguous()
        a_shape, b_shape = tuple(input.shape), tuple(other.shape)
        kernel = self._kernel((input, other), input.dtype, a_shape, b_shape)
        result = kernel(input, other)
        self._note_call(input.dtype, input_shape=a_shape, other_shape=b_shape)
        return result

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``other``."""
        return type(self)._wrapped(input, other, self._instance_key)


class FusedGatedOp(_PerDtypeKernels, Op):
    """Template base class for fused gated elementwise ops.

    Input: x of shape (M, 2*N). gate = x[:, :N], value = x[:, N:].
    Output: y = activation(gate) * value, shape (M, N).

    Subclass must set ``kernel_cls`` and ``_op_name``. Both dimensions arrive with the
    tensor.

    """

    kernel_cls: type
    _op_name: str
    _wrapped = None  # Set by _register_fused_gated_custom_op at class definition
    FLOPS_PER_ELEM: int = 6

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
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.x_shape: Optional[tuple] = None
        self.dispatch_kernel(kernel_map)

    def _infer_output_shapes(self, x_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == [x.shape[0], x.shape[1] // 2]``."""
        return {"output": (int(x_shape[0]), int(x_shape[1]) // 2)}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def M(self) -> int:
        """Row count of the most recent forward."""
        return self._dims()[0]

    @property
    def N(self) -> int:
        """Output width of the most recent forward."""
        return self._dims()[1]

    def _dims(self) -> tuple:
        if self.x_shape is None:
            raise RuntimeError(
                f"{type(self).__name__} needs a prior forward() call: both dimensions "
                "arrive with the tensor"
            )
        return self._infer_output_shapes(self.x_shape)["output"]

    @property
    def total_memory(self) -> float:
        """Read x (M*2N) + write y (M*N)."""
        if self.dtype is None:
            raise RuntimeError("Fused gated dimensions are available after first forward")
        m, n = self._dims()
        out_elem = resolve_output_dtype(type(self).__name__, self.dtype).itemsize
        return m * 2 * n * self.dtype.itemsize + m * n * out_elem

    def eval_roofline(self) -> tuple[int, int]:
        if self.dtype is None:
            raise RuntimeError("Fused gated roofline is available after first forward")
        m, n = self._dims()
        return self.FLOPS_PER_ELEM * m * n, int(self.total_memory)

    def _build(self, dtype: torch.dtype, m: int, n: int):
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        supported = impl.SUPPORTED_DTYPES
        if supported is not None and ctor_dtype not in supported:
            names = ", ".join(str(dt) for dt in supported)
            raise ValueError(
                f"{self._op_name} does not support dtype {dtype}. Supported: [{names}]"
            )
        return impl(m, n, ctor_dtype, tune=self.tune)

    def _validate_input(self, x: torch.Tensor) -> None:
        self._validate_dtypes(x)
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got {x.ndim}D")
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Expected x.shape[1] to be even, got {x.shape[1]}")

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Validate, normalize, resolve the kernel and launch, inside the operator."""
        self._validate_input(x)
        x = x.contiguous()
        x_shape = tuple(x.shape)
        m, n = self._infer_output_shapes(x_shape)["output"]
        result = self._kernel((x,), x.dtype, m, n)(x)
        self._note_call(x.dtype, x_shape=x_shape)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the op on ``x``."""
        return type(self)._wrapped(x, self._instance_key)


# Intermediate (private) base classes shared by leaf op modules


class _UnaryActivationMixin:
    """Shared ``forward`` / inplace dispatch for unary activation Ops.

    The inplace path dispatches through ``_wrapped_inplace`` (registered
    ``mutates_args=("x",)`` so ``torch.compile`` traces the mutation) and
    returns the original ``input``, so callers see ``y is x``.

    Which of the two operators runs is decided by ``self.inplace``, a construction
    parameter — read on the traced side, never written there. Leaves without
    ``inplace`` in their signature default it to ``False``.
    """

    # Set by ``_register_unary_inplace_custom_op`` for leaves that
    # declare ``inplace`` in their manifest signature.
    _wrapped_inplace = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        if self.inplace:
            type(self)._wrapped_inplace(input, self._instance_key)
            return input
        return type(self)._wrapped(input, self._instance_key)


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
        *,
        inplace: bool = False,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)
        self.inplace = inplace


class _ParametricActivationOp(_UnaryActivationMixin, UnaryOp):
    """Shared base for the parametric activation Op group.

    Used by activations that take one or more scalar construction-time
    parameters (LeakyReLU, ELU, Hardtanh, Softplus). Leaves own their
    ``__init__`` because scalar names and defaults vary per leaf: each records
    its scalars on ``self``, then delegates to ``UnaryOp.__init__``.

    Leaves that declare ``inplace`` in the manifest signature accept it
    in ``__init__``. ``forward`` and ``_eager_forward`` are inherited from the
    mixin and ``UnaryOp``.
    """

    # Names of the scalar parameters baked into the kernel; each names both the
    # attribute on ``self`` and the kernel kwarg. The entry builder validates
    # them against the element type before baking, which is why the check
    # cannot live in ``__init__``: it needs a dtype, and none exists until a
    # tensor arrives.
    _scalar_params: tuple[str, ...] = ()

    def _build(self, dtype: torch.dtype, n_total: int):
        kwargs = {}
        for name in type(self)._scalar_params:
            value = getattr(self, name)
            _validate_scalar_param_repr(name, value, dtype, self._op_name)
            kwargs[name] = value
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(n_total, ctor_dtype, tune=self.tune, **kwargs)


class _AlphaScaledBinaryOp(BinaryOp):
    """Shared base for ops that take a scalar ``alpha`` multiplier on ``other``.

    PyTorch ``torch.add(input, other, alpha=1)`` and ``torch.sub(input,
    other, alpha=1)`` scale ``other`` by ``alpha`` before the binary op.
    ``alpha`` is baked into the kernel — one specialization per
    ``(alpha, element type, broadcast)`` — so non-default alpha runs through the
    same fast kernel as the default. It stays out of the memory key because it is
    fixed for the instance.
    """

    def __init__(
        self,
        *,
        alpha: int | float = 1,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.alpha = alpha
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _build_kernel_instance(self, tune, dtype, impl, a_shape, b_shape):
        return impl(a_shape, b_shape, dtype, tune=tune, alpha=self.alpha)


_MANIFEST_INT_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _int_identity(input: torch.Tensor) -> torch.Tensor:
    """The default integer answer: the op leaves such a value unchanged."""
    return input.clone()


_PREDICATE_FALLBACK_DTYPES = _MANIFEST_INT_DTYPES + (torch.bool,)


class _IntFallbackCall:
    """What ``_IntIdentityUnaryOp`` builds for a dtype the shipped kernels do not serve.

    Callable with the op's tensors, like a kernel, but not a ``Kernel``: ``autotune``
    walks past it. Only the in-tree path builds one — a target that registers the op is
    asked for a kernel instead.
    """

    def __init__(self, handler):
        """Build the op. Shapes and dtype are taken from the first call."""
        self._handler = handler

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        # Contiguous like the kernel path: the layout must not depend on which dtype
        # the op was handed, and it has to agree with the registered fake.
        return self._handler(input).contiguous()


class _IntIdentityUnaryOp(UnaryOp):
    """Base for unary ops whose manifest declares integer dtypes the shipped
    float-only kernels do not serve.

    Such a dtype builds ``_IntFallbackCall``; subclasses set ``_int_handler`` and
    ``_int_output_dtype``. Every other dtype goes to the kernel, which raises on its
    own dtype check. A ``kernel_map`` override that declares integer support in
    ``SUPPORTED_DTYPES`` is used instead — the choice is made in ``_build``, which
    only the in-tree path reaches.
    """

    _int_handler: Callable[[torch.Tensor], torch.Tensor] = staticmethod(_int_identity)
    _int_output_dtype: Optional[torch.dtype] = None
    # Subclasses may extend the fallback dtype set when the manifest
    # signature includes additional non-float dtypes (e.g. torch.bool for
    # the is{nan,inf,finite} predicates).
    _fallback_dtypes: tuple = _MANIFEST_INT_DTYPES

    def _build(self, dtype: torch.dtype, n_total: int):
        if dtype in type(self)._fallback_dtypes:
            impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
            supported = impl.SUPPORTED_DTYPES
            if supported is None or ctor_dtype in supported:
                return super()._build(dtype, n_total)
            return _IntFallbackCall(type(self)._int_handler)
        return super()._build(dtype, n_total)


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
        *,
        approximate: str = "none",
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        if approximate not in ("none", "tanh"):
            raise ValueError(
                f"{type(self).__name__}: approximate must be 'none' or 'tanh', got {approximate!r}"
            )
        self.approximate = approximate
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)
