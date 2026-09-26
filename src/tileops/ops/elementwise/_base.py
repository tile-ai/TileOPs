"""Elementwise op infrastructure: umbrella bases, helpers, registration factories.

Three umbrella Op base classes, one per shape the family's kernels take:

- ``UnaryOp`` — one tensor in, the same shape out
- ``BinaryOp`` — two tensors broadcast against each other
- ``FusedGatedOp`` — one ``(M, 2N)`` tensor split into gate and value

The checks, output shapes and roofline are generated from each op's manifest
signature, which also registers its compile-boundary operators. An op normalizes
contiguity and hands the *manifest-declared* shapes to its kernel; flattening,
broadcasting and restoring the output shape are the kernel's own business. Element
type is not a construction parameter: an instance serves whichever dtype its caller
passes, one specialization per element type, built on first use.
"""

import math
from typing import Callable, ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel

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


class _PerDtypeKernels:
    """The family's one way to reach a kernel: ``self._kernel(inputs, dtype, *dims)``.

    A subclass supplies ``_build(dtype, *dims)`` for one specialization. What comes
    back is called with the manifest-declared tensors, so the in-tree path and a
    target's path hand back the same kind of thing.
    """

    @property
    def _slot(self) -> str:
        """The one dispatch key this op's ``kernel_map`` holds; also its memory role."""
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
        return self.kernel_for(self._slot, inputs, (dtype, *dims))

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built for the dtype and the extents it bakes in."""
        return call, lambda: self._build(*call)

    def _build(self, dtype: torch.dtype, *dims):
        """Construct the in-tree kernel for one specialization."""
        raise NotImplementedError(f"{type(self).__name__} must implement _build")

    def _check_kernel_dtype(self, impl: type, dtype: torch.dtype, compute: torch.dtype) -> None:
        """Refuse a dtype the selected kernel class does not serve."""
        supported = impl.SUPPORTED_DTYPES
        if supported is not None and compute not in supported:
            names = ", ".join(str(dt) for dt in supported)
            raise ValueError(f"{self._slot} does not support dtype {dtype}. Supported: [{names}]")


class UnaryOp(_PerDtypeKernels, Op):
    """Template base class for unary elementwise ops.

    A subclass sets ``kernel_types``, its one dispatch key. The element count arrives
    with the tensor, so nothing about shape is a construction parameter.
    """

    compile_boundary: ClassVar[bool] = True

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
        self.dispatch_kernel(kernel_map)

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

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        input = input.contiguous()
        return self._kernel((input,), input.dtype, input.numel())(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)


class BinaryOp(_PerDtypeKernels, Op):
    """Template base class for binary elementwise ops with broadcast.

    A subclass sets ``kernel_types``, its one dispatch key. Both operand shapes arrive
    with the tensors; the broadcast *lowering* — dim coalescing and stride synthesis —
    is the kernel's, so this class only hands the two shapes down.
    """

    compile_boundary: ClassVar[bool] = True

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
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, a_shape: tuple, b_shape: tuple):
        """Build one specialization for the semantic *dtype* and this broadcast."""
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        self._check_kernel_dtype(impl, dtype, ctor_dtype)
        return self._build_kernel_instance(self.tune, ctor_dtype, impl, a_shape, b_shape)

    def _build_kernel_instance(self, tune, dtype, impl, a_shape, b_shape):
        """Construct the kernel. Subclasses override to inject extra kwargs."""
        return impl(a_shape, b_shape, dtype, tune=tune)

    def _eager_forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        input = input.contiguous()
        other = other.contiguous()
        kernel = self._kernel((input, other), input.dtype, tuple(input.shape), tuple(other.shape))
        return kernel(input, other)

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``other``."""
        return self._call_boundary(input, other)


class FusedGatedOp(_PerDtypeKernels, Op):
    """Template base class for fused gated elementwise ops.

    Input: x of shape (M, 2*N). gate = x[:, :N], value = x[:, N:].
    Output: y = activation(gate) * value, shape (M, N).

    A subclass sets ``kernel_types``, its one dispatch key. Both dimensions arrive with
    the tensor.
    """

    compile_boundary: ClassVar[bool] = True

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
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, m: int, n: int):
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        self._check_kernel_dtype(impl, dtype, ctor_dtype)
        return impl(m, n, ctor_dtype, tune=self.tune)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        x = x.contiguous()
        return self._kernel((x,), x.dtype, x.shape[0], x.shape[1] // 2)(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the op on ``x``."""
        return self._call_boundary(x)


# Intermediate (private) base classes shared by leaf op modules


class _UnaryActivationMixin:
    """The ``inplace`` switch of a unary activation.

    The kernel writes a fresh buffer; with ``inplace`` set the result is copied back into
    ``input`` and ``input`` is returned, so callers see ``y is x``. The manifest marks
    ``input`` written exactly when ``inplace`` holds, which selects the operator that
    carries the write in a traced graph.
    """

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        result = super()._eager_forward(input)
        if not self.inplace:
            return result
        input.copy_(result)
        return input


class _ParamFreeActivationOp(_UnaryActivationMixin, UnaryOp):
    """Shared base for the activations whose only parameter is ``inplace``.

    ReLU, SiLU, HardSwish, HardSigmoid, Mish and SELU: each leaf declares only its
    ``kernel_types`` and docstring.
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
            inplace: When True, write the result into ``input`` and return ``input``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.inplace = inplace
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)


class _ParametricActivationOp(UnaryOp):
    """Shared base for the activations with scalar construction parameters.

    LeakyReLU, ELU, Hardtanh and Softplus. Leaves own their ``__init__`` because scalar
    names and defaults vary per leaf: each records its parameters on ``self``, then
    delegates to ``UnaryOp.__init__``.
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
            _validate_scalar_param_repr(name, value, dtype, self._slot)
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
            alpha: Multiplier on ``other`` (default 1).
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.alpha = alpha
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _build(self, dtype: torch.dtype, a_shape: tuple, b_shape: tuple):
        # torch.add / torch.sub refuse a floating-point alpha for integral and bool inputs.
        if isinstance(self.alpha, float) and not dtype.is_floating_point:
            raise ValueError(
                f"{type(self).__name__}: alpha={self.alpha!r} is a float, which a {dtype} "
                "input does not take"
            )
        return super()._build(dtype, a_shape, b_shape)

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
        """Serve a call with *handler*, a function of the op's input."""
        self._handler = handler

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        # Contiguous like the kernel path: the layout must not depend on which dtype
        # the op was handed.
        return self._handler(input).contiguous()


class _IntIdentityUnaryOp(UnaryOp):
    """Base for unary ops whose manifest declares integer dtypes the shipped
    float-only kernels do not serve.

    Such a dtype builds ``_IntFallbackCall``; subclasses set ``_int_handler``. Every
    other dtype goes to the kernel, which raises on its own dtype check. A
    ``kernel_map`` override that declares integer support in ``SUPPORTED_DTYPES`` is
    used instead — the choice is made in ``_build``, which only the in-tree path
    reaches.
    """

    _int_handler: Callable[[torch.Tensor], torch.Tensor] = staticmethod(_int_identity)
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
