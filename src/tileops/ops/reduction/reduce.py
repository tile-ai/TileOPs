"""Reduce ops: SumFwdOp, MeanFwdOp, AminFwdOp, AmaxFwdOp, ProdFwdOp, StdFwdOp, VarFwdOp, VarMeanFwdOp.

Each op reduces along the configured ``dim`` and supports arbitrary-rank input.
The ``dim`` parameter accepts ``int``, ``list[int]``, or ``tuple[int, ...]``
for multi-dim reduction. Constructor ``dim`` defaults to ``None`` (full
reduction) for the ten ops whose manifest declares ``default: null``;
``ProdFwdOp`` preserves ``dim=-1``.

The Op layer validates the input, normalizes its contiguity, and hands it over as the
manifest declares it. Moving the reduced axes to the end, flattening to ``(M, N)``, the
alignment padding and shaping the result back all belong to the kernel, so both sides of
the op/backend boundary speak the declared shape. Kernels are cached by shape, axes,
dtype and device, so one op instance handles varying shapes.
"""

import warnings
from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.reduce import ReduceKernel
from tileops.manifest.shape_rules import reduced_shape

from ..op_base import Op
from ._boundary import register_reduction_op
from ._multidim import EmptyDimPolicy, normalize_dim

# Op kinds that accept 0-D (scalar) input. The kernel path assumes
# ``ndim >= 1`` (and the Welford kernel's Bessel correction is undefined for
# ``N == 1``), so the Op layer computes the scalar result directly without
# invoking PyTorch's reduction ops. Mapping a degenerate single-element
# reduction to its closed-form result is pure arithmetic, not a fallback.
_SCALAR_REDUCE_KINDS = frozenset(
    {
        "sum",
        "mean",
        "amin",
        "amax",
        "prod",
        "std",
        "var",
        "var_mean",
        "all",
        "any",
        "count_nonzero",
    }
)


__all__ = [
    "AmaxFwdOp",
    "AminFwdOp",
    "MeanFwdOp",
    "ProdFwdOp",
    "StdFwdOp",
    "SumFwdOp",
    "VarFwdOp",
    "VarMeanFwdOp",
]


class _ReduceOpBase(Op):
    """Common base for all reduce ops (simple, Welford, argreduce, logical, vector_norm).

    Holds the shared init params, the reading of ``dim``, and the one place a kernel is
    resolved. Subclasses declare ``_op_kind``, ``_kernel_key``, ``_kernel_cls``, and
    override hooks as needed. ``forward`` is one call to the operator the op registers;
    an op whose returns are not a single tensor (``VarMeanFwdOp``) overrides
    ``_eager_forward``, which runs behind that operator.

    Hooks for subclass customization:

    - ``_kernel_key``: kernel map key (default ``"reduce"``).
    - ``_kernel_cls``: kernel class (default ``ReduceKernel``).
    - ``_validate_dim()``: validate ``dim`` at init (default: accept int/list/None).
    - ``_build_kernel_kwargs(x, axes)``: extra kwargs for the kernel constructor.
    """

    # Set by ``register_reduction_op`` on each concrete op; a base registers none.
    _wrapped = None

    _op_kind: str = ""  # overridden by subclasses
    _kernel_key: str = "reduce"  # overridden by subclasses for different kernel families
    _kernel_cls: type = ReduceKernel  # overridden by subclasses for different kernel classes
    _empty_dim_policy: EmptyDimPolicy = "reject"

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct a reduce op.

        Args:
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or
                ``None``.
            keepdim: Whether to retain reduced dims as size 1.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        self.dim = dim
        self.keepdim = keepdim
        self.target = target
        self.tune = tune
        self._validate_dim()
        self.dispatch_kernel(kernel_map)
        self._last_roofline_mn: tuple[int, int] | None = None

    def _infer_output_shapes(self, x_shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: the reduced axes leave, or stay as size 1."""
        return {"output": self._reduced_shape(x_shape)}

    def _reduced_shape(self, x_shape: tuple[int, ...]) -> tuple[int, ...]:
        """The output shape, read with this op's empty-``dim`` policy."""
        return reduced_shape(
            x_shape,
            self.dim,
            self.keepdim,
            "noop" if self._empty_dim_policy == "noop" else "full",
        )

    # Dim validation (subclasses may override)

    def _validate_dim(self) -> None:
        """Validate the ``dim`` parameter.

        Default: accept ``int``, ``list[int]``/``tuple[int]``, or ``None``.
        Subclasses that only support single-dim reduction (e.g. argreduce)
        should override to reject non-scalar values.

        ``bool`` values are rejected explicitly. Python's ``bool`` subclasses
        ``int`` (so ``isinstance(True, int)`` is true), but a boolean dim has
        no meaningful interpretation as a tensor axis and almost always
        signals a caller bug.
        """
        dim = self.dim
        if isinstance(dim, bool):
            raise TypeError(
                f"dim must not be bool (subclasses int but is not a valid axis), got {dim!r}"
            )
        if dim is None or isinstance(dim, int):
            return
        if isinstance(dim, (list, tuple)):
            for d in dim:
                if isinstance(d, bool) or not isinstance(d, int):
                    raise TypeError(f"All elements of dim must be int (not bool), got {dim!r}")
            return
        raise TypeError(
            f"dim must be int, list[int], tuple[int, ...], or None, got {type(dim).__name__}"
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_cls}

    # Forward (subclasses with non-standard returns, e.g. VarMeanFwdOp,
    # must override ``_eager_forward``)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the reduce op on *x* along the configured dim.

        One call to the operator this op registers: this is as far as dynamo traces.
        """
        return type(self)._wrapped(x, self._instance_key)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        scalar_out = self._maybe_scalar(x)
        if scalar_out is not None:
            return scalar_out
        noop_out = self._maybe_noop(x)
        if noop_out is not None:
            return noop_out
        x, kernel = self._prepare_input(x)
        return kernel(x)

    # Empty-dim no-op short-circuit

    def _noop_output_dtype(self) -> Optional[torch.dtype]:
        """Manifest-declared output dtype for the dtype-altering short-circuits.

        Consulted by both the empty-dim no-op path (``_maybe_noop``) and
        the scalar 0-D path (``_scalar_forward``) so the manifest output
        dtype contract is honored without dispatching to the kernel.
        Subclasses with a fixed output dtype (e.g. All/Any -> bool,
        CountNonzero -> int64) MUST override. The default ``None`` means
        "preserve input dtype".
        """
        return None

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate device, dtype, and rank of the forward input.

        Shared by ``_prepare_input`` and the ``dim=[]`` noop short-circuit
        so both paths enforce the same forward contract. Which devices a set of kernels
        runs on is the kernel's own statement, so no device kind is checked here.
        """
        self._validate_dtypes(x)
        self.dtype = x.dtype
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

    # Scalar (0-D) input fast path

    def _validate_scalar_dim(self) -> None:
        """Validate that ``self.dim`` is an accepted form for a 0-D input.

        PyTorch accepts ``None``, ``0``, ``-1``, ``()``, and ``[]`` on a
        0-D tensor, plus singleton list/tuple forms (``[0]``, ``(0,)``,
        ``[-1]``, ``(-1,)``). Integers outside ``{0, -1}`` raise
        ``IndexError``. Multi-entry sequences whose canonical dims
        collide (``0`` and ``-1`` both alias axis ``0`` on a 0-D tensor)
        raise ``RuntimeError`` to match PyTorch's
        ``"dim 0 appears multiple times in the list of dims"``.
        """
        dim = self.dim
        if dim is None:
            return
        if isinstance(dim, int):
            if dim not in (0, -1):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
                )
            return
        if isinstance(dim, (list, tuple)):
            seen: set = set()
            for d in dim:
                if d not in (0, -1):
                    raise IndexError(
                        f"Dimension out of range (expected to be in range of [-1, 0], but got {d})"
                    )
                canon = 0  # 0 and -1 alias the same axis on a 0-D tensor.
                if canon in seen:
                    raise RuntimeError(f"dim {canon} appears multiple times in the list of dims")
                seen.add(canon)
            return

    def _scalar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward result for a 0-D input natively.

        Single-element reductions are degenerate: every arithmetic family
        collapses to the input value, the logical families collapse to
        ``x != 0`` cast to the manifest output dtype, and the Welford
        family follows a closed form in ``correction``. This method
        computes the closed-form result directly so the kernel path
        (undefined for ``N == 1``) is bypassed without delegating to
        PyTorch's reduction ops.

        Arithmetic reductions (``sum``, ``mean``, ``amin``, ``amax``,
        ``prod``) over one element return the element itself. Logical /
        count ops override ``_noop_output_dtype`` so this default applies
        the ``x != 0`` predicate and casts to the declared output dtype.
        Welford ops (``std``, ``var``, ``var_mean``) override this hook
        because their result depends on ``correction``.
        """
        out_dtype = self._noop_output_dtype()
        if out_dtype is None:
            return x.clone()
        return (x != 0).to(out_dtype)

    def _maybe_scalar(self, x: torch.Tensor):
        """Short-circuit a 0-D input to the native scalar forward.

        Returns the scalar-path output when ``x.ndim == 0``; returns
        ``None`` otherwise so the caller proceeds with the kernel path.
        The roofline state is bound to ``(1, 1)`` so ``eval_roofline()``
        after a scalar forward stays well-defined.
        """
        if x.ndim != 0:
            return None
        if self._op_kind not in _SCALAR_REDUCE_KINDS:
            # Subclasses without a defined 0-D contract (e.g. argmax/argmin/
            # l1/l2/inf) fall through to the kernel path, which raises the
            # pre-existing ``ValueError("Input tensor must be at least 1D")``.
            return None
        self._validate_dtypes(x)
        self.dtype = x.dtype
        self._validate_scalar_dim()
        self._last_roofline_mn = (1, 1)
        return self._scalar_forward(x)

    def _maybe_noop(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return *x* (cast to the manifest output dtype) when ``dim`` is
        an empty list/tuple and the op's ``_empty_dim_policy`` is
        ``"noop"``; return ``None`` otherwise so the caller proceeds with
        the normal kernel path.

        Runs the same input validation as ``_prepare_input`` (CUDA / dtype
        / ndim) and binds ``_last_roofline_mn`` before short-circuiting, so
        the noop path still honors the public forward contract -- bad
        inputs raise, and ``eval_roofline()`` works after a noop forward.
        """
        if self._empty_dim_policy != "noop":
            return None
        if not isinstance(self.dim, (list, tuple)) or len(self.dim) != 0:
            return None
        self._validate_input_tensor(x)
        # Bind roofline state. The noop performs no reduction but still
        # reads every input element and writes an equal-shape result
        # (cast to bool for All/Any, the only ops whose ``_empty_dim_policy``
        # is ``"noop"``; other reduce ops, including ``CountNonzero``, keep
        # ``"full"`` and never enter this branch). Model this as a
        # degenerate reduction over an axis of length 1: M = numel, N = 1.
        # Under the existing per-op-kind
        # formulas this yields mem_bytes proportional to numel * elem_bytes
        # for the read plus the output term, instead of collapsing to
        # zero, which would under-count the actual data-movement cost.
        self._last_roofline_mn = (x.numel(), 1)
        # ``copy=True`` because the operator this runs inside may not return an alias of
        # its input, and ``to`` hands back the same object when the dtype already matches
        # — which a bool input to All or Any does.
        out_dtype = self._noop_output_dtype()
        return x.clone() if out_dtype is None else x.to(out_dtype, copy=True)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        M, N = self._last_roofline_mn
        if self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dtype"
            )
        elem_bytes = self.dtype.itemsize
        op_kind = self._op_kind

        if op_kind == "mean":
            flops = M * (N + 1)
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "std":
            flops = 5 * M * N + M
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "var":
            flops = 5 * M * N
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "var_mean":
            flops = 5 * M * N
            mem_bytes = (M * N + 2 * M) * elem_bytes
        elif op_kind in {"argmax", "argmin"}:
            flops = M * N
            mem_bytes = M * N * elem_bytes + M * 8
        elif op_kind in {"all", "any"}:
            flops = M * N
            mem_bytes = M * N * elem_bytes + M
        elif op_kind == "count_nonzero":
            flops = 2 * M * N
            mem_bytes = M * N * elem_bytes + M * 8
        elif op_kind == "l1":
            flops = 2 * M * N
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "l2":
            flops = 2 * M * N + M
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "inf":
            flops = 2 * M * N
            mem_bytes = (M * N + M) * elem_bytes
        else:
            flops = M * N
            mem_bytes = (M * N + M) * elem_bytes

        return flops, mem_bytes

    # Kernel cache

    def _build_kernel_kwargs(self, x: torch.Tensor, axes: "tuple[int, ...]") -> dict:
        """What this op's kernel takes beyond the shared arguments.

        The device is one of them: a kernel that plans against shared memory has to plan
        against the device the input lives on, not whichever one is current.
        """
        return {"device_index": x.device.index}

    def _select_kernel_key(
        self,
        x: torch.Tensor,
        axes: "tuple[int, ...]",
        m: int,
        n: int,
    ) -> str:
        """Choose the implementation key for this reduce call."""

        return self._kernel_key

    def _reduce_axes(self, x: torch.Tensor) -> "tuple[int, ...]":
        """The axes this call reduces, ascending and non-negative.

        Raises:
            IndexError: ``dim`` names an axis this rank does not have. Raised before any
                kernel is built, so an out-of-range call reaches no backend.
        """
        return tuple(normalize_dim(self.dim, x.ndim, empty_dim_policy=self._empty_dim_policy))

    # Input preparation (validate → normalize contiguity → resolve the kernel)

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Validate, normalize contiguity, and resolve the kernel for this call.

        The row layout the kernel wants is the kernel's business, so what comes back is
        the declared tensor and a kernel that takes it.

        Returns:
            ``(x, kernel)``, where *x* is the contiguous declared input.
        """
        self._validate_input_tensor(x)
        # Normalized here and handed over as the manifest declares it; how a kernel wants
        # that laid out is its own business.
        x = x.contiguous()
        axes = self._reduce_axes(x)
        # From the shape, not from ``numel``: an empty reduced axis makes ``n`` zero.
        n = prod(x.shape[a] for a in axes)
        m = prod(d for i, d in enumerate(x.shape) if i not in axes)
        self._last_roofline_mn = (m, n)
        extra = self._build_kernel_kwargs(x, axes)
        selected_key = self._select_kernel_key(x, axes, m, n)
        kernel = self.get_or_build_kernel(
            selected_key,
            (x,),
            # The kernel now owns the permute, so the whole shape decides what it is,
            # not just the row count and width it reduces. The device is in the key
            # because the kernel plans against that device's shared memory.
            key=(selected_key, tuple(x.shape), axes, self.keepdim, x.dtype, x.device.index),
            build=lambda: self.kernel_map[selected_key](
                m,
                n,
                self._op_kind,
                x.dtype,
                reduce_axes=axes,
                keepdim=self.keepdim,
                tune=self.tune,
                **extra,
            ),
        )
        return x, kernel


# Simple reduce ops (sum, mean, amin, amax, prod)


class _SimpleReduceOp(_ReduceOpBase):
    """Base for single-output reduce ops (sum, mean, amin, amax, prod).

    The ``dim`` default follows each op's manifest entry: ``sum``, ``mean``,
    ``amin``, and ``amax`` default to ``None`` (full reduction); ``prod``
    overrides to ``dim=-1`` and restricts the type to ``int``.

    Args:
        dim: Reduction dimension. Accepts ``int``, ``list[int]``,
            ``tuple[int, ...]``, or ``None`` on the base class; subclasses
            may narrow this (see ``ProdFwdOp``).
        keepdim: Whether to retain the reduced dimension as size 1.
        target: Which set of kernels serves this op — a target name, ``BUILTIN``
            for the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """


class SumFwdOp(_SimpleReduceOp):
    """Sum reduction along dim=-1."""

    _op_kind = "sum"
    _empty_dim_policy: EmptyDimPolicy = "full"


class MeanFwdOp(_SimpleReduceOp):
    """Mean reduction along dim=-1."""

    _op_kind = "mean"
    _empty_dim_policy: EmptyDimPolicy = "full"


class AminFwdOp(_SimpleReduceOp):
    """Amin (element-wise minimum) reduction along dim=-1."""

    _op_kind = "amin"
    _empty_dim_policy: EmptyDimPolicy = "full"


class AmaxFwdOp(_SimpleReduceOp):
    """Amax (element-wise maximum) reduction along dim=-1."""

    _op_kind = "amax"
    _empty_dim_policy: EmptyDimPolicy = "full"


class ProdFwdOp(_SimpleReduceOp):
    """Product reduction.

    Unlike the other simple reduce ops, ``ProdFwdOp`` defaults to
    ``dim=-1`` (manifest declares ``default: -1`` for ``prod``).
    """

    _op_kind = "prod"

    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct ProdFwdOp.

        Args:
            dim: Reduction dimension (default ``-1``).
            keepdim: Whether to retain reduced dims as size 1.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    def _validate_dim(self) -> None:
        # Manifest declares prod.signature.params.dim as int; reject the
        # multi-dim and full-reduction overloads inherited from the base.
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise TypeError(f"ProdFwdOp.dim must be int, got {type(self.dim).__name__}")


# Welford-based ops (std, var, var_mean)


class _WelfordReduceOp(_ReduceOpBase):
    """Base for Welford-based reduce ops (std, var, var_mean).

    Construction: ``op(dim=None, correction=1, keepdim=False)``.

    """

    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        correction: int = 1,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct a Welford-based reduce op.

        Args:
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or
                ``None``.
            correction: Bessel's correction (default 1).
            keepdim: Whether to retain reduced dims as size 1.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).

        Args:
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, or ``tuple[int, ...]`` for
                multi-dim reduction.
            correction: Bessel's correction (default 1).
            keepdim: Whether to retain the reduced dimension as size 1.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default False).
        """
        self.correction = correction
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    def _build_kernel_kwargs(self, x: torch.Tensor, axes: "tuple[int, ...]") -> dict:
        """Pass correction to the kernel constructor."""
        return {**super()._build_kernel_kwargs(x, axes), "correction": self.correction}

    def _scalar_forward(self, x: torch.Tensor):
        """Compute Welford ops on a 0-D input from closed-form.

        For a single-element reduction with reduction factor ``N = 1`` and
        Bessel ``correction``:

        - ``N - correction <= 0`` (i.e. ``correction >= 1``): variance
          and standard deviation are mathematically undefined; the
          contract returns ``nan`` (computed as ``x * nan`` so the
          output stays connected to ``x``).
        - ``correction == 0``: the unbiased denominator is ``N``, so the
          deviation from the mean (which equals the element itself) is
          zero for finite inputs. The result is computed as ``x - x``
          so non-finite inputs propagate (``nan`` / ``inf`` → ``nan``).

        ``VarMeanFwdOp`` overrides this hook to additionally return the
        mean (the input element).
        """
        if self.correction >= 1:
            warnings.warn(
                f"{self._op_kind}(): degrees of freedom is <= 0. Correction "
                "should be strictly less than the reduction factor (input "
                "numel divided by output numel).",
                UserWarning,
                stacklevel=2,
            )
            return x * float("nan")
        return x - x

    def _invalid_dof_output(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return PyTorch-compatible NaNs when ``N - correction <= 0``.

        The TileLang Welford kernel bakes ``N`` and ``correction`` into the
        generated code, so a zero denominator fails at compile time. PyTorch
        defines this degree-of-freedom case as NaN; handle it before kernel
        dispatch. The shape comes from the manifest's rule, the same source the
        kernel path's result is shaped by.
        """
        if self._last_roofline_mn is None:
            return None
        _M, N = self._last_roofline_mn
        if self.correction < N:
            return None
        shape = self._reduced_shape(tuple(x.shape))
        return torch.full(shape, float("nan"), dtype=x.dtype, device=x.device)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        scalar_out = self._maybe_scalar(x)
        if scalar_out is not None:
            return scalar_out
        noop_out = self._maybe_noop(x)
        if noop_out is not None:
            return noop_out
        x, kernel = self._prepare_input(x)
        invalid_dof = self._invalid_dof_output(x)
        if invalid_dof is not None:
            return invalid_dof
        return kernel(x)


class StdFwdOp(_WelfordReduceOp):
    """Standard deviation reduction with Bessel's correction."""

    _op_kind = "std"


class VarFwdOp(_WelfordReduceOp):
    """Variance reduction with Bessel's correction."""

    _op_kind = "var"


class VarMeanFwdOp(_WelfordReduceOp):
    """Variance and mean reduction."""

    def _infer_output_shapes(self, x_shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: both outputs are the reduced shape."""
        shape = self._reduced_shape(x_shape)
        return {"var": shape, "mean": shape}

    _op_kind = "var_mean"

    def _scalar_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(var, mean)`` on a 0-D input.

        Variance follows the Welford closed form (``nan`` when
        ``correction >= 1``, ``0`` otherwise); the mean of a single element
        is the element itself.
        """
        var_out = super()._scalar_forward(x)
        return var_out, x.clone()

    def _eager_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar_out = self._maybe_scalar(x)
        if scalar_out is not None:
            return scalar_out
        x, kernel = self._prepare_input(x)
        invalid_dof = self._invalid_dof_output(x)
        if invalid_dof is not None:
            axes = self._reduce_axes(x)
            mean_out = x.float().mean(dim=axes, keepdim=self.keepdim).to(x.dtype)
            return invalid_dof, mean_out.reshape(invalid_dof.shape)
        return kernel(x)


for _op_cls in (
    SumFwdOp,
    MeanFwdOp,
    AminFwdOp,
    AmaxFwdOp,
    ProdFwdOp,
    StdFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
):
    register_reduction_op(_op_cls)
