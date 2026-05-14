"""Reduce ops: SumFwdOp, MeanFwdOp, AminFwdOp, AmaxFwdOp, ProdFwdOp, StdFwdOp, VarFwdOp, VarMeanFwdOp.

Each op reduces along the configured ``dim`` and supports arbitrary-rank input.
The ``dim`` parameter accepts ``int``, ``list[int]``, or ``tuple[int, ...]``
for multi-dim reduction. Constructor ``dim`` defaults to ``None`` (full
reduction) for the ten ops whose manifest declares ``default: null``;
``ProdFwdOp`` preserves ``dim=-1``.
The Op layer validates inputs, reshapes to 2D (M, N), and calls the kernel.
For simple, Welford, logical reduce, and vector norm ops, alignment padding is
handled inside the kernel via masked loads with identity-element fills,
eliminating host-side ``F.pad`` from the forward path. Other ops that inherit
``_ReduceOpBase`` continue to use host-side padding until their kernels are
converted.
Kernels are cached by ``(M, N)`` so that the same op instance can handle
varying shapes.
"""

import warnings
from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.reduce import ReduceKernel

from ..op_base import Op
from ._multidim import EmptyDimPolicy, flatten_for_multidim, normalize_dim, restore_multidim_shape

# Op kinds that accept 0-D (scalar) input. The kernel path assumes
# ``ndim >= 1`` (and the Welford kernel's Bessel correction is undefined for
# ``N == 1``), so the Op layer computes the scalar result directly without
# invoking PyTorch's reduction ops. Mapping a degenerate single-element
# reduction to its closed-form result is pure arithmetic, not a fallback.
_SCALAR_REDUCE_KINDS = frozenset({
    "sum", "mean", "amin", "amax", "prod", "std", "var", "var_mean",
    "all", "any", "count_nonzero",
})


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


# ---------------------------------------------------------------------------
# Shared base class for all reduce ops
# ---------------------------------------------------------------------------


class _ReduceOpBase(Op):
    """Common base for all reduce ops (simple, Welford, argreduce, logical, vector_norm).

    Consolidates shared init params (dtype, dim, keepdim, tune), initializes
    and owns an internal kernel cache, and handles input preparation
    (validate, transpose, reshape to 2D, pad) and output reshaping.
    Subclasses declare ``_op_kind``, ``_kernel_key``, ``_kernel_cls``, and
    override hooks as needed.  ``forward()`` is provided by this base class;
    only ops with non-standard returns (e.g. ``VarMeanFwdOp``) need to
    override it.

    Hooks for subclass customization:

    - ``_kernel_key``: kernel map key (default ``"reduce"``).
    - ``_kernel_cls``: kernel class (default ``ReduceKernel``).
    - ``_validate_dim()``: validate ``dim`` at init (default: accept int/list/None).
    - ``_pad_value()``: identity element for alignment padding (default ``0.0``).
    - ``_build_kernel_kwargs()``: extra kwargs for kernel constructor.
    - ``_pre_kernel(x)``: transform 2D input before kernel call (default identity).
      Returns ``(x, context)`` where *context* is passed to ``_post_kernel``.
    - ``_post_kernel(y, context)``: transform kernel output (default identity).
    """

    _op_kind: str = ""  # overridden by subclasses
    _kernel_key: str = "reduce"  # overridden by subclasses for different kernel families
    _kernel_cls: type = ReduceKernel  # overridden by subclasses for different kernel classes
    _kernel_handles_padding: bool = False  # True when kernel accepts (M, N) with masked loads
    _empty_dim_policy: EmptyDimPolicy = "reject"

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct a reduce op.

        Args:
            dtype: Input data type.
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or
                ``None``.
            keepdim: Whether to retain reduced dims as size 1.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        self.dtype = dtype
        self.dim = dim
        self.keepdim = keepdim
        self._tune = tune
        self._validate_dim()
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}
        self._last_roofline_mn: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Dim validation (subclasses may override)
    # ------------------------------------------------------------------

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
                f"dim must not be bool (subclasses int but is not a valid "
                f"axis), got {dim!r}"
            )
        if dim is None or isinstance(dim, int):
            return
        if isinstance(dim, (list, tuple)):
            for d in dim:
                if isinstance(d, bool) or not isinstance(d, int):
                    raise TypeError(
                        f"All elements of dim must be int (not bool), "
                        f"got {dim!r}"
                    )
            return
        raise TypeError(
            f"dim must be int, list[int], tuple[int, ...], or None, "
            f"got {type(dim).__name__}"
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_cls}

    # ------------------------------------------------------------------
    # Pad value (subclasses may override; used only when
    # _kernel_handles_padding is False)
    # ------------------------------------------------------------------

    def _pad_value(self) -> float:
        """Return the identity element used when padding to alignment.

        Only used when ``_kernel_handles_padding`` is ``False`` (i.e. the
        kernel expects pre-padded input from the Op layer).
        """
        return 0.0

    # ------------------------------------------------------------------
    # Extra kernel kwargs (subclasses may override)
    # ------------------------------------------------------------------

    def _build_kernel_kwargs(self) -> dict:
        """Return extra keyword arguments for the kernel constructor.

        Override in subclasses to pass additional params like ``correction``.
        """
        return {}

    # ------------------------------------------------------------------
    # Pre/post kernel hooks (subclasses may override)
    # ------------------------------------------------------------------

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Transform 2D input before kernel call.

        Returns ``(x, context)`` where *context* is an opaque value
        passed through to ``_post_kernel``.  Default: identity.
        """
        return x, None

    def _post_kernel(self, y: torch.Tensor, context: object) -> torch.Tensor:
        """Transform kernel output.  Default: identity."""
        return y

    # ------------------------------------------------------------------
    # Forward (subclasses with non-standard returns, e.g. VarMeanFwdOp,
    # must override)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the reduce op on *x* along the configured dim."""
        scalar_out = self._maybe_scalar(x)
        if scalar_out is not None:
            return scalar_out
        noop_out = self._maybe_noop(x)
        if noop_out is not None:
            return noop_out
        x, orig_shape, dim_info, kernel = self._prepare_input(x)
        x, ctx = self._pre_kernel(x)
        y = kernel(x)
        y = self._post_kernel(y, ctx)
        return self._reshape_output(y, orig_shape, dim_info)

    # ------------------------------------------------------------------
    # Empty-dim no-op short-circuit
    # ------------------------------------------------------------------

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
        so both paths enforce the same forward contract.
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

    # ------------------------------------------------------------------
    # Scalar (0-D) input fast path
    # ------------------------------------------------------------------

    def _validate_scalar_dim(self) -> None:
        """Validate that ``self.dim`` is an accepted form for a 0-D input.

        PyTorch accepts ``None``, ``0``, ``-1``, ``()``, and ``[]`` on a
        0-D tensor, plus singleton list/tuple forms (``[0]``, ``(0,)``,
        ``[-1]``, ``(-1,)``). Integers outside ``{0, -1}`` raise
        ``IndexError`` here; multi-entry sequences (e.g. ``[0, -1]``)
        are passed through to torch, which may raise ``RuntimeError`` on
        duplicate dims because ``0`` and ``-1`` alias the same axis on a
        0-D tensor.
        """
        dim = self.dim
        if dim is None:
            return
        if isinstance(dim, int):
            if dim not in (0, -1):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[-1, 0], but got {dim})"
                )
            return
        if isinstance(dim, (list, tuple)):
            for d in dim:
                if d not in (0, -1):
                    raise IndexError(
                        f"Dimension out of range (expected to be in range of "
                        f"[-1, 0], but got {d})"
                    )
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
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
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
        out_dtype = self._noop_output_dtype()
        if out_dtype is None:
            return x
        return x.to(out_dtype)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        M, N = self._last_roofline_mn
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

    # ------------------------------------------------------------------
    # Kernel cache
    # ------------------------------------------------------------------

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed."""
        key = (M, N)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map[self._kernel_key]
            self._kernel_cache[key] = kernel_cls(
                M, N, self._op_kind, self.dtype,
                tune=self._tune, **self._build_kernel_kwargs(),
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Input preparation (validate → transpose → reshape → pad)
    # ------------------------------------------------------------------

    def _prepare_input(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Size, object, object]:
        """Validate, derive M/N, transpose, reshape to 2D, optionally pad.

        Returns ``(x_2d, orig_shape, dim_info, kernel)`` where
        *dim_info* is either an ``int`` (single-dim) or ``list[int]``
        (multi-dim).

        When ``_kernel_handles_padding`` is ``True``, the raw ``(M, N)``
        tensor is passed through -- the kernel handles alignment internally
        via masked loads.  Otherwise, host-side ``F.pad`` is applied for
        backward compatibility with kernels that expect ``(M, N_padded)``.
        """
        self._validate_input_tensor(x)

        orig_shape = x.shape

        # --- multi-dim path (includes dim=None for full reduction) ---
        if isinstance(self.dim, (list, tuple)) or self.dim is None:
            dims = normalize_dim(
                self.dim, x.ndim, empty_dim_policy=self._empty_dim_policy,
            )
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            self._last_roofline_mn = (M, N)
            x = x.reshape(M, N)
            kernel = self._get_or_create_kernel(M, N)
            if not self._kernel_handles_padding:
                N_padded = align_up(N, DEFAULT_ALIGNMENT)
                if N_padded != N:
                    pv = self._pad_value()
                    pad = (0, N_padded - N)
                    x = F.pad(x, pad) if pv == 0.0 else F.pad(x, pad, value=pv)
            return x, orig_shape, dims, kernel

        # --- single-dim path ---
        if self.dim < -x.ndim or self.dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {self.dim})"
            )
        dim = self.dim % x.ndim

        N = x.shape[dim]
        M = prod(s for i, s in enumerate(x.shape) if i != dim)
        self._last_roofline_mn = (M, N)

        if dim != x.ndim - 1:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        kernel = self._get_or_create_kernel(M, N)

        if not self._kernel_handles_padding:
            N_padded = align_up(N, DEFAULT_ALIGNMENT)
            if N_padded != N:
                pv = self._pad_value()
                pad = (0, N_padded - N)
                x = F.pad(x, pad) if pv == 0.0 else F.pad(x, pad, value=pv)

        return x, orig_shape, dim, kernel

    # ------------------------------------------------------------------
    # Output reshape
    # ------------------------------------------------------------------

    def _reshape_output(
        self, y: torch.Tensor, orig_shape: torch.Size, dim_info: Union[int, List[int]],
    ) -> torch.Tensor:
        """Reshape (M,) kernel output to match keepdim setting.

        *dim_info* is either an ``int`` (single-dim) or ``list[int]``
        (multi-dim).
        """
        if isinstance(dim_info, list):
            return restore_multidim_shape(y, orig_shape, dim_info, self.keepdim)

        dim = dim_info
        if self.keepdim:
            kept_shape = list(orig_shape)
            kept_shape[dim] = 1
            return y.reshape(kept_shape)
        else:
            reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
            return y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)


# ---------------------------------------------------------------------------
# Simple reduce ops (sum, mean, amin, amax, prod)
# ---------------------------------------------------------------------------


class _SimpleReduceOp(_ReduceOpBase):
    """Base for single-output reduce ops (sum, mean, amin, amax, prod).

    M and N are derived from the input tensor at forward time, and kernels
    are cached by ``(M, N)`` to avoid rebuilds. Alignment padding is handled
    inside the kernel via masked loads with identity-element fills, so no
    host-side ``F.pad`` is needed.

    The ``dim`` default follows each op's manifest entry: ``sum``, ``mean``,
    ``amin``, and ``amax`` default to ``None`` (full reduction); ``prod``
    overrides to ``dim=-1`` and restricts the type to ``int``.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension. Accepts ``int``, ``list[int]``,
            ``tuple[int, ...]``, or ``None`` on the base class; subclasses
            may narrow this (see ``ProdFwdOp``).
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _kernel_handles_padding = True


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
        *,
        dtype: torch.dtype,
        dim: int = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct ProdFwdOp.

        Args:
            dtype: Input data type.
            dim (int): reduction dimension (default ``-1``).
            keepdim: Whether to retain reduced dims as size 1.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        super().__init__(
            dtype=dtype, dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _validate_dim(self) -> None:
        # Manifest declares prod.signature.params.dim as int; reject the
        # multi-dim and full-reduction overloads inherited from the base.
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise TypeError(
                f"ProdFwdOp.dim must be int, got "
                f"{type(self.dim).__name__}"
            )


# ---------------------------------------------------------------------------
# Welford-based ops (std, var, var_mean)
# ---------------------------------------------------------------------------


class _WelfordReduceOp(_ReduceOpBase):
    """Base for Welford-based reduce ops (std, var, var_mean).

    Construction: ``op(dtype=..., dim=None, correction=1, keepdim=False)``.
    M and N are derived from the input tensor at forward time, and kernels
    are cached by ``(M, N)`` to avoid rebuilds.

    Alignment padding is handled inside the kernel via masked loads,
    so no host-side ``F.pad`` is needed.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default ``None``, i.e. full reduction).
            Accepts ``int``, ``list[int]``, or ``tuple[int, ...]`` for
            multi-dim reduction.
        correction: Bessel's correction (default 1).
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _kernel_handles_padding = True
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        correction: int = 1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct a Welford-based reduce op.

        Args:
            dtype: Input data type.
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or
                ``None``.
            correction: Bessel's correction (default 1).
            keepdim: Whether to retain reduced dims as size 1.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        self.correction = correction
        super().__init__(
            dtype=dtype, dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _build_kernel_kwargs(self) -> dict:
        """Pass correction to the kernel constructor."""
        return {"correction": self.correction}

    def _scalar_forward(self, x: torch.Tensor):
        """Compute Welford ops on a 0-D input from closed-form.

        For a single-element reduction with reduction factor ``N = 1`` and
        Bessel ``correction``:

        - ``N - correction <= 0`` (i.e. ``correction >= 1``): variance and
          standard deviation are mathematically undefined; the contract
          returns ``nan`` to match the standard reduction convention.
        - ``correction == 0``: the unbiased denominator is ``N``, so the
          deviation from the mean (which equals the element itself) is
          zero for finite inputs. The result is computed as ``x - x`` so
          non-finite inputs propagate (``nan`` / ``inf`` → ``nan``) and
          autograd history on ``x`` is preserved.

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
            return torch.full((), float("nan"), dtype=x.dtype, device=x.device)
        return x - x

    def _invalid_dof_output(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return PyTorch-compatible NaNs when ``N - correction <= 0``.

        The TileLang Welford kernel bakes ``N`` and ``correction`` into the
        generated code, so a zero denominator fails at compile time. PyTorch
        defines this degree-of-freedom case as NaN; handle it before kernel
        dispatch.
        """
        if self._last_roofline_mn is None:
            return None
        M, N = self._last_roofline_mn
        if self.correction < N:
            return None
        return torch.full((M,), float("nan"), dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalar_out = self._maybe_scalar(x)
        if scalar_out is not None:
            return scalar_out
        noop_out = self._maybe_noop(x)
        if noop_out is not None:
            return noop_out
        x, orig_shape, dim_info, kernel = self._prepare_input(x)
        invalid_dof = self._invalid_dof_output(x)
        if invalid_dof is not None:
            return self._reshape_output(invalid_dof, orig_shape, dim_info)
        y = kernel(x)
        return self._reshape_output(y, orig_shape, dim_info)


class StdFwdOp(_WelfordReduceOp):
    """Standard deviation reduction with Bessel's correction."""

    _op_kind = "std"


class VarFwdOp(_WelfordReduceOp):
    """Variance reduction with Bessel's correction."""

    _op_kind = "var"


class VarMeanFwdOp(_WelfordReduceOp):
    """Variance and mean reduction."""

    _op_kind = "var_mean"

    def _scalar_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(var, mean)`` on a 0-D input.

        Variance follows the Welford closed form (``nan`` when
        ``correction >= 1``, ``0`` otherwise); the mean of a single element
        is the element itself.
        """
        var_out = super()._scalar_forward(x)
        return var_out, x.clone()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar_out = self._maybe_scalar(x)
        if scalar_out is not None:
            return scalar_out
        x, orig_shape, dim_info, kernel = self._prepare_input(x)
        invalid_dof = self._invalid_dof_output(x)
        if invalid_dof is not None:
            mean_out = x.float().mean(dim=-1).to(x.dtype)
            return (
                self._reshape_output(invalid_dof, orig_shape, dim_info),
                self._reshape_output(mean_out, orig_shape, dim_info),
            )
        var_out, mean_out = kernel(x)
        return (
            self._reshape_output(var_out, orig_shape, dim_info),
            self._reshape_output(mean_out, orig_shape, dim_info),
        )
