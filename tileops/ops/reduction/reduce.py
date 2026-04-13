"""Reduce ops: SumFwdOp, MeanFwdOp, AminFwdOp, AmaxFwdOp, ProdFwdOp, StdFwdOp, VarFwdOp, VarMeanFwdOp.

Each op reduces along the configured ``dim`` and supports arbitrary-rank input.
The ``dim`` parameter accepts ``int`` or ``list[int]`` for multi-dim reduction.
The Op layer validates inputs, reshapes to 2D (M, N), and calls the kernel.
For simple and Welford reduce ops, alignment padding is handled inside the
kernel via masked loads with identity-element fills, eliminating host-side
``F.pad`` from the forward path.  Other ops that inherit ``_ReduceOpBase``
(argreduce, logical, vector_norm) continue to use host-side padding until
their kernels are converted.
Kernels are cached by ``(M, N)`` so that the same op instance can handle
varying shapes.
"""

from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.reduce import ReduceKernel

from ..op import Op
from ._multidim import flatten_for_multidim, normalize_dim, restore_multidim_shape

__all__ = [
    "SumFwdOp",
    "MeanFwdOp",
    "AminFwdOp",
    "AmaxFwdOp",
    "ProdFwdOp",
    "StdFwdOp",
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

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.dtype = dtype
        self.dim = dim
        self.keepdim = keepdim
        self._tune = tune
        self._validate_dim()
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}

    # ------------------------------------------------------------------
    # Dim validation (subclasses may override)
    # ------------------------------------------------------------------

    def _validate_dim(self) -> None:
        """Validate the ``dim`` parameter.

        Default: accept ``int``, ``list[int]``/``tuple[int]``, or ``None``.
        Subclasses that only support single-dim reduction (e.g. argreduce)
        should override to reject non-scalar values.
        """
        dim = self.dim
        if dim is None or isinstance(dim, int):
            return
        if isinstance(dim, (list, tuple)):
            if not all(isinstance(d, int) for d in dim):
                raise TypeError(
                    f"All elements of dim must be int, got {dim}"
                )
            return
        raise TypeError(
            f"dim must be int, list[int], or None, got {type(dim).__name__}"
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
        x, orig_shape, dim_info, kernel = self._prepare_input(x)
        x, ctx = self._pre_kernel(x)
        y = kernel(x)
        y = self._post_kernel(y, ctx)
        return self._reshape_output(y, orig_shape, dim_info)

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
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

        orig_shape = x.shape

        # --- multi-dim path (includes dim=None for full reduction) ---
        if isinstance(self.dim, (list, tuple)) or self.dim is None:
            dims = normalize_dim(self.dim, x.ndim)
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
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

    Construction: ``op(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Alignment padding is handled inside the kernel via masked loads with
    identity-element fills, so no host-side ``F.pad`` is needed.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _kernel_handles_padding = True


class SumFwdOp(_SimpleReduceOp):
    """Sum reduction along dim=-1."""

    _op_kind = "sum"


class MeanFwdOp(_SimpleReduceOp):
    """Mean reduction along dim=-1."""

    _op_kind = "mean"


class AminFwdOp(_SimpleReduceOp):
    """Amin (element-wise minimum) reduction along dim=-1."""

    _op_kind = "amin"


class AmaxFwdOp(_SimpleReduceOp):
    """Amax (element-wise maximum) reduction along dim=-1."""

    _op_kind = "amax"


class ProdFwdOp(_SimpleReduceOp):
    """Product reduction along dim=-1."""

    _op_kind = "prod"


# ---------------------------------------------------------------------------
# Welford-based ops (std, var, var_mean)
# ---------------------------------------------------------------------------


class _WelfordReduceOp(_ReduceOpBase):
    """Base for Welford-based reduce ops (std, var, var_mean).

    Construction: ``op(dtype=..., dim=-1, correction=1, keepdim=False)``.
    M and N are derived from the input tensor at forward time, and kernels
    are cached by ``(M, N)`` to avoid rebuilds.

    Alignment padding is handled inside the kernel via masked loads,
    so no host-side ``F.pad`` is needed.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
        correction: Bessel's correction (default 1).
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _kernel_handles_padding = True

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
        correction: int = 1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.correction = correction
        super().__init__(
            dtype=dtype, dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _build_kernel_kwargs(self) -> dict:
        """Pass correction to the kernel constructor."""
        return {"correction": self.correction}


class StdFwdOp(_WelfordReduceOp):
    """Standard deviation reduction with Bessel's correction."""

    _op_kind = "std"


class VarFwdOp(_WelfordReduceOp):
    """Variance reduction with Bessel's correction."""

    _op_kind = "var"


class VarMeanFwdOp(_WelfordReduceOp):
    """Variance and mean reduction."""

    _op_kind = "var_mean"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, orig_shape, dim_info, kernel = self._prepare_input(x)
        var_out, mean_out = kernel(x)
        return (
            self._reshape_output(var_out, orig_shape, dim_info),
            self._reshape_output(mean_out, orig_shape, dim_info),
        )
