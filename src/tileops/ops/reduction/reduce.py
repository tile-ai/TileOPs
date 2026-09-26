"""Reduce ops: SumFwdOp, MeanFwdOp, AminFwdOp, AmaxFwdOp, ProdFwdOp, StdFwdOp, VarFwdOp, VarMeanFwdOp.

Each op reduces the axes ``dim`` names of an arbitrary-rank input. The generated signature
checks have run before ``_eager_forward``: dtype, ``dim`` range and uniqueness, and every
refinement. The op normalizes contiguity and hands the input over as the manifest declares
it; moving the reduced axes to the end, flattening to ``(M, N)`` and shaping the result back
belong to the kernel. Kernels are cached by shape, axes, dtype and device.
"""

import math
import warnings
from typing import ClassVar, Dict, List, Mapping, Optional, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.reduction.reduce import ReduceKernel
from tileops.manifest.primitives import normalize_axis, reduced

from ..op_base import Op

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

Dim = Union[int, List[int], Tuple[int, ...], None]


def reduce_axes(dim: Dim, rank: int, empty: str) -> "tuple[int, ...]":
    """The axes *dim* names at *rank*, ascending and non-negative.

    ``None`` names every axis; an empty sequence names every axis when *empty* is
    ``'full'`` and none when it is ``'noop'``.
    """
    if dim is None:
        return tuple(range(rank))
    dims = [dim] if isinstance(dim, int) else list(dim)
    if not dims:
        return tuple(range(rank)) if empty == "full" else ()
    return tuple(sorted({normalize_axis(d, rank) for d in dims}))


class _ReduceOpBase(Op):
    """Shared call flow of the reductions (simple, Welford, argreduce, logical, vector norm).

    A subclass declares ``_op_kind``, ``_kernel_key``, ``kernel_types`` and the empty-``dim``
    mode of its manifest output shape, and overrides the hooks below where it differs.

    - ``_output_dtype(x)``: the output dtype; the input's by default.
    - ``_identity``: the result over an empty reduced extent.
    - ``_scalar_forward(x)``: the result on a 0-d input.
    - ``_build_kernel_kwargs(shape, axes, device_index)``: extra kernel constructor kwargs.
    - ``_call_kwargs(n)``: kernel constructor kwargs that depend on the call's extent.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"reduce": ReduceKernel}

    _op_kind: str = ""
    _kernel_key: str = "reduce"
    # The manifest's empty-``dim`` mode of ``reduced``: ``'full'`` or ``'noop'``.
    _empty: str = "full"
    _identity: "float | bool | int" = 0

    def __init__(
        self,
        dim: Dim = None,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct a reduce op.

        Args:
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        self.dim = dim
        self.keepdim = keepdim
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce *x* over the configured axes.

        Args:
            x: Input tensor of any rank.

        Returns:
            The reduction, shaped by ``dim`` and ``keepdim``.
        """
        return self._call_boundary(x)

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        """*x* in the dtype the reduction runs in: the ``dtype`` parameter's when passed."""
        dtype = getattr(self, "dtype", None)
        return x if dtype is None or x.dtype == dtype else x.to(dtype)

    def _output_dtype(self, x: torch.Tensor) -> torch.dtype:
        return x.dtype

    def _output_shape(self, x: torch.Tensor) -> "tuple[int, ...]":
        return reduced(tuple(x.shape), self.dim, self.keepdim, self._empty)

    def _scalar_forward(self, x: torch.Tensor):
        """A 0-d input reduces one element: the element itself."""
        return x.clone()

    def _eager_forward(self, x: torch.Tensor):
        """Resolve the kernel and launch, inside the operator; closed forms need no kernel."""
        x = self._cast(x)
        if x.ndim == 0:
            return self._scalar_forward(x)
        axes = reduce_axes(self.dim, x.ndim, self._empty)
        if not axes:
            return self._noop_forward(x)
        if x.numel() == 0:
            return self._empty_forward(x)
        x = x.contiguous()
        n = math.prod(x.shape[a] for a in axes)
        m = x.numel() // n
        return self._launch(x, axes, m, n)

    def _noop_forward(self, x: torch.Tensor):
        """An empty ``dim`` under the ``'noop'`` mode keeps every element."""
        return x.to(self._output_dtype(x), copy=True)

    def _empty_forward(self, x: torch.Tensor):
        """An empty input: the identity over each empty reduced extent, or an empty output."""
        return torch.full(
            self._output_shape(x), self._identity, dtype=self._output_dtype(x), device=x.device
        )

    def _launch(self, x: torch.Tensor, axes: "tuple[int, ...]", m: int, n: int):
        return self.kernel_for("reduce", (x,), self._call(x, axes, m, n))(x)

    def _build_kernel_kwargs(
        self, shape: "tuple[int, ...]", axes: "tuple[int, ...]", device_index: "int | None"
    ) -> dict:
        """What this op's kernel takes beyond the shared arguments.

        The device is one of them: a kernel that plans against shared memory has to plan
        against the device the input lives on, not whichever one is current.
        """
        return {"device_index": device_index}

    def _call_kwargs(self, n: int) -> tuple:
        """Kernel constructor arguments this call decides, as ``(name, value)`` pairs."""
        return ()

    def _call(self, x: torch.Tensor, axes: "tuple[int, ...]", m: int, n: int) -> object:
        """What this call is, for :meth:`entry_for`: the facts the kernel is built from."""
        return (
            tuple(x.shape),
            axes,
            self.keepdim,
            x.dtype,
            x.device.index,
            m,
            n,
            self._call_kwargs(n),
        )

    def entry_for(self, role: str, call: object) -> Entry:
        """One implementation, built from the whole shape and the axes it reduces.

        The kernel owns the permute, so the whole shape decides what it is. The device is
        in the identity because the kernel plans against that device's shared memory.
        """
        shape, axes, keepdim, dtype, device_index, m, n, extra = call
        cls = self.kernel_map[self._kernel_key]
        return call, lambda: cls(
            m,
            n,
            self._op_kind,
            dtype,
            reduce_axes=axes,
            keepdim=keepdim,
            tune=self.tune,
            **self._build_kernel_kwargs(shape, axes, device_index),
            **dict(extra),
        )


class _CastReduceOp(_ReduceOpBase):
    """A reduction taking torch's keyword-only ``dtype``: the input is cast to it first."""

    def __init__(
        self,
        dim: Dim = None,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct the op.

        Args:
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        self.dtype = dtype
        super().__init__(dim, keepdim, target=target, kernel_map=kernel_map, tune=tune)


class SumFwdOp(_CastReduceOp):
    """Sum over ``dim``, following ``torch.sum``."""

    _op_kind = "sum"


class MeanFwdOp(_CastReduceOp):
    """Mean over ``dim``, following ``torch.mean``; an empty reduction is NaN."""

    _op_kind = "mean"
    _identity = math.nan


class AminFwdOp(_ReduceOpBase):
    """Minimum over ``dim``, following ``torch.amin``."""

    _op_kind = "amin"


class AmaxFwdOp(_ReduceOpBase):
    """Maximum over ``dim``, following ``torch.amax``."""

    _op_kind = "amax"


class ProdFwdOp(_CastReduceOp):
    """Product over one axis, following ``torch.prod(input, dim, keepdim, *, dtype)``."""

    _op_kind = "prod"
    _identity = 1

    def __init__(
        self,
        dim: int,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct ProdFwdOp.

        Args:
            dim: The axis to reduce.
            keepdim: Whether the reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        super().__init__(dim, keepdim, dtype=dtype, target=target, kernel_map=kernel_map, tune=tune)


class _WelfordReduceOp(_ReduceOpBase):
    """Base for the variance family: ``op(dim=None, *, correction=1, keepdim=False)``."""

    _identity = math.nan

    def __init__(
        self,
        dim: Dim = None,
        *,
        correction: "float | None" = 1,
        keepdim: bool = False,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct a variance-family op.

        Args:
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            correction: Difference between the sample size and the degrees of freedom;
                ``None`` means 1, as in torch.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        self.correction = correction
        super().__init__(dim, keepdim, target=target, kernel_map=kernel_map, tune=tune)

    @property
    def _dof_correction(self) -> float:
        return 1 if self.correction is None else self.correction

    def _call_kwargs(self, n: int) -> tuple:
        # With no degrees of freedom the kernel runs uncorrected and the op divides by zero.
        return (("correction", self._dof_correction if self._dof_correction < n else 0),)

    def _warn_dof(self) -> None:
        warnings.warn(
            f"{self._op_kind}(): degrees of freedom is <= 0. Correction should be strictly "
            "less than the reduction factor (input numel divided by output numel).",
            UserWarning,
            stacklevel=2,
        )

    def _scalar_forward(self, x: torch.Tensor):
        """One element: no spread, over ``max(0, 1 - correction)`` degrees of freedom."""
        if self._dof_correction >= 1:
            self._warn_dof()
            return (x - x) / 0.0
        return x - x

    def _no_dof(self, variance: torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
        """torch divides the squared deviations by ``max(0, n - correction)``: zero here, so
        a spread is ``inf`` and no spread is NaN. *variance* is the uncorrected float32 one,
        in which a small spread does not round to zero."""
        self._warn_dof()
        return ((variance * n) / 0.0).to(dtype)

    def _launch(self, x, axes, m, n):
        if self._dof_correction < n:
            return super()._launch(x, axes, m, n)
        out = super()._launch(x.float(), axes, m, n)
        if self._op_kind == "std":
            return self._no_dof(out * out, n, torch.float32).sqrt().to(x.dtype)
        return self._no_dof(out, n, x.dtype)


class StdFwdOp(_WelfordReduceOp):
    """Standard deviation over ``dim``, following ``torch.std``."""

    _op_kind = "std"


class VarFwdOp(_WelfordReduceOp):
    """Variance over ``dim``, following ``torch.var``."""

    _op_kind = "var"


class VarMeanFwdOp(_WelfordReduceOp):
    """Variance and mean over ``dim``, following ``torch.var_mean``."""

    _op_kind = "var_mean"

    def _scalar_forward(self, x: torch.Tensor):
        return super()._scalar_forward(x), x.clone()

    def _empty_forward(self, x: torch.Tensor):
        nan = super()._empty_forward(x)
        return nan, nan.clone()

    def _launch(self, x, axes, m, n):
        if self._dof_correction < n:
            return _ReduceOpBase._launch(self, x, axes, m, n)
        var, mean = _ReduceOpBase._launch(self, x.float(), axes, m, n)
        return self._no_dof(var, n, x.dtype), mean.to(x.dtype)
