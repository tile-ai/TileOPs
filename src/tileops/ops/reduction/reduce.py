"""Reduce ops: SumFwdOp, MeanFwdOp, AminFwdOp, AmaxFwdOp, ProdFwdOp, StdFwdOp, VarFwdOp, VarMeanFwdOp.

Each op reduces the axes ``dim`` names of an arbitrary-rank input. The generated signature
checks have run before ``forward``: dtype, ``dim`` range and uniqueness, and every
refinement. The op normalizes contiguity and hands the input over as the manifest declares
it; moving the reduced axes to the end, flattening to ``(M, N)`` and shaping the result back
belong to the kernel. Kernels are cached by shape, axes, dtype and device.
"""

import math
import warnings
from typing import ClassVar, List, Mapping, Optional, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.reduction.call_spec import (
    ProdFwdInterface,
    ReduceCall,
    ReduceFwdInterface,
    VarianceFwdInterface,
    VarMeanFwdInterface,
)
from tileops.kernels.reduction.reduce import (
    ReduceEdgeKernel,
    ReduceFoldKernel,
    ReduceKernel,
    ReduceLeadingKernel,
    ReduceProdKernel,
    WelfordEdgeKernel,
    WelfordReduceKernel,
)
from tileops.manifest.primitives import normalize_axis, reduced
from tileops.ops.op_base import Op

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

    A subclass declares ``_op_kind``, ``kernel_types``, ``interfaces`` and the empty-``dim``
    mode of its manifest output shape, and overrides the hooks below where it differs.

    - ``_call(x, axes, n)``: the call spec the ``reduce`` interface takes.
    - ``_output_dtype(x)``: the output dtype; the input's by default.
    - ``_identity``: the result over an empty reduced extent.
    - ``_scalar_forward(x)``: the result on a 0-d input.
    """

    _op_kind: str = ""
    # The manifest's empty-``dim`` mode of ``reduced``: ``'full'`` or ``'noop'``.
    _empty: str = "full"
    _identity: "float | bool | int" = 0

    def __init__(
        self,
        dim: Dim = None,
        keepdim: bool = False,
        *,
        target: Target = None,
    ):
        """Construct a reduce op.

        Args:
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.dim = dim
        self.keepdim = keepdim
        super().__init__(target=target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce *x* over the configured axes.

        Args:
            x: Input tensor of any rank.

        Returns:
            The reduction, shaped by ``dim`` and ``keepdim``.
        """
        if x.ndim == 0:
            return self._scalar_forward(self._cast(x))
        axes = reduce_axes(self.dim, x.ndim, self._empty)
        if not axes:
            return self._noop_forward(self._cast(x))
        if x.numel() == 0:
            return self._empty_forward(self._cast(x))
        x = self._cast(x, for_kernel=True).contiguous()
        n = math.prod(x.shape[a] for a in axes)
        return self._launch(x, axes, n)

    def _cast(self, x: torch.Tensor, *, for_kernel: bool = False) -> torch.Tensor:
        """*x* in the dtype the reduction runs in: the ``dtype`` parameter's when passed.

        A float32 ``dtype`` is not cast first: widening is exact and the kernel reads the
        input as stored. Any other cast runs first, as in torch.
        """
        dtype = self.dtype
        if dtype is None or x.dtype == dtype or (for_kernel and dtype == torch.float32):
            return x
        return x.to(dtype)

    def _output_dtype(self, x: torch.Tensor) -> torch.dtype:
        return x.dtype

    def _output_shape(self, x: torch.Tensor) -> "tuple[int, ...]":
        return reduced(tuple(x.shape), self.dim, self.keepdim, self._empty)

    def _scalar_forward(self, x: torch.Tensor):
        """A 0-d input reduces one element: the element itself."""
        return x.clone()

    def _noop_forward(self, x: torch.Tensor):
        """An empty ``dim`` under the ``'noop'`` mode keeps every element."""
        return x.to(self._output_dtype(x), copy=True)

    def _empty_forward(self, x: torch.Tensor):
        """A non-empty output of an empty input: the identity over each empty reduced extent."""
        return torch.full(
            self._output_shape(x), self._identity, dtype=self._output_dtype(x), device=x.device
        )

    def _launch(self, x: torch.Tensor, axes: "tuple[int, ...]", n: int):
        return self.kernel_for("reduce", self._call(x, axes, n))(x)

    def _call(self, x: torch.Tensor, axes: "tuple[int, ...]", n: int) -> CallSpec:
        """The call spec of reducing *axes* of *x*, ``n`` elements to each output."""
        raise NotImplementedError


class ReduceCallOp(_ReduceOpBase):
    """A reduction over a `ReduceCall`: sum, mean, amax or amin unless a subclass says otherwise."""

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "reduce_fold": ReduceFoldKernel,
        "reduce": ReduceKernel,
        "reduce_leading": ReduceLeadingKernel,
        "reduce_edge": ReduceEdgeKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"reduce": ReduceFwdInterface}

    def _call_kwargs(self, n: int) -> tuple:
        """Call spec fields this call decides, as ``(name, value)`` pairs."""
        return () if self.dtype is None else (("out_dtype", self.dtype),)

    def _call(self, x: torch.Tensor, axes: "tuple[int, ...]", n: int) -> ReduceCall:
        return ReduceCall(
            device=x.device,
            shape=tuple(x.shape),
            axes=axes,
            keepdim=self.keepdim,
            op_kind=self._op_kind,
            dtype=x.dtype,
            **dict(self._call_kwargs(n)),
        )


class _CastReduceOp(ReduceCallOp):
    """A reduction taking torch's keyword-only ``dtype``: the input is cast to it first."""

    def __init__(
        self,
        dim: Dim = None,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
    ):
        """Construct the op.

        Args:
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.dtype = dtype
        super().__init__(dim, keepdim, target=target)


class SumFwdOp(_CastReduceOp):
    """Sum over ``dim``, following ``torch.sum``."""

    _op_kind = "sum"


class MeanFwdOp(_CastReduceOp):
    """Mean over ``dim``, following ``torch.mean``; an empty reduction is NaN."""

    _op_kind = "mean"
    _identity = math.nan


class AminFwdOp(ReduceCallOp):
    """Minimum over ``dim``, following ``torch.amin``."""

    _op_kind = "amin"


class AmaxFwdOp(ReduceCallOp):
    """Maximum over ``dim``, following ``torch.amax``."""

    _op_kind = "amax"


class ProdFwdOp(_CastReduceOp):
    """Product over one axis, following ``torch.prod(input, dim, keepdim, *, dtype)``."""

    _op_kind = "prod"
    _identity = 1
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "reduce_fold": ReduceFoldKernel,
        "reduce_prod": ReduceProdKernel,
        "reduce_leading": ReduceLeadingKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"reduce": ProdFwdInterface}

    def __init__(
        self,
        dim: int,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
    ):
        """Construct ProdFwdOp.

        Args:
            dim: The axis to reduce.
            keepdim: Whether the reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(dim, keepdim, dtype=dtype, target=target)


class _WelfordReduceOp(ReduceCallOp):
    """Base for the variance family: ``op(dim=None, *, correction=1, keepdim=False)``."""

    _identity = math.nan
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "reduce_welford": WelfordReduceKernel,
        "reduce_welford_edge": WelfordEdgeKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"reduce": VarianceFwdInterface}

    def __init__(
        self,
        dim: Dim = None,
        *,
        correction: "float | None" = 1,
        keepdim: bool = False,
        target: Target = None,
    ):
        """Construct a variance-family op.

        Args:
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            correction: Difference between the sample size and the degrees of freedom;
                ``None`` means 1, as in torch.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.correction = correction
        super().__init__(dim, keepdim, target=target)

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

    def _launch(self, x, axes, n):
        if self._dof_correction < n:
            return super()._launch(x, axes, n)
        out = super()._launch(x.float(), axes, n)
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
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"reduce": VarMeanFwdInterface}

    def _scalar_forward(self, x: torch.Tensor):
        return super()._scalar_forward(x), x.clone()

    def _empty_forward(self, x: torch.Tensor):
        nan = super()._empty_forward(x)
        return nan, nan.clone()

    def _launch(self, x, axes, n):
        if self._dof_correction < n:
            return ReduceCallOp._launch(self, x, axes, n)
        var, mean = ReduceCallOp._launch(self, x.float(), axes, n)
        return self._no_dof(var, n, x.dtype), mean.to(x.dtype)
