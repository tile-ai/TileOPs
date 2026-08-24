"""Softmax-family operators (softmax, log_softmax, logsumexp)."""

import warnings
from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.logsumexp import LogSumExpKernel
from tileops.kernels.reduction.softmax import SoftmaxKernel
from tileops.manifest.shape_rules import reduced_shape

from ..op_base import Op
from ._boundary import register_reduction_op
from ._multidim import EmptyDimPolicy, normalize_dim

__all__ = ["LogSoftmaxFwdOp", "LogSumExpFwdOp", "SoftmaxFwdOp", "_SoftmaxBaseOp"]


def _resolve_implicit_softmax_dim(name: str, ndim: int) -> int:
    """Mirror ``torch.nn.functional._get_softmax_dim``."""
    warnings.warn(
        f"Implicit dimension choice for {name} has been deprecated. "
        "Change the call to include dim=X as an argument.",
        UserWarning,
        stacklevel=3,
    )
    if ndim in (0, 1, 3):
        return 0
    return 1


class _SoftmaxBaseOp(Op):
    """Base class for softmax-family ops.

    Holds the shared validation, the reading of ``dim``, and the one place a kernel is
    resolved. A subclass sets ``_op_kind``, ``_kernel_key`` and ``_kernel_cls``, and
    overrides ``_kernel_ctor_kwargs`` when its kernel takes something else.

    """

    #: Set by ``register_reduction_op`` on each concrete op; a base registers none.
    _wrapped = None

    _op_kind: str  # set by subclass
    _kernel_key: str  # set by subclass
    _kernel_cls: type  # set by subclass
    _supports_multidim: bool = False  # override to True in reduced-dim ops (e.g. LogSumExpFwdOp)
    _empty_dim_policy: EmptyDimPolicy = "reject"

    def __init__(
        self,
        dim: Union[int, List[int]] = -1,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction dimension (default -1).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default False).
        """
        self.dim = dim
        self.keepdim = False
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._last_roofline_spec: tuple[int, int, torch.dtype] | None = None

    def _infer_output_shapes(self, x_shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: normalizing over an axis keeps the shape."""
        return {"output": tuple(x_shape)}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_cls}

    # Validation

    def _validate(self, x: torch.Tensor) -> None:
        """Validate the input against the manifest dtype union and the minimum rank.

        Which devices a set of kernels runs on is the kernel's own statement, so no
        device kind is checked here.
        """
        self._validate_dtypes(x)
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")
        self.dtype = x.dtype

    # Forward

    def _reduce_axes(self, x: torch.Tensor) -> "tuple[int, ...]":
        """The axes this call runs over, ascending and non-negative.

        ``dim=None`` means two different things and neither is "every axis": for softmax
        and log-softmax it is PyTorch's implicit-axis rule, for logsumexp it is the full
        reduction. Getting this wrong keeps the output shape and changes the values.

        Raises:
            IndexError: ``dim`` names an axis this rank does not have.
            ValueError: A sequence ``dim`` reached an op that reduces one axis.
        """
        # Resolved per call rather than written back to ``self.dim``, so one instance
        # accepts inputs of different ranks, matching F.softmax.
        dim: Union[int, List[int], Tuple[int, ...], None] = self.dim
        if dim is None and not self._supports_multidim:
            dim = _resolve_implicit_softmax_dim(self._op_kind, x.ndim)
        if (isinstance(dim, (list, tuple)) or dim is None) and not self._supports_multidim:
            raise ValueError(
                f"{type(self).__name__} does not support multi-dim reduction. Use a scalar dim."
            )
        return tuple(normalize_dim(dim, x.ndim, empty_dim_policy=self._empty_dim_policy))

    def _kernel_ctor_kwargs(self, axes: "tuple[int, ...]") -> dict:
        """What this op's kernel takes beyond the shared arguments."""
        (axis,) = axes
        return {"norm_axis": axis}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax-family op.

        One call to the operator this op registers: this is as far as dynamo traces.
        """
        return type(self)._wrapped(x, self._instance_key)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        self._validate(x)
        x = x.contiguous()  # handed over as the manifest declares it
        axes = self._reduce_axes(x)
        # From the shape, not from ``numel``: an empty reduced axis makes ``n`` zero.
        n = prod(x.shape[a] for a in axes)
        m = prod(d for i, d in enumerate(x.shape) if i not in axes)
        self._last_roofline_spec = (m, n, x.dtype)
        kernel = self.get_or_build_kernel(
            self._kernel_key,
            (x,),
            # The kernel owns the permute, so the whole shape decides which kernel it is.
            key=(tuple(x.shape), axes, self.keepdim, x.dtype, x.device.index),
            build=lambda: self.kernel_map[self._kernel_key](
                m,
                n,
                self._op_kind,
                x.dtype,
                tune=self.tune,
                device_index=x.device.index,
                **self._kernel_ctor_kwargs(axes),
            ),
        )
        return kernel(x)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        M, N, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        if self._op_kind == "softmax":
            return 5 * M * N, 2 * M * N * elem_bytes
        if self._op_kind == "log_softmax":
            return 5 * M * N, 2 * M * N * elem_bytes
        if self._op_kind == "logsumexp":
            return 4 * M * N, (M * N + M) * elem_bytes
        raise NotImplementedError(
            f"{type(self).__name__} has unknown roofline op kind {self._op_kind!r}"
        )


class SoftmaxFwdOp(_SoftmaxBaseOp):
    """Softmax operator: y = softmax(x, dim).

    Output has the same shape and dtype as input. The reduction-dim extent
    ``N`` and dtype are inferred from ``x`` during ``forward()``.

    """

    _op_kind = "softmax"
    _kernel_key = "softmax_fwd"
    _kernel_cls = SoftmaxKernel

    def __init__(
        self,
        dim: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction dimension (default ``None``, matching PyTorch's
                ``torch.nn.functional.softmax``). When ``None``, the axis is
                resolved at forward time using PyTorch's implicit-axis rule
                (``0`` for ``ndim in {0, 1, 3}`` else ``1``) and the same
                deprecation ``UserWarning`` is emitted.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default False).
        """
        super().__init__(dim=dim, target=target, kernel_map=kernel_map, tune=tune)


class LogSoftmaxFwdOp(_SoftmaxBaseOp):
    """Log-softmax operator: y = log_softmax(x, dim).

    Output has the same shape and dtype as input. The reduction-dim extent
    ``N`` and dtype are inferred from ``x`` during ``forward()``.

    """

    _op_kind = "log_softmax"
    _kernel_key = "softmax_fwd"
    _kernel_cls = SoftmaxKernel

    def __init__(
        self,
        dim: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction dimension (default ``None``, matching PyTorch's
                ``torch.nn.functional.log_softmax``). When ``None``, the axis is
                resolved at forward time using PyTorch's implicit-axis rule
                (``0`` for ``ndim in {0, 1, 3}`` else ``1``) and the same
                deprecation ``UserWarning`` is emitted.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default False).
        """
        super().__init__(dim=dim, target=target, kernel_map=kernel_map, tune=tune)


class LogSumExpFwdOp(_SoftmaxBaseOp):
    """LogSumExp operator: y = logsumexp(x, dim, keepdim).

    Output shape is input shape without the reduction dimension
    (or with size-1 if keepdim=True).

    """

    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _kernel_cls = LogSumExpKernel
    _supports_multidim = True

    def __init__(
        self,
        dim: Union[int, List[int]] = -1,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction dimension (default -1).
            keepdim: Retain reduced dimension (default False).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default False).
        """
        super().__init__(dim=dim, target=target, kernel_map=kernel_map, tune=tune)
        self.keepdim = keepdim

    def _infer_output_shapes(self, x_shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: this one reduces, unlike its siblings."""
        return {"output": reduced_shape(x_shape, self.dim, self.keepdim)}

    def _kernel_ctor_kwargs(self, axes: "tuple[int, ...]") -> dict:
        """This kernel reduces the axes away, so it is told which and whether they stay."""
        return {"reduce_axes": axes, "keepdim": self.keepdim}


for _op_cls in (SoftmaxFwdOp, LogSoftmaxFwdOp, LogSumExpFwdOp):
    register_reduction_op(_op_cls)
