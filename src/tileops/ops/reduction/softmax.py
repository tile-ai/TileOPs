"""Softmax-family operators (softmax, log_softmax, logsumexp)."""

import math
import warnings
from typing import ClassVar, Dict, List, Mapping, Optional, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.reduction.logsumexp import LogSumExpKernel
from tileops.kernels.reduction.softmax import SoftmaxKernel
from tileops.manifest.primitives import normalize_axis

from ..op_base import Op
from .reduce import _ReduceOpBase

__all__ = ["LogSoftmaxFwdOp", "LogSumExpFwdOp", "SoftmaxFwdOp", "_SoftmaxBaseOp"]


class _SoftmaxBaseOp(Op):
    """Softmax and log-softmax: normalize along one axis, keeping the shape.

    The generated signature checks have run before ``_eager_forward``. The input is cast
    to ``dtype`` first when one is passed, as torch does.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"softmax_fwd": SoftmaxKernel}
    _op_kind: ClassVar[str]

    def __init__(
        self,
        dim: Optional[int] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: The axis to normalize along. ``None`` takes torch's implicit axis (``0``
                at rank 0, 1 or 3, else ``1``) and warns, as torch does.
            dtype: The dtype the input is cast to first, and the output's; ``None`` keeps
                the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default False).
        """
        self.dim = dim
        self.dtype = dtype
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize *x* along the configured axis.

        Args:
            x: Input tensor of any rank.

        Returns:
            A tensor of *x*'s shape, in ``dtype`` when one was passed.
        """
        return self._call_boundary(x)

    def _axis(self, rank: int) -> int:
        if self.dim is not None:
            return normalize_axis(self.dim, rank)
        warnings.warn(
            f"Implicit dimension choice for {self._op_kind} has been deprecated. "
            "Change the call to include dim=X as an argument.",
            UserWarning,
            stacklevel=3,
        )
        return 0 if rank in (0, 1, 3) else 1

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator; closed forms need no kernel."""
        if self.dtype is not None and x.dtype != self.dtype:
            x = x.to(self.dtype)
        axis = self._axis(x.ndim)
        if x.ndim == 0:
            # One element normalizes to probability one; NaN and inf propagate as in torch.
            shifted = x - x
            return shifted.exp() if self._op_kind == "softmax" else shifted
        if x.numel() == 0:
            return torch.empty_like(x)
        x = x.contiguous()
        n = x.shape[axis]
        m = x.numel() // n
        call = (tuple(x.shape), axis, x.dtype, x.device.index, m, n)
        return self.kernel_for("softmax", (x,), call)(x)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built from the whole shape and the axis it normalizes.

        The kernel owns the permute, so the whole shape decides which kernel it is.
        """
        _shape, axis, dtype, device_index, m, n = call
        cls = self.kernel_map["softmax_fwd"]
        return call, lambda: cls(
            m, n, self._op_kind, dtype, norm_axis=axis, tune=self.tune, device_index=device_index
        )


class SoftmaxFwdOp(_SoftmaxBaseOp):
    """Softmax along ``dim``, following ``torch.nn.functional.softmax``."""

    _op_kind = "softmax"


class LogSoftmaxFwdOp(_SoftmaxBaseOp):
    """Log-softmax along ``dim``, following ``torch.nn.functional.log_softmax``."""

    _op_kind = "log_softmax"


class LogSumExpFwdOp(_ReduceOpBase):
    """LogSumExp over ``dim``, following ``torch.logsumexp``; an empty reduction is ``-inf``."""

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"logsumexp_fwd": LogSumExpKernel}
    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _empty = "reject"
    _identity = -math.inf

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...]],
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Axes to reduce: an ``int`` or a non-empty sequence of them.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default False).
        """
        super().__init__(dim, keepdim, target=target, kernel_map=kernel_map, tune=tune)
