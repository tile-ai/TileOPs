"""Arg-reduction operators (argmax, argmin)."""

from typing import Dict, Optional

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.argreduce import ArgreduceKernel

from .reduce import _ReduceOpBase

__all__ = ["ArgmaxFwdOp", "ArgminFwdOp"]


class ArgmaxFwdOp(_ReduceOpBase):
    """Argmax reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgmaxFwdOp(dtype=..., dim=None, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dim: Reduction dimension. ``None`` (the default) matches
            ``torch.argmax(x)`` semantics: the input is treated as a
            contiguous flattened 1D buffer and the returned index is into
            that flattened tensor.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "argmax"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel

    def __init__(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _validate_dim(self) -> None:
        """Argmax accepts a scalar ``int`` dim or ``None`` (full-tensor reduction).

        ``dim=None`` matches ``torch.argmax(x)`` semantics: the input is
        treated as a contiguous flattened 1D buffer and the returned index
        is into that flattened tensor.
        """
        if self.dim is None or isinstance(self.dim, int):
            return
        raise ValueError(
            f"ArgmaxFwdOp only supports scalar dim (int) or None, "
            f"got {type(self.dim).__name__}: {self.dim!r}"
        )

    def _pad_value(self) -> float:
        """Pad with -inf so padded positions never win argmax."""
        return float("-inf")



class ArgminFwdOp(_ReduceOpBase):
    """Argmin reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgminFwdOp(dtype=..., dim=None, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dim: Reduction dimension. ``None`` (the default) matches
            ``torch.argmin(x)`` semantics: the input is treated as a
            contiguous flattened 1D buffer and the returned index is into
            that flattened tensor.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "argmin"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel

    def __init__(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _validate_dim(self) -> None:
        """Argmin accepts a scalar ``int`` dim or ``None`` (full-tensor reduction).

        ``dim=None`` matches ``torch.argmin(x)`` semantics: the input is
        treated as a contiguous flattened 1D buffer and the returned index
        is into that flattened tensor.
        """
        if self.dim is None or isinstance(self.dim, int):
            return
        raise ValueError(
            f"ArgminFwdOp only supports scalar dim (int) or None, "
            f"got {type(self.dim).__name__}: {self.dim!r}"
        )

    def _pad_value(self) -> float:
        """Pad with +inf so padded positions never win argmin."""
        return float("inf")
