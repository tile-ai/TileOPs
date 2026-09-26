"""Vector-norm reduction operators (L1, L2, inf), each one ``ord`` of ``torch.linalg.vector_norm``."""

from math import inf
from typing import ClassVar, Dict, List, Mapping, Optional, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.vector_norm import VectorNormKernel

from .reduce import _ReduceOpBase

__all__ = ["InfNormFwdOp", "L1NormFwdOp", "L2NormFwdOp"]


class _VectorNormOp(_ReduceOpBase):
    """``torch.linalg.vector_norm`` at the one ``ord`` the subclass computes.

    ``ord`` is taken to mirror torch, and the signature accepts only that value.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"vector_norm": VectorNormKernel}
    _kernel_key = "vector_norm"

    def __init__(
        self,
        ord: Union[int, float],
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            ord: Norm order; only this op's order is accepted.
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                it may not narrow the input's. ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        self.ord = ord
        self.dtype = dtype
        super().__init__(dim, keepdim, target=target, kernel_map=kernel_map, tune=tune)

    def _scalar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The norm of one element is its magnitude."""
        return x.abs()


class L1NormFwdOp(_VectorNormOp):
    """L1 norm over ``dim``: ``torch.linalg.vector_norm(x, 1, dim, keepdim, dtype=dtype)``."""

    _op_kind = "l1"

    def __init__(
        self,
        ord: Union[int, float] = 1,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            ord: Norm order; only 1 is accepted.
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                it may not narrow the input's. ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        super().__init__(
            ord, dim, keepdim, dtype=dtype, target=target, kernel_map=kernel_map, tune=tune
        )


class L2NormFwdOp(_VectorNormOp):
    """L2 norm over ``dim``: ``torch.linalg.vector_norm(x, 2, dim, keepdim, dtype=dtype)``."""

    _op_kind = "l2"

    def __init__(
        self,
        ord: Union[int, float] = 2,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            ord: Norm order; only 2 is accepted.
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                it may not narrow the input's. ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        super().__init__(
            ord, dim, keepdim, dtype=dtype, target=target, kernel_map=kernel_map, tune=tune
        )


class InfNormFwdOp(_VectorNormOp):
    """Infinity norm over ``dim``, as ``torch.linalg.vector_norm(x, inf, ...)``.

    A row holding a NaN yields NaN, as in torch.
    """

    _op_kind = "inf"

    def __init__(
        self,
        ord: Union[int, float] = inf,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            ord: Norm order; only inf is accepted.
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                it may not narrow the input's. ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        super().__init__(
            ord, dim, keepdim, dtype=dtype, target=target, kernel_map=kernel_map, tune=tune
        )
