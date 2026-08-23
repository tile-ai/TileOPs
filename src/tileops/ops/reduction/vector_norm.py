"""Vector-norm reduction operators (L1, L2, inf)."""

from math import inf
from typing import Dict, List, Optional, Union

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.vector_norm import VectorNormKernel

from ._boundary import register_reduction_op
from ._multidim import EmptyDimPolicy
from .reduce import _ReduceOpBase

__all__ = ["InfNormFwdOp", "L1NormFwdOp", "L2NormFwdOp"]


class L1NormFwdOp(_ReduceOpBase):
    """L1 norm reduction along a configurable dim.

    Construction: ``L1NormFwdOp(dim=None, keepdim=False)``.

    """

    _op_kind = "l1"
    _kernel_key = "vector_norm"
    _kernel_cls = VectorNormKernel
    _required_ord: Union[int, float] = 1
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        ord: Union[int, float] = 1,
        dim: Union[int, List[int], None] = None,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction dimension (default ``None`` -> full reduction, matching
                ``torch.linalg.vector_norm``). Accepts ``int``, ``list[int]``, or
                ``None``.
            keepdim: Whether to retain the reduced dimension as size 1.
            ord: Norm order. Must equal 1 for ``L1NormFwdOp`` (manifest fixes
                ``ord == 1``); accepted as a kwarg to mirror
                ``torch.linalg.vector_norm``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        if ord != self._required_ord:
            raise ValueError(
                f"{type(self).__name__} only supports ord={self._required_ord!r}, got ord={ord!r}"
            )
        self.ord = ord
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )


class L2NormFwdOp(_ReduceOpBase):
    """L2 norm reduction along a configurable dim.

    Construction: ``L2NormFwdOp(dim=None, keepdim=False)``.

    """

    _op_kind = "l2"
    _kernel_key = "vector_norm"
    _kernel_cls = VectorNormKernel
    _required_ord: Union[int, float] = 2
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        ord: Union[int, float] = 2,
        dim: Union[int, List[int], None] = None,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction dimension (default ``None`` -> full reduction, matching
                ``torch.linalg.vector_norm``). Accepts ``int``, ``list[int]``, or
                ``None``.
            keepdim: Whether to retain the reduced dimension as size 1.
            ord: Norm order. Must equal 2 for ``L2NormFwdOp`` (manifest fixes
                ``ord == 2``); accepted as a kwarg to mirror
                ``torch.linalg.vector_norm``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        if ord != self._required_ord:
            raise ValueError(
                f"{type(self).__name__} only supports ord={self._required_ord!r}, got ord={ord!r}"
            )
        self.ord = ord
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )


class InfNormFwdOp(_ReduceOpBase):
    """Infinity norm reduction along a configurable dim.

    Construction: ``InfNormFwdOp(dim=None, keepdim=False)``.

    NaN handling: rows containing any NaN produce NaN output, matching
    torch.linalg.vector_norm(ord=inf) semantics. The kernel drops NaN values and patches
    those rows itself, so the compensation stays with the implementation that needs it.

    """

    _op_kind = "inf"
    _kernel_key = "vector_norm"
    _kernel_cls = VectorNormKernel
    _required_ord: Union[int, float] = inf
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        ord: Union[int, float] = inf,
        dim: Union[int, List[int], None] = None,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction dimension (default ``None`` -> full reduction, matching
                ``torch.linalg.vector_norm``). Accepts ``int``, ``list[int]``, or
                ``None``.
            keepdim: Whether to retain the reduced dimension as size 1.
            ord: Norm order. Must equal ``float('inf')`` for ``InfNormFwdOp``
                (manifest fixes ``ord == float('inf')``); accepted as a kwarg to
                mirror ``torch.linalg.vector_norm``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        if ord != self._required_ord:
            raise ValueError(
                f"{type(self).__name__} only supports ord={self._required_ord!r}, got ord={ord!r}"
            )
        self.ord = ord
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )


for _op_cls in (
    L1NormFwdOp,
    L2NormFwdOp,
    InfNormFwdOp,
):
    register_reduction_op(_op_cls)
