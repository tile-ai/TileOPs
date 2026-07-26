"""Vector-norm reduction operators (L1, L2, inf)."""

from math import inf
from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.vector_norm import VectorNormKernel

from ._multidim import EmptyDimPolicy
from .reduce import _ReduceOpBase

__all__ = ["InfNormFwdOp", "L1NormFwdOp", "L2NormFwdOp"]


class L1NormFwdOp(_ReduceOpBase):
    """L1 norm reduction along a configurable dim.

    Construction: ``L1NormFwdOp(dtype=..., dim=None, keepdim=False)``.  M and N
    are derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Input data type (float16, bfloat16, float32).
        dim: Reduction dimension (default ``None`` -> full reduction, matching
            ``torch.linalg.vector_norm``). Accepts ``int``, ``list[int]``, or
            ``None``.
        keepdim: Whether to retain the reduced dimension as size 1.
        ord: Norm order. Must equal 1 for ``L1NormFwdOp`` (manifest fixes
            ``ord == 1``); accepted as a kwarg to mirror
            ``torch.linalg.vector_norm``.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "l1"
    _kernel_key = "vector_norm"
    _kernel_cls = VectorNormKernel
    _kernel_handles_padding = True
    _required_ord: Union[int, float] = 1
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        dtype: torch.dtype,
        ord: Union[int, float] = 1,
        dim: Union[int, List[int], None] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if ord != self._required_ord:
            raise ValueError(
                f"{type(self).__name__} only supports ord={self._required_ord!r}, "
                f"got ord={ord!r}"
            )
        self.ord = ord
        super().__init__(
            dtype=dtype, dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )



class L2NormFwdOp(_ReduceOpBase):
    """L2 norm reduction along a configurable dim.

    Construction: ``L2NormFwdOp(dtype=..., dim=None, keepdim=False)``.  M and N
    are derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Input data type (float16, bfloat16, float32).
        dim: Reduction dimension (default ``None`` -> full reduction, matching
            ``torch.linalg.vector_norm``). Accepts ``int``, ``list[int]``, or
            ``None``.
        keepdim: Whether to retain the reduced dimension as size 1.
        ord: Norm order. Must equal 2 for ``L2NormFwdOp`` (manifest fixes
            ``ord == 2``); accepted as a kwarg to mirror
            ``torch.linalg.vector_norm``.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "l2"
    _kernel_key = "vector_norm"
    _kernel_cls = VectorNormKernel
    _kernel_handles_padding = True
    _required_ord: Union[int, float] = 2
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        dtype: torch.dtype,
        ord: Union[int, float] = 2,
        dim: Union[int, List[int], None] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if ord != self._required_ord:
            raise ValueError(
                f"{type(self).__name__} only supports ord={self._required_ord!r}, "
                f"got ord={ord!r}"
            )
        self.ord = ord
        super().__init__(
            dtype=dtype, dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )



class InfNormFwdOp(_ReduceOpBase):
    """Infinity norm reduction along a configurable dim.

    Construction: ``InfNormFwdOp(dtype=..., dim=None, keepdim=False)``.  M and
    N are derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    NaN handling: rows containing any NaN produce NaN output, matching
    torch.linalg.vector_norm(ord=inf) semantics.

    Args:
        dtype: Input data type (float16, bfloat16, float32).
        dim: Reduction dimension (default ``None`` -> full reduction, matching
            ``torch.linalg.vector_norm``). Accepts ``int``, ``list[int]``, or
            ``None``.
        keepdim: Whether to retain the reduced dimension as size 1.
        ord: Norm order. Must equal ``float('inf')`` for ``InfNormFwdOp``
            (manifest fixes ``ord == float('inf')``); accepted as a kwarg to
            mirror ``torch.linalg.vector_norm``.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "inf"
    _kernel_key = "vector_norm"
    _kernel_cls = VectorNormKernel
    _kernel_handles_padding = True
    _required_ord: Union[int, float] = inf
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        dtype: torch.dtype,
        ord: Union[int, float] = inf,
        dim: Union[int, List[int], None] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if ord != self._required_ord:
            raise ValueError(
                f"{type(self).__name__} only supports ord={self._required_ord!r}, "
                f"got ord={ord!r}"
            )
        self.ord = ord
        super().__init__(
            dtype=dtype, dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Detect NaN rows before kernel call (kernel drops NaN values)."""
        nan_mask = x.isnan().any(dim=-1)  # shape (M,)
        return x, nan_mask

    def _post_kernel(self, y: torch.Tensor, context: object) -> torch.Tensor:
        """Patch NaN into output rows that had NaN in the input."""
        nan_mask = context
        if nan_mask.any():
            y[nan_mask] = float("nan")
        return y
