"""Logical reduction operators (all, any, count_nonzero)."""

from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.logical_reduce import (
    _UNSUPPORTED_STORAGE_DTYPES,
    LogicalReduceKernel,
    to_logical_float32,
)
from tileops.manifest.shape_rules import reduced_shape

from ._multidim import EmptyDimPolicy
from .reduce import _ReduceOpBase

__all__ = ["AllFwdOp", "AnyFwdOp", "CountNonzeroFwdOp"]


class AllFwdOp(_ReduceOpBase):
    """All reduction along ``dim``, returning bool.

    Construction: ``AllFwdOp(dim=None, keepdim=False)``.  M and N
    are derived from the input tensor at forward time, and kernels are
    cached by ``(M, N)`` to avoid rebuilds.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Empty-dim contract: ``dim=[]`` / ``dim=()`` is a no-op -- forward returns
    ``x.bool()`` with the input shape, matching ``torch.all`` semantics.

    Args:
        dim: Reduction dimension (default ``None``, i.e. full reduction).
            Accepts ``int``, ``list[int]``, or ``tuple[int, ...]`` for
            multi-dim reduction.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "all"
    _kernel_key = "logical_reduce"
    _kernel_cls = LogicalReduceKernel
    _empty_dim_policy: EmptyDimPolicy = "noop"

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct AllFwdOp.

        Args:
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or
                ``None``.
            keepdim: Whether to retain reduced dims as size 1.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            kernel_map=kernel_map,
            tune=tune,
        )

    def _noop_output_dtype(self) -> torch.dtype:
        """All returns bool per manifest contract."""
        return torch.bool

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Convert unsupported storage dtypes to float32."""
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)
        return x, None


class AnyFwdOp(_ReduceOpBase):
    """Any reduction along ``dim``, returning bool.

    Construction: ``AnyFwdOp(dim=None, keepdim=False)``.  M and N
    are derived from the input tensor at forward time, and kernels are
    cached by ``(M, N)`` to avoid rebuilds.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Empty-dim contract: ``dim=[]`` / ``dim=()`` is a no-op -- forward returns
    ``x.bool()`` with the input shape, matching ``torch.any`` semantics.

    Args:
        dim: Reduction dimension (default ``None``, i.e. full reduction).
            Accepts ``int``, ``list[int]``, or ``tuple[int, ...]`` for
            multi-dim reduction.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "any"
    _kernel_key = "logical_reduce"
    _kernel_cls = LogicalReduceKernel
    _empty_dim_policy: EmptyDimPolicy = "noop"

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Construct AnyFwdOp.

        Args:
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or
                ``None``.
            keepdim: Whether to retain reduced dims as size 1.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune (default ``False``).
        """
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            kernel_map=kernel_map,
            tune=tune,
        )

    def _noop_output_dtype(self) -> torch.dtype:
        """Any returns bool per manifest contract."""
        return torch.bool

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Convert unsupported storage dtypes to float32."""
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)
        return x, None


class CountNonzeroFwdOp(_ReduceOpBase):
    """Count nonzero reduction along ``dim``, returning int64.

    Construction: ``CountNonzeroFwdOp(dim=None)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Note: No ``keepdim`` parameter -- the reduction dimension is always
    removed, matching ``torch.count_nonzero`` semantics.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Args:
        dim: Reduction dimension (default ``None``, i.e. full reduction).
            Accepts ``int``, ``list[int]``, or ``tuple[int, ...]`` for
            multi-dim reduction.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "count_nonzero"
    _kernel_key = "logical_reduce"
    _kernel_cls = LogicalReduceKernel
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        # count_nonzero never keeps dim (matches torch.count_nonzero)
        super().__init__(
            dim=dim,
            keepdim=False,
            kernel_map=kernel_map,
            tune=tune,
        )

    def _infer_output_shapes(self, x_shape: Tuple[int, ...]) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: no ``keepdim`` param, so a reduced axis always goes."""
        return {"output": reduced_shape(x_shape, self.dim, False, self._empty_dim_policy)}

    def _noop_output_dtype(self) -> torch.dtype:
        """count_nonzero returns int64 per manifest contract.

        Although count_nonzero's ``_empty_dim_policy`` is ``"full"`` (so the
        empty-dim no-op short-circuit never fires), the shared scalar
        forward in the base class consults this hook to cast the
        ``x != 0`` predicate to the declared output dtype.
        """
        return torch.int64

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Convert unsupported storage dtypes to float32."""
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)
        return x, None
