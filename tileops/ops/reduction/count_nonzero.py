"""CountNonzeroFwdOp: counts non-zero elements along ``dim``, returning int64.

The Op layer validates inputs, normalizes ``dim``, reshapes to 2D (M, N),
pads to alignment (with 0, which is neutral for sum/count), calls the kernel,
and reshapes the output back.  Output dtype is always int64.

Supports any numeric dtype as input including torch.bool, int32, int64, and
complex types. Inputs with unsupported TileLang storage dtypes (bool, int32,
int64, complex64, complex128) are pre-converted to float32 before the kernel
call.

Kernels are cached by ``(M, N)`` so that the same op instance can handle
varying shapes.

Note: Unlike AllFwdOp/AnyFwdOp, CountNonzeroFwdOp does NOT accept ``keepdim``.
The reduction dimension is always removed, matching ``torch.count_nonzero``.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.logical_reduce import (
    _UNSUPPORTED_STORAGE_DTYPES,
    LogicalReduceKernel,
    to_logical_float32,
)

from ._multidim import EmptyDimPolicy
from .reduce import _ReduceOpBase

__all__ = ["CountNonzeroFwdOp"]


class CountNonzeroFwdOp(_ReduceOpBase):
    """Count nonzero reduction along ``dim``, returning int64.

    Construction: ``CountNonzeroFwdOp(dtype=..., dim=-1)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Padded positions use 0, which is neutral for sum/count.

    Note: No ``keepdim`` parameter -- the reduction dimension is always
    removed, matching ``torch.count_nonzero`` semantics.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Args:
        dtype: Input data type (float16, bfloat16, float32, int32, int64,
               bool, complex64, complex128).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "count_nonzero"
    _kernel_key = "logical_reduce"
    _kernel_cls = LogicalReduceKernel
    # torch.count_nonzero(x, dim=()) full-reduces.
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        # count_nonzero never keeps dim (matches torch.count_nonzero)
        super().__init__(
            dtype=dtype, dim=dim, keepdim=False,
            kernel_map=kernel_map, tune=tune,
        )

    def _pad_value(self) -> float:
        """Pad with 0, neutral for sum/count."""
        return 0.0

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Convert unsupported storage dtypes to float32."""
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)
        return x, None
