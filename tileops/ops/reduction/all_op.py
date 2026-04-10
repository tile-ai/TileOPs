"""AllFwdOp: returns bool indicating if all elements are non-zero along ``dim``.

The Op layer validates inputs, normalizes ``dim``, reshapes to 2D (M, N),
pads to alignment (with 1, which is neutral for AND/all), calls the kernel,
and reshapes the output back.  Output dtype is always bool.

Supports any numeric dtype as input including torch.bool, int32, int64, and
complex types. Inputs with unsupported TileLang storage dtypes (bool, int32,
int64, complex64, complex128) are pre-converted to float32 before the kernel
call.

Kernels are cached by ``(M, N)`` so that the same op instance can handle
varying shapes.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction.logical_reduce import LogicalReduceKernel
from tileops.kernels.reduction.logical_reduce.fwd import (
    _UNSUPPORTED_STORAGE_DTYPES,
    to_logical_float32,
)

from .reduce import _ReduceOpBase

__all__ = ["AllFwdOp"]


class AllFwdOp(_ReduceOpBase):
    """All reduction along ``dim``, returning bool.

    Construction: ``AllFwdOp(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Padded positions use 1 (True), which is neutral for AND/all.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Args:
        dtype: Input data type (float16, bfloat16, float32, int32, int64,
               bool, complex64, complex128).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "all"
    _kernel_key = "logical_reduce"

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            dtype=dtype, dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"logical_reduce": LogicalReduceKernel}

    def _pad_value(self) -> float:
        """Pad with 1 (True), neutral for AND/all."""
        return 1.0

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Convert unsupported storage dtypes to float32."""
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)
        return x, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all along the configured dim."""
        x, orig_shape, dim_info, kernel = self._prepare_input(x)
        x, ctx = self._pre_kernel(x)
        y = kernel(x)
        y = self._post_kernel(y, ctx)
        return self._reshape_output(y, orig_shape, dim_info)
