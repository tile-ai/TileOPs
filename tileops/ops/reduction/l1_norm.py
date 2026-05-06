"""L1NormFwdOp: computes L1 norm (sum of absolute values) along a given dim.

The Op layer validates inputs, reshapes to 2D (M, N), calls the kernel, and
reshapes the output back. Alignment padding is handled inside the kernel with
0.0, which is neutral for sum. Output dtype matches input dtype; internal
computation in fp32.
"""

from typing import Dict, List, Optional, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.vector_norm import VectorNormKernel

from ._multidim import EmptyDimPolicy
from .reduce import _ReduceOpBase

__all__ = ["L1NormFwdOp"]


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
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = None,
        keepdim: bool = False,
        ord: Union[int, float] = 1,
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
