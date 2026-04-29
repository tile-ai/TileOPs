"""L2NormFwdOp: computes L2 norm (Euclidean norm) along a given dim.

The Op layer validates inputs, reshapes to 2D (M, N), pads to alignment
(with 0.0, which is neutral for sum of squares), calls the kernel, and reshapes
the output back. Output dtype matches input dtype; internal computation in fp32.
"""

from typing import Dict, List, Optional, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.vector_norm import VectorNormKernel

from ._multidim import EmptyDimPolicy
from .reduce import _ReduceOpBase

__all__ = ["L2NormFwdOp"]


class L2NormFwdOp(_ReduceOpBase):
    """L2 norm reduction along a configurable dim.

    Construction: ``L2NormFwdOp(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Input data type (float16, bfloat16, float32).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
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
    _required_ord: Union[int, float] = 2
    # torch.linalg.vector_norm(x, dim=[]) full-reduces.
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
        keepdim: bool = False,
        ord: Union[int, float] = 2,
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
