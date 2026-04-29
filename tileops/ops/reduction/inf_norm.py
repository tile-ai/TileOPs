"""InfNormFwdOp: computes infinity norm (max absolute value) along a given dim.

The Op layer validates inputs, reshapes to 2D (M, N), pads to alignment
(with 0.0, which is neutral for max of absolute values), calls the kernel,
and reshapes the output back. Output dtype matches input dtype; internal
computation in fp32.

NaN propagation: T.reduce_max in TileLang does not propagate NaN (it drops
NaN values). To match torch.linalg.vector_norm(ord=inf) semantics, the Op
layer detects rows containing NaN before the kernel call and patches the
output to NaN for those rows.
"""

from math import inf
from typing import Dict, List, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.vector_norm import VectorNormKernel

from ._multidim import EmptyDimPolicy
from .reduce import _ReduceOpBase

__all__ = ["InfNormFwdOp"]


class InfNormFwdOp(_ReduceOpBase):
    """Infinity norm reduction along a configurable dim.

    Construction: ``InfNormFwdOp(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    NaN handling: rows containing any NaN produce NaN output, matching
    torch.linalg.vector_norm(ord=inf) semantics.

    Args:
        dtype: Input data type (float16, bfloat16, float32).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
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
    _required_ord: Union[int, float] = inf
    _empty_dim_policy: EmptyDimPolicy = "full"

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
        keepdim: bool = False,
        ord: Union[int, float] = inf,
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
