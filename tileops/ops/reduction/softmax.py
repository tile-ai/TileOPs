"""Softmax operator (L2 Op layer).

Provides:
  - SoftmaxFwdOp: y = softmax(x, dim)

Example:
    >>> op = SoftmaxFwdOp(dim=-1)
    >>> x = torch.randn(2, 32, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (2, 32, 4096)
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.softmax import SoftmaxKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["SoftmaxFwdOp"]


class SoftmaxFwdOp(_SoftmaxBaseOp):
    """Softmax operator: y = softmax(x, dim).

    Output has the same shape and dtype as input. The reduction-dim extent
    ``N`` and dtype are inferred from ``x`` during ``forward()``. Optional
    ``N`` and ``dtype`` constructor arguments are retained as compatibility
    guards and, when provided, are validated against the input tensor.

    Args:
        dim: Reduction dimension (default ``None``, matching PyTorch's
            ``torch.nn.functional.softmax``). When ``None``, the axis is
            resolved at forward time using PyTorch's implicit-axis rule
            (``0`` for ``ndim in {0, 1, 3}`` else ``1``) and the same
            deprecation ``UserWarning`` is emitted.
        N: Optional committed reduction-dim size for compatibility.
        dtype: Optional committed data type for compatibility.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind = "softmax"
    _kernel_key = "softmax_fwd"
    _kernel_class = SoftmaxKernel

    def __init__(
        self,
        N: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        dim: Optional[int] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            N=N, dtype=dtype, dim=dim, kernel_map=kernel_map, tune=tune,
        )
