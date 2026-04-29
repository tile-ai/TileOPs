"""Log-softmax operator (L2 Op layer).

Provides:
  - LogSoftmaxFwdOp: y = log_softmax(x, dim)

Example:
    >>> op = LogSoftmaxFwdOp(N=4096, dtype=torch.float16, dim=-1)
    >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (1024, 4096)
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.softmax import SoftmaxKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSoftmaxFwdOp"]


class LogSoftmaxFwdOp(_SoftmaxBaseOp):
    """Log-softmax operator: y = log_softmax(x, dim).

    Output has the same shape and dtype as input. The reduction-dim extent
    ``N`` is committed at construction time per manifest
    ``static_dims.N = "x.shape[dim]"`` (R20); ``forward()`` validates the
    actual tensor against the committed value.

    Args:
        N: Reduction-dim size (statically committed at ctor; corresponds to
            manifest ``static_dims.N = "x.shape[dim]"``).
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default ``None``, matching PyTorch's
            ``torch.nn.functional.log_softmax``). When ``None``, the axis is
            resolved at forward time using PyTorch's implicit-axis rule
            (``0`` for ``ndim in {0, 1, 3}`` else ``1``) and the same
            deprecation ``UserWarning`` is emitted.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind = "log_softmax"
    _kernel_key = "softmax_fwd"
    _kernel_class = SoftmaxKernel

    def __init__(
        self,
        *,
        N: int,
        dtype: torch.dtype,
        dim: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            N=N, dtype=dtype, dim=dim, kernel_map=kernel_map, tune=tune,
        )
