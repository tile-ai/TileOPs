"""LogSumExp operator (L2 Op layer).

Provides:
  - LogSumExpFwdOp: y = logsumexp(x, dim, keepdim)

Example:
    >>> op = LogSumExpFwdOp(dtype=torch.float16, dim=-1)
    >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (1024,)
"""

from typing import Dict, List, Optional, Union

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction.softmax import LogSumExpKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSumExpFwdOp"]


class LogSumExpFwdOp(_SoftmaxBaseOp):
    """LogSumExp operator: y = logsumexp(x, dim, keepdim).

    Output shape is input shape without the reduction dimension
    (or with size-1 if keepdim=True).

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        keepdim: Retain reduced dimension (default False).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _kernel_class = LogSumExpKernel
    _supports_multidim = True

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int]] = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(dtype=dtype, dim=dim, kernel_map=kernel_map, tune=tune)
        self.keepdim = keepdim
