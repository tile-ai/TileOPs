"""Softmax operator (L2 Op layer).

Provides:
  - SoftmaxOp: y = softmax(x, dim)

Supports two construction paths:

- **Spec path**: ``SoftmaxOp(dtype=torch.float16, dim=-1)`` -- M, N are
  derived from the input tensor at forward time.
- **Legacy path**: ``SoftmaxOp(M=1024, N=4096, dtype=torch.float16)`` --
  caller pre-computes M, N.

Example (spec path):
    >>> op = SoftmaxOp(dtype=torch.float16, dim=-1)
    >>> x = torch.randn(2, 32, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (2, 32, 4096)

Example (legacy path):
    >>> op = SoftmaxOp(M=1024, N=4096, dtype=torch.float16)
    >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (1024, 4096)
"""

from tileops.kernels.reduction.softmax import SoftmaxKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["SoftmaxOp"]


class SoftmaxOp(_SoftmaxBaseOp):
    """Softmax operator: y = softmax(x, dim).

    Output has the same shape and dtype as input.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        M: Number of rows (legacy path, optional).
        N: Hidden dimension (legacy path, optional).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind = "softmax"
    _kernel_key = "softmax_fwd"
    _kernel_class = SoftmaxKernel
