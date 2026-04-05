"""Softmax operator (L2 Op layer).

Provides:
  - SoftmaxOp: y = softmax(x, dim)

Example:
    >>> op = SoftmaxOp(dtype=torch.float16, dim=-1)
    >>> x = torch.randn(2, 32, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (2, 32, 4096)
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
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind = "softmax"
    _kernel_key = "softmax_fwd"
    _kernel_class = SoftmaxKernel
