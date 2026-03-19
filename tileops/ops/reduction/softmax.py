"""Softmax operator (L2 Op layer).

Provides:
  - SoftmaxOp: y = softmax(x, dim=-1)

Follows the validate -> reshape -> pad -> kernel -> trim -> reshape pattern
and supports 1D-4D input with dim=-1.
"""

from tileops.kernels.reduction.softmax import SoftmaxKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["SoftmaxOp"]


class SoftmaxOp(_SoftmaxBaseOp):
    """Softmax operator: y = softmax(x, dim=-1).

    Output has the same shape and dtype as input.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        dtype: Data type (float32, float16, or bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = SoftmaxOp(M=1024, N=4096, dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> y = op(x)  # shape: (1024, 4096)
    """

    _op_kind = "softmax"
    _kernel_key = "softmax_fwd"
    _kernel_class = SoftmaxKernel
