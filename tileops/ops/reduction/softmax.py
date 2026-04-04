"""Softmax operator (L2 Op layer).

Provides:
  - SoftmaxOp: y = softmax(x, dim)

Follows the validate -> reshape -> pad -> kernel -> trim -> reshape pattern
and supports 1D-4D input with arbitrary reduction dim.
"""

from tileops.kernels.reduction.softmax import SoftmaxKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["SoftmaxOp"]


class SoftmaxOp(_SoftmaxBaseOp):
    """Softmax operator: y = softmax(x, dim).

    Output has the same shape and dtype as input.

    Args:
        dim: Reduction dimension (default -1).
        dtype: Data type (float32, float16, or bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = SoftmaxOp(dim=-1, dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> y = op(x)  # shape: (1024, 4096)
    """

    _op_kind = "softmax"
    _kernel_key = "softmax_fwd"
    _kernel_class = SoftmaxKernel
