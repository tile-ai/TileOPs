"""Log-softmax operator (L2 Op layer).

Provides:
  - LogSoftmaxOp: y = log_softmax(x, dim=-1)

Follows the validate -> reshape -> pad -> kernel -> trim -> reshape pattern
and supports 1D-4D input with dim=-1.
"""

from tileops.kernels.reduction.softmax import SoftmaxKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSoftmaxOp"]


class LogSoftmaxOp(_SoftmaxBaseOp):
    """Log-softmax operator: y = log_softmax(x, dim=-1).

    Output has the same shape and dtype as input.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        dtype: Data type (float32, float16, or bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = LogSoftmaxOp(M=1024, N=4096, dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> y = op(x)  # shape: (1024, 4096)
    """

    _op_kind = "log_softmax"
    _kernel_key = "softmax_fwd"
    _kernel_class = SoftmaxKernel
