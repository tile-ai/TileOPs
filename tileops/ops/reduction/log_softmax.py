"""Log-softmax operator (L2 Op layer).

Provides:
  - LogSoftmaxOp: y = log_softmax(x, dim)

Supports both spec path (dtype + dim) and legacy path (M, N, dtype).

Example (legacy):
    >>> op = LogSoftmaxOp(M=1024, N=4096, dtype=torch.float16)
    >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (1024, 4096)
"""

from tileops.kernels.reduction.softmax import SoftmaxKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSoftmaxOp"]


class LogSoftmaxOp(_SoftmaxBaseOp):
    """Log-softmax operator: y = log_softmax(x, dim).

    Output has the same shape and dtype as input.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        M: Number of rows (legacy path, optional).
        N: Hidden dimension (legacy path, optional).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind = "log_softmax"
    _kernel_key = "softmax_fwd"
    _kernel_class = SoftmaxKernel
