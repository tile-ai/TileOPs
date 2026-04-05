"""LogSumExp operator (L2 Op layer).

Provides:
  - LogSumExpOp: y = logsumexp(x, dim=-1)

Follows the validate -> reshape -> pad -> kernel -> trim -> reshape pattern
and supports 1D-4D input with dim=-1.
"""

import torch

from tileops.kernels.reduction.softmax import LogSumExpKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSumExpOp"]


class LogSumExpOp(_SoftmaxBaseOp):
    """LogSumExp operator: y = logsumexp(x, dim=-1).

    Output shape is input shape without the last dimension (*leading,).

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        dtype: Data type (float32, float16, or bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = LogSumExpOp(M=1024, N=4096, dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> y = op(x)  # shape: (1024,)
    """

    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _kernel_class = LogSumExpKernel

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """Restore leading dims without the last dimension."""
        # y is (M,) -- reshape to (*leading_dims,)
        leading_shape = orig_shape[:-1]
        if len(leading_shape) == 0:
            # 1D input -> scalar output
            return y.squeeze()
        return y.reshape(leading_shape)
