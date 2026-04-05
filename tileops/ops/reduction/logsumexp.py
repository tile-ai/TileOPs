"""LogSumExp operator (L2 Op layer).

Provides:
  - LogSumExpOp: y = logsumexp(x, dim)

Supports both spec path (dtype + dim) and legacy path (M, N, dtype).

Example (legacy):
    >>> op = LogSumExpOp(M=1024, N=4096, dtype=torch.float16)
    >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (1024,)
"""

import torch

from tileops.kernels.reduction.softmax import LogSumExpKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSumExpOp"]


class LogSumExpOp(_SoftmaxBaseOp):
    """LogSumExp operator: y = logsumexp(x, dim).

    Output shape is input shape without the reduction dimension.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        M: Number of rows (legacy path, optional).
        N: Hidden dimension (legacy path, optional).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _kernel_class = LogSumExpKernel

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """Restore leading dims without the last dimension (legacy path)."""
        # y is (M,) -- reshape to (*leading_dims,)
        leading_shape = orig_shape[:-1]
        if len(leading_shape) == 0:
            # 1D input -> scalar output
            return y.squeeze()
        return y.reshape(leading_shape)
