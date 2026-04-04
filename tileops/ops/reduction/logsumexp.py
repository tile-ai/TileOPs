"""LogSumExp operator (L2 Op layer).

Provides:
  - LogSumExpOp: y = logsumexp(x, dim)

Follows the validate -> reshape -> pad -> kernel -> trim -> reshape pattern
and supports 1D-4D input with arbitrary reduction dim.
"""

from typing import Optional

import torch

from tileops.kernels.reduction.softmax import LogSumExpKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSumExpOp"]


class LogSumExpOp(_SoftmaxBaseOp):
    """LogSumExp operator: y = logsumexp(x, dim).

    Output shape is input shape with the reduce dimension removed.

    Args:
        dim: Reduction dimension (default -1).
        dtype: Data type (float32, float16, or bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = LogSumExpOp(dim=-1, dtype=torch.float16)
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
        dim: int,
        N: Optional[int] = None,
        N_padded: Optional[int] = None,
    ) -> torch.Tensor:
        """Restore shape with the reduce dimension removed."""
        # y is (M,) -- reshape to orig_shape with dim removed
        leading_shape = list(orig_shape)
        del leading_shape[dim]
        if len(leading_shape) == 0:
            # 1D input -> scalar output
            return y.squeeze()
        return y.reshape(leading_shape)
