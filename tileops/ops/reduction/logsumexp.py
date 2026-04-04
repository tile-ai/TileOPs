"""LogSumExp operator (L2 Op layer).

Provides:
  - LogSumExpOp: y = logsumexp(x, dim)

Follows the validate -> reshape -> pad -> kernel -> trim -> reshape pattern
and supports 1D-4D input with arbitrary reduction dim.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction.softmax import LogSumExpKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSumExpOp"]


class LogSumExpOp(_SoftmaxBaseOp):
    """LogSumExp operator: y = logsumexp(x, dim).

    Output shape is input shape with the reduce dimension removed
    (or kept as size 1 when ``keepdim=True``).

    Args:
        dim: Reduction dimension (default -1).
        keepdim: Whether to retain the reduced dimension (default False).
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

    def __init__(
        self,
        dim: int = -1,
        dtype: torch.dtype = torch.float16,
        *,
        keepdim: bool = False,
        M: Optional[int] = None,
        N: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(dim=dim, dtype=dtype, M=M, N=N, kernel_map=kernel_map, tune=tune)
        self.keepdim = keepdim

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        dim: int,
        N: Optional[int] = None,
        N_padded: Optional[int] = None,
    ) -> torch.Tensor:
        """Restore shape with the reduce dimension removed (or kept)."""
        # y is (M,) from the kernel.
        ndim = len(orig_shape)

        if ndim == 1:
            # 1D input -> scalar output (or (1,) if keepdim)
            return y.unsqueeze(0) if self.keepdim else y.squeeze()

        # The base class transposes dim to the last position before flattening
        # to 2D. After reduction the kernel output (M,) is ordered according to
        # the *transposed* leading dimensions, not the original ones. We must
        # reshape into that transposed order first, then transpose back.
        if dim == ndim - 1:
            # Reduce dim was already last -- leading dims are in original order.
            out_shape = list(orig_shape[:-1])
            y = y.reshape(out_shape)
        else:
            # Build the transposed shape (with dim swapped to last), then drop
            # the last axis (the one that was reduced).
            transposed_shape = list(orig_shape)
            transposed_shape[dim], transposed_shape[-1] = (
                transposed_shape[-1],
                transposed_shape[dim],
            )
            # Drop the reduced (last) dim to get the transposed leading shape.
            transposed_leading = transposed_shape[:-1]
            y = y.reshape(transposed_leading)
            # Transpose back: the original dim position now holds what was the
            # last dim's size, swap them back.
            y = y.transpose(dim, -1).contiguous()

        if self.keepdim:
            y = y.unsqueeze(dim)

        return y
