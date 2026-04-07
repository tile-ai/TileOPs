"""LogSumExp operator (L2 Op layer).

Provides:
  - LogSumExpOp: y = logsumexp(x, dim, keepdim)

Supports both single-int and multi-dim (list[int]) reduction, matching
``torch.logsumexp`` semantics.

Example:
    >>> op = LogSumExpOp(dtype=torch.float16, dim=-1)
    >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (1024,)

    >>> op = LogSumExpOp(dtype=torch.float16, dim=[1, 2])
    >>> x = torch.randn(4, 16, 256, dtype=torch.float16, device="cuda")
    >>> y = op(x)  # shape: (4,)
"""

from math import prod
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.softmax import LogSumExpKernel

from ._softmax_base import _SoftmaxBaseOp

__all__ = ["LogSumExpOp"]


class LogSumExpOp(_SoftmaxBaseOp):
    """LogSumExp operator: y = logsumexp(x, dim, keepdim).

    Output shape is input shape without the reduction dimension(s)
    (or with size-1 if keepdim=True).

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension(s) (default -1).  Accepts a single int
            or a list of ints for multi-dim reduction.
        keepdim: Retain reduced dimension (default False).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind = "logsumexp"
    _kernel_key = "logsumexp_fwd"
    _kernel_class = LogSumExpKernel

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int]] = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        # For single-int dim, delegate fully to the base class.
        # For multi-dim, store the raw dim list and pass dim=-1 as a
        # placeholder (the base __init__ only uses dim for storage;
        # we override forward to handle multi-dim).
        if isinstance(dim, (list, tuple)):
            if len(dim) == 0:
                raise ValueError(
                    "dim must be non-empty when provided as a list; "
                    "pass a single int to reduce over one dimension"
                )
            super().__init__(
                dtype=dtype, dim=-1, kernel_map=kernel_map, tune=tune
            )
            self.dim = list(dim)
        else:
            super().__init__(
                dtype=dtype, dim=dim, kernel_map=kernel_map, tune=tune
            )
        self.keepdim = keepdim

    # ------------------------------------------------------------------
    # Multi-dim forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run logsumexp, supporting both single-int and list[int] dim."""
        if isinstance(self.dim, int):
            return super().forward(x)
        return self._forward_multi_dim(x)

    def _forward_multi_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-dim logsumexp: flatten reduction dims, run kernel, reshape."""
        self._validate(x)
        orig_shape = x.shape
        ndim = x.ndim

        # Normalize and validate each dim.
        dims = []
        for d in self.dim:
            if d < -ndim or d >= ndim:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[{-ndim}, {ndim - 1}], but got {d})"
                )
            dims.append(d % ndim)

        if len(set(dims)) != len(dims):
            raise ValueError("Repeated dim in reduction dimensions")

        dims_sorted = sorted(dims)

        # Permute so that reduction dims are at the end (in their
        # original relative order), and non-reduction dims come first.
        kept_dims = [i for i in range(ndim) if i not in dims_sorted]
        perm = kept_dims + dims_sorted
        x = x.permute(perm)

        # Flatten: kept dims -> M, reduction dims -> N.
        N = prod(orig_shape[d] for d in dims_sorted)
        M = prod(orig_shape[d] for d in kept_dims) if kept_dims else 1

        x = x.contiguous().reshape(M, N)

        # Get or create cached kernel for this (M, N).
        kernel = self._get_or_create_kernel(M, N)

        # Pad hidden dim to alignment.
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N), value=float("-inf"))

        y = kernel(x)

        # Reshape output.
        if self.keepdim:
            out_shape = list(orig_shape)
            for d in dims_sorted:
                out_shape[d] = 1
            y = y.reshape(out_shape)
        else:
            out_shape = [orig_shape[d] for d in kept_dims]
            y = y.squeeze() if len(out_shape) == 0 else y.reshape(out_shape)

        return y
