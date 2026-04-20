from typing import Dict, Hashable, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import FusedAddRMSNormKernel

from ..op_base import Op

__all__ = ["FusedAddRMSNormFwdOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class FusedAddRMSNormFwdOp(Op):
    """Fused residual addition and RMS Normalization operator.

    Computes the residual sum followed by RMS normalization in a single
    fused kernel:

    .. math::

        \\begin{aligned}
        r &= x + \\mathrm{residual} \\\\
        y &= \\frac{r}{\\sqrt{\\mathrm{mean}(r^2) + \\epsilon}} \\cdot w
        \\end{aligned}

    Returns dual outputs ``(y, residual_out)`` so downstream residual connections can
    reuse the pre-norm sum without recomputation.

    Supported dtypes:
        ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims
        by padding to 256-element alignment.

    Args:
        N: Hidden dimension (last dim). Committed at construction per
            manifest ``static_dims``; forward validates ``x.shape[-1] == N``.
        dtype: Data type (``torch.float16`` or ``torch.bfloat16``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        *,
        N: int,
        dtype: torch.dtype,
        eps: float = 1e-6,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N = N
        self.dtype = dtype
        self.eps = eps
        self._tune = tune
        self.N_padded = _align_up(N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[Hashable, Kernel] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_add_rms_norm": FusedAddRMSNormKernel}

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Kernel cache key: the (M,) product of leading dims of ``x``."""
        x_shape = input_shapes[0]
        M = 1
        for s in x_shape[:-1]:
            M *= s
        return (M,)

    def _get_or_create_kernel(self, M: int) -> Kernel:
        key = (M,)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["fused_add_rms_norm"](
                M, self.N, self.eps, self.dtype, tune=self._tune,
            )
            self._kernel_cache[key] = kernel
        return kernel

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual addition and RMS normalization.

        Args:
            x: Input tensor of shape ``(*leading, N)`` on CUDA.
            residual: Residual tensor of the same shape as *x* on CUDA.
            weight: Affine scale of shape ``(N,)`` on CUDA.

        Returns:
            Tuple of ``(y, residual_out)`` where *y* is the normalized
            output and *residual_out* is ``x + residual``, both of the
            same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch,
                or shapes are incompatible with the configured dimensions.
        """
        for name, tensor in [("x", x), ("residual", residual), ("weight", weight)]:
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if tensor.dtype != self.dtype:
                raise ValueError(
                    f"Expected {name}.dtype {self.dtype}, got {tensor.dtype}"
                )
        if weight.ndim != 1:
            raise ValueError(
                f"Expected weight to be 1D, got {weight.ndim}D"
            )
        # static_dims validation: x.shape[-1] == N (committed at ctor).
        if x.shape[-1] != self.N:
            raise ValueError(
                f"Expected hidden dim {self.N}, got {x.shape[-1]}"
            )
        if residual.shape != x.shape:
            raise ValueError(
                f"Expected residual shape {x.shape}, got {residual.shape}"
            )
        if weight.shape[0] != self.N:
            raise ValueError(
                f"Expected weight dim {self.N}, got {weight.shape[0]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, self.N)
        residual = residual.contiguous().reshape(-1, self.N)
        M = x.shape[0]
        kernel = self._get_or_create_kernel(M)

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            residual = F.pad(residual, (0, self.N_padded - self.N))
            weight = F.pad(weight, (0, self.N_padded - self.N))

        y, residual_out = kernel(x, residual, weight)

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]
            residual_out = residual_out[:, :self.N]

        return y.reshape(orig_shape), residual_out.reshape(orig_shape)
