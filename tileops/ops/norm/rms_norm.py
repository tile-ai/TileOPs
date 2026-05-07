import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import RMSNormKernel

from ..op_base import Op
from .norm_base import ALIGNMENT, align_up

__all__ = ["RMSNormFwdOp"]

_DEFAULT_EPS = 1e-6


class RMSNormFwdOp(Op):
    """Standalone Root Mean Square (RMS) Norm operator.

    Mirrors :func:`torch.nn.functional.rms_norm`. Computes::

        y = x * rsqrt(mean(x ** 2, trailing_axes) + eps) * weight

    where the reduction runs over the trailing ``len(normalized_shape)``
    axes; ``normalized_shape`` is the only entry point (the manifest spec).

    Args:
        normalized_shape: Trailing-axis shape tuple over which the
            reduction runs (manifest ``params.normalized_shape``).
        eps: Epsilon for numerical stability (manifest ``params.eps``).
            ``None`` selects the implementation default ``1e-6``.
        dtype: Data type (``torch.float16`` or ``torch.bfloat16``).
        kernel_map: Optional kernel override dictionary.
        tune: Whether to autotune (default ``False``).

    Example:
        >>> op = RMSNormFwdOp(normalized_shape=(4096,), dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> w = torch.randn(4096, dtype=torch.float16, device="cuda")
        >>> y = op(x, w)  # shape: (1024, 4096)
    """

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: Optional[float] = None,
        *,
        dtype: torch.dtype,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        if len(self.normalized_shape) == 0:
            raise ValueError("normalized_shape must be non-empty")
        self.N = math.prod(self.normalized_shape)
        self.dtype = dtype
        self.eps = _DEFAULT_EPS if eps is None else float(eps)
        self.tune = tune
        self.N_padded = align_up(self.N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[int, Kernel] = {}
        self._last_roofline_mn: Optional[Tuple[int, int]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"rms_norm": RMSNormKernel}

    def _get_kernel(self, m: int) -> Kernel:
        if m not in self._kernel_cache:
            self._kernel_cache[m] = self.kernel_map["rms_norm"](
                m, self.N, self.eps, self.dtype, tune=self.tune,
            )
        return self._kernel_cache[m]

    def eval_roofline(self) -> Tuple[int, int]:
        if self._last_roofline_mn is None:
            raise RuntimeError(
                "RMSNormFwdOp.eval_roofline() requires a prior forward() "
                "call to bind the leading-dims product."
            )
        m, n = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        return (4 * m * n, (2 * m * n + n) * elem_bytes)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization over the trailing ``normalized_shape``.

        Args:
            x: Input tensor with trailing shape equal to
                ``normalized_shape`` on CUDA.
            weight: Affine scale of shape ``normalized_shape`` on CUDA.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch, or
                shapes are incompatible with the configured
                ``normalized_shape``.
        """
        ns = self.normalized_shape
        k = len(ns)
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if not weight.is_cuda or weight.dtype != self.dtype:
            raise ValueError(
                f"weight must be a CUDA tensor of dtype {self.dtype}"
            )
        if x.ndim < k or tuple(x.shape[-k:]) != ns:
            raise ValueError(
                f"Expected x trailing shape {ns}, "
                f"got {tuple(x.shape[-k:]) if x.ndim >= k else tuple(x.shape)}"
            )
        if tuple(weight.shape) != ns:
            raise ValueError(
                f"Expected weight shape {ns}, got {tuple(weight.shape)}"
            )

        orig_shape = tuple(x.shape)
        x_flat = x.contiguous().reshape(-1, self.N)
        w_flat = weight.contiguous().reshape(self.N)
        m = x_flat.shape[0]
        if self.N_padded != self.N:
            x_flat = F.pad(x_flat, (0, self.N_padded - self.N))
            w_flat = F.pad(w_flat, (0, self.N_padded - self.N))
        y = self._get_kernel(m)(x_flat, w_flat)
        if self.N_padded != self.N:
            y = y[:, : self.N]
        self._last_roofline_mn = (m, self.N)
        return y.reshape(orig_shape)
