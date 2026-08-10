import math
from typing import Optional, Sequence, Tuple

import torch

from ..op_base import Op

__all__ = ["RMSNormFwdOp"]

_DEFAULT_EPS = 1e-6


class RMSNormFwdOp(Op):
    """Standalone Root Mean Square (RMS) Norm operator.

    Mirrors :func:`torch.nn.functional.rms_norm`. Computes::

        y = x * rsqrt(mean(x ** 2, trailing_axes) + eps) * weight

    where the reduction runs over the trailing ``len(normalized_shape)``
    axes; ``normalized_shape`` is the only entry point (the manifest spec).

    Dispatches through the backend registry: the target's ``get_kernel`` decides which
    kernel runs, so this class names none.

    Args:
        normalized_shape: Trailing-axis shape tuple over which the
            reduction runs (manifest ``params.normalized_shape``).
        eps: Epsilon for numerical stability (manifest ``params.eps``).
            ``None`` selects the documented default ``1e-6``. Canonicalized here, so
            the same call means the same thing on every target.
        target: Which backend serves this op, or ``None`` to detect from the input device.
        tune: Whether kernels this op builds tune themselves (default ``False``).

    Example:
        >>> op = RMSNormFwdOp(normalized_shape=(4096,))
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> w = torch.randn(4096, dtype=torch.float16, device="cuda")
        >>> y = op(x, w)  # shape: (1024, 4096)
    """

    OP_NAME = "RMSNormFwdOp"

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: Optional[float] = None,
        *,
        target: Optional[str] = None,
        tune: bool = False,
    ) -> None:
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        if len(self.normalized_shape) == 0:
            raise ValueError("normalized_shape must be non-empty")
        self.N = math.prod(self.normalized_shape)
        self.eps = _DEFAULT_EPS if eps is None else float(eps)
        self.target = target
        self.tune = tune
        self.dispatch_kernel()
        self._last_roofline_mn: Optional[Tuple[int, int]] = None

    def eval_roofline(self) -> Tuple[int, int]:
        if self._last_roofline_mn is None or self.dtype is None:
            raise RuntimeError(
                "RMSNormFwdOp.eval_roofline() requires a prior forward() "
                "call to bind the leading-dims product and the dtype."
            )
        m, n = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        return (4 * m * n, (2 * m * n + n) * elem_bytes)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization over the trailing ``normalized_shape``.

        Args:
            x: Input tensor whose trailing shape equals ``normalized_shape``.
            weight: Affine scale of shape ``normalized_shape``.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: Dtypes mismatch, or shapes are incompatible with the configured
                ``normalized_shape``.
        """
        tensors, params = self._bind_call(x, weight)
        x, weight = tensors
        ns = self.normalized_shape
        k = len(ns)
        self._validate_dtypes(x, weight)
        self.dtype = x.dtype
        if weight.device != x.device or weight.dtype != x.dtype:
            raise ValueError(
                f"weight must be on {x.device} with dtype {x.dtype}, "
                f"got {weight.device} and {weight.dtype}"
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

        # Lower the public call to what a kernel is handed: 2-D, contiguous. The kernel is
        # asked for after this, so the descriptions it is chosen by are the real ones.
        orig_shape = tuple(x.shape)
        x_flat = x.contiguous().reshape(-1, self.N)
        w_flat = weight.contiguous().reshape(self.N)

        y = self.backend_kernel(x_flat, w_flat, **params)(x_flat, w_flat)
        self._last_roofline_mn = (x_flat.shape[0], self.N)
        return y.reshape(orig_shape)
