import math
from typing import Dict, Optional, Sequence, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import RMSNormKernel

from ..op_base import Op

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
            ``None`` selects the documented default ``1e-6``. Normalized here, so a
            backend is handed the number rather than ``None``.
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
            in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional kernel override dictionary.
        tune: Whether to autotune (default ``False``).

    Example:
        >>> op = RMSNormFwdOp(normalized_shape=(4096,))
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> w = torch.randn(4096, dtype=torch.float16, device="cuda")
        >>> y = op(x, w)  # shape: (1024, 4096)
    """

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: Optional[float] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        if len(self.normalized_shape) == 0:
            raise ValueError("normalized_shape must be non-empty")
        self.N = math.prod(self.normalized_shape)
        self.eps = _DEFAULT_EPS if eps is None else float(eps)
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._last_roofline_mn: Optional[Tuple[int, int]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"rms_norm": RMSNormKernel}

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
            ValueError: Dtypes or devices disagree, or shapes are incompatible with the
                configured ``normalized_shape``.
        """
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

        # The op normalizes contiguity and hands over what the manifest declares; how a
        # kernel wants that laid out is its own business.
        x = x.contiguous()
        weight = weight.contiguous()
        kernel = self.get_or_build_kernel(
            "rms_norm",
            (x, weight),
            key=x.dtype,
            build=lambda: self.kernel_map["rms_norm"](
                self.N, self.eps, x.dtype, tune=self.tune,
            ),
        )
        self._last_roofline_mn = (x.numel() // self.N, self.N)
        return kernel(x, weight)
