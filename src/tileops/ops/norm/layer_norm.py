from typing import Optional, Sequence

import torch

from ..op_base import Op
from .norm_base import normalized_shape_to_n

__all__ = ["LayerNormFwdOp"]


class LayerNormFwdOp(Op):
    """Layer Normalization operator.

    Computes layer normalization over the trailing ``normalized_shape``
    axes:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    Mirrors :func:`torch.nn.functional.layer_norm`. ``normalized_shape``
    is the only entry point (the manifest spec).

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims. For
        non-aligned hidden dims, boundary handling is performed inside the
        kernel rather than by allocating padded tensors in the Op layer. The
        leading-dims product ``M`` is bound on the first forward call; if a
        subsequent call uses a different ``M``, the kernel is rebuilt for the
        new value.

    Args:
        normalized_shape: Trailing-axis shape tuple over which the
            reduction runs (manifest ``params.normalized_shape``).
        eps: Epsilon for numerical stability (manifest ``params.eps``).
            ``None`` uses the PyTorch default ``1e-5``.
        target: Which backend serves this op, or ``None`` to detect from the input device.
        tune: If ``True``, autotune tile configurations.
    """

    OP_NAME = "LayerNormFwdOp"

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: Optional[float] = 1e-5,
        *,
        target: Optional[str] = None,
        tune: bool = False,
    ):
        self.N = normalized_shape_to_n(normalized_shape)
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        # Manifest declares ``eps: float | None`` with PyTorch default 1e-5.
        self.eps = 1e-5 if eps is None else float(eps)
        self.target = target
        self.tune = tune
        self.dispatch_kernel()
        self._last_m: Optional[int] = None


    def eval_roofline(self) -> tuple[int, int]:
        if self._last_m is None or self.dtype is None:
            raise RuntimeError(
                "LayerNormFwdOp.eval_roofline() requires a prior forward() "
                "call to bind the leading-dims product and the dtype."
            )
        elem_bytes = self.dtype.itemsize
        m = self._last_m
        return (
            5 * m * self.N,
            (2 * m * self.N + 2 * self.N) * elem_bytes,
        )

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    ) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor with trailing shape equal to
                ``normalized_shape`` on CUDA.
            weight: Affine scale of shape ``normalized_shape`` on CUDA.
            bias: Affine shift of shape ``normalized_shape`` on CUDA.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch, or
                shapes are incompatible with the configured
                ``normalized_shape``.
        """
        (x, weight, bias), params = self._bind_call(x, weight, bias)
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if not bias.is_cuda:
            raise ValueError("bias must be a CUDA tensor")
        self._validate_dtypes(x, weight, bias)
        self.dtype = x.dtype
        if weight.dtype != x.dtype:
            raise ValueError(
                f"Expected weight.dtype {x.dtype}, got {weight.dtype}"
            )
        if bias.dtype != x.dtype:
            raise ValueError(
                f"Expected bias.dtype {x.dtype}, got {bias.dtype}"
            )

        ns = self.normalized_shape
        k = len(ns)
        if x.ndim < k or tuple(x.shape[-k:]) != ns:
            raise ValueError(
                f"Expected x trailing shape {ns}, "
                f"got {tuple(x.shape[-k:]) if x.ndim >= k else tuple(x.shape)}"
            )
        if tuple(weight.shape) != ns:
            raise ValueError(
                f"Expected weight shape {ns}, got {tuple(weight.shape)}"
            )
        if tuple(bias.shape) != ns:
            raise ValueError(
                f"Expected bias shape {ns}, got {tuple(bias.shape)}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, self.N)
        weight = weight.contiguous().reshape(self.N)
        bias = bias.contiguous().reshape(self.N)
        m_actual = x.shape[0]
        kernel = self.backend_kernel(x, weight, bias, **params)
        self._last_m = m_actual

        y = kernel(x, weight, bias)

        return y.reshape(orig_shape)
