from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import LayerNormKernel

from ..op_base import Op
from .norm_base import ALIGNMENT, align_up, normalized_shape_to_n

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
        Handles non-contiguous inputs and non-power-of-two hidden dims by
        padding to 256-element alignment. The leading-dims product ``M``
        is bound on the first forward call; if a subsequent call uses a
        different ``M``, the kernel is rebuilt for the new value.

    Args:
        normalized_shape: Trailing-axis shape tuple over which the
            reduction runs (manifest ``params.normalized_shape``).
        eps: Epsilon for numerical stability (manifest ``params.eps``).
            ``None`` uses the PyTorch default ``1e-5``.
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: Optional[float] = 1e-5,
        *,
        dtype: torch.dtype,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N = normalized_shape_to_n(normalized_shape)
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        self.dtype = dtype
        # Manifest declares ``eps: float | None`` with PyTorch default 1e-5.
        self.eps = 1e-5 if eps is None else float(eps)
        self.tune = tune
        self.N_padded = align_up(self.N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel: Optional[Kernel] = None
        self._last_m: Optional[int] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"layer_norm": LayerNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_m is None:
            raise RuntimeError(
                "LayerNormFwdOp.eval_roofline() requires a prior forward() "
                "call to bind the leading-dims product."
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
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if not bias.is_cuda:
            raise ValueError("bias must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        if weight.dtype != self.dtype:
            raise ValueError(
                f"Expected weight.dtype {self.dtype}, got {weight.dtype}"
            )
        if bias.dtype != self.dtype:
            raise ValueError(
                f"Expected bias.dtype {self.dtype}, got {bias.dtype}"
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
        if self.kernel is None or m_actual != self._last_m:
            self.kernel = self.kernel_map["layer_norm"](
                m_actual, self.N, self.eps, self.dtype, tune=self.tune,
            )
        self._last_m = m_actual

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            weight = F.pad(weight, (0, self.N_padded - self.N))
            bias = F.pad(bias, (0, self.N_padded - self.N))

        y = self.kernel(x, weight, bias)

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]

        return y.reshape(orig_shape)
