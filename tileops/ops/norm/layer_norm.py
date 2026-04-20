from typing import Dict, Hashable, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import LayerNormKernel

from ..op_base import Op

__all__ = ["LayerNormFwdOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class LayerNormFwdOp(Op):
    """Layer Normalization operator.

    Computes layer normalization over the last dimension:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    where the mean and variance are computed over the last dimension.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims
        by padding to 256-element alignment.

    Args:
        N: Hidden dimension (last dim). Committed at construction per
            manifest ``static_dims``; forward validates ``x.shape[-1] == N``.
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    # The committed static axis (last of x) is not a fixed non-negative
    # index; `_cache_key` is overridden below, so `_static_axes` remains
    # empty (the default path is unused).

    def __init__(
        self,
        *,
        N: int,
        dtype: torch.dtype,
        eps: float = 1e-5,
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
        return {"layer_norm": LayerNormKernel}

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Kernel cache key: the (M,) product of leading dims of ``x``.

        The kernel math depends only on ``(M, N)`` and ``N`` is committed
        at construction, so keying by ``M`` alone is sufficient.
        """
        x_shape = input_shapes[0]
        M = 1
        for s in x_shape[:-1]:
            M *= s
        return (M,)

    def _get_or_create_kernel(self, M: int) -> Kernel:
        key = (M,)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["layer_norm"](
                M, self.N, self.eps, self.dtype, tune=self._tune,
            )
            self._kernel_cache[key] = kernel
        return kernel

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape ``(*leading, N)`` on CUDA.
            weight: Affine scale of shape ``(N,)`` on CUDA.
            bias: Affine shift of shape ``(N,)`` on CUDA.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch,
                or shapes are incompatible with the configured dimensions.
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
        if weight.ndim != 1:
            raise ValueError(
                f"Expected weight to be 1D, got {weight.ndim}D"
            )
        if bias.ndim != 1:
            raise ValueError(
                f"Expected bias to be 1D, got {bias.ndim}D"
            )
        # static_dims validation: x.shape[-1] == N (committed at ctor).
        if x.shape[-1] != self.N:
            raise ValueError(
                f"Expected hidden dim {self.N}, got {x.shape[-1]}"
            )
        if weight.shape[0] != self.N:
            raise ValueError(
                f"Expected weight dim {self.N}, got {weight.shape[0]}"
            )
        if bias.shape[0] != self.N:
            raise ValueError(
                f"Expected bias dim {self.N}, got {bias.shape[0]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, self.N)
        M = x.shape[0]
        kernel = self._get_or_create_kernel(M)

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            weight = F.pad(weight, (0, self.N_padded - self.N))
            bias = F.pad(bias, (0, self.N_padded - self.N))

        y = kernel(x, weight, bias)

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]

        return y.reshape(orig_shape)
