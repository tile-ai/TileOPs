from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import LayerNormKernel

from ..op_base import Op
from .norm_base import normalized_shape_to_n

__all__ = ["LayerNormFwdOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class LayerNormFwdOp(Op):
    """Layer Normalization operator.

    Computes layer normalization over the trailing ``normalized_shape``
    axes:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    Mirrors :func:`torch.nn.functional.layer_norm`.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims
        by padding to 256-element alignment. The leading-dims product
        ``M`` is bound at forward time and kernels are cached per ``M``.

    Args:
        normalized_shape: Trailing-axis shape tuple over which the reduction
            runs (manifest ``params.normalized_shape``). Either this or
            legacy ``N`` must be set.
        eps: Epsilon for numerical stability (manifest ``params.eps``).
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        N: Legacy single-axis reduction size; mutually exclusive with
            ``normalized_shape``.
        M: Optional pre-bound leading-dims product. When omitted, ``M`` is
            derived from the input at forward time.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        normalized_shape: Optional[Sequence[int]] = None,
        eps: float = 1e-5,
        *,
        dtype: torch.dtype,
        N: Optional[int] = None,
        M: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.normalized_shape = (
            tuple(int(d) for d in normalized_shape)
            if normalized_shape is not None else None
        )
        self.N = normalized_shape_to_n(normalized_shape, n_fallback=N)
        self.dtype = dtype
        self.eps = float(eps)
        self.tune = tune
        self.N_padded = _align_up(self.N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.M: Optional[int] = M
        self._kernel_cache: Dict[int, Kernel] = {}
        if M is not None:
            self._kernel_cache[M] = self.kernel_map["layer_norm"](
                M, self.N, self.eps, dtype, tune=tune,
            )
        self._last_m: Optional[int] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"layer_norm": LayerNormKernel}

    def _get_kernel(self, m: int) -> Kernel:
        if m not in self._kernel_cache:
            self._kernel_cache[m] = self.kernel_map["layer_norm"](
                m, self.N, self.eps, self.dtype, tune=self.tune,
            )
        return self._kernel_cache[m]

    def eval_roofline(self) -> tuple[int, int]:
        m = self._last_m if self._last_m is not None else self.M
        if m is None:
            raise RuntimeError(
                "LayerNormFwdOp.eval_roofline() requires a prior "
                "forward() call (or ctor M) to bind the leading-dims product."
            )
        elem_bytes = self.dtype.itemsize
        return (
            5 * m * self.N,
            (2 * m * self.N + 2 * self.N) * elem_bytes,
        )

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    ) -> torch.Tensor:
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
        if self.normalized_shape is not None:
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
        else:
            if weight.ndim != 1:
                raise ValueError(
                    f"Expected weight to be 1D, got {weight.ndim}D"
                )
            if bias.ndim != 1:
                raise ValueError(
                    f"Expected bias to be 1D, got {bias.ndim}D"
                )
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
        weight = weight.contiguous().reshape(self.N)
        bias = bias.contiguous().reshape(self.N)
        m_actual = x.shape[0]
        if self.M is not None and m_actual != self.M:
            raise ValueError(
                f"Expected M={self.M} (product of leading dims), "
                f"got {m_actual}"
            )
        self._last_m = m_actual

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            weight = F.pad(weight, (0, self.N_padded - self.N))
            bias = F.pad(bias, (0, self.N_padded - self.N))

        y = self._get_kernel(m_actual)(x, weight, bias)

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]

        return y.reshape(orig_shape)
