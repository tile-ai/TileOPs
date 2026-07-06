from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import FusedAddLayerNormKernel

from ..op_base import Op
from .norm_base import ALIGNMENT, align_up

__all__ = ["FusedAddLayerNormFwdOp"]


class FusedAddLayerNormFwdOp(Op):
    """Fused residual addition and Layer Normalization operator.

    Computes the residual sum followed by layer normalization in a single
    fused kernel:

    .. math::

        \\begin{aligned}
        r &= x + \\mathrm{residual} \\\\
        y &= \\frac{r - \\mathrm{E}[r]}{\\sqrt{\\mathrm{Var}[r] + \\epsilon}}
            \\cdot w + b
        \\end{aligned}

    Returns dual outputs ``(y, residual_out)`` so downstream residual connections can
    reuse the pre-norm sum without recomputation.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims
        by padding to 256-element alignment.

    Args:
        M: Optional committed row count for strict compatibility. Preferred
            API infers it from ``x.shape[:-1]``.
        N: Optional committed hidden dimension. Preferred API infers it from
            ``x.shape[-1]``.
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        M: Optional[int] = None,
        N: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self._committed_M = M
        self._committed_N = N
        self._committed_dtype = dtype
        self.eps = eps
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple[int, int, torch.dtype, int | None], Kernel] = {}
        self._last_roofline_mn: Optional[tuple[int, int]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_add_layer_norm": FusedAddLayerNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        return (
            6 * M * N,
            (4 * M * N + 2 * N) * elem_bytes,
        )

    def _get_kernel(
        self, M: int, N: int, dtype: torch.dtype, device_index: int | None,
    ) -> Kernel:
        key = (M, N, dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["fused_add_layer_norm"](
                M, N, self.eps, dtype, tune=self.tune,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual addition and layer normalization.

        Args:
            x: Input tensor of shape ``(*leading, N)`` on CUDA.
            residual: Residual tensor of the same shape as *x* on CUDA.
            weight: Affine scale of shape ``(N,)`` on CUDA.
            bias: Affine shift of shape ``(N,)`` on CUDA.

        Returns:
            Tuple of ``(y, residual_out)`` where *y* is the normalized
            output and *residual_out* is ``x + residual``, both of the
            same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch,
                or shapes are incompatible with the configured dimensions.
        """
        expected_dtype = self._committed_dtype
        for name, tensor in [("x", x), ("residual", residual), ("weight", weight), ("bias", bias)]:
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if expected_dtype is not None and tensor.dtype != expected_dtype:
                raise ValueError(
                    f"Expected {name}.dtype {expected_dtype}, got {tensor.dtype}"
                )
            if expected_dtype is None:
                expected_dtype = tensor.dtype
        if weight.ndim != 1:
            raise ValueError(
                f"Expected weight to be 1D, got {weight.ndim}D"
            )
        if bias.ndim != 1:
            raise ValueError(
                f"Expected bias to be 1D, got {bias.ndim}D"
            )
        N = x.shape[-1]
        if self._committed_N is not None and self._committed_N != N:
            raise ValueError(
                f"Expected hidden dim {self._committed_N}, got {N}"
            )
        if residual.shape != x.shape:
            raise ValueError(
                f"Expected residual shape {x.shape}, got {residual.shape}"
            )
        if weight.shape[0] != N:
            raise ValueError(
                f"Expected weight dim {N}, got {weight.shape[0]}"
            )
        if bias.shape[0] != N:
            raise ValueError(
                f"Expected bias dim {N}, got {bias.shape[0]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, N)
        residual = residual.contiguous().reshape(-1, N)
        M_actual = x.shape[0]
        if self._committed_M is not None and M_actual != self._committed_M:
            raise ValueError(
                f"Expected M={self._committed_M} (product of leading dims), got {M_actual}"
            )
        self.M = M_actual
        self.N = N
        dtype = expected_dtype
        assert dtype is not None
        self.dtype = dtype
        N_padded = align_up(N, ALIGNMENT)

        # Pad hidden dim to 256-element alignment if needed
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N))
            residual = F.pad(residual, (0, N_padded - N))
            weight = F.pad(weight, (0, N_padded - N))
            bias = F.pad(bias, (0, N_padded - N))

        kernel = self._get_kernel(M_actual, N, dtype, x.device.index)
        y, residual_out = kernel(x, residual, weight, bias)
        self._last_roofline_mn = (M_actual, N)

        # Trim padding
        if N_padded != N:
            y = y[:, :N]
            residual_out = residual_out[:, :N]

        return y.reshape(orig_shape), residual_out.reshape(orig_shape)
