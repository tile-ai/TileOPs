from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm import FusedAddLayerNormKernel

from ..op import Op

__all__ = ["FusedAddLayerNormFwdOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


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
        M: Number of rows (product of all dims except the last).
        N: Hidden dimension (last dim).
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        eps: float = 1e-5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self.eps = eps
        self.N_padded = _align_up(N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["fused_add_layer_norm"](
            M, N, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_add_layer_norm": FusedAddLayerNormKernel}

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
        for name, tensor in [("x", x), ("residual", residual), ("weight", weight), ("bias", bias)]:
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
        if bias.ndim != 1:
            raise ValueError(
                f"Expected bias to be 1D, got {bias.ndim}D"
            )
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
        if bias.shape[0] != self.N:
            raise ValueError(
                f"Expected bias dim {self.N}, got {bias.shape[0]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, self.N)
        residual = residual.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(
                f"Expected M={self.M} (product of leading dims), got {M_actual}"
            )

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            residual = F.pad(residual, (0, self.N_padded - self.N))
            weight = F.pad(weight, (0, self.N_padded - self.N))
            bias = F.pad(bias, (0, self.N_padded - self.N))

        y, residual_out = self.kernel(x, residual, weight, bias)

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]
            residual_out = residual_out[:, :self.N]

        return y.reshape(orig_shape), residual_out.reshape(orig_shape)
