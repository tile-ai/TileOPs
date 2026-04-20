from math import prod
from typing import Dict, Hashable, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import AdaLayerNormKernel

from ..op_base import Op

__all__ = ["AdaLayerNormZeroFwdOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class AdaLayerNormZeroFwdOp(Op):
    """Adaptive Layer Normalization-Zero (AdaLN-Zero) operator.

    Applies layer normalization with per-token adaptive scale, shift, and
    gating:

    .. math::

        y = g \\cdot \\left( s \\cdot \\frac{x - \\mathrm{E}[x]}
            {\\sqrt{\\mathrm{Var}[x] + \\epsilon}} + d \\right)

    where *s* (scale), *d* (shift), and *g* (gate) are per-token tensors of
    shape ``(*leading, N)``, pre-computed by the caller from a conditioning
    signal.  Linear projection from the conditioning input to scale/shift/gate
    is the caller's responsibility.

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
        return {"ada_layer_norm": AdaLayerNormKernel}

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Kernel cache key: the (M,) product of leading dims of ``x``."""
        x_shape = input_shapes[0]
        M = prod(x_shape[:-1])
        return (M,)

    def _get_or_create_kernel(self, M: int) -> Kernel:
        key = (M,)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["ada_layer_norm"](
                M, self.N, self.eps, self.dtype, has_gate=True, tune=self._tune,
            )
            self._kernel_cache[key] = kernel
        return kernel

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive layer normalization with zero-init gating.

        Args:
            x: Input tensor of shape ``(*leading, N)`` on CUDA.
            scale: Per-token scale tensor of shape ``(*leading, N)`` on CUDA.
            shift: Per-token shift tensor of shape ``(*leading, N)`` on CUDA.
            gate: Per-token gate tensor of shape ``(*leading, N)`` on CUDA.

        Returns:
            Normalized, modulated, and gated tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch,
                or shapes are incompatible with the configured dimensions.
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not scale.is_cuda:
            raise ValueError("scale must be a CUDA tensor")
        if not shift.is_cuda:
            raise ValueError("shift must be a CUDA tensor")
        if not gate.is_cuda:
            raise ValueError("gate must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        if scale.dtype != self.dtype:
            raise ValueError(
                f"Expected scale.dtype {self.dtype}, got {scale.dtype}"
            )
        if shift.dtype != self.dtype:
            raise ValueError(
                f"Expected shift.dtype {self.dtype}, got {shift.dtype}"
            )
        if gate.dtype != self.dtype:
            raise ValueError(
                f"Expected gate.dtype {self.dtype}, got {gate.dtype}"
            )
        # static_dims validation: x.shape[-1] == N (committed at ctor).
        if x.shape[-1] != self.N:
            raise ValueError(
                f"Expected hidden dim {self.N}, got {x.shape[-1]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, self.N)
        scale = scale.contiguous().reshape(-1, self.N)
        shift = shift.contiguous().reshape(-1, self.N)
        gate = gate.contiguous().reshape(-1, self.N)
        M = x.shape[0]
        kernel = self._get_or_create_kernel(M)

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            scale = F.pad(scale, (0, self.N_padded - self.N))
            shift = F.pad(shift, (0, self.N_padded - self.N))
            gate = F.pad(gate, (0, self.N_padded - self.N))

        y = kernel(x, scale, shift, gate)

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]

        return y.reshape(orig_shape)
