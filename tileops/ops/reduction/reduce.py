"""Reduce ops: SumOp, MeanOp, AminOp, AmaxOp, ProdOp, StdOp, VarOp, VarMeanOp.

Each op reduces along the configured ``dim`` and supports arbitrary-rank input.
The Op layer validates inputs, reshapes to 2D (M, N), pads to alignment,
calls the kernel, trims padding, and reshapes the output back.  Kernels are
cached by ``(M, N)`` so that the same op instance can handle varying shapes.
"""

from math import prod
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction.reduce import ReduceKernel

from ..op import Op

__all__ = [
    "SumOp",
    "MeanOp",
    "AminOp",
    "AmaxOp",
    "ProdOp",
    "StdOp",
    "VarOp",
    "VarMeanOp",
]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Base class for simple reduce ops (sum, mean, amin, amax, prod)
# ---------------------------------------------------------------------------


class _SimpleReduceOp(Op):
    """Base for single-output reduce ops (sum, mean, amin, amax, prod).

    Construction: ``op(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).  Only a single ``int`` is
            supported; passing ``list[int]`` raises ``NotImplementedError``.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind: str = ""  # overridden by subclasses

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: int = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if isinstance(dim, (list, tuple)):
            raise NotImplementedError("Multi-dim reduction not yet supported")
        self.dtype = dtype
        self.dim = dim
        self.keepdim = keepdim
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"reduce": ReduceKernel}

    # ------------------------------------------------------------------
    # Kernel cache
    # ------------------------------------------------------------------

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed."""
        key = (M, N)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map["reduce"]
            self._kernel_cache[key] = kernel_cls(
                M, N, self._op_kind, self.dtype, tune=self._tune
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Pad value
    # ------------------------------------------------------------------

    def _pad_value(self) -> float:
        """Return the identity element used when padding to alignment."""
        if self._op_kind == "prod":
            return 1.0
        if self._op_kind == "amin":
            return float("inf")
        if self._op_kind == "amax":
            return float("-inf")
        return 0.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the reduce op on *x* along the configured dim."""
        # --- validation ---
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

        orig_shape = x.shape

        # Validate and normalize dim (match PyTorch IndexError message).
        if self.dim < -x.ndim or self.dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {self.dim})"
            )
        dim = self.dim % x.ndim

        # N = size along reduction dim, M = product of all other dims.
        N = x.shape[dim]
        M = prod(s for i, s in enumerate(x.shape) if i != dim)

        # If reduction dim is not the last, move it to the end.
        needs_transpose = dim != x.ndim - 1
        if needs_transpose:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        # Get or create cached kernel for this (M, N).
        kernel = self._get_or_create_kernel(M, N)

        # Pad to alignment.
        N_padded = _align_up(N, ALIGNMENT)
        if N_padded != N:
            pv = self._pad_value()
            x = F.pad(x, (0, N_padded - N)) if pv == 0.0 else F.pad(x, (0, N_padded - N), value=pv)

        y = kernel(x)

        # --- reshape output ---
        if self.keepdim:
            kept_shape = list(orig_shape)
            kept_shape[dim] = 1
            y = y.reshape(kept_shape)
        else:
            reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
            y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)

        return y


class SumOp(_SimpleReduceOp):
    """Sum reduction along dim=-1."""

    _op_kind = "sum"


class MeanOp(_SimpleReduceOp):
    """Mean reduction along dim=-1."""

    _op_kind = "mean"


class AminOp(_SimpleReduceOp):
    """Amin (element-wise minimum) reduction along dim=-1."""

    _op_kind = "amin"


class AmaxOp(_SimpleReduceOp):
    """Amax (element-wise maximum) reduction along dim=-1."""

    _op_kind = "amax"


class ProdOp(_SimpleReduceOp):
    """Product reduction along dim=-1."""

    _op_kind = "prod"


# ---------------------------------------------------------------------------
# Welford-based ops (std, var, var_mean)
# ---------------------------------------------------------------------------


class _WelfordReduceOp(Op):
    """Base for Welford-based reduce ops along dim=-1."""

    _op_kind: str = ""

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        correction: int = 1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self.correction = correction
        self.N_padded = _align_up(N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["reduce"](
            M,
            N,
            self._op_kind,
            dtype,
            correction=correction,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"reduce": ReduceKernel}

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """Validate, reshape, and pad input. Returns (x_2d_padded, orig_shape)."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.shape[-1] != self.N:
            raise ValueError(f"Expected last dim {self.N}, got {x.shape[-1]}")

        orig_shape = x.shape[:-1]
        x = x.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(f"Expected M={self.M} (product of leading dims), got {M_actual}")

        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))

        return x, orig_shape


class StdOp(_WelfordReduceOp):
    """Standard deviation reduction along dim=-1 with Bessel's correction."""

    _op_kind = "std"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, orig_shape = self._prepare_input(x)
        y = self.kernel(x)
        return y.reshape(orig_shape)


class VarOp(_WelfordReduceOp):
    """Variance reduction along dim=-1 with Bessel's correction."""

    _op_kind = "var"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, orig_shape = self._prepare_input(x)
        y = self.kernel(x)
        return y.reshape(orig_shape)


class VarMeanOp(_WelfordReduceOp):
    """Variance and mean reduction along dim=-1."""

    _op_kind = "var_mean"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, orig_shape = self._prepare_input(x)
        var_out, mean_out = self.kernel(x)
        return var_out.reshape(orig_shape), mean_out.reshape(orig_shape)
