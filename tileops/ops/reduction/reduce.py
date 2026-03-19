"""Reduce ops: SumOp, MeanOp, AminOp, AmaxOp, ProdOp, StdOp, VarOp, VarMeanOp.

Each op reduces along dim=-1 and supports 1D-4D input.  The Op layer
validates inputs, reshapes to 2D (M_flat, N), pads to alignment, calls the
kernel, trims padding, and reshapes the output back.
"""

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
    """Base for single-output reduce ops along dim=-1."""

    _op_kind: str = ""  # overridden by subclasses

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self.N_padded = _align_up(N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["reduce"](
            M,
            N,
            self._op_kind,
            dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"reduce": ReduceKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.shape[-1] != self.N:
            raise ValueError(f"Expected last dim {self.N}, got {x.shape[-1]}")

        orig_shape = x.shape[:-1]  # output shape (leading dims)
        x = x.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(f"Expected M={self.M} (product of leading dims), got {M_actual}")

        # Pad to alignment
        if self.N_padded != self.N:
            pad_value = 1.0 if self._op_kind == "prod" else 0.0
            if self._op_kind in ("amin", "amax"):
                # For min/max, pad with appropriate extreme values
                pad_value = float("inf") if self._op_kind == "amin" else float("-inf")
            if pad_value == 0.0:
                x = F.pad(x, (0, self.N_padded - self.N))
            else:
                x = F.pad(x, (0, self.N_padded - self.N), value=pad_value)

        y = self.kernel(x)

        return y.reshape(orig_shape)


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
