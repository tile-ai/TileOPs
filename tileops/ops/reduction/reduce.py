"""Reduce ops: SumOp, MeanOp, AminOp, AmaxOp, ProdOp, StdOp, VarOp, VarMeanOp.

Each op reduces along the configured ``dim`` and supports arbitrary-rank input.
The ``dim`` parameter accepts ``int`` or ``list[int]`` for multi-dim reduction.
The Op layer validates inputs, reshapes to 2D (M, N), pads to alignment,
calls the kernel, trims padding, and reshapes the output back.  Kernels are
cached by ``(M, N)`` so that the same op instance can handle varying shapes.
"""

from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction.reduce import ReduceKernel

from ..op import Op
from ._multidim import flatten_for_multidim, normalize_dim, restore_multidim_shape

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
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind: str = ""  # overridden by subclasses

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int]] = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
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

        # --- multi-dim path ---
        if isinstance(self.dim, (list, tuple)):
            dims = normalize_dim(self.dim, x.ndim)
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            # x now has target dims flattened into the last dim.
            # Reduce along the last dim (single-dim path on reshaped tensor).
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            x = x.reshape(M, N)
            kernel = self._get_or_create_kernel(M, N)
            N_padded = _align_up(N, ALIGNMENT)
            if N_padded != N:
                pv = self._pad_value()
                pad = (0, N_padded - N)
                x = F.pad(x, pad) if pv == 0.0 else F.pad(x, pad, value=pv)
            y = kernel(x)
            return restore_multidim_shape(y, orig_shape, dims, self.keepdim)

        # --- single-dim path ---
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
        if dim != x.ndim - 1:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        # Get or create cached kernel for this (M, N).
        kernel = self._get_or_create_kernel(M, N)

        # Pad to alignment.
        N_padded = _align_up(N, ALIGNMENT)
        if N_padded != N:
            pv = self._pad_value()
            pad = (0, N_padded - N)
            x = F.pad(x, pad) if pv == 0.0 else F.pad(x, pad, value=pv)

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
    """Base for Welford-based reduce ops (std, var, var_mean).

    Construction: ``op(dtype=..., dim=-1, correction=1, keepdim=False)``.
    M and N are derived from the input tensor at forward time, and kernels
    are cached by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
        correction: Bessel's correction (default 1).
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind: str = ""

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int]] = -1,
        correction: int = 1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.dtype = dtype
        self.dim = dim
        self.correction = correction
        self.keepdim = keepdim
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"reduce": ReduceKernel}

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed."""
        key = (M, N)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map["reduce"]
            self._kernel_cache[key] = kernel_cls(
                M, N, self._op_kind, self.dtype,
                correction=self.correction, tune=self._tune,
            )
        return self._kernel_cache[key]

    def _forward_common(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Size, object, object]:
        """Validate, derive M/N, transpose, reshape, pad.

        Returns (x_2d_padded, orig_shape, dim_info, kernel) where
        dim_info is either an int (single-dim) or list[int] (multi-dim).
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

        orig_shape = x.shape

        # --- multi-dim path ---
        if isinstance(self.dim, (list, tuple)):
            dims = normalize_dim(self.dim, x.ndim)
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            x = x.reshape(M, N)
            kernel = self._get_or_create_kernel(M, N)
            N_padded = _align_up(N, ALIGNMENT)
            if N_padded != N:
                x = F.pad(x, (0, N_padded - N))
            return x, orig_shape, dims, kernel

        # --- single-dim path ---
        if self.dim < -x.ndim or self.dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {self.dim})"
            )
        dim = self.dim % x.ndim

        N = x.shape[dim]
        M = prod(s for i, s in enumerate(x.shape) if i != dim)

        if dim != x.ndim - 1:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        kernel = self._get_or_create_kernel(M, N)

        N_padded = _align_up(N, ALIGNMENT)
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N))

        return x, orig_shape, dim, kernel

    def _reshape_output(
        self, y: torch.Tensor, orig_shape: torch.Size, dim_info: object,
    ) -> torch.Tensor:
        """Reshape (M,) kernel output to match keepdim setting.

        dim_info is either an int (single-dim) or list[int] (multi-dim).
        """
        if isinstance(dim_info, list):
            return restore_multidim_shape(y, orig_shape, dim_info, self.keepdim)

        dim = dim_info
        if self.keepdim:
            kept_shape = list(orig_shape)
            kept_shape[dim] = 1
            return y.reshape(kept_shape)
        else:
            reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
            return y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)


class StdOp(_WelfordReduceOp):
    """Standard deviation reduction with Bessel's correction."""

    _op_kind = "std"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, orig_shape, dim, kernel = self._forward_common(x)
        y = kernel(x)
        return self._reshape_output(y, orig_shape, dim)


class VarOp(_WelfordReduceOp):
    """Variance reduction with Bessel's correction."""

    _op_kind = "var"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, orig_shape, dim, kernel = self._forward_common(x)
        y = kernel(x)
        return self._reshape_output(y, orig_shape, dim)


class VarMeanOp(_WelfordReduceOp):
    """Variance and mean reduction."""

    _op_kind = "var_mean"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, orig_shape, dim, kernel = self._forward_common(x)
        var_out, mean_out = kernel(x)
        return (
            self._reshape_output(var_out, orig_shape, dim),
            self._reshape_output(mean_out, orig_shape, dim),
        )
