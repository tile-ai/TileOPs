"""Arg-reduction operators (argmax, argmin)."""

from math import prod
from typing import Dict, Optional

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.argreduce import ArgreduceKernel

from .reduce import _ReduceOpBase

__all__ = ["ArgmaxFwdOp", "ArgminFwdOp"]


class _ArgreduceOpBase(_ReduceOpBase):
    """Prepare argreduce inputs without materializing a transposed tensor.

    A contiguous non-last-axis input is flattened in its existing layout and
    the product of trailing dimensions is passed to :class:`ArgreduceKernel`
    as ``inner_stride``.  Truly non-contiguous inputs retain the compatibility
    fallback through ``movedim(...).contiguous()``.
    """

    def _get_or_create_strided_kernel(
        self,
        M: int,
        N: int,
        inner_stride: int,
    ) -> object:
        key = (M, N, inner_stride)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map[self._kernel_key]
            self._kernel_cache[key] = kernel_cls(
                M,
                N,
                self._op_kind,
                self.dtype,
                inner_stride=inner_stride,
                tune=self._tune,
                **self._build_kernel_kwargs(),
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run argreduce with stride-aware input/output traversal."""
        self._validate_input_tensor(x)
        orig_shape = x.shape

        if self.dim is None:
            # torch.argmax/argmin(dim=None) use logical contiguous order.
            N = x.numel()
            M = 1
            dim_info = list(range(x.ndim))
            x_flat = x.contiguous().reshape(-1)
            inner_stride = 1
        else:
            if self.dim < -x.ndim or self.dim >= x.ndim:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[{-x.ndim}, {x.ndim - 1}], but got {self.dim})"
                )
            dim = self.dim % x.ndim
            N = x.shape[dim]
            M = prod(s for i, s in enumerate(x.shape) if i != dim)
            dim_info = dim

            if x.is_contiguous():
                # Keep the original layout.  Adjacent output threads read
                # adjacent elements for non-last-axis reductions.
                inner_stride = prod(x.shape[dim + 1 :])
                x_flat = x.reshape(-1)
            else:
                # Storage strides are not part of the public Kernel contract;
                # compact unusual views once, then use the contiguous path.
                if dim != x.ndim - 1:
                    x = x.movedim(dim, -1)
                x_flat = x.contiguous().reshape(-1)
                inner_stride = 1

        self._last_roofline_mn = (M, N)
        kernel = self._get_or_create_strided_kernel(M, N, inner_stride)
        y = kernel(x_flat)
        return self._reshape_output(y, orig_shape, dim_info)


class ArgmaxFwdOp(_ArgreduceOpBase):
    """Argmax reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgmaxFwdOp(dim=None, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dim: Reduction dimension. ``None`` (the default) matches
            ``torch.argmax(x)`` semantics: the input is treated as a
            contiguous flattened 1D buffer and the returned index is into
            that flattened tensor.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "argmax"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel

    def __init__(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _validate_dim(self) -> None:
        """Argmax accepts a scalar ``int`` dim or ``None`` (full-tensor reduction).

        ``dim=None`` matches ``torch.argmax(x)`` semantics: the input is
        treated as a contiguous flattened 1D buffer and the returned index
        is into that flattened tensor.
        """
        if self.dim is None or isinstance(self.dim, int):
            return
        raise ValueError(
            f"ArgmaxFwdOp only supports scalar dim (int) or None, "
            f"got {type(self.dim).__name__}: {self.dim!r}"
        )

    def _pad_value(self) -> float:
        """Pad with -inf so padded positions never win argmax."""
        return float("-inf")



class ArgminFwdOp(_ArgreduceOpBase):
    """Argmin reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgminFwdOp(dim=None, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dim: Reduction dimension. ``None`` (the default) matches
            ``torch.argmin(x)`` semantics: the input is treated as a
            contiguous flattened 1D buffer and the returned index is into
            that flattened tensor.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "argmin"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel

    def __init__(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _validate_dim(self) -> None:
        """Argmin accepts a scalar ``int`` dim or ``None`` (full-tensor reduction).

        ``dim=None`` matches ``torch.argmin(x)`` semantics: the input is
        treated as a contiguous flattened 1D buffer and the returned index
        is into that flattened tensor.
        """
        if self.dim is None or isinstance(self.dim, int):
            return
        raise ValueError(
            f"ArgminFwdOp only supports scalar dim (int) or None, "
            f"got {type(self.dim).__name__}: {self.dim!r}"
        )

    def _pad_value(self) -> float:
        """Pad with +inf so padded positions never win argmin."""
        return float("inf")
