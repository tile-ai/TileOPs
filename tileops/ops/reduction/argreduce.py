"""Arg-reduction operators (argmax, argmin)."""

from math import prod
from typing import Dict, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.argreduce import ArgreduceKernel

from .reduce import _ReduceOpBase

__all__ = ["ArgmaxFwdOp", "ArgminFwdOp"]


class _ArgreduceOpBase(_ReduceOpBase):
    """Reduce a non-last axis in place rather than transposing to reach it.

    The base transposes so the reduction axis is last, which copies the whole
    tensor. When the input is contiguous the copy is avoidable: the output axis
    is the contiguous one, so a kernel that assigns a thread per output element
    and strides along the reduction axis reads the original buffer coalesced.
    Only the preparation differs, so that is all this overrides.
    """

    def _get_or_create_strided_kernel(self, M: int, N: int, inner_stride: int, dtype):
        key = (M, N, dtype, inner_stride)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map[self._kernel_key](
                M, N, self._op_kind, dtype, inner_stride=inner_stride,
                tune=self._tune, **self._build_kernel_kwargs(),
            )
        return self._kernel_cache[key]

    def _prepare_input(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Size, Union[int, list], object]:
        if self.dim is None or not x.is_contiguous():
            return super()._prepare_input(x)
        if self.dim < -x.ndim or self.dim >= x.ndim:
            return super()._prepare_input(x)  # the base raises with the right message
        dim = self.dim % x.ndim
        inner_stride = prod(x.shape[dim + 1:])
        if inner_stride == 1:
            return super()._prepare_input(x)

        N = x.shape[dim]
        M = prod(s for i, s in enumerate(x.shape) if i != dim)
        self._last_roofline_mn = (M, N)
        kernel = self._get_or_create_strided_kernel(M, N, inner_stride, x.dtype)
        return x.reshape(-1), x.shape, dim, kernel


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
    _kernel_handles_padding = True

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
    _kernel_handles_padding = True

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
