"""Concatenation operator (host-side Op layer).

Provides:
  - ConcatFwdOp: output = torch.cat(tensors, dim)
"""

import functools
import math
from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.indexing import ConcatCall, ConcatFwdInterface, ConcatKernel
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target
from tileops.manifest.primitives import normalize_axis

__all__ = ["ConcatFwdOp"]


class ConcatFwdOp(Op):
    """Tensors joined along one axis: ``torch.cat(tensors, dim)``.

    The elements have one rank and are equal in every axis but ``dim``. The output takes the
    dtype the elements' dtypes promote to, as ``torch.cat`` does; each element converts to it
    exactly.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"concat": ConcatKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"concat": ConcatFwdInterface}

    def __init__(self, dim: int = 0, *, target: Target = None) -> None:
        """Build the op. The shapes and dtypes are taken from each call.

        Args:
            dim: The axis to join along; negative counts from the last.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.dim = dim
        super().__init__(target=target)

    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Join *tensors* along ``dim``.

        Args:
            tensors: One or more float16, bfloat16 or float32 tensors of one rank, equal in
                every axis but ``dim``.

        Returns:
            A new contiguous tensor in the promoted dtype.
        """
        tensors = [t.contiguous() for t in tensors]
        dim = normalize_axis(self.dim, tensors[0].ndim)
        return self.kernel_for(
            "concat",
            lambda: ConcatCall(
                dim=dim,
                widths=tuple(math.prod(t.shape[dim:]) for t in tensors),
                dtypes=tuple(t.dtype for t in tensors),
                out_dtype=functools.reduce(torch.promote_types, (t.dtype for t in tensors)),
                device=tensors[0].device,
            ),
        )(tensors)
