"""Hadamard transform operator (host-side Op layer).

Provides:
  - HadamardTransformFwdOp: y = x @ kron(B, H_m) / sqrt(n) along the last axis
"""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.transform import (
    HADAMARD_BASES,
    HadamardBaseExchangeKernel,
    HadamardExchangeKernel,
    HadamardMultiPassKernel,
    HadamardSharedRowsKernel,
    HadamardTransformCall,
    HadamardTransformFwdInterface,
    HadamardWarpRowsKernel,
)
from tileops.ops.op_base import Op

__all__ = ["HadamardTransformFwdOp"]


class HadamardTransformFwdOp(Op):
    """Fast Walsh-Hadamard transform of the last axis, scaled by ``1 / sqrt(n)``.

    The row length is ``n = base_order * m`` with ``m`` a power of two. Read as ``base_order``
    segments of ``m``, a row becomes ``x @ kron(B, H_m) / sqrt(n)``: ``H_m`` is the Sylvester
    Hadamard matrix and ``B`` is :meth:`base_matrix`, the order-``base_order`` Hadamard matrix
    ``fast_hadamard_transform``'s ``hadamard_transform_12N`` / ``20N`` / ``28N`` / ``40N`` use.
    Those kernels do not reduce to this product below ``m = 4``; this op does at every ``m``.

    Every add and subtract is a float32 operation on the 16-bit input, and the result rounds
    once to the input dtype. An output is a sum of ``n`` terms of the input's magnitude taken
    ``log2(m)`` levels deep, plus ``base_order`` terms where it is above 1, so a cancellation
    to far below the input's magnitude keeps only float32's precision of those terms.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "hadamard_warp_rows": HadamardWarpRowsKernel,
        "hadamard_exchange": HadamardExchangeKernel,
        "hadamard_base_exchange": HadamardBaseExchangeKernel,
        "hadamard_shared_rows": HadamardSharedRowsKernel,
        "hadamard_multi_pass": HadamardMultiPassKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "hadamard_transform": HadamardTransformFwdInterface
    }

    def __init__(self, base_order: int = 1, *, target: Target = None) -> None:
        """Build the op. The row length and dtype are taken from each call.

        Args:
            base_order: The order of the Hadamard matrix that mixes the row's segments:
                1 for a power-of-two row, or 12, 20, 28 or 40.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.base_order = base_order
        super().__init__(target=target)

    @staticmethod
    def base_matrix(base_order: int) -> torch.Tensor:
        """The ``base_order x base_order`` matrix ``B`` of the transform, as float32 ``+-1``.

        Args:
            base_order: 1, 12, 20, 28 or 40.

        Returns:
            ``B`` on the CPU; ``[[1.0]]`` for base order 1.
        """
        if base_order == 1:
            return torch.ones(1, 1)
        rows = HADAMARD_BASES[base_order]
        return torch.tensor([[1.0 if s == "+" else -1.0 for s in row] for row in rows])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the last axis of ``x``.

        Args:
            x: Input of shape ``(..., n)``, float16 or bfloat16.

        Returns:
            A new tensor shaped and typed as ``x``.
        """
        x = x.contiguous()
        call = HadamardTransformCall(
            n=x.shape[-1], base_order=self.base_order, dtype=x.dtype, device=x.device
        )
        return self.kernel_for("hadamard_transform", call)(x)
