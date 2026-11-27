"""Masked scatter operator (host-side Op layer).

Provides:
  - MaskedScatterFwdOp: output = input with its mask-true positions taken from source in order
"""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.indexing import (
    MaskedScatterCall,
    MaskedScatterFwdInterface,
    MaskedScatterKernel,
    mask_runs,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op

__all__ = ["MaskedScatterFwdOp"]


class MaskedScatterFwdOp(Op):
    """Fill the true positions of ``mask`` with consecutive elements of ``source``.

    The out-of-place forward of ``torch.Tensor.masked_scatter``: ``input`` and ``mask``
    broadcast together, and the k-th true position in row-major order takes
    ``source.flatten()[k]``. The other positions keep ``input``.

    The caller guarantees that ``source`` holds at least as many elements as the broadcast
    mask has true positions; the op does not count them.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"masked_scatter": MaskedScatterKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "masked_scatter": MaskedScatterFwdInterface
    }

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Scatter ``source`` into the true positions of ``mask``.

        Args:
            input: Any shape ``[*A]``, float16, bfloat16 or float32.
            mask: bool ``[*M]``, broadcastable with ``input``.
            source: Any shape ``[*R]`` in ``input``'s dtype.

        Returns:
            A new tensor of shape ``broadcast(A, M)`` in ``input``'s dtype.
        """
        input, mask, source = input.contiguous(), mask.contiguous(), source.contiguous()
        shape = torch.broadcast_shapes(input.shape, mask.shape)
        return self.kernel_for(
            "masked_scatter",
            lambda: MaskedScatterCall(
                run_len=mask_runs(shape, mask.shape)[1], dtype=input.dtype, device=input.device
            ),
        )(input, mask, source)
