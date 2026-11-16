"""Masked scatter operator (host-side Op layer).

Provides:
  - MaskedScatterFwdOp: output = input with source's elements, in order, at mask's true positions
"""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.indexing import (
    MaskedScatterCall,
    MaskedScatterFwdInterface,
    MaskedScatterKernel,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["MaskedScatterFwdOp"]


class MaskedScatterFwdOp(Op):
    """Masked scatter (``torch.masked_scatter(input, mask, source)``), out of place.

    The output has the broadcast shape of ``input`` and ``mask``. Its true positions, in
    row-major order, take ``source``'s elements in row-major order; the others keep
    ``input``'s. ``source`` must hold at least as many elements as the broadcast mask has
    true positions. The op does not check this: with fewer, the output is unspecified and
    the op reads nothing past ``source``'s end, where torch raises a device-side assertion.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"masked_scatter": MaskedScatterKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "masked_scatter": MaskedScatterFwdInterface
    }

    def __init__(self, *, target: Target = None) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Scatter ``source`` into the true positions of ``mask``.

        Args:
            input: The values the false positions keep; float16, bfloat16 or float32.
            mask: ``torch.bool``, broadcastable with ``input``.
            source: Any shape in ``input``'s dtype, read in row-major order.

        Returns:
            A new tensor of the broadcast shape of ``input`` and ``mask``, in ``input``'s dtype.
        """
        input = input.contiguous()
        mask = mask.contiguous()
        source = source.contiguous()
        return self.kernel_for(
            "masked_scatter",
            lambda: MaskedScatterCall(
                device=input.device,
                input_shape=tuple(input.shape),
                mask_shape=tuple(mask.shape),
                dtype=input.dtype,
            ),
        )(input, mask, source)
