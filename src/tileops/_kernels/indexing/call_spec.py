"""The masked scatter's call spec and kernel interface."""

import dataclasses
from abc import abstractmethod

import torch

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import KernelInterface

__all__ = ["MaskedScatterCall", "MaskedScatterFwdInterface"]


@dataclasses.dataclass(frozen=True)
class MaskedScatterCall(CallSpec):
    """The shapes of ``input`` and ``mask`` and the element type; source's length stays
    device data."""

    input_shape: tuple[int, ...] = ()
    mask_shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.bfloat16


class MaskedScatterFwdInterface(KernelInterface):
    """``torch.masked_scatter``: source's elements, in order, at the true positions of mask."""

    request = MaskedScatterCall

    @abstractmethod
    def forward(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Write source's elements into the true positions of mask, out of place.

        The output has the broadcast shape of input and mask. Its true positions, in
        row-major order, take source's elements in row-major order; the rest are input's.

        Args:
            input: ``call.input_shape`` in ``call.dtype`` on ``call.device``, contiguous.
            mask: ``call.mask_shape``, bool, contiguous.
            source: Any shape in ``call.dtype``, contiguous, holding at least as many
                elements as the broadcast mask has true positions.

        Returns:
            A new contiguous tensor of the broadcast shape, in ``call.dtype``.
        """
