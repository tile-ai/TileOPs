"""The concatenation's call spec and kernel interface."""

import dataclasses
from abc import abstractmethod

import torch

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import KernelInterface

__all__ = ["ConcatCall", "ConcatFwdInterface"]


@dataclasses.dataclass(frozen=True)
class ConcatCall(CallSpec):
    """The joined axis, each element's width and dtype, and the output dtype.

    An element's width is the product of its axes from ``dim`` on: the run of contiguous
    elements it holds per row of the axes before ``dim``.
    """

    dim: int = 0
    widths: tuple[int, ...] = ()
    dtypes: tuple[torch.dtype, ...] = ()
    out_dtype: torch.dtype = torch.bfloat16


class ConcatFwdInterface(KernelInterface):
    """Tensors joined along one axis into a new tensor."""

    request = ConcatCall

    @abstractmethod
    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Join *tensors* along ``call.dim``.

        Args:
            tensors: Contiguous tensors on ``call.device`` of one rank, equal in every axis
                but ``call.dim``. Element ``j`` has dtype ``call.dtypes[j]`` and holds
                ``call.widths[j]`` elements from ``call.dim`` on per leading row.

        Returns:
            A new contiguous tensor in ``call.out_dtype``, each input's elements converted
            to it and placed after the previous input's along ``call.dim``.
        """
