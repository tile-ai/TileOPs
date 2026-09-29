"""Call records for quantization kernels."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface
from tileops.kernels.quantization.dequant_call import DequantizeCall

__all__ = ["INT8DequantFwdInterface", "INT8QuantPerTensorFwdInterface", "QuantizeCall"]


@dataclasses.dataclass(frozen=True)
class QuantizeCall(CallSpec):
    """The facts of one quantize call: the 2-D input's shape and dtype, and its group size.

    The block shapes and output dtypes are fixed by each op's signature, so the op a
    kernel serves states them and the record does not.
    """

    # The input's leading extent, ``M`` or ``N``.
    rows: int = 0
    # The extent the scales group along, ``K``.
    cols: int = 0
    dtype: torch.dtype = torch.float16
    # ``INT4QuantPerGroupFwdOp``'s ``group_size``; ``None`` for an op without one.
    group_size: "int | None" = None


class INT8DequantFwdInterface(KernelInterface):
    """INT8 dequantize: ``x = (q.float() * scale).to(out_dtype)``, a scale per group of codes."""

    request = DequantizeCall

    @abstractmethod
    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize *q*; nothing is written in place.

        Both tensors are contiguous on ``call.device``.

        Args:
            q: ``int8`` ``(call.m, call.k)``.
            scale: ``float32``, one value per group ``call.granularity`` names.

        Returns:
            A new ``(call.m, call.k)`` tensor in ``call.out_dtype``.
        """


class INT8QuantPerTensorFwdInterface(KernelInterface):
    """Symmetric INT8 quantization of a whole matrix against one amax."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize *x*; nothing is written in place.

        Args:
            x: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *x* in ``int8`` and a new ``scale`` ``[1]`` in
            ``float32``.
        """
