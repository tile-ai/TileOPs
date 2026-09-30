"""Call records for quantization kernels."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface
from tileops.kernels.quantization.dequant_call import DequantizeCall

__all__ = [
    "FP8QuantPerBlockFwdInterface",
    "INT4QuantPerGroupFwdInterface",
    "INT8DequantPerBlockFwdInterface",
    "INT8DequantPerChannelFwdInterface",
    "INT8DequantPerTensorFwdInterface",
    "INT8QuantPerBlockFwdInterface",
    "INT8QuantPerChannelFwdInterface",
    "INT8QuantPerTensorFwdInterface",
    "QuantizeCall",
    "SmoothQuantFwdInterface",
]


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


class INT8DequantPerTensorFwdInterface(KernelInterface):
    """INT8 dequantize with one scale for the matrix: ``x = (q.float() * scale).to(out_dtype)``."""

    request = DequantizeCall

    @abstractmethod
    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize *q*; nothing is written in place.

        Both tensors are contiguous on ``call.device``.

        Args:
            q: ``int8`` ``(call.m, call.k)``.
            scale: ``float32`` ``(1,)``.

        Returns:
            A new ``(call.m, call.k)`` tensor in ``call.out_dtype``.
        """


class INT8DequantPerChannelFwdInterface(KernelInterface):
    """INT8 dequantize with one scale per row: ``x = (q.float() * scale[:, None]).to(out_dtype)``."""

    request = DequantizeCall

    @abstractmethod
    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize *q*; nothing is written in place.

        Both tensors are contiguous on ``call.device``.

        Args:
            q: ``int8`` ``(call.m, call.k)``.
            scale: ``float32`` ``(call.m,)``.

        Returns:
            A new ``(call.m, call.k)`` tensor in ``call.out_dtype``.
        """


class INT8DequantPerBlockFwdInterface(KernelInterface):
    """INT8 dequantize with one scale per 128 codes of a row: ``x = (q.float() * scale[m, k // 128]).to(out_dtype)``."""

    request = DequantizeCall

    @abstractmethod
    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize *q*; nothing is written in place.

        Both tensors are contiguous on ``call.device``.

        Args:
            q: ``int8`` ``(call.m, call.k)``.
            scale: ``float32`` ``(call.m, ceil(call.k / 128))``.

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


class INT8QuantPerChannelFwdInterface(KernelInterface):
    """Symmetric INT8 quantization of each row against its own amax."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each of the ``call.rows`` rows of ``w``; nothing is written in place.

        Args:
            w: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *w* in ``int8`` and a new ``scale`` ``[call.rows]`` in
            ``float32``.
        """


class INT4QuantPerGroupFwdInterface(KernelInterface):
    """Asymmetric INT4 quantization of each group of a row into ``GemmW4A16FwdOp``'s operands."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize and pack the ``call.rows`` rows of ``w``; nothing is written in place.

        Args:
            w: ``[call.rows, call.cols]``, contiguous, ``float16`` on ``call.device``.

        Returns:
            A new ``packed_weight`` ``[call.rows, call.cols // 2]`` in ``uint8``, in the order
            ``GemmW4A16FwdOp.repack`` produces, and a new ``weight_scale`` in ``float16`` and
            ``weight_zero`` in ``uint8``, both ``[call.rows, call.cols // call.group_size]``.
        """


class INT8QuantPerBlockFwdInterface(KernelInterface):
    """Symmetric INT8 quantization of each 128-element block of a row against its own amax."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each block of the ``call.rows`` rows of ``x``; nothing is written in place.

        Args:
            x: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *x* in ``int8`` and a new ``scale``
            ``[call.rows, ceil(call.cols / 128)]`` in ``float32``.
        """


class FP8QuantPerBlockFwdInterface(KernelInterface):
    """Block-scaled ``float8_e4m3fn`` quantization of each 128x128 tile against its own amax."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each tile of the ``call.rows x call.cols`` weight *w*; nothing is written in place.

        Args:
            w: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *w* in ``float8_e4m3fn`` and a new ``scale``
            ``[ceil(call.rows / 128), ceil(call.cols / 128)]`` in ``float32``.
        """


class SmoothQuantFwdInterface(KernelInterface):
    """SmoothQuant: divide each column by its smoothing factor, then INT8 per row."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each of the ``call.rows`` rows of ``x / smooth``; nothing is written in place.

        Args:
            x: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.
            smooth: ``[call.cols]``, contiguous, in ``float32`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *x* in ``int8`` and a new ``scale`` ``[call.rows]`` in
            ``float32``.
        """
