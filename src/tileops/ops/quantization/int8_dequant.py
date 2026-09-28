"""Symmetric INT8 dequantize operators."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization import DequantizeCall

from ..op_base import Op

__all__ = [
    "INT8DequantPerBlockFwdOp",
    "INT8DequantPerChannelFwdOp",
    "INT8DequantPerTensorFwdOp",
]


class INT8DequantPerTensorFwdOp(Op):
    """Dequantize an INT8 matrix with one scale: ``x = (q.float() * scale).to(out_dtype)``.

    The multiply is in float32 and its result is cast once to ``out_dtype``. The op has
    no in-tree kernel yet, so a call needs a target that registers one.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {}

    def __init__(
        self,
        out_dtype: torch.dtype,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes are taken from each call.

        Args:
            out_dtype: Dtype of ``x``: ``torch.float16``, ``torch.bfloat16`` or
                ``torch.float32``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.out_dtype = out_dtype
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize ``q``.

        Args:
            q: $[M \\times K]$, ``int8``.
            scale: $[1]$, ``float32``.

        Returns:
            ``x`` $[M \\times K]$ in ``out_dtype``.
        """
        q, scale = q.contiguous(), scale.contiguous()
        call = DequantizeCall(
            m=q.shape[0],
            k=q.shape[1],
            granularity="tensor",
            out_dtype=self.out_dtype,
            tune=self.tune,
            device=q.device,
        )
        return self.kernel_for("dequant", (q, scale), call)(q, scale)


class INT8DequantPerChannelFwdOp(Op):
    """Dequantize an INT8 matrix with one scale per row: ``x[m, k] = q[m, k] * scale[m]``.

    The multiply is in float32 and its result is cast once to ``out_dtype``. The op has
    no in-tree kernel yet, so a call needs a target that registers one.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {}

    def __init__(
        self,
        out_dtype: torch.dtype,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes are taken from each call.

        Args:
            out_dtype: Dtype of ``x``: ``torch.float16``, ``torch.bfloat16`` or
                ``torch.float32``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.out_dtype = out_dtype
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize ``q`` row by row.

        Args:
            q: $[M \\times K]$, ``int8``.
            scale: $[M]$, ``float32``.

        Returns:
            ``x`` $[M \\times K]$ in ``out_dtype``.
        """
        q, scale = q.contiguous(), scale.contiguous()
        call = DequantizeCall(
            m=q.shape[0],
            k=q.shape[1],
            granularity="channel",
            out_dtype=self.out_dtype,
            tune=self.tune,
            device=q.device,
        )
        return self.kernel_for("dequant", (q, scale), call)(q, scale)


class INT8DequantPerBlockFwdOp(Op):
    """Dequantize an INT8 matrix with one scale per 128 contiguous elements of a row.

    ``x[m, k] = q[m, k] * scale[m, k // 128]``; the last block of a row may be partial.
    The multiply is in float32 and its result is cast once to ``out_dtype``. The op has
    no in-tree kernel yet, so a call needs a target that registers one.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {}

    def __init__(
        self,
        out_dtype: torch.dtype,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes are taken from each call.

        Args:
            out_dtype: Dtype of ``x``: ``torch.float16``, ``torch.bfloat16`` or
                ``torch.float32``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.out_dtype = out_dtype
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize ``q`` block by block.

        Args:
            q: $[M \\times K]$, ``int8``.
            scale: $[M \\times \\lceil K / 128 \\rceil]$, ``float32``.

        Returns:
            ``x`` $[M \\times K]$ in ``out_dtype``.
        """
        q, scale = q.contiguous(), scale.contiguous()
        call = DequantizeCall(
            m=q.shape[0],
            k=q.shape[1],
            granularity="block",
            out_dtype=self.out_dtype,
            tune=self.tune,
            device=q.device,
        )
        return self.kernel_for("dequant", (q, scale), call)(q, scale)
