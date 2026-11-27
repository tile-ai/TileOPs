"""Symmetric INT8 dequantize operators."""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.quantization import (
    DequantizeCall,
    INT8DequantPerBlockFwdInterface,
    INT8DequantPerBlockFwdKernel,
    INT8DequantPerBlockSmallFwdKernel,
    INT8DequantPerChannelFwdInterface,
    INT8DequantPerChannelFwdKernel,
    INT8DequantPerTensorFwdInterface,
    INT8DequantPerTensorFwdKernel,
    INT8DequantPerTensorSmallFwdKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = [
    "INT8DequantPerBlockFwdOp",
    "INT8DequantPerChannelFwdOp",
    "INT8DequantPerTensorFwdOp",
]


class _INT8DequantFwdOpBase(Op):
    """Symmetric INT8 dequantize. The multiply is in float32 and its result is cast once to
    ``out_dtype``, so ``x`` is bit-identical to the torch expression each op states.

    Concrete subclasses set ``kernel_types`` and ``interfaces``.
    """

    def __init__(
        self,
        out_dtype: torch.dtype,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes are taken from each call.

        Args:
            out_dtype: Dtype of ``x``: ``torch.float16``, ``torch.bfloat16`` or
                ``torch.float32``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.out_dtype = out_dtype
        super().__init__(target=target)

    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize ``q``.

        Args:
            q: $[M \\times K]$, ``int8``.
            scale: ``float32``, in the shape the op states.

        Returns:
            ``x`` $[M \\times K]$ in ``out_dtype``.
        """
        q, scale = q.contiguous(), scale.contiguous()
        return self.kernel_for(
            "dequant",
            lambda: DequantizeCall(
                m=q.shape[0], k=q.shape[1], out_dtype=self.out_dtype, device=q.device
            ),
        )(q, scale)


class INT8DequantPerTensorFwdOp(_INT8DequantFwdOpBase):
    """Dequantize an INT8 matrix with one scale: ``x = (q.float() * scale).to(out_dtype)``,
    ``scale`` $[1]$.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_dequant_per_tensor": INT8DequantPerTensorFwdKernel,
        "int8_dequant_per_tensor_small": INT8DequantPerTensorSmallFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "dequant": INT8DequantPerTensorFwdInterface
    }


class INT8DequantPerChannelFwdOp(_INT8DequantFwdOpBase):
    """Dequantize an INT8 matrix with one scale per row:
    ``x = (q.float() * scale[:, None]).to(out_dtype)``, ``scale`` $[M]$.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_dequant_per_channel": INT8DequantPerChannelFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "dequant": INT8DequantPerChannelFwdInterface
    }


class INT8DequantPerBlockFwdOp(_INT8DequantFwdOpBase):
    """Dequantize an INT8 matrix with one scale per 128 contiguous elements of a row:
    ``x = (q.float() * scale.repeat_interleave(128, dim=1)[:, :K]).to(out_dtype)``,
    ``scale`` $[M \\times \\lceil K / 128 \\rceil]$; the last block of a row may be partial.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_dequant_per_block": INT8DequantPerBlockFwdKernel,
        "int8_dequant_per_block_small": INT8DequantPerBlockSmallFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "dequant": INT8DequantPerBlockFwdInterface
    }
