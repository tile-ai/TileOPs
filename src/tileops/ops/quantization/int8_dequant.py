"""Symmetric INT8 dequantize operators."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.quantization import (
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
from tileops.ops.op_base import Op

__all__ = [
    "INT8DequantPerBlockFwdOp",
    "INT8DequantPerChannelFwdOp",
    "INT8DequantPerTensorFwdOp",
]


class INT8DequantPerTensorFwdOp(Op):
    """Dequantize an INT8 matrix with one scale: ``x = (q.float() * scale).to(out_dtype)``.

    The multiply is in float32 and its result is cast once to ``out_dtype``, so ``x`` is
    bit-identical to that expression.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_dequant_per_tensor": INT8DequantPerTensorFwdKernel,
        "int8_dequant_per_tensor_small": INT8DequantPerTensorSmallFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "dequant": INT8DequantPerTensorFwdInterface
    }

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
        return self._call_boundary(q, scale)

    def _eager_forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        q, scale = q.contiguous(), scale.contiguous()
        call = DequantizeCall(
            m=q.shape[0],
            k=q.shape[1],
            out_dtype=self.out_dtype,
            device=q.device,
        )
        return self.kernel_for("dequant", call)(q, scale)


class INT8DequantPerChannelFwdOp(Op):
    """Dequantize an INT8 matrix with one scale per row: ``x[m, k] = q[m, k] * scale[m]``.

    The multiply is in float32 and its result is cast once to ``out_dtype``, so ``x`` is
    bit-identical to ``(q.float() * scale[:, None]).to(out_dtype)``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_dequant_per_channel": INT8DequantPerChannelFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "dequant": INT8DequantPerChannelFwdInterface
    }

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
        return self._call_boundary(q, scale)

    def _eager_forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        q, scale = q.contiguous(), scale.contiguous()
        call = DequantizeCall(
            m=q.shape[0],
            k=q.shape[1],
            out_dtype=self.out_dtype,
            device=q.device,
        )
        return self.kernel_for("dequant", call)(q, scale)


class INT8DequantPerBlockFwdOp(Op):
    """Dequantize an INT8 matrix with one scale per 128 contiguous elements of a row.

    ``x[m, k] = q[m, k] * scale[m, k // 128]``; the last block of a row may be partial.
    The multiply is in float32 and its result is cast once to ``out_dtype``, so ``x`` is
    bit-identical to ``(q.float() * scale.repeat_interleave(128, dim=1)[:, :K]).to(out_dtype)``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_dequant_per_block": INT8DequantPerBlockFwdKernel,
        "int8_dequant_per_block_small": INT8DequantPerBlockSmallFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "dequant": INT8DequantPerBlockFwdInterface
    }

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
        return self._call_boundary(q, scale)

    def _eager_forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        q, scale = q.contiguous(), scale.contiguous()
        call = DequantizeCall(
            m=q.shape[0],
            k=q.shape[1],
            out_dtype=self.out_dtype,
            device=q.device,
        )
        return self.kernel_for("dequant", call)(q, scale)
