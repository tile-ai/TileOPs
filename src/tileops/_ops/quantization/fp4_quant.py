from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.quantization.fp4_quant import (
    FP4QuantCall,
    FP4QuantFwdInterface,
    FP4QuantKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["FP4QuantPerGroupFwdOp"]


class FP4QuantPerGroupFwdOp(Op):
    """Quantize each block of columns to FP4 e2m1, or fake-quantize ``x`` in place.

    DeepSeek-V4-Flash ``inference/kernel.py`` ``fp4_act_quant``, bit for bit. A block is
    ``block_size`` adjacent columns of the last axis. Its scale is
    $s = 2^{\\lceil \\log_2(\\max(amax, 6 \\cdot 2^{-126}) \\cdot r) \\rceil}$, ``r`` the
    float32 nearest ``1 / 6``, and each element's code is $x / s$ clamped to $[-6, 6]$ and
    rounded to the nearest e2m1 value, ties to the even code. A zero keeps its sign: an
    element of $x / s$ in $[-0.25, 0)$ has code 8, $-0$.

    ``q`` holds two codes a byte, the even column in the low nibble, as ``uint8``: the
    logical dtype is ``float4_e2m1fn_x2``, which ``q.view(torch.float4_e2m1fn_x2)`` gives.
    ``scale`` is ``float8_e8m0fnu``. With ``inplace``, ``x`` is overwritten with
    $code \\cdot s$, exact in ``bfloat16`` below $|x| = 1.75 \\cdot 2^{127}$ and infinite
    from there, and returned in the third slot, ``q`` and ``scale`` being ``None``; otherwise
    the third slot is ``None``. A block holding an infinity or a NaN has no defined result.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"fp4_quant": FP4QuantKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"fp4_quant": FP4QuantFwdInterface}

    def __init__(
        self,
        block_size: int = 32,
        inplace: bool = False,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes are taken from the first call.

        Args:
            block_size: Columns per block, each under one scale; divides the last axis.
            inplace: Overwrite ``x`` with its fake-quantized values instead of returning
                ``q`` and ``scale``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.block_size = block_size
        self.inplace = inplace
        super().__init__(target=target)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Quantize ``x`` block by block.

        Args:
            x: $[*S \\times N]$, ``bfloat16``; written in place under ``inplace``, a
                strided view included.

        Returns:
            ``(q, scale, None)``: ``q`` $[*S \\times N / 2]$ in ``uint8`` and ``scale``
            $[*S \\times N / block\\_size]$ in ``float8_e8m0fnu``; with ``inplace``,
            ``(None, None, x)``.

        Example:
            ```python linenums="1"
            import torch
            from tileops.quantization import FP4QuantPerGroupFwdOp

            x = torch.randn(4096, 64, 128, dtype=torch.bfloat16, device="cuda")
            q, scale, _ = FP4QuantPerGroupFwdOp()(x)
            FP4QuantPerGroupFwdOp(inplace=True)(x)
            ```
        """
        if not self.inplace:
            x = x.contiguous()
        kernel = self.kernel_for(
            "fp4_quant",
            lambda: FP4QuantCall(
                n=x.shape[-1],
                block_size=self.block_size,
                dtype=x.dtype,
                inplace=self.inplace,
                device=x.device,
            ),
        )
        q, scale = kernel(x)
        if self.inplace:
            return None, None, x
        return q, scale, None
