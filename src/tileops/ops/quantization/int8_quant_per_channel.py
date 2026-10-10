"""Per-channel symmetric INT8 quantization operator."""

from typing import ClassVar, Mapping, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.quantization import (
    INT8QuantPerChannelFwdInterface,
    INT8QuantPerChannelFwdKernel,
    QuantizeCall,
)
from tileops.ops.op_base import Op

__all__ = ["INT8QuantPerChannelFwdOp"]


class INT8QuantPerChannelFwdOp(Op):
    """Symmetric INT8 quantization of a weight with one scale per output channel (row).

    ``scale[n]`` is ``amax(|w[n, :]|) / 127`` in float32, or 1.0 for an all-zero row, and is
    the dequantization multiplier: ``w[n, :] ~= q[n, :] * scale[n]``. ``q`` is the row divided
    by its scale, rounded half to even and clamped to ``[-127, 127]``.

    Both are bit-equal to the torch expression.

    A NaN or an infinity in a row reaches that row's ``scale`` as it does in torch; the
    row's ``q`` is then unspecified, as it is when the row's amax is so small (below about
    ``8.8e-44``, reachable only in float32) that its ``scale`` rounds to zero.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_quant_per_channel_fwd": INT8QuantPerChannelFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "int8_quant_per_channel_fwd": INT8QuantPerChannelFwdInterface
    }

    def __init__(
        self,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize each row of ``w`` against its own absolute maximum.

        Args:
            w: $[N \\times K]$, ``float16``, ``bfloat16`` or ``float32``.

        Returns:
            ``q`` $[N \\times K]$ in ``int8`` and ``scale`` $[N]$ in ``float32``.
        """
        w = w.contiguous()
        call = QuantizeCall(
            device=w.device,
            rows=w.shape[0],
            cols=w.shape[1],
            dtype=w.dtype,
        )
        kernel = self.kernel_for("int8_quant_per_channel_fwd", call)
        return kernel(w)
