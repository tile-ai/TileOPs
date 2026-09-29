"""Per-channel symmetric INT8 quantization operator."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization import QuantizeCall
from tileops.ops.op_base import Op

__all__ = ["INT8QuantPerChannelFwdOp"]


class INT8QuantPerChannelFwdOp(Op):
    """Symmetric INT8 quantization of a weight with one scale per output channel (row).

    ``scale[n]`` is ``amax(|w[n, :]|) / 127`` in float32, or 1.0 for an all-zero row, and is
    the dequantization multiplier: ``w[n, :] ~= q[n, :] * scale[n]``. ``q`` is the row divided
    by its scale, rounded half to even and clamped to ``[-127, 127]``.

    The op has no in-tree kernel yet: a call raises ``OpNotAvailableError`` unless a
    target serves it.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {}

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

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
            tune=self.tune,
        )
        kernel = self.kernel_for("int8_quant_per_channel_fwd", (w,), call)
        return kernel(w)
