"""Per-block (128x128) FP8 quantization operator."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization import QuantizeCall

from ..op_base import Op

__all__ = ["FP8QuantPerBlockFwdOp"]


class FP8QuantPerBlockFwdOp(Op):
    """Block-scaled ``float8_e4m3fn`` quantization of a weight, one scale per 128x128 tile.

    This is the block-scaled FP8 weight format of DeepSeek-V3 checkpoints, where ``scale`` is
    stored as ``weight_scale_inv``. Edge tiles cover the ``N % 128`` rows and ``K % 128``
    columns that remain. A tile's scale is ``amax(|tile|) / 448`` in float32, or 1.0 for an
    all-zero tile, and is the dequantization multiplier. ``q`` is each element divided by its
    tile's scale, clamped to ``[-448, 448]`` and rounded to the nearest ``float8_e4m3fn``.

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
        """Quantize each 128x128 tile of ``w`` against its own absolute maximum.

        Args:
            w: $[N \\times K]$, ``bfloat16``, ``float16`` or ``float32``.

        Returns:
            ``q`` $[N \\times K]$ in ``float8_e4m3fn`` and ``scale``
                $[\\lceil N / 128 \\rceil \\times \\lceil K / 128 \\rceil]$ in ``float32``.
        """
        w = w.contiguous()
        call = QuantizeCall(
            device=w.device,
            rows=w.shape[0],
            cols=w.shape[1],
            dtype=w.dtype,
            tune=self.tune,
        )
        kernel = self.kernel_for("fp8_quant_per_block_fwd", (w,), call)
        return kernel(w)
