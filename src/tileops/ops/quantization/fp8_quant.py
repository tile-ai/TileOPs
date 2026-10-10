from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.quantization.fp8_quant import (
    FP8QuantCall,
    FP8QuantFwdInterface,
    FP8QuantKernel,
)
from tileops.ops.op_base import Op

__all__ = ["FP8QuantFwdOp"]


class FP8QuantFwdOp(Op):
    """Quantize each row of an index tensor to ``float8_e4m3fn`` against its own maximum.

    ``scale_tensor`` is the row's absolute maximum, floored at ``1e-4``, over 448, to
    within one float32 ulp. ``output_tensor`` is the row scaled by the reciprocal of that
    scale, so for a finite row every element is within one ``float8_e4m3fn`` code of
    ``clamp(input / scale, -448, 448)`` rather than equal to it.

    A row holding an infinity or a NaN has no defined result: whether the row maximum and
    the clamp carry the non-finite value depends on the kernel serving the call.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"fp8_quant_kernel": FP8QuantKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"fp8_quant": FP8QuantFwdInterface}

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``input_tensor`` row by row.

        Args:
            input_tensor: $[B \\times S \\times G \\times D]$, ``float16``, ``bfloat16`` or
                ``float32``.

        Returns:
            ``scale_tensor`` $[B \\times S \\times G]$ in ``float32`` and ``output_tensor``
            $[B \\times S \\times G \\times D]$ in ``float8_e4m3fn``.
        """
        input_tensor = input_tensor.contiguous()
        batch, seq_len_kv, kv_group, index_dim = input_tensor.shape
        call = FP8QuantCall(
            batch=batch,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            index_dim=index_dim,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        kernel = self.kernel_for("fp8_quant", call)
        return kernel(input_tensor)
