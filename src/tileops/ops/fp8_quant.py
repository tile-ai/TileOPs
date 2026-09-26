from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.fp8_quant import FP8QuantKernel
from tileops.kernels.kernel_base import Entry, Kernel

from .op_base import Op

__all__ = ["FP8QuantFwdOp"]


class FP8QuantFwdOp(Op):
    """Quantize each row of an index tensor to ``float8_e4m3fn`` against its own maximum.

    ``scale_tensor`` is the row's absolute maximum, floored at ``1e-4``, over 448, to
    within one float32 ulp. ``output_tensor`` is the row scaled by the reciprocal of that
    scale, so for a finite row every element is within one ``float8_e4m3fn`` code of
    ``clamp(input / scale, -448, 448)`` rather than equal to it.

    Neither output propagates a non-finite input: a row holding an infinity or a NaN is
    quantized against the maximum of its finite values.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"fp8_quant_kernel": FP8QuantKernel}

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
        self.kernel = None

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device."""
        batch, seq_len_kv, kv_group, index_dim, in_dtype, _device_index = call
        return call, lambda: self.kernel_map["fp8_quant_kernel"](
            batch, seq_len_kv, kv_group, index_dim, in_dtype, tune=self.tune
        )

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
        call = (*input_tensor.shape, input_tensor.dtype, input_tensor.device.index)
        self.kernel = self.kernel_for("fp8_quant_kernel", (input_tensor,), call)
        return self.kernel(input_tensor)
