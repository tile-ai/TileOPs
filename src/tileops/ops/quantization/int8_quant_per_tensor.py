"""Per-tensor symmetric INT8 quantization operator."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.quantization import (
    INT8QuantPerTensorFwdInterface,
    INT8QuantPerTensorFwdKernel,
    QuantizeCall,
)
from tileops.ops.op_base import Op

__all__ = ["INT8QuantPerTensorFwdOp"]


class INT8QuantPerTensorFwdOp(Op):
    """Symmetric INT8 quantization of a 2-D tensor against one scale for the whole tensor.

    ``scale`` is ``amax(|x|) / 127`` in float32, or 1.0 when ``x`` is all zero, and is the
    dequantization multiplier: ``x ~= q * scale``. ``q`` is ``x / scale`` rounded half to
    even and clamped to ``[-127, 127]``. Both are bit-equal to the torch expression.

    A NaN or an infinity in ``x`` reaches ``scale`` as it does in torch; ``q`` is then
    unspecified.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_quant_per_tensor_fwd": INT8QuantPerTensorFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "int8_quant_per_tensor_fwd": INT8QuantPerTensorFwdInterface
    }

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``x`` against its absolute maximum.

        Args:
            x: $[M \\times K]$, ``float16``, ``bfloat16`` or ``float32``.

        Returns:
            ``q`` $[M \\times K]$ in ``int8`` and ``scale`` $[1]$ in ``float32``.
        """
        x = x.contiguous()
        call = QuantizeCall(
            device=x.device,
            rows=x.shape[0],
            cols=x.shape[1],
            dtype=x.dtype,
        )
        kernel = self.kernel_for("int8_quant_per_tensor_fwd", call)
        return kernel(x)
