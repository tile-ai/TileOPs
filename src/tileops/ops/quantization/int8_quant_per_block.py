"""Per-block (1x128) symmetric INT8 quantization operator."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.quantization import (
    INT8QuantPerBlockFwdInterface,
    INT8QuantPerBlockFwdKernel,
    INT8QuantPerBlockShiftedFwdKernel,
    QuantizeCall,
)
from tileops.ops.op_base import Op

__all__ = ["INT8QuantPerBlockFwdOp"]


class INT8QuantPerBlockFwdOp(Op):
    """Symmetric INT8 quantization with one scale per 128 contiguous elements along K.

    Each row of ``x`` is split into blocks of 128 elements along ``K``; the last block of a
    row holds ``K % 128`` elements when ``K`` is not a multiple of 128. A block's scale is
    ``amax(|block|) / 127`` in float32, or 1.0 for an all-zero block, and is the dequantization
    multiplier. ``q`` is each element divided by its block's scale, rounded half to even and
    clamped to ``[-127, 127]``.

    Both are bit-equal to the torch expression.

    A NaN or an infinity in a block reaches that block's ``scale`` as it does in torch; the
    block's ``q`` is then unspecified, as it is when the block's amax is so small (below
    about ``8.8e-44``, reachable only in float32) that its ``scale`` rounds to zero.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int8_quant_per_block_fwd": INT8QuantPerBlockFwdKernel,
        "int8_quant_per_block_shifted_fwd": INT8QuantPerBlockShiftedFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "int8_quant_per_block_fwd": INT8QuantPerBlockFwdInterface
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
        """Quantize each 128-element block of ``x`` against its own absolute maximum.

        Args:
            x: $[M \\times K]$, ``float16``, ``bfloat16`` or ``float32``.

        Returns:
            ``q`` $[M \\times K]$ in ``int8`` and ``scale`` $[M \\times \\lceil K / 128 \\rceil]$ in
                ``float32``.
        """
        return self._call_boundary(x)

    def _eager_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator."""
        x = x.contiguous()
        call = QuantizeCall(
            device=x.device,
            rows=x.shape[0],
            cols=x.shape[1],
            dtype=x.dtype,
        )
        kernel = self.kernel_for("int8_quant_per_block_fwd", call)
        return kernel(x)
