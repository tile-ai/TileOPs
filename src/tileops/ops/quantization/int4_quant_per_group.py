"""Per-group asymmetric INT4 quantization operator."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.quantization import (
    INT4QuantPerGroupFwdInterface,
    INT4QuantPerGroupFwdKernel,
    INT4QuantPerGroupRowFwdKernel,
    QuantizeCall,
)
from tileops.ops.op_base import Op

__all__ = ["INT4QuantPerGroupFwdOp"]


class INT4QuantPerGroupFwdOp(Op):
    """DeepSpeed's asymmetric INT4 quantization, grouped along K.

    For a group's minimum ``lo`` and maximum ``hi``, the quantization multiplier is
    ``16 / (hi - lo)`` (1 for a constant group), and its offset is ``(hi + lo) / 2``.
    Codes are ``clamp(round((w - offset) * multiplier), -8, 7)``, with ties to even.
    Adjacent signed codes occupy the high and low nibbles of each int8 byte.
    ``params`` stores the inverse multiplier and offset as float32, one row per group.
    Floating-point division may move a code at a rounding boundary by one step.

    The in-tree kernels serve power-of-two group sizes from 32 to 1024 or multiples
    of 128 up to 65536, with ``N * K <= 2**31 - 1``. Nonfinite groups are unspecified.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "int4_quant_per_group_fwd": INT4QuantPerGroupFwdKernel,
        "int4_quant_per_group_row_fwd": INT4QuantPerGroupRowFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "int4_quant_per_group_fwd": INT4QuantPerGroupFwdInterface
    }

    def __init__(
        self,
        group_size: int = 128,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from each call.

        Args:
            group_size: Elements per group along ``K`` (default 128).
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.group_size = group_size
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and pack the groups of ``w``.

        Args:
            w: $[N \\times K]$, ``float16``.

        Returns:
            ``packed_weight`` $[N \\times K / 2]$ in ``int8`` and ``params``
                $[(N K / group\\_size) \\times 2]$ in ``float32``.
        """
        return self._call_boundary(w)

    def _eager_forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator."""
        w = w.contiguous()
        call = QuantizeCall(
            device=w.device,
            rows=w.shape[0],
            cols=w.shape[1],
            dtype=w.dtype,
            group_size=self.group_size,
        )
        kernel = self.kernel_for("int4_quant_per_group_fwd", call)
        return kernel(w)
