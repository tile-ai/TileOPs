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
    """Asymmetric INT4 quantization of a weight with a scale and a zero point per group along K.

    Each row of ``w`` is split into groups of ``group_size`` elements along ``K``;
    ``group_size == K`` is per-channel quantization. A group's range ``[lo, hi]`` is its
    minimum and maximum widened to include 0. ``scale = (hi - lo) / 15``, at least the
    smallest normal of the input dtype and 1.0 for an all-zero group, rounded to the input
    dtype; ``zero = round(-lo / scale)`` lies in ``[0, 15]``. Each value is
    ``clamp(round(w / scale) + zero, 0, 15)``, rounded half to even, so
    ``w ~= (q - zero) * scale``. The outputs are ``GemmW4A16FwdOp``'s weight operands:
    ``packed_weight`` holds two values per byte in the order ``GemmW4A16FwdOp.repack``
    produces, ``weight_scale`` is ``scale`` and ``weight_zero`` is ``zero``.

    The three outputs are bit-equal to that expression computed in float32 by torch. A NaN
    or an infinity in a group leaves that group's outputs unspecified.

    The in-tree kernels serve a ``group_size`` that is a power of two from 32 to 1024 or a
    multiple of 128 up to 65536, and ``N * K`` up to ``2**31 - 1``; any other call raises
    ``ValueError`` on a CUDA device unless a target serves it.
    """

    compile_boundary: ClassVar[bool] = True

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

    def forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize each group of ``w`` and pack the values for ``GemmW4A16FwdOp``.

        Args:
            w: $[N \\times K]$, ``float16``.

        Returns:
            ``packed_weight`` $[N \\times K / 2]$ in ``uint8``, ``weight_scale``
                $[N \\times K / group\\_size]$ in ``float16`` and ``weight_zero``
                $[N \\times K / group\\_size]$ in ``uint8``.
        """
        return self._call_boundary(w)

    def _eager_forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
