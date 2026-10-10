"""SmoothQuant activation quantization operator."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.quantization import (
    QuantizeCall,
    SmoothQuantFwdInterface,
    SmoothQuantFwdKernel,
)
from tileops.ops.op_base import Op

__all__ = ["SmoothQuantFwdOp"]


class SmoothQuantFwdOp(Op):
    """SmoothQuant activation quantization: per-channel smoothing, then symmetric INT8 per row.

    Each column of ``x`` is divided by its smoothing factor, ``xs = x / smooth`` in float32,
    as in SmoothQuant (Xiao et al., 2023). ``scale[m]`` is ``amax(|xs[m, :]|) / 127``, or
    1.0 for an all-zero row, and is the dequantization multiplier of ``xs``. ``q`` is
    ``xs[m, :] / scale[m]`` rounded half to even and clamped to ``[-127, 127]``.

    Both are bit-equal to the torch expression.

    A NaN or an infinity in ``xs`` (a zero smoothing factor gives one) reaches that row's
    ``scale`` as it does in torch; the row's ``q`` is then unspecified, as it is when the
    row's amax is so small (below about ``8.8e-44``) that its ``scale`` rounds to zero.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"smooth_quant_fwd": SmoothQuantFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "smooth_quant_fwd": SmoothQuantFwdInterface
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

    def forward(self, x: torch.Tensor, smooth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Smooth ``x`` by channel, then quantize each row against its absolute maximum.

        Args:
            x: $[M \\times K]$, ``float16`` or ``bfloat16``.
            smooth: $[K]$, ``float32``: the per-channel smoothing factors.

        Returns:
            ``q`` $[M \\times K]$ in ``int8`` and ``scale`` $[M]$ in ``float32``.
        """
        x = x.contiguous()
        smooth = smooth.contiguous()
        call = QuantizeCall(
            device=x.device,
            rows=x.shape[0],
            cols=x.shape[1],
            dtype=x.dtype,
        )
        kernel = self.kernel_for("smooth_quant_fwd", call)
        return kernel(x, smooth)
