from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.norm import AdaLayerNormZeroKernel
from tileops.kernels.norm.call_spec import AdaLayerNormZeroFwdInterface, LayerNormCall
from tileops.ops.op_base import Op

__all__ = ["AdaLayerNormZeroFwdOp"]


class AdaLayerNormZeroFwdOp(Op):
    """Adaptive Layer Normalization-Zero (AdaLN-Zero) operator.

    Applies layer normalization with per-token adaptive scale, shift, and
    gating:

    $$
    y = g \\cdot \\left( s \\cdot \\frac{x - \\mathrm{E}[x]}
    {\\sqrt{\\mathrm{Var}[x] + \\epsilon}} + d \\right)
    $$

    where *s* (scale), *d* (shift), and *g* (gate) are per-token tensors of
    shape $[M \\times N]$, pre-computed by the caller from a conditioning signal.
    Linear projection from the conditioning input to scale/shift/gate is the
    caller's responsibility.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims.

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"ada_layer_norm": AdaLayerNormZeroKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "ada_layer_norm": AdaLayerNormZeroFwdInterface
    }

    def __init__(
        self,
        eps: float = 1e-5,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            eps: Epsilon for numerical stability (manifest ``params.eps``).
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.eps = eps
        super().__init__(target=target)

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """Apply adaptive layer normalization with zero-init gating.

        Args:
            x: Tensor of shape ``(*leading, N)``.
            scale: Tensor of shape ``(*leading, N)``.
            shift: Tensor of shape ``(*leading, N)``.
            gate: Tensor of shape ``(*leading, N)``.

        Returns:
            Tensor of the same shape as *x*.

        Raises:
            ValueError: Dtypes or shapes disagree. Raised by the generated signature checks.
        """
        x = x.contiguous()
        scale = scale.contiguous()
        shift = shift.contiguous()
        gate = gate.contiguous()
        call = LayerNormCall(device=x.device, n=x.shape[-1], eps=self.eps, dtype=x.dtype)
        kernel = self.kernel_for("ada_layer_norm", call)
        return kernel(x, scale, shift, gate)
