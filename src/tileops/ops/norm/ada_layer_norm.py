from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.norm import AdaLayerNormKernel
from tileops.kernels.norm.call_spec import AdaLayerNormFwdInterface, LayerNormCall
from tileops.ops.op_base import Op

__all__ = ["AdaLayerNormFwdOp"]


class AdaLayerNormFwdOp(Op):
    """Adaptive Layer Normalization (AdaLN) operator.

    Applies layer normalization with per-token adaptive scale and shift:

    $$
    y = s \\cdot \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x]
    + \\epsilon}} + d
    $$

    where *s* (scale) and *d* (shift) are per-token tensors of shape
    ``(M, N)``, pre-computed by the caller from a conditioning signal.
    Linear projection from the conditioning input to scale/shift is the
    caller's responsibility.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims.

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"ada_layer_norm": AdaLayerNormKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "ada_layer_norm": AdaLayerNormFwdInterface
    }

    def __init__(
        self,
        eps: float = 1e-5,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            eps: Epsilon for numerical stability (manifest ``params.eps``).
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: If ``True``, autotune tile configurations.
        """
        self.eps = eps
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Tensor of shape ``(*leading, N)``.
            scale: Tensor of shape ``(*leading, N)``.
            shift: Tensor of shape ``(*leading, N)``.

        Returns:
            Tensor of the same shape as *x*.

        Raises:
            ValueError: Dtypes or shapes disagree. Raised by the generated signature checks.
        """
        x = x.contiguous()
        scale = scale.contiguous()
        shift = shift.contiguous()
        call = LayerNormCall(device=x.device, n=x.shape[-1], eps=self.eps, dtype=x.dtype)
        kernel = self.kernel_for("ada_layer_norm", call)
        return kernel(x, scale, shift)
