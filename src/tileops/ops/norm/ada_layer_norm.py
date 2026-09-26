from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm import AdaLayerNormKernel

from ..op_base import Op

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

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"ada_layer_norm": AdaLayerNormKernel}

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
        return self._call_boundary(x, scale, shift)

    def _eager_forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        if x.numel() == 0:
            return torch.empty_like(x)
        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        scale = scale.contiguous()
        shift = shift.contiguous()
        n = x.shape[-1]
        kernel = self.kernel_for("ada_layer_norm", (x, scale, shift), (n, x.dtype))
        return kernel(x, scale, shift)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per row width and dtype; epsilon is the op's."""
        n, dtype = call
        return call, lambda: self.kernel_map["ada_layer_norm"](
            n, self.eps, dtype, has_gate=False, tune=self.tune
        )
