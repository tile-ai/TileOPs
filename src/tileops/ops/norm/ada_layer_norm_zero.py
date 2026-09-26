from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm import AdaLayerNormKernel

from ..op_base import Op

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
        return self._call_boundary(x, scale, shift, gate)

    def _eager_forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, gate: torch.Tensor
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
        gate = gate.contiguous()
        n = x.shape[-1]
        kernel = self.kernel_for("ada_layer_norm", (x, scale, shift, gate), (n, x.dtype))
        return kernel(x, scale, shift, gate)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per row width and dtype; epsilon is the op's."""
        n, dtype = call
        return call, lambda: self.kernel_map["ada_layer_norm"](
            n, self.eps, dtype, has_gate=True, tune=self.tune
        )
