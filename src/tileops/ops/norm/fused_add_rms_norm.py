from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm import FusedAddRMSNormKernel

from ..op_base import Op

__all__ = ["FusedAddRMSNormFwdOp"]


class FusedAddRMSNormFwdOp(Op):
    """Fused residual addition and RMS Normalization operator.

    Computes the residual sum followed by RMS normalization in a single
    fused kernel:

    $$
    \\begin{aligned}
    r &= x + \\mathrm{residual} \\\\
    y &= \\frac{r}{\\sqrt{\\mathrm{mean}(r^2) + \\epsilon}} \\cdot w
    \\end{aligned}
    $$

    Returns dual outputs ``(y, residual_out)`` so downstream residual connections can
    reuse the pre-norm sum without recomputation.

    Supported dtypes:
        ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims
        by padding to 256-element alignment.

    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fused_add_rms_norm": FusedAddRMSNormKernel
    }

    def __init__(
        self,
        eps: float = 1e-6,
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
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual addition and normalization.

        Args:
            x: Input tensor of shape ``(*leading, N)``.
            residual: Residual tensor of the same shape as *x*.
            weight: Affine scale of shape $[N]$.

        Returns:
            ``(y, residual_out)``, where *residual_out* is ``x + residual``, both of the
            same shape as *x*.

        Raises:
            ValueError: Dtypes or shapes disagree. Raised by the generated signature checks.
        """
        return self._call_boundary(x, residual, weight)

    def _eager_forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        if x.numel() == 0:
            return torch.empty_like(x), x + residual
        n = x.shape[-1]
        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        kernel = self.kernel_for(
            "fused_add_rms_norm",
            (x, residual, weight),
            (n, x.dtype),
        )
        y, residual_out = kernel(x, residual, weight)
        return y, residual_out

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per row width and dtype; epsilon is the op's."""
        n, dtype = call
        return call, lambda: self.kernel_map["fused_add_rms_norm"](
            n, self.eps, dtype, tune=self.tune
        )
