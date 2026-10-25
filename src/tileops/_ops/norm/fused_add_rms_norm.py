from typing import ClassVar, Mapping, Tuple

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.norm import FusedAddRMSNormKernel
from tileops._kernels.norm.call_spec import FusedAddRMSNormFwdInterface, LayerNormCall
from tileops._ops.op_base import Op
from tileops.backend import Target

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

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fused_add_rms_norm": FusedAddRMSNormKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "fused_add_rms_norm": FusedAddRMSNormFwdInterface
    }

    def __init__(
        self,
        eps: float = 1e-6,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            eps: Epsilon for numerical stability. The default ``1e-6`` is that of
                ``flashinfer.norm.fused_add_rmsnorm``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.eps = eps
        super().__init__(target=target)

    def forward(
        self, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual addition and normalization.

        Args:
            input: Input tensor of shape ``(*leading, N)``.
            residual: Residual tensor of the same shape as *input*.
            weight: Affine scale of shape $[N]$.

        Returns:
            ``(y, residual_out)``, where *residual_out* is ``input + residual``, both of the
            same shape as *input*.

        Raises:
            ValueError: Dtypes or shapes disagree. Raised by the generated signature checks.
        """
        n = input.shape[-1]
        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        input = input.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        kernel = self.kernel_for(
            "fused_add_rms_norm",
            lambda: LayerNormCall(device=input.device, n=n, eps=self.eps, dtype=input.dtype),
        )
        y, residual_out = kernel(input, residual, weight)
        return y, residual_out
