from typing import ClassVar, Mapping, Tuple

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.norm import FusedAddLayerNormHeldKernel, FusedAddLayerNormKernel
from tileops._kernels.norm.call_spec import FusedAddLayerNormFwdInterface, LayerNormCall
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["FusedAddLayerNormFwdOp"]


class FusedAddLayerNormFwdOp(Op):
    """Fused residual addition and Layer Normalization operator.

    Computes the residual sum followed by layer normalization in a single
    fused kernel:

    $$
    \\begin{aligned}
    r &= x + \\mathrm{residual} \\\\
    y &= \\frac{r - \\mathrm{E}[r]}{\\sqrt{\\mathrm{Var}[r] + \\epsilon}}
    \\cdot w + b
    \\end{aligned}
    $$

    Returns dual outputs ``(y, residual_out)`` so downstream residual connections can
    reuse the pre-norm sum without recomputation.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and any hidden size.

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fused_add_layer_norm": FusedAddLayerNormKernel,
        "fused_add_layer_norm_held": FusedAddLayerNormHeldKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "fused_add_layer_norm": FusedAddLayerNormFwdInterface
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
                ``flash_attn``'s ``layer_norm_fn``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.eps = eps
        super().__init__(target=target)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual addition and normalization.

        Args:
            x: Input tensor of shape ``(*leading, N)``.
            residual: Residual tensor of the same shape as *x*.
            weight: Affine scale of shape $[N]$.
            bias: Affine shift of shape $[N]$.

        Returns:
            ``(y, residual_out)``, where *residual_out* is ``x + residual``, both of the
            same shape as *x*.

        Raises:
            ValueError: Dtypes or shapes disagree. Raised by the generated signature checks.
        """
        n = x.shape[-1]
        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        kernel = self.kernel_for(
            "fused_add_layer_norm",
            lambda: LayerNormCall(
                device=x.device, n=n, eps=self.eps, dtype=x.dtype, rows=x.numel() // n
            ),
        )
        y, residual_out = kernel(x, residual, weight, bias)
        return y, residual_out
