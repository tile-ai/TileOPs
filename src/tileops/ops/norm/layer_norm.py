"""Layer normalization operator."""

import math
from typing import ClassVar, Dict, Mapping, Optional, Sequence

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm import LayerNormKernel

from ..op_base import Op
from .norm_base import affine_or_constant

__all__ = ["LayerNormFwdOp"]


class LayerNormFwdOp(Op):
    """Layer Normalization operator.

    Computes layer normalization over the trailing ``normalized_shape`` axes:

    $$
    y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
    \\cdot w + b
    $$

    Follows `torch.nn.functional.layer_norm`: an absent ``weight`` scales by one and an
    absent ``bias`` shifts by zero.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"layer_norm": LayerNormKernel}

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: float = 1e-5,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            normalized_shape: Trailing-axis shape over which the statistics run.
            eps: Epsilon for numerical stability.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: Whether to autotune (default ``False``).
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply layer normalization over the trailing ``normalized_shape``.

        Args:
            x: Input tensor whose trailing shape equals ``normalized_shape``.
            weight: Affine scale of shape ``normalized_shape``, or ``None``.
            bias: Affine shift of shape ``normalized_shape``, or ``None``.

        Returns:
            Normalized tensor of the same shape as *x*.
        """
        return self._call_boundary(x, weight, bias)

    def _eager_forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        if x.numel() == 0:
            return torch.empty_like(x)
        ns = tuple(self.normalized_shape)
        weight = affine_or_constant(weight, ns, 1.0, x.dtype, x.device)
        bias = affine_or_constant(bias, ns, 0.0, x.dtype, x.device)
        x = x.contiguous()
        kernel = self.kernel_for("layer_norm", (x, weight, bias), x.dtype)
        return kernel(x, weight, bias)

    def entry_for(self, role: str, call: torch.dtype) -> Entry:
        """One implementation, built per dtype; the row width and epsilon are the op's."""
        n = math.prod(self.normalized_shape)
        return call, lambda: self.kernel_map["layer_norm"](n, self.eps, call, tune=self.tune)
