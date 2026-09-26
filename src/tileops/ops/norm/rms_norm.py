"""Root Mean Square (RMS) norm operator."""

import math
from typing import ClassVar, Dict, Mapping, Optional, Sequence

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm import RMSNormKernel

from ..op_base import Op
from .norm_base import affine_or_constant

__all__ = ["RMSNormFwdOp"]


class RMSNormFwdOp(Op):
    """Standalone Root Mean Square (RMS) Norm operator.

    Follows `torch.nn.functional.rms_norm`. Computes::

        y = x * rsqrt(mean(x ** 2, trailing_axes) + eps) * weight

    where the reduction runs over the trailing ``len(normalized_shape)`` axes. An absent
    ``weight`` scales by one; ``eps=None`` is the machine epsilon of the float32 accumulation,
    as in torch.

    Example:
        ```python linenums="1"
        op = RMSNormFwdOp(normalized_shape=(4096,), eps=1e-6)
        x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        w = torch.randn(4096, dtype=torch.float16, device="cuda")
        y = op(x, w)  # shape: (1024, 4096)
        ```
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"rms_norm": RMSNormKernel}

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: Optional[float] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            normalized_shape: Trailing-axis shape over which the reduction runs.
            eps: Epsilon for numerical stability; ``None`` takes the machine epsilon of
                float32, the dtype the reduction accumulates in, as torch does.
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

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply RMS normalization over the trailing ``normalized_shape``.

        Args:
            x: Input tensor whose trailing shape equals ``normalized_shape``.
            weight: Affine scale of shape ``normalized_shape``, or ``None``.

        Returns:
            Normalized tensor of the same shape as *x*.
        """
        return self._call_boundary(x, weight)

    def _eager_forward(
        self, x: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        if x.numel() == 0:
            return torch.empty_like(x)
        weight = affine_or_constant(weight, tuple(self.normalized_shape), 1.0, x.dtype, x.device)
        x = x.contiguous()
        kernel = self.kernel_for("rms_norm", (x, weight), x.dtype)
        return kernel(x, weight)

    def entry_for(self, role: str, call: torch.dtype) -> Entry:
        """One implementation, built per dtype; the row width and epsilon are the op's."""
        n = math.prod(self.normalized_shape)
        eps = torch.finfo(torch.float32).eps if self.eps is None else float(self.eps)
        return call, lambda: self.kernel_map["rms_norm"](n, eps, call, tune=self.tune)
