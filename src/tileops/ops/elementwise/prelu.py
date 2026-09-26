"""PReLU op: y = x if x > 0 else weight[channel] * x."""

from math import prod
from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import PreluFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _PerDtypeKernels


class PreluFwdOp(_PerDtypeKernels, Op):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel dimension follows PyTorch convention: dimension 1 for inputs
    with ndim >= 2, dimension 0 for 1-D inputs. Both the shape and the channel
    count arrive with the tensors.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"prelu": PreluFwdKernel}

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional dispatch override mapping kernel keys to
                ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int, num_channels: int, inner_size: int):
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(n_total, num_channels, inner_size, ctor_dtype, tune=self.tune)

    def _eager_forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        weight = weight.contiguous()
        # Elements per channel per row: PyTorch puts the channel at dim 1.
        inner_size = prod(input.shape[2:]) if input.ndim > 2 else 1
        kernel = self._kernel(
            (input, weight), input.dtype, input.numel(), weight.numel(), inner_size
        )
        return kernel(input, weight)

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``weight``."""
        return self._call_boundary(input, weight)
