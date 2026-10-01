"""PReLU op: y = x if x > 0 else weight[channel] * x."""

from math import prod
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import PreluFwdKernel
from tileops.kernels.elementwise.call_spec import PreluCall, PreluFwdInterface
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.elementwise._base import ELEMENTWISE
from tileops.ops.op_base import Op


class PreluFwdOp(Op):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel dimension follows PyTorch convention: dimension 1 for inputs
    with ndim >= 2, dimension 0 for 1-D inputs. Both the shape and the channel
    count arrive with the tensors.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"prelu": PreluFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: PreluFwdInterface}

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

    def _eager_forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        weight = weight.contiguous()
        # Elements per channel per row: PyTorch puts the channel at dim 1.
        inner_size = prod(input.shape[2:]) if input.ndim > 2 else 1
        call = PreluCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            num_channels=weight.numel(),
            inner_size=inner_size,
        )
        return self.kernel_for(ELEMENTWISE, (input, weight), call)(input, weight)

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``weight``."""
        return self._call_boundary(input, weight)
