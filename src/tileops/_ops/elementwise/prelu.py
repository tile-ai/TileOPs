"""PReLU op: y = x if x > 0 else weight[channel] * x."""

from math import prod
from typing import ClassVar, Mapping

import torch

from tileops._kernels.elementwise import PreluFwdKernel
from tileops._kernels.elementwise.call_spec import PreluCall, PreluFwdInterface
from tileops._kernels.kernel_base import KernelInterface
from tileops._ops.elementwise._base import ELEMENTWISE
from tileops._ops.op_base import Op
from tileops.backend import Target


class PreluFwdOp(Op):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel dimension follows PyTorch convention: dimension 1 for inputs
    with ndim >= 2, dimension 0 for 1-D inputs. Both the shape and the channel
    count arrive with the tensors.
    """

    kernel_types = {"prelu": PreluFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: PreluFwdInterface}

    def __init__(
        self,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``weight``."""
        input = input.contiguous()
        weight = weight.contiguous()
        # Elements per channel per row: PyTorch puts the channel at dim 1.
        inner_size = prod(input.shape[2:]) if input.ndim > 2 else 1
        return self.kernel_for(
            ELEMENTWISE,
            lambda: PreluCall(
                device=input.device,
                n_total=input.numel(),
                dtype=input.dtype,
                num_channels=weight.numel(),
                inner_size=inner_size,
            ),
        )(input, weight)
