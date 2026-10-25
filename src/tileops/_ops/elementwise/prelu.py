"""PReLU op: y = x if x > 0 else weight[channel] * x."""

from math import prod
from typing import ClassVar, Mapping

import torch

from tileops._kernels.elementwise import PReLUFwdKernel
from tileops._kernels.elementwise.call_spec import PReLUCall, PReLUFwdInterface
from tileops._kernels.kernel_base import KernelInterface
from tileops._ops.elementwise._base import ELEMENTWISE
from tileops._ops.op_base import Op
from tileops.backend import Target


class PReLUFwdOp(Op):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel dimension follows PyTorch convention: dimension 1 for inputs
    with ndim >= 2, dimension 0 for 1-D inputs. Both the shape and the channel
    count arrive with the tensors.
    """

    kernel_types = {"prelu": PReLUFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: PReLUFwdInterface}

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
        """Run the op on ``input`` and ``weight``.

        Args:
            input: Input tensor; its channel dimension is dim 1, or dim 0 when 1-D.
            weight: Negative slopes, same dtype as ``input``: one element, or one per
                channel when ``input`` has two or more dims.

        Returns:
            A new tensor of the shape and dtype of ``input``.
        """
        input = input.contiguous()
        weight = weight.contiguous()
        # Elements per channel per row: PyTorch puts the channel at dim 1.
        inner_size = prod(input.shape[2:]) if input.ndim > 2 else 1
        return self.kernel_for(
            ELEMENTWISE,
            lambda: PReLUCall(
                device=input.device,
                n_total=input.numel(),
                dtype=input.dtype,
                num_channels=weight.numel(),
                inner_size=inner_size,
            ),
        )(input, weight)
