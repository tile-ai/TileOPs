"""Elementwise op infrastructure: umbrella bases and the shared construction parameters.

Three umbrella Op base classes, one per shape the family's kernels take:

- ``UnaryOp`` — one tensor in, the same shape out
- ``BinaryOp`` — two tensors broadcast against each other
- ``FusedGatedOp`` — one ``(M, 2N)`` tensor split into gate and value

The checks, output shapes and roofline are generated from each op's manifest
signature, which also registers its compile-boundary operators. An op normalizes
contiguity and hands the *manifest-declared* shapes to its kernel; flattening,
broadcasting and restoring the output shape are the kernel's own business. Element
type is not a construction parameter: an instance serves whichever dtype its caller
passes, one specialization per element type, built on first use.
"""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.elementwise.call_spec import (
    BinaryElementwiseFwdInterface,
    BroadcastCall,
    ElementwiseCall,
    FusedGatedCall,
    FusedGatedFwdInterface,
    UnaryElementwiseFwdInterface,
)
from tileops.kernels.kernel_base import KernelInterface
from tileops.ops.op_base import Op

# The one name every elementwise op calls its kernel through.
ELEMENTWISE = "elementwise"


def generated_on(op: Op) -> "torch.device | None":
    """Where an op with no tensor input produces its output.

    The ``device`` parameter it declares, else the current CUDA device. The call spec
    names it rather than leaving it unset, because a build reads the device it compiles
    for and the entry is keyed by the call.
    """
    declared = op._declared_device()
    if declared is not None:
        return declared
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None


class UnaryOp(Op):
    """Template base class for unary elementwise ops.

    A subclass sets ``kernel_types``, its dispatch keys. The element count arrives
    with the tensor, so nothing about shape is a construction parameter.
    """

    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: UnaryElementwiseFwdInterface
    }

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

    def _call_spec(self, input: torch.Tensor) -> ElementwiseCall:
        """The call record for *input*; a subclass with parameters widens it."""
        return ElementwiseCall(device=input.device, n_total=input.numel(), dtype=input.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        input = input.contiguous()
        return self.kernel_for(ELEMENTWISE, self._call_spec(input))(input)


class BinaryOp(Op):
    """Template base class for binary elementwise ops with broadcast.

    A subclass sets ``kernel_types``, its dispatch keys. Both operand shapes arrive
    with the tensors; the broadcast *lowering* — dim coalescing and stride synthesis —
    is the kernel's, so this class only hands the two shapes down.
    """

    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryElementwiseFwdInterface
    }

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

    def _call_spec(self, input: torch.Tensor, other: torch.Tensor) -> BroadcastCall:
        """The call record for the two operands; a subclass with parameters widens it."""
        return BroadcastCall(
            device=input.device,
            a_shape=tuple(input.shape),
            b_shape=tuple(other.shape),
            dtype=input.dtype,
        )

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input`` and ``other``."""
        input = input.contiguous()
        other = other.contiguous()
        call = self._call_spec(input, other)
        return self.kernel_for(ELEMENTWISE, call)(input, other)


class FusedGatedOp(Op):
    """Template base class for fused gated elementwise ops.

    Input: x of shape (M, 2*N). gate = x[:, :N], value = x[:, N:].
    Output: y = activation(gate) * value, shape (M, N).

    A subclass sets ``kernel_types``, its one dispatch key. Both dimensions arrive with
    the tensor.
    """

    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: FusedGatedFwdInterface
    }

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the op on ``x``."""
        x = x.contiguous()
        call = FusedGatedCall(device=x.device, m=x.shape[0], n=x.shape[1] // 2, dtype=x.dtype)
        return self.kernel_for(ELEMENTWISE, call)(x)


# Intermediate (private) base classes shared by leaf op modules


class _UnaryActivationMixin:
    """The ``inplace`` switch of a unary activation.

    The kernel writes a fresh buffer; with ``inplace`` set the result is copied back into
    ``input`` and ``input`` is returned, so callers see ``y is x``. The manifest marks
    ``input`` written exactly when ``inplace`` holds, which selects the operator that
    carries the write in a traced graph.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the activation on ``input``, into ``input`` itself when ``inplace`` is set."""
        result = super().forward(input)
        if not self.inplace:
            return result
        input.copy_(result)
        return input


class _ParamFreeActivationOp(_UnaryActivationMixin, UnaryOp):
    """Shared base for the activations whose only parameter is ``inplace``.

    ReLU, SiLU, HardSwish, HardSigmoid, Mish and SELU: each leaf declares only its
    ``kernel_types`` and docstring.
    """

    def __init__(
        self,
        *,
        inplace: bool = False,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            inplace: When True, write the result into ``input`` and return ``input``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
        """
        self.inplace = inplace
        super().__init__(target=target)
