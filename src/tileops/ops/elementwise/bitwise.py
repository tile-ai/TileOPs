"""Element-wise bitwise ops."""

from tileops.kernels.elementwise import (
    BitwiseAndFwdKernel,
    BitwiseNotFwdKernel,
    BitwiseOrFwdKernel,
    BitwiseXorFwdKernel,
)

from ._base import BinaryOp, UnaryOp


class BitwiseAndFwdOp(BinaryOp):
    """Element-wise bitwise AND with broadcast: y = a & b."""

    kernel_types = {"bitwise_and": BitwiseAndFwdKernel}


class BitwiseOrFwdOp(BinaryOp):
    """Element-wise bitwise OR with broadcast: y = a | b."""

    kernel_types = {"bitwise_or": BitwiseOrFwdKernel}


class BitwiseXorFwdOp(BinaryOp):
    """Element-wise bitwise XOR with broadcast: y = a ^ b."""

    kernel_types = {"bitwise_xor": BitwiseXorFwdKernel}


class BitwiseNotFwdOp(UnaryOp):
    """Element-wise bitwise NOT (~x) for bool/integer inputs."""

    kernel_types = {"bitwise_not": BitwiseNotFwdKernel}
