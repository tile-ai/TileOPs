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

    _op_name = "bitwise_and"
    kernel_cls = BitwiseAndFwdKernel


class BitwiseOrFwdOp(BinaryOp):
    """Element-wise bitwise OR with broadcast: y = a | b."""

    _op_name = "bitwise_or"
    kernel_cls = BitwiseOrFwdKernel


class BitwiseXorFwdOp(BinaryOp):
    """Element-wise bitwise XOR with broadcast: y = a ^ b."""

    _op_name = "bitwise_xor"
    kernel_cls = BitwiseXorFwdKernel


class BitwiseNotFwdOp(UnaryOp):
    """Element-wise bitwise NOT (~x) for bool/integer inputs."""

    _op_name = "bitwise_not"
    kernel_cls = BitwiseNotFwdKernel
