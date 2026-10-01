"""Element-wise bitwise ops."""

from tileops.kernels.elementwise import (
    BitwiseAndBoolStorageFwdKernel,
    BitwiseAndFwdKernel,
    BitwiseNotFwdKernel,
    BitwiseOrBoolStorageFwdKernel,
    BitwiseOrFwdKernel,
    BitwiseXorBoolStorageFwdKernel,
    BitwiseXorFwdKernel,
)
from tileops.ops.elementwise._base import BinaryOp, UnaryOp


class BitwiseAndFwdOp(BinaryOp):
    """Element-wise bitwise AND with broadcast: y = a & b."""

    kernel_types = {
        "bitwise_and": BitwiseAndFwdKernel,
        "bitwise_and_bool": BitwiseAndBoolStorageFwdKernel,
    }


class BitwiseOrFwdOp(BinaryOp):
    """Element-wise bitwise OR with broadcast: y = a | b."""

    kernel_types = {
        "bitwise_or": BitwiseOrFwdKernel,
        "bitwise_or_bool": BitwiseOrBoolStorageFwdKernel,
    }


class BitwiseXorFwdOp(BinaryOp):
    """Element-wise bitwise XOR with broadcast: y = a ^ b."""

    kernel_types = {
        "bitwise_xor": BitwiseXorFwdKernel,
        "bitwise_xor_bool": BitwiseXorBoolStorageFwdKernel,
    }


class BitwiseNotFwdOp(UnaryOp):
    """Element-wise bitwise NOT (~x) for bool/integer inputs."""

    kernel_types = {"bitwise_not": BitwiseNotFwdKernel}
