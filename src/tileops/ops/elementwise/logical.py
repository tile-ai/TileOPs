"""Element-wise logical ops (output bool)."""

from tileops.kernels.elementwise import (
    LogicalAndFwdKernel,
    LogicalNotFwdKernel,
    LogicalOrFwdKernel,
)

from ._base import BinaryOp, UnaryOp


class LogicalAndFwdOp(BinaryOp):
    """Element-wise logical AND with broadcast using non-zero truthiness."""

    kernel_types = {"logical_and": LogicalAndFwdKernel}


class LogicalOrFwdOp(BinaryOp):
    """Element-wise logical OR with broadcast using non-zero truthiness."""

    kernel_types = {"logical_or": LogicalOrFwdKernel}


class LogicalNotFwdOp(UnaryOp):
    """Element-wise logical NOT with bool output."""

    kernel_types = {"logical_not": LogicalNotFwdKernel}
