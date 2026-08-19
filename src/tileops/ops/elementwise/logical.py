"""Element-wise logical ops (output bool)."""

from tileops.kernels.elementwise import (
    LogicalAndFwdKernel,
    LogicalNotFwdKernel,
    LogicalOrFwdKernel,
)

from ._base import UnaryOp, _BoolOutputBinaryOp


class LogicalAndFwdOp(_BoolOutputBinaryOp):
    """Element-wise logical AND with broadcast using non-zero truthiness."""

    _op_name = "logical_and"
    kernel_cls = LogicalAndFwdKernel


class LogicalOrFwdOp(_BoolOutputBinaryOp):
    """Element-wise logical OR with broadcast using non-zero truthiness."""

    _op_name = "logical_or"
    kernel_cls = LogicalOrFwdKernel


class LogicalNotFwdOp(UnaryOp):
    """Element-wise logical NOT with bool output."""

    _op_name = "logical_not"
    kernel_cls = LogicalNotFwdKernel
