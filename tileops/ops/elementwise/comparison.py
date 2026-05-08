"""Element-wise comparison ops (output bool).

Kernels produce int8 (1/0) because TileLang cannot vectorize bool.
The Op forward() casts to torch.bool after the kernel call.
"""

from tileops.kernels.elementwise import (
    EqFwdKernel,
    GeFwdKernel,
    GtFwdKernel,
    LeFwdKernel,
    LtFwdKernel,
    NeFwdKernel,
)

from ._base import _BoolOutputBinaryOp


class EqFwdOp(_BoolOutputBinaryOp):
    """Element-wise equality with broadcast: y = (a == b)."""

    _op_name = "eq"
    kernel_cls = EqFwdKernel


class NeFwdOp(_BoolOutputBinaryOp):
    """Element-wise not-equal with broadcast: y = (a != b)."""

    _op_name = "ne"
    kernel_cls = NeFwdKernel


class GtFwdOp(_BoolOutputBinaryOp):
    """Element-wise greater-than with broadcast: y = (a > b)."""

    _op_name = "gt"
    kernel_cls = GtFwdKernel


class LtFwdOp(_BoolOutputBinaryOp):
    """Element-wise less-than with broadcast: y = (a < b)."""

    _op_name = "lt"
    kernel_cls = LtFwdKernel


class GeFwdOp(_BoolOutputBinaryOp):
    """Element-wise greater-equal with broadcast: y = (a >= b)."""

    _op_name = "ge"
    kernel_cls = GeFwdKernel


class LeFwdOp(_BoolOutputBinaryOp):
    """Element-wise less-equal with broadcast: y = (a <= b)."""

    _op_name = "le"
    kernel_cls = LeFwdKernel
