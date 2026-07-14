"""Element-wise comparison ops (output bool)."""

from tileops.kernels.elementwise import (
    EqBoolStorageFwdKernel,
    EqFwdKernel,
    GeBoolStorageFwdKernel,
    GeFwdKernel,
    GtBoolStorageFwdKernel,
    GtFwdKernel,
    LeBoolStorageFwdKernel,
    LeFwdKernel,
    LtBoolStorageFwdKernel,
    LtFwdKernel,
    NeBoolStorageFwdKernel,
    NeFwdKernel,
)

from ._base import _BoolOutputBinaryOp


class EqFwdOp(_BoolOutputBinaryOp):
    """Element-wise equality with broadcast: y = (a == b)."""

    _op_name = "eq"
    kernel_cls = EqFwdKernel
    bool_storage_kernel_cls = EqBoolStorageFwdKernel


class NeFwdOp(_BoolOutputBinaryOp):
    """Element-wise not-equal with broadcast: y = (a != b)."""

    _op_name = "ne"
    kernel_cls = NeFwdKernel
    bool_storage_kernel_cls = NeBoolStorageFwdKernel


class GtFwdOp(_BoolOutputBinaryOp):
    """Element-wise greater-than with broadcast: y = (a > b)."""

    _op_name = "gt"
    kernel_cls = GtFwdKernel
    bool_storage_kernel_cls = GtBoolStorageFwdKernel


class LtFwdOp(_BoolOutputBinaryOp):
    """Element-wise less-than with broadcast: y = (a < b)."""

    _op_name = "lt"
    kernel_cls = LtFwdKernel
    bool_storage_kernel_cls = LtBoolStorageFwdKernel


class GeFwdOp(_BoolOutputBinaryOp):
    """Element-wise greater-equal with broadcast: y = (a >= b)."""

    _op_name = "ge"
    kernel_cls = GeFwdKernel
    bool_storage_kernel_cls = GeBoolStorageFwdKernel


class LeFwdOp(_BoolOutputBinaryOp):
    """Element-wise less-equal with broadcast: y = (a <= b)."""

    _op_name = "le"
    kernel_cls = LeFwdKernel
    bool_storage_kernel_cls = LeBoolStorageFwdKernel
