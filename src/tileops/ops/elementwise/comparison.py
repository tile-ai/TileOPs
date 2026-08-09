"""Element-wise comparison ops (output bool)."""

import torch

from tileops.kernels.elementwise import (
    EqFwdKernel,
    GeFwdKernel,
    GtFwdKernel,
    IsfiniteFwdKernel,
    IsinfFwdKernel,
    IsnanFwdKernel,
    LeFwdKernel,
    LtFwdKernel,
    NeFwdKernel,
)

from ._base import (
    _PREDICATE_FALLBACK_DTYPES,
    _BoolOutputBinaryOp,
    _int_all_false,
    _int_all_true,
    _IntIdentityUnaryOp,
)


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


class IsnanFwdOp(_IntIdentityUnaryOp):
    """Element-wise isnan with bool output.

    Always False on integer / bool input (no NaN representation in those
    dtypes).
    """

    _op_name = "isnan"
    kernel_cls = IsnanFwdKernel
    _int_handler = staticmethod(_int_all_false)
    _int_output_dtype = torch.bool
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES


class IsinfFwdOp(_IntIdentityUnaryOp):
    """Element-wise isinf with bool output.

    Always False on integer / bool input (no Inf representation in those
    dtypes).
    """

    _op_name = "isinf"
    kernel_cls = IsinfFwdKernel
    _int_handler = staticmethod(_int_all_false)
    _int_output_dtype = torch.bool
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES


class IsfiniteFwdOp(_IntIdentityUnaryOp):
    """Element-wise isfinite with bool output.

    Always True on integer / bool input (every value in those dtypes is
    finite).
    """

    _op_name = "isfinite"
    kernel_cls = IsfiniteFwdKernel
    _int_handler = staticmethod(_int_all_true)
    _int_output_dtype = torch.bool
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES
