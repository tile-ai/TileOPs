"""Special predicate ops: isnan, isinf, isfinite (output bool)."""

import torch

from tileops.kernels.elementwise import (
    IsfiniteFwdKernel,
    IsinfFwdKernel,
    IsnanFwdKernel,
)

from ._base import (
    _PREDICATE_FALLBACK_DTYPES,
    _int_all_false,
    _int_all_true,
    _IntIdentityUnaryOp,
)


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
