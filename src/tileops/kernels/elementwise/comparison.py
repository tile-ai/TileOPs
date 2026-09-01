"""Comparison and float-predicate kernels."""

import tilelang.language as T
import torch

from ._base import (
    BinaryKernel,
    FloatPredicateKernel,
    _Uint8StorageBinaryKernel,
)
from ._dtype import _BINARY_FULL_DTYPES

__all__ = [
    "EqBoolStorageFwdKernel",
    "EqFwdKernel",
    "GeBoolStorageFwdKernel",
    "GeFwdKernel",
    "GtBoolStorageFwdKernel",
    "GtFwdKernel",
    "IsfiniteFwdKernel",
    "IsinfFwdKernel",
    "IsnanFwdKernel",
    "LeBoolStorageFwdKernel",
    "LeFwdKernel",
    "LtBoolStorageFwdKernel",
    "LtFwdKernel",
    "NeBoolStorageFwdKernel",
    "NeFwdKernel",
]


class EqFwdKernel(BinaryKernel):
    """Element-wise equality: y = (a == b)."""

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a == b


class EqBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise equality on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_xor(T.bitwise_xor(a, b), T.cast(1, "uint8"))


class NeFwdKernel(BinaryKernel):
    """Element-wise not-equal: y = (a != b).

    A half operand is compared in float32, which is where ``!=`` means what
    IEEE 754 says. CUDA's ``__hne`` is an *ordered* comparison and answers
    false when either operand is NaN; ``!=`` is the unordered one and answers
    true, since NaN is unequal to everything, itself included. Widening is
    exact for float16 and bfloat16, and float32's own ``!=`` already carries
    the case, so only the half formats pay it.

    The other five comparisons want the ordered answer -- IEEE reads ``<``,
    ``<=``, ``>``, ``>=`` and ``==`` as false against a NaN -- and take the
    operator directly. Negating equality does not work here: the simplifier
    folds ``not (a == b)`` back to ``a != b``, which is the comparison being
    avoided.
    """

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        if str(a.dtype) in ("float16", "bfloat16"):
            return T.Cast("float32", a) != T.Cast("float32", b)
        return a != b


class NeBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise not-equal on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_xor(a, b)


class GtFwdKernel(BinaryKernel):
    """Element-wise greater-than: y = (a > b)."""

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a > b


class GtBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise greater-than on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_and(a, T.bitwise_xor(b, T.cast(1, "uint8")))


class LtFwdKernel(BinaryKernel):
    """Element-wise less-than: y = (a < b)."""

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a < b


class LtBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise less-than on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_and(T.bitwise_xor(a, T.cast(1, "uint8")), b)


class GeFwdKernel(BinaryKernel):
    """Element-wise greater-equal: y = (a >= b)."""

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a >= b


class GeBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise greater-equal on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_or(a, T.bitwise_xor(b, T.cast(1, "uint8")))


class LeFwdKernel(BinaryKernel):
    """Element-wise less-equal: y = (a <= b)."""

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a <= b


class LeBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise less-equal on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_or(T.bitwise_xor(a, T.cast(1, "uint8")), b)


class IsnanFwdKernel(FloatPredicateKernel):
    """Element-wise isnan with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isnan(T.cast(x, "float32"))


class IsinfFwdKernel(FloatPredicateKernel):
    """Element-wise isinf with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isinf(T.cast(x, "float32"))


class IsfiniteFwdKernel(FloatPredicateKernel):
    """Element-wise isfinite with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isfinite(T.cast(x, "float32"))
