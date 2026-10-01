"""Logical and/or/not kernels."""

import tilelang.language as T
import torch

from tileops.kernels.elementwise._base import (
    LOGICAL_DTYPES,
    BinaryKernel,
    LogicalUnaryKernel,
    Uint8StorageBinaryKernel,
    Uint8StorageUnaryKernel,
)
from tileops.kernels.elementwise.call_spec import (
    BinaryPredicateFwdInterface,
    UnaryPredicateFwdInterface,
)

__all__ = [
    "LogicalAndBoolStorageFwdKernel",
    "LogicalAndFwdKernel",
    "LogicalNotBoolStorageFwdKernel",
    "LogicalNotFwdKernel",
    "LogicalOrBoolStorageFwdKernel",
    "LogicalOrFwdKernel",
]


class LogicalAndFwdKernel(BinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise logical AND with non-zero truthiness."""

    SUPPORTED_DTYPES = LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        a_nonzero = a != T.cast(0, a.dtype)
        b_nonzero = b != T.cast(0, b.dtype)
        return a_nonzero & b_nonzero


class LogicalAndBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise logical AND on uint8-backed bool storage."""

    preferred_over = frozenset({"logical_and"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_and(a, b)


class LogicalOrFwdKernel(BinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise logical OR with non-zero truthiness."""

    SUPPORTED_DTYPES = LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        a_nonzero = a != T.cast(0, a.dtype)
        b_nonzero = b != T.cast(0, b.dtype)
        return a_nonzero | b_nonzero


class LogicalOrBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise logical OR on uint8-backed bool storage."""

    preferred_over = frozenset({"logical_or"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_or(a, b)


class LogicalNotFwdKernel(LogicalUnaryKernel, UnaryPredicateFwdInterface):
    """Element-wise logical NOT with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return x == T.cast(0, x.dtype)


class LogicalNotBoolStorageFwdKernel(Uint8StorageUnaryKernel, UnaryPredicateFwdInterface):
    """Element-wise logical NOT on uint8-backed bool storage."""

    preferred_over = frozenset({"logical_not"})

    @staticmethod
    def op_func(x):
        return T.bitwise_xor(x, T.cast(1, "uint8"))
