"""Logical and/or/not kernels."""

import tilelang.language as T
import torch

from ._base import (
    _LOGICAL_DTYPES,
    BinaryKernel,
    LogicalUnaryKernel,
    _Uint8StorageBinaryKernel,
    _Uint8StorageUnaryKernel,
)

__all__ = [
    "LogicalAndBoolStorageFwdKernel",
    "LogicalAndFwdKernel",
    "LogicalNotBoolStorageFwdKernel",
    "LogicalNotFwdKernel",
    "LogicalOrBoolStorageFwdKernel",
    "LogicalOrFwdKernel",
]


class LogicalAndFwdKernel(BinaryKernel):
    """Element-wise logical AND with non-zero truthiness."""

    SUPPORTED_DTYPES = _LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        a_nonzero = a != T.cast(0, a.dtype)
        b_nonzero = b != T.cast(0, b.dtype)
        return a_nonzero & b_nonzero


class LogicalAndBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise logical AND on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_and(a, b)


class LogicalOrFwdKernel(BinaryKernel):
    """Element-wise logical OR with non-zero truthiness."""

    SUPPORTED_DTYPES = _LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        a_nonzero = a != T.cast(0, a.dtype)
        b_nonzero = b != T.cast(0, b.dtype)
        return a_nonzero | b_nonzero


class LogicalOrBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise logical OR on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_or(a, b)


class LogicalNotFwdKernel(LogicalUnaryKernel):
    """Element-wise logical NOT with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return x == T.cast(0, x.dtype)


class LogicalNotBoolStorageFwdKernel(_Uint8StorageUnaryKernel):
    """Element-wise logical NOT on uint8-backed bool storage."""

    @staticmethod
    def op_func(x):
        return T.bitwise_xor(x, T.cast(1, "uint8"))
