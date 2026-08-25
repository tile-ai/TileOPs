"""Unary math kernels: exp/log family, roots, rounding, trigonometry."""

import tilelang.language as T
import torch

from ._base import (
    FloatUnaryKernel,
)
from ._dtype import log_for_output_precision

__all__ = [
    "AbsFwdKernel",
    "CeilFwdKernel",
    "CosFwdKernel",
    "ErfFwdKernel",
    "ExpFwdKernel",
    "Expm1FwdKernel",
    "FloorFwdKernel",
    "Log1pFwdKernel",
    "LogFwdKernel",
    "NegFwdKernel",
    "ReciprocalFwdKernel",
    "RoundFwdKernel",
    "RsqrtFwdKernel",
    "SignFwdKernel",
    "SinFwdKernel",
    "SqrtFwdKernel",
    "TruncFwdKernel",
]


class ExpFwdKernel(FloatUnaryKernel):
    """Element-wise exp(x)."""

    @staticmethod
    def op_func(x):
        return T.exp(T.cast(x, "float32"))


class LogFwdKernel(FloatUnaryKernel):
    """Element-wise log(x)."""

    @staticmethod
    def op_func(x):
        return log_for_output_precision(x, T.cast(x, "float32"))


class SqrtFwdKernel(FloatUnaryKernel):
    """Element-wise sqrt(x)."""

    @staticmethod
    def op_func(x):
        return T.sqrt(T.cast(x, "float32"))


class RsqrtFwdKernel(FloatUnaryKernel):
    """Element-wise 1/sqrt(x)."""

    @staticmethod
    def op_func(x):
        return T.rsqrt(T.cast(x, "float32"))


class AbsFwdKernel(FloatUnaryKernel):
    """Element-wise |x|."""

    @staticmethod
    def op_func(x):
        return T.abs(x)


class NegFwdKernel(FloatUnaryKernel):
    """Element-wise -x."""

    @staticmethod
    def op_func(x):
        return -x


class ReciprocalFwdKernel(FloatUnaryKernel):
    """Element-wise 1/x.

    Integral inputs are this backend's business: it has no integer kernel, so it
    declares float32 as the type it computes them in and converts at the
    boundary. A backend with a native integer-input reciprocal declares nothing
    and receives the integers.
    """

    _INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

    @classmethod
    def specialize(cls, dtype: torch.dtype) -> tuple:
        if dtype in cls._INT_DTYPES:
            return cls, torch.float32
        return super().specialize(dtype)

    @staticmethod
    def op_func(x):
        return T.cast(1.0, "float32") / x

    def forward(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return super().forward(x)


class SignFwdKernel(FloatUnaryKernel):
    """Element-wise sign(x): -1, 0, or +1."""

    @staticmethod
    def op_func(x):
        zero = T.cast(0.0, x.dtype)
        one = T.cast(1.0, x.dtype)
        neg_one = T.cast(-1.0, x.dtype)
        return T.if_then_else(
            x > zero,
            one,
            T.if_then_else(x < zero, neg_one, zero),
        )


class SinFwdKernel(FloatUnaryKernel):
    """Element-wise sin(x)."""

    @staticmethod
    def op_func(x):
        return T.sin(T.cast(x, "float32"))


class CosFwdKernel(FloatUnaryKernel):
    """Element-wise cos(x)."""

    @staticmethod
    def op_func(x):
        return T.cos(T.cast(x, "float32"))


class FloorFwdKernel(FloatUnaryKernel):
    """Element-wise floor(x).

    Casts to fp32 before calling ``T.floor`` because ``hfloor`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.floor(T.cast(x, "float32"))


class CeilFwdKernel(FloatUnaryKernel):
    """Element-wise ceil(x).

    Casts to fp32 before calling ``T.ceil`` because ``hceil`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.ceil(T.cast(x, "float32"))


class RoundFwdKernel(FloatUnaryKernel):
    """Element-wise round(x) with banker's rounding (round-to-nearest-even).

    Uses ``T.nearbyint`` (maps to ``nearbyintf`` in CUDA) to match
    PyTorch's ``torch.round`` semantics. Casts to fp32 because
    ``hnearbyint`` is not available for ``cutlass::half_t``.
    """

    @staticmethod
    def op_func(x):
        return T.nearbyint(T.cast(x, "float32"))


class TruncFwdKernel(FloatUnaryKernel):
    """Element-wise trunc(x) -- integer part toward zero.

    Casts to fp32 before calling ``T.trunc`` because ``htrunc`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.trunc(T.cast(x, "float32"))


class ErfFwdKernel(FloatUnaryKernel):
    """Element-wise erf(x).

    Casts to fp32 before calling ``T.erf`` because the half-precision
    intrinsic ``herf`` is not a valid CUDA built-in.
    """

    @staticmethod
    def op_func(x):
        return T.erf(T.cast(x, "float32"))


class Log1pFwdKernel(FloatUnaryKernel):
    """Element-wise log(1 + x).

    fp32 takes ``T.log1p``, which keeps the small values ``log(1 + x)`` rounds away
    once x falls under the epsilon of 1. A narrower result cannot hold them either way,
    so it takes the composite over the faster logarithm.
    """

    @staticmethod
    def op_func(x):
        wide = T.cast(x, "float32")
        if x.dtype == "float32":
            return T.log1p(wide)
        return log_for_output_precision(x, T.cast(1.0, "float32") + wide)


class Expm1FwdKernel(FloatUnaryKernel):
    """Element-wise exp(x) - 1."""

    @staticmethod
    def op_func(x):
        return T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")
