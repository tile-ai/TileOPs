"""Comparison and float-predicate kernels."""

import tilelang.language as T
import torch

from tileops.kernels.elementwise._base import (
    BinaryKernel,
    FloatPredicateKernel,
    TorchFallbackKernel,
    Uint8StorageBinaryKernel,
)
from tileops.kernels.elementwise._dtype import BINARY_FULL_DTYPES
from tileops.kernels.elementwise.call_spec import (
    BinaryPredicateFwdInterface,
    UnaryPredicateFwdInterface,
)
from tileops.kernels.elementwise.math_unary import INT_DTYPES

# Neither an integer nor a bool can hold a NaN or an infinity, so the float predicates
# answer for them without reading the input.
_PREDICATE_FALLBACK_DTYPES = INT_DTYPES + (torch.bool,)

__all__ = [
    "AlwaysFalseFwdKernel",
    "AlwaysTrueFwdKernel",
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


class EqFwdKernel(BinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise equality: y = (a == b)."""

    SUPPORTED_DTYPES = BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a == b


class EqBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise equality on uint8-backed bool storage."""

    preferred_over = frozenset({"eq"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_xor(T.bitwise_xor(a, b), T.cast(1, "uint8"))


class NeFwdKernel(BinaryKernel, BinaryPredicateFwdInterface):
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

    SUPPORTED_DTYPES = BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        if str(a.dtype) in ("float16", "bfloat16"):
            return T.Cast("float32", a) != T.Cast("float32", b)
        return a != b


class NeBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise not-equal on uint8-backed bool storage."""

    preferred_over = frozenset({"ne"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_xor(a, b)


class GtFwdKernel(BinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise greater-than: y = (a > b)."""

    SUPPORTED_DTYPES = BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a > b


class GtBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise greater-than on uint8-backed bool storage."""

    preferred_over = frozenset({"gt"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_and(a, T.bitwise_xor(b, T.cast(1, "uint8")))


class LtFwdKernel(BinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise less-than: y = (a < b)."""

    SUPPORTED_DTYPES = BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a < b


class LtBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise less-than on uint8-backed bool storage."""

    preferred_over = frozenset({"lt"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_and(T.bitwise_xor(a, T.cast(1, "uint8")), b)


class GeFwdKernel(BinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise greater-equal: y = (a >= b)."""

    SUPPORTED_DTYPES = BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a >= b


class GeBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise greater-equal on uint8-backed bool storage."""

    preferred_over = frozenset({"ge"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_or(a, T.bitwise_xor(b, T.cast(1, "uint8")))


class LeFwdKernel(BinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise less-equal: y = (a <= b)."""

    SUPPORTED_DTYPES = BINARY_FULL_DTYPES
    OUTPUT_DTYPE = torch.bool

    @staticmethod
    def op_func(a, b):
        return a <= b


class LeBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryPredicateFwdInterface):
    """Element-wise less-equal on uint8-backed bool storage."""

    preferred_over = frozenset({"le"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_or(T.bitwise_xor(a, T.cast(1, "uint8")), b)


class IsnanFwdKernel(FloatPredicateKernel, UnaryPredicateFwdInterface):
    """Element-wise isnan with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isnan(T.cast(x, "float32"))


class IsinfFwdKernel(FloatPredicateKernel, UnaryPredicateFwdInterface):
    """Element-wise isinf with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isinf(T.cast(x, "float32"))


class IsfiniteFwdKernel(FloatPredicateKernel, UnaryPredicateFwdInterface):
    """Element-wise isfinite with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isfinite(T.cast(x, "float32"))


class AlwaysFalseFwdKernel(TorchFallbackKernel, UnaryPredicateFwdInterface):
    """False everywhere: what isnan and isinf answer for an integer or bool input."""

    SUPPORTED_DTYPES = _PREDICATE_FALLBACK_DTYPES

    @staticmethod
    def handler(input: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input.shape, dtype=torch.bool, device=input.device)


class AlwaysTrueFwdKernel(TorchFallbackKernel, UnaryPredicateFwdInterface):
    """True everywhere: what isfinite answers for an integer or bool input."""

    SUPPORTED_DTYPES = _PREDICATE_FALLBACK_DTYPES

    @staticmethod
    def handler(input: torch.Tensor) -> torch.Tensor:
        return torch.ones(input.shape, dtype=torch.bool, device=input.device)
