"""Unary math kernels: exp/log family, roots, rounding, trigonometry."""

import tilelang.language as T
import torch

from tileops.kernels.elementwise._base import FloatUnaryKernel, TorchFallbackKernel
from tileops.kernels.elementwise._dtype import FLOAT_DTYPES, log_for_output_precision
from tileops.kernels.elementwise._erf import erf
from tileops.kernels.elementwise.call_spec import (
    ElementwiseCall,
    ReciprocalFwdInterface,
    RoundCall,
    RoundFwdInterface,
    UnaryElementwiseFwdInterface,
)
from tileops.kernels.kernel_base import Entry, Kernel

# The integral element types the manifest admits for the unary math ops. This backend
# compiles float programs only, so a torch fallback answers for them.
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

__all__ = [
    "INT_DTYPES",
    "AbsFwdKernel",
    "AbsIntFwdKernel",
    "CeilFwdKernel",
    "CosFwdKernel",
    "ErfFwdKernel",
    "ExpFwdKernel",
    "Expm1FwdKernel",
    "FloorFwdKernel",
    "Log1pFwdKernel",
    "LogFwdKernel",
    "IntIdentityFwdKernel",
    "NegFwdKernel",
    "NegIntFwdKernel",
    "ReciprocalFwdKernel",
    "RoundDecimalsFwdKernel",
    "RoundFwdKernel",
    "RsqrtFwdKernel",
    "SignFwdKernel",
    "SignIntFwdKernel",
    "SinFwdKernel",
    "SqrtFwdKernel",
    "TruncFwdKernel",
]


class ExpFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise exp(x)."""

    @staticmethod
    def op_func(x):
        return T.exp(T.cast(x, "float32"))


class LogFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise log(x)."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        return log_for_output_precision(x, T.cast(x, "float32"))


class SqrtFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise sqrt(x)."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        return T.sqrt(T.cast(x, "float32"))


class RsqrtFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise 1/sqrt(x)."""

    @staticmethod
    def op_func(x):
        return T.rsqrt(T.cast(x, "float32"))


class AbsFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise |x|."""

    @staticmethod
    def op_func(x):
        return T.abs(x)


class NegFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise -x."""

    @staticmethod
    def op_func(x):
        return -x


class ReciprocalFwdKernel(FloatUnaryKernel, ReciprocalFwdInterface):
    """Element-wise 1/x.

    Integral inputs are this backend's business: it has no integer kernel, so it
    declares float32 as the type it computes them in and converts at the
    boundary. A backend with a native integer-input reciprocal declares nothing
    and receives the integers.
    """

    BYTES_PER_THREAD = 32

    @classmethod
    def refusal(cls, call) -> "str | None":
        return None if call.dtype in INT_DTYPES else super().refusal(call)

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        dtype = torch.float32 if call.dtype in INT_DTYPES else call.dtype
        return call, lambda: cls(call.n_total, dtype)

    @staticmethod
    def op_func(x):
        return T.cast(1.0, "float32") / x

    def forward(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return super().forward(x)


class SignFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise sign(x): -1, 0, or +1."""

    @staticmethod
    def op_func(x):
        # In float32, not x.dtype: the comparisons are exact in either width, and a
        # half-native select lowers to scalar code instead of vectorising.
        zero = T.cast(0.0, "float32")
        one = T.cast(1.0, "float32")
        neg_one = T.cast(-1.0, "float32")
        wide = T.cast(x, "float32")
        return T.if_then_else(
            wide > zero,
            one,
            T.if_then_else(wide < zero, neg_one, zero),
        )


class SinFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise sin(x)."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        return T.sin(T.cast(x, "float32"))


class CosFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise cos(x)."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        return T.cos(T.cast(x, "float32"))


class FloorFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise floor(x).

    Casts to fp32 before calling ``T.floor`` because ``hfloor`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.floor(T.cast(x, "float32"))


class CeilFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise ceil(x).

    Casts to fp32 before calling ``T.ceil`` because ``hceil`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.ceil(T.cast(x, "float32"))


class RoundFwdKernel(FloatUnaryKernel, RoundFwdInterface):
    """Element-wise round(x) with banker's rounding (round-to-nearest-even).

    Uses ``T.nearbyint`` (maps to ``nearbyintf`` in CUDA) to match
    PyTorch's ``torch.round`` semantics. Casts to fp32 because
    ``hnearbyint`` is not available for ``cutlass::half_t``.
    """

    @classmethod
    def refusal(cls, call: RoundCall) -> "str | None":
        if call.decimals != 0:
            return f"rounds to whole numbers, not {call.decimals} decimal places"
        return super().refusal(call)

    @staticmethod
    def op_func(x):
        return T.nearbyint(T.cast(x, "float32"))


class TruncFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise trunc(x) -- integer part toward zero.

    Casts to fp32 before calling ``T.trunc`` because ``htrunc`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.trunc(T.cast(x, "float32"))


class ErfFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise erf(x)."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        return erf(x, x.dtype)


class Log1pFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise log(1 + x).

    fp32 takes ``T.log1p``, which keeps the small values ``log(1 + x)`` rounds away
    once x falls under the epsilon of 1. A narrower result cannot hold them either way,
    so it takes the composite over the faster logarithm.
    """

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        wide = T.cast(x, "float32")
        if x.dtype == "float32":
            return T.log1p(wide)
        return log_for_output_precision(x, T.cast(1.0, "float32") + wide)


class Expm1FwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise exp(x) - 1."""

    @staticmethod
    def op_func(x):
        return T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")


class AbsIntFwdKernel(TorchFallbackKernel, UnaryElementwiseFwdInterface):
    """The absolute value of an integral input."""

    SUPPORTED_DTYPES = INT_DTYPES
    handler = staticmethod(torch.abs)


class NegIntFwdKernel(TorchFallbackKernel, UnaryElementwiseFwdInterface):
    """The negation of an integral input."""

    SUPPORTED_DTYPES = INT_DTYPES
    handler = staticmethod(torch.neg)


class SignIntFwdKernel(TorchFallbackKernel, UnaryElementwiseFwdInterface):
    """The sign of an integral input."""

    SUPPORTED_DTYPES = INT_DTYPES
    handler = staticmethod(torch.sign)


class IntIdentityFwdKernel(TorchFallbackKernel, UnaryElementwiseFwdInterface, RoundFwdInterface):
    """An integral input, unchanged: floor, ceil, round and trunc all leave it alone."""

    SUPPORTED_DTYPES = INT_DTYPES
    handler = staticmethod(torch.Tensor.clone)

    @classmethod
    def refusal(cls, call) -> "str | None":
        if getattr(call, "decimals", 0) != 0:
            return f"leaves whole numbers alone, which {call.decimals} decimal places do not"
        return super().refusal(call)


class RoundDecimalsFwdKernel(Kernel, RoundFwdInterface):
    """Rounding to a non-zero number of decimal places.

    ``round(x, decimals=k)`` is ``round(x * 10 ** k) / 10 ** k``, which the
    round-to-nearest-integer program does not do; torch computes it. The class compiles
    nothing, so there is nothing to tune.
    """

    SUPPORTED_DTYPES = FLOAT_DTYPES + INT_DTYPES

    @classmethod
    def refusal(cls, call: RoundCall) -> "str | None":
        if call.decimals == 0:
            return "rounds to a non-zero number of decimal places"
        if call.dtype not in cls.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in cls.SUPPORTED_DTYPES)
            return f"serves dtypes [{supported}], not {call.dtype}"
        return None

    @classmethod
    def applies(cls, call: RoundCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def entry_for(cls, call: RoundCall) -> Entry:
        return call.decimals, lambda: cls(call.decimals)

    def __init__(self, decimals: int) -> None:
        """Round to *decimals* places."""
        super().__init__()
        self.decimals = decimals

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Through float32, so scaling by ``10 ** decimals`` cannot overflow a narrow
        # input; the single down-cast restores the element type.
        return torch.round(input.float(), decimals=self.decimals).to(input.dtype)
