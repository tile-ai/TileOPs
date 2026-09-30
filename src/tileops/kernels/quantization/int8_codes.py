"""Float32 arithmetic of the symmetric INT8 quantize kernels, bit-equal to the torch reference."""

import tilelang.language as T
import torch

__all__ = ["INV_QMAX", "SCALE_UP", "SMALL_SCALE", "abs_bits", "quantize", "widen"]

# torch's ``amax / 127`` multiplies by this float32 reciprocal, since the divisor is a CPU
# scalar; a kernel's scale is that product so that ``q`` divides by the reference's scale.
INV_QMAX = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(127.0))
# ``quantize`` is correctly rounded while its residual stays normal, which holds for a scale
# of at least 2**-100. A scale below SMALL_SCALE is multiplied, with every element it
# divides, by SCALE_UP first, which is exact and lifts any nonzero float32 scale to at
# least 2**-85.
SMALL_SCALE = 2.0**-60
SCALE_UP = 2.0**64


def abs_bits(value):
    """``|value|`` of a float32 as its int32 bit pattern, which orders as the magnitude does
    and puts a NaN above every number."""
    return T.reinterpret(T.abs(value), "int32")


def quantize(value, scale, rcp, clamp: bool = False):
    """``value / scale`` of a float32, rounded half to even, as int8.

    ``rcp`` is the correctly rounded reciprocal of ``scale``; two FMAs make the quotient
    correctly rounded. Adding 1.5 * 2**23 to a float of magnitude below 2**22 rounds it
    half to even to an integer, which the low byte of the sum's bit pattern then holds in
    two's complement. A normal scale is at least ``amax / 127`` rounded, so the quotient
    rounds to at most 127 and needs no clamp; a subnormal scale keeps too few bits for
    that, and ``clamp`` bounds the quotient to ``[-127, 127]`` first.
    """
    q0 = value * rcp
    quotient = T.ieee_fmaf(T.ieee_fmaf(-q0, scale, value), rcp, q0)
    if clamp:
        quotient = T.clamp(quotient, T.float32(-127.0), T.float32(127.0))
    return T.cast(T.reinterpret(quotient + T.float32(12582912.0), "int32"), "int8")


def widen(word, half: int, dtype: str):
    """The float32 of element ``half`` of a 32-bit word that holds ``dtype`` elements."""
    if dtype == "bfloat16":
        # A bfloat16 is the high half of the float32 it widens to.
        if half:
            return T.reinterpret(word & T.uint32(0xFFFF0000), "float32")
        return T.reinterpret(word << T.uint32(16), "float32")
    if dtype == "float16":
        bits = (word >> T.uint32(16)) if half else word
        return T.cast(T.reinterpret(T.cast(bits, "uint16"), "float16"), "float32")
    return T.reinterpret(word, "float32")
