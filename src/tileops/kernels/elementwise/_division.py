"""Float32 division for a quotient rounded to an element type, at the cost that type needs."""

import tilelang.language as T
import tvm.tirx as tirx

from tileops.kernels.elementwise._nan import bound

__all__ = [
    "approx_fdiv",
    "float32_quotient",
    "full_range_fdiv",
    "in_range_fdiv",
    "refined_reciprocal",
]


def approx_fdiv(num, den):
    """CUDA's ``__fdividef``: one ``div.approx.f32``, two ulp.

    It is defined for a divisor in ``[2**-126, 2**126]``. float16 spans ``[2**-24, 2**16]`` and rounds its result to eleven bits, so
    it meets both bounds. bfloat16 carries float32's exponent and a float32
    result keeps all 24 bits.
    """
    return T.call_extern("float32", "__fdividef", num, den)


def full_range_fdiv(num, den):
    """PTX's ``div.full.f32``: two ulp for every float32 divisor.

    It runs ``approx_fdiv`` on both operands scaled by a power of two that brings
    the divisor into that function's range. Rounded to bfloat16 it gives the IEEE
    quotient's bfloat16 for every pair of bfloat16 operands, all 2**32 checked
    against torch. An IEEE divide costs a range check and a branch per element.
    """

    def scaled(magnitude):
        big = T.cast(2.0**126, "float32")
        scale = tirx.Select(
            magnitude > big,
            T.cast(0.25, "float32"),
            tirx.Select(
                magnitude < T.cast(2.0**-126, "float32"),
                T.cast(2.0**24, "float32"),
                T.cast(1.0, "float32"),
            ),
        )
        return bound(scale, lambda s: approx_fdiv(num * s, den * s))

    return bound(T.abs(den), scaled)


def in_range_fdiv(num, den):
    """``(approx_fdiv(num, den), holds)``: ``full_range_fdiv`` without its scaling.

    ``holds`` is a divisor magnitude in ``[2**-126, 2**126]``, where ``full_range_fdiv``
    scales by one and returns this value bit for bit; a NaN divisor fails it.
    """
    magnitude = T.abs(den)
    holds = tirx.And(
        magnitude >= T.cast(2.0**-126, "float32"), magnitude <= T.cast(2.0**126, "float32")
    )
    return approx_fdiv(num, den), holds


def refined_reciprocal(den):
    """``1 / den`` in float32 for *den* of at least one: a reciprocal and one Newton step.

    The step costs two fused multiply-adds and no branch; an IEEE divide costs a
    range check and a branch to its slow path per element. ``approx_fdiv`` returns
    zero for a divisor past ``2**126``, where the step would form ``inf * 0``, so a
    zero reciprocal is kept; a NaN divisor gives NaN. Below one the domain ends: a
    zero divisor gives NaN, not infinity. The result is not the IEEE quotient in
    every bit, so a caller checks its rounded results.
    """

    def step(d, r):
        one = T.cast(1.0, "float32")
        return tirx.Select(r == T.cast(0.0, "float32"), r, r + r * (one - d * r))

    return bound(den, lambda d: bound(approx_fdiv(T.cast(1.0, "float32"), d), lambda r: step(d, r)))


def float32_quotient(dtype: str, num, den):
    """``num / den`` in float32, for a result stored as *dtype*.

    float16 takes ``approx_fdiv`` and bfloat16 ``full_range_fdiv``. Rounded to *dtype*,
    the quotient of two *dtype* values is the IEEE quotient's: checked over all 2**32
    bfloat16 pairs and over every reciprocal of either dtype. An operand computed in
    float32 can leave a two-ulp quotient on the other side of a rounding boundary, so
    such a caller checks its own results. A float32 result keeps the IEEE divide.
    """
    if dtype == "float16":
        return approx_fdiv(num, den)
    if dtype == "bfloat16":
        return full_range_fdiv(num, den)
    return num / den
