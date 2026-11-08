"""Float32 division for a quotient rounded to an element type, at the cost that type needs."""

import tilelang.language as T
import tvm.tirx as tirx

from tileops._kernels.elementwise._nan import bound

__all__ = [
    "QUOTIENT_NUM_PER_THREAD",
    "approx_fdiv",
    "fast_quotient",
    "float32_quotient",
    "full_range_fdiv",
    "half_quotient",
    "quartered_fdiv",
    "refined_reciprocal",
]

# Elements a register-copy thread of a quotient body carries. The body outlasts a second
# load of a 2-byte operand, and eight is still two loads of a 4-byte one.
QUOTIENT_NUM_PER_THREAD = 8


def approx_fdiv(num, den):
    """CUDA's ``__fdividef``: one ``div.approx.f32``, two ulp.

    It is defined for a divisor in ``[2**-126, 2**126]``. float16 spans ``[2**-24, 2**16]`` and rounds its result to eleven bits, so
    it meets both bounds. bfloat16 carries float32's exponent and a float32
    result keeps all 24 bits.
    """
    return T.call_extern("float32", "__fdividef", num, den)


def half_quotient(num, den):
    """``num / den`` in float32 for two float16 operands, torch's float16 quotient once rounded.

    ``approx_fdiv`` can move a quotient that lies exactly on a float16 rounding tie, which
    only a subnormal float16 result can, to the wrong side of it. One correction by the
    exact residual ``num - den * q`` puts such a quotient back on the tie. Where the residual
    is zero or NaN (an exact quotient, a zero, infinite or NaN operand, a zero divisor) the
    uncorrected quotient stands, with its sign. The caller's program pre-includes
    ``approx_math.h``.
    """

    def corrected(rcp, q):
        def pick(residual):
            # An ordered compare: false for a NaN residual as for a zero one.
            exact = T.abs(residual) > T.cast(0.0, "float32")
            return tirx.Select(exact, T.ieee_fmaf(residual, rcp, q), q)

        return bound(T.ieee_fmaf(-den, q, num), pick)

    rcp = T.call_extern("float32", "tileops::approx_rcp", den)
    return bound(rcp, lambda rcp: bound(num * rcp, lambda q: corrected(rcp, q)))


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


def quartered_fdiv(num, den):
    """``full_range_fdiv(num, den)`` for a divisor of at least 1, without its range check.

    A quarter of a finite such divisor stays within ``[2**-2, 2**126]``, where one
    reciprocal covers it, and scaling both operands by a power of two keeps the
    quotient's bits wherever ``num / 4`` is exact, as it is for every 16-bit ``num``.
    An infinite divisor gives a signed zero for a finite numerator and NaN for an
    infinite one, as ``full_range_fdiv`` does.
    """
    quarter = T.cast(0.25, "float32")
    return (num * quarter) * T.fast_rcp(den * quarter)


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


def fast_quotient(dtype: str, num, den):
    """``(value, holds)``: ``num / den`` in float32 for a bfloat16 or float32 result, without a branch.

    Wherever ``holds``, *value* stored as *dtype* is torch's quotient, the IEEE float32
    quotient rounded once; elsewhere the caller divides exactly.

    float32 runs the IEEE divide's own fast path: a reciprocal refined by one Newton
    step, the quotient, its exact residual and one correction. ``holds`` is both
    magnitudes in ``[2**-31, 2**33)``, where every nonzero intermediate is a normal
    float, so the value scales with the operands' exponents; it equals the IEEE quotient
    for all 2**46 pairs of significands. The test is bits 31 and 30 of
    ``2 * bits - 0x60000000``, clear for both operands, which nvcc folds into one OR a
    thread.

    bfloat16 takes ``num * rcp(den)``, which rounds to torch's bfloat16 for all 2**32
    operand pairs with a divisor magnitude in ``[2**-126, 2**126)``, where the reciprocal
    is normal; ``holds`` is that range.
    """
    rcp = T.fast_rcp(den)
    if dtype == "float32":
        one = T.cast(1.0, "float32")

        def quotient(r):
            q = num * r
            return T.ieee_fmaf(T.ieee_fmaf(-den, q, num), r, q)

        def window(x):
            bits = T.reinterpret(x, "uint32")
            return bits * T.cast(2, "uint32") + T.cast(0xA0000000, "uint32")

        value = bound(rcp, lambda r: bound(T.ieee_fmaf(T.ieee_fmaf(-den, r, one), r, r), quotient))
        holds = tirx.bitwise_or(window(num), window(den)) < T.cast(1 << 30, "uint32")
        return value, holds
    magnitude = T.abs(den)
    holds = tirx.And(
        magnitude >= T.cast(2.0**-126, "float32"), magnitude < T.cast(2.0**126, "float32")
    )
    return num * rcp, holds


def float32_quotient(dtype: str, num, den):
    """``num / den`` in float32, for a result stored as *dtype*.

    float16 takes ``approx_fdiv`` and bfloat16 ``full_range_fdiv``. Rounded to *dtype*,
    the result is the IEEE quotient's for all 2**32 bfloat16 pairs and for every
    reciprocal of either dtype; a float16 quotient on a subnormal rounding tie can round
    the wrong way, which ``half_quotient`` corrects. An operand computed in
    float32 can leave a two-ulp quotient on the other side of a rounding boundary, so
    such a caller checks its own results. A float32 result keeps the IEEE divide.
    """
    if dtype == "float16":
        return approx_fdiv(num, den)
    if dtype == "bfloat16":
        return full_range_fdiv(num, den)
    return num / den
