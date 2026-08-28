"""erf(x) at the accuracy the result's storage dtype can hold."""

import tilelang.language as T

__all__ = ["erf"]

#: |x| at which erf is taken as saturated. erf(3.6) = 1 - 7.4e-7, three hundred
#: times finer than half a float16 ulp at 1.0, so the clamp costs neither dtype
#: anything it could store.
_CLAMP = 3.6

#: erf(x) = clip(t * P(w), -1, 1) for t = clamp(x, -_CLAMP, _CLAMP) and
#: w = 1 - (t / _CLAMP)**2, coefficients highest degree first. Three properties of
#: this form, in this order:
#:
#: - erf saturates to exactly +-1 at the clamp, which GELU depends on: it scales
#:   the 1 - erf(x) residual by x, so a tail off by eps carries an error of
#:   |x| * eps / 2, unbounded in |x|. Two things make it exact. _CLAMP is scaled
#:   out of x before squaring rather than after, so w is exactly zero at the clamp
#:   and P is exactly its constant term float32(1 / _CLAMP), whose product with
#:   _CLAMP is 1.0; and the clip stops the backend contracting that product into
#:   the caller's add, which would evaluate it to full width and land 7e-9 short.
#: - w rather than x**2 as the polynomial variable keeps the Horner chain
#:   conditioned in float32. The same fit in unnormalised x**2 loses four digits
#:   to cancellation at the clamp.
#: - t * P(w) rather than a polynomial in x alone keeps the *relative* error small
#:   near zero, where erf(x) -> x.
#:
#: Worst case over the real line is 1.7e-5, an order below half a float16 ulp at
#: 1.0 (2.4e-4). tests/ops/test_unary_math.py measures the resulting ulp distance
#: on device, over every float16 and every bfloat16 value.
_POLY_COEFFS = (
    8.971932411193848,
    -33.1397590637207,
    53.60123062133789,
    -48.02798080444336,
    26.02545738220215,
    -8.488715171813965,
    1.7499406337738037,
    -0.09359221160411835,
    0.11330155283212662,
    0.1387363225221634,
    0.2777777910232544,
)


def erf(x, out_dtype):
    """Element-wise erf(x), evaluated and returned in float32.

    ``erff`` is correct to 2 ulp of float32 and costs roughly twice the
    polynomial's instruction count. A float16 or bfloat16 result cannot hold that
    accuracy, so only float32 pays for it.

    NaN is not preserved: the clamp lowers to ``fminf``/``fmaxf``, which return
    their non-NaN operand, so a NaN argument reads back as +-1. That matches what
    the other clamp-shaped bodies in this package already do -- relu, hardtanh and
    hardsigmoid all answer a NaN input with a finite value. Guarding it costs 23%,
    because ``T.if_then_else`` in an element body scalarises the loop.

    Args:
        x: The argument. Any float dtype; promoted to float32 before evaluation.
        out_dtype: The dtype the caller stores the result in. It selects the
            evaluation, not the return dtype.

    Returns:
        erf(x) in float32.

    Example:
        >>> erf(T.cast(x, "float32") * inv_sqrt_2, x.dtype)  # doctest: +SKIP
    """
    wide = T.cast(x, "float32")
    if out_dtype == "float32":
        return T.erf(wide)
    one = T.cast(1.0, "float32")
    clamped = T.min(T.max(wide, T.cast(-_CLAMP, "float32")), T.cast(_CLAMP, "float32"))
    scaled = clamped * T.cast(1.0 / _CLAMP, "float32")
    w = one - scaled * scaled
    acc = T.cast(_POLY_COEFFS[0], "float32")
    for coeff in _POLY_COEFFS[1:]:
        acc = acc * w + T.cast(coeff, "float32")
    return T.min(T.max(clamped * acc, -one), one)
