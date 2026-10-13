"""erf(x) and GELU(x) at the accuracy the result's storage dtype can hold."""

import tilelang.language as T

from tileops.kernels.constants import INV_SQRT2
from tileops.kernels.elementwise._nan import bound, keep_nan

__all__ = ["erf", "gelu"]

# |x| taken as saturated. erf(3.6) = 1 - 7.4e-7, far below half a float16 ulp.
_CLAMP = 3.6

# erf(x) = clip(t * P(w), -1, 1), t = clamp(x, -_CLAMP, _CLAMP), w = 1 - (t / _CLAMP)**2,
# highest degree first. Worst case over the real line is 1.7e-5, an order below half
# a float16 ulp at 1.0.
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
    polynomial's instruction count, which only a float32 result can hold.

    A NaN argument gives NaN, as in ``torch.erf``. The clamps lower to
    ``fminf``/``fmaxf``, which return their non-NaN operand, so one select on the
    argument restores it.

    Args:
        x: The argument. Any float dtype; promoted to float32 before evaluation.
        out_dtype: The dtype the caller stores the result in. It selects the
            evaluation, not the return dtype.

    Returns:
        erf(x) in float32.
    """

    def _clamped_polynomial(wide):
        """erf of float32 *wide* by the clamped polynomial; the clamps drop a NaN argument."""
        one = T.cast(1.0, "float32")
        clamped = T.min(T.max(wide, T.cast(-_CLAMP, "float32")), T.cast(_CLAMP, "float32"))
        # Scaling the clamp out before squaring, not after, is what makes w exactly zero
        # at the clamp: float32(1 / _CLAMP) * _CLAMP is 1.0, where 1 - x**2 / _CLAMP**2
        # leaves 1e-8 once the backend contracts it into an FMA.
        scaled = clamped * T.cast(1.0 / _CLAMP, "float32")
        w = one - scaled * scaled
        acc = T.cast(_POLY_COEFFS[0], "float32")
        for coeff in _POLY_COEFFS[1:]:
            acc = acc * w + T.cast(coeff, "float32")
        # The clip keeps the backend from contracting the product into a caller's add,
        # which would evaluate it to full width and land the tail 7e-9 short of +-1.
        # GELU scales the 1 - erf(x) residual by x, so that error is unbounded in |x|.
        return T.min(T.max(clamped * acc, -one), one)

    wide = T.cast(x, "float32")
    if out_dtype == "float32":
        return T.erf(wide)
    return keep_nan(wide, _clamped_polynomial)


def gelu(x, out_dtype):
    """Element-wise GELU(x) = x / 2 * (1 + erf(x / sqrt(2))), evaluated and returned in float32.

    The polynomial is ``erf``'s with its argument scaled to s = clamp(x / (sqrt(2) *
    _CLAMP), -1, 1) by one multiply and ``_CLAMP`` folded into the coefficients, so
    that erf = s * P(1 - s**2) and P(0) rounds to 1.0. The result's dependent chain
    is five instructions shorter than ``x / 2 * (1 + erf(x / sqrt(2), out_dtype))``:

    - the two argument scales are one multiply;
    - ``1 + s * P`` contracts into the last FMA;
    - no NaN select: the clamp takes a NaN to -1, and the x / 2 factor carries the
      NaN into the result;
    - one clip side: s * P overshoots 1 by up to 3.2e-6 short of the clamp, which
      only matters where 1 + erf would turn negative and flip a negative x's sign.

    ``erf`` keeps its form: the bound argument this needs costs Erf and GeluAndMul
    registers they do not have to spare.

    Args:
        x: The argument. Any float dtype; promoted to float32 before evaluation.
        out_dtype: The dtype the caller stores the result in. It selects the
            evaluation, not the return dtype.

    Returns:
        GELU(x) in float32.
    """
    wide = T.cast(x, "float32")
    half_x = T.cast(0.5, "float32") * wide
    one = T.cast(1.0, "float32")
    if out_dtype == "float32":
        return half_x * (one + T.erf(wide * T.cast(INV_SQRT2, "float32")))

    def one_plus_erf(s):
        w = one - s * s
        acc = T.cast(_POLY_COEFFS[0] * _CLAMP, "float32")
        for coeff in _POLY_COEFFS[1:]:
            acc = acc * w + T.cast(coeff * _CLAMP, "float32")
        return T.max(s * acc + one, T.cast(0.0, "float32"))

    scaled = wide * T.cast(INV_SQRT2 / _CLAMP, "float32")
    return half_x * bound(T.min(T.max(scaled, -one), one), one_plus_erf)
