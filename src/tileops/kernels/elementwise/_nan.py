"""NaN-propagating max and min, the ones torch's maximum, minimum and clamp follow."""

import tilelang.language as T
import tvm.tirx as tirx

__all__ = ["nan_max", "nan_min"]


def _propagate_nan(a, b, result):
    """Return NaN where either operand is NaN, and *result* everywhere else.

    ``fminf``/``fmaxf`` return the non-NaN operand, so the NaN torch propagates
    has to be put back. One select, not two: each ``T.if_then_else`` scalarises
    the element loop. ``isnan`` takes float32 because bfloat16 has no native
    form; a self-compare is not a guard, since TileLang lowers ``!=`` as an
    ordered compare.

    The NaN is canonical where torch returns the offending operand, visible only
    to a caller reading raw bits. Naming which operand would cost a second select.
    """
    either_is_nan = tirx.any(T.isnan(T.Cast("float32", a)), T.isnan(T.Cast("float32", b)))
    return T.if_then_else(either_is_nan, T.Cast(a.dtype, T.cast(float("nan"), "float32")), result)


def _nan_intrin(name, a, b):
    """A TileLang NaN-propagating builtin, named the way TileLang names its own.

    ``tl.max_nan`` and ``tl.min_nan`` lower to CUDA's ``__hmax_nan`` /
    ``__hmin_nan``: one instruction where the guarded form is a compare, an or
    and a select. TileLang registers both builtins and wraps neither in Python,
    so there is nothing to import; naming the op through ``tirx.op.Op.get`` is
    the line its own ``math_intrinsics`` uses for ``tl.max2`` and the ieee_*
    family.
    """
    return tirx.call_intrin(a.dtype, tirx.op.Op.get(name), a, b)


def _select(a, b, plain, intrin):
    dtype = str(a.dtype)
    if not dtype.startswith(("float", "bfloat")):
        # Integer and bool have no NaN.
        return plain(a, b)
    if dtype in ("float16", "bfloat16"):
        return _nan_intrin(intrin, a, b)
    return _propagate_nan(a, b, plain(a, b))


def nan_max(a, b):
    """``max(a, b)``, NaN where either operand is NaN."""
    return _select(a, b, T.max, "tl.max_nan")


def nan_min(a, b):
    """``min(a, b)``, NaN where either operand is NaN."""
    return _select(a, b, T.min, "tl.min_nan")
