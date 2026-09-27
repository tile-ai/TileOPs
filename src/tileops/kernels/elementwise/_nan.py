"""NaN-propagating max and min, the ones torch's maximum, minimum and clamp follow."""

import tilelang.language as T
import tvm.tirx as tirx

__all__ = ["keep_nan", "nan_max", "nan_min"]


def _bound(x, body):
    """``body(x)`` with *x* evaluated once, however often *body* mentions it.

    A select names its operand in the test and in an arm, and a nested call (a
    clamp is ``nan_min(nan_max(x, lo), hi)``) names the inner select three times.
    Written out unbound, nvcc keeps the copies and branches around each one; bound
    to a local, the select compiles to a compare and a predicated instruction.
    """
    if isinstance(x, (tirx.Var, tirx.FloatImm, tirx.IntImm)):
        return body(x)
    var = tirx.Var("nan_operand", x.dtype)
    return tirx.Let(var, x, body(var))


def _is_nan(x):
    # isnan takes float32 because bfloat16 has no native form. A self-compare is
    # not a guard: TileLang lowers ``!=`` as an ordered compare.
    return T.isnan(T.Cast("float32", x))


def _propagate_nan(a, b, plain):
    """``plain(a, b)``, and NaN where either operand is NaN.

    ``fminf``/``fmaxf`` return the non-NaN operand, so the NaN torch propagates
    has to be put back. The NaN is canonical where torch returns the offending
    operand, visible only to a caller reading raw bits.
    """

    def select(a, b):
        nan = T.Cast(a.dtype, T.cast(float("nan"), "float32"))
        return tirx.Select(tirx.any(_is_nan(a), _is_nan(b)), nan, plain(a, b))

    return _bound(a, lambda a: _bound(b, lambda b: select(a, b)))


def _nan_intrin(name, a, b):
    """A TileLang NaN-propagating builtin, named the way TileLang names its own.

    ``tl.max_nan`` and ``tl.min_nan`` lower to CUDA's ``__hmax_nan`` /
    ``__hmin_nan``: one instruction where the guarded form is a compare, an or
    and a select. TileLang registers both builtins and wraps neither in Python,
    so there is nothing to import; naming the op through ``tirx.op.Op.get`` is
    the line its own ``math_intrinsics`` uses for ``tl.max2`` and the ieee_*
    family. For float32 both lower to ``fmaxf``/``fminf``, which drop NaN.
    """
    return tirx.call_intrin(a.dtype, tirx.op.Op.get(name), a, b)


def _select(a, b, plain, intrin):
    dtype = str(a.dtype)
    if not dtype.startswith(("float", "bfloat")):
        # Integer and bool have no NaN.
        return plain(a, b)
    if dtype in ("float16", "bfloat16"):
        return _nan_intrin(intrin, a, b)
    return _propagate_nan(a, b, plain)


def nan_max(a, b):
    """``max(a, b)``, NaN where either operand is NaN."""
    return _select(a, b, T.max, "tl.max_nan")


def nan_min(a, b):
    """``min(a, b)``, NaN where either operand is NaN."""
    return _select(a, b, T.min, "tl.min_nan")


def keep_nan(x, fn):
    """``fn(x)``, and *x* itself where *x* is NaN.

    For a body that drops NaN somewhere inside, such as a clamp ahead of a
    polynomial: one select on the argument restores it. Only *x* is bound. A
    bound ``fn(x)`` keeps every unrolled element's value live at once, which
    raises the erf body from 29 to 40 registers and slows GeluAndMul by 2%.
    """
    return _bound(x, lambda x: tirx.Select(_is_nan(x), x, fn(x)))
