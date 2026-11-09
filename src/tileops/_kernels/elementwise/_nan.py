"""NaN-propagating max and min, the ones torch's maximum, minimum and clamp follow."""

import tilelang.language as T
import tvm.tirx as tirx

__all__ = [
    "HALF_COMPARE_HEADER",
    "keep_nan",
    "lane_compare",
    "nan_max",
    "nan_min",
    "ne_unordered",
]

# The header ``ne_unordered`` calls into; a kernel whose body calls it lists it in
# ``CSRC_HEADERS``.
HALF_COMPARE_HEADER = "half_compare.h"


def bound(x, body):
    """``body(x)`` with *x* evaluated once, however often *body* mentions it.

    Bind a non-leaf *x* that *body* mentions more than once. In a body a guarded
    builder replicates per vector -- the floored tiers -- bind only an ``asm volatile``
    value such as ``T.fast_rcp``; nvcc merges a repeated pure expression itself.
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

    return bound(a, lambda a: bound(b, lambda b: select(a, b)))


def _select(a, b, plain, intrin):
    dtype = str(a.dtype)
    if not dtype.startswith(("float", "bfloat")):
        # Integer and bool have no NaN.
        return plain(a, b)
    if dtype in ("float16", "bfloat16"):
        # ``tl.max_nan`` / ``tl.min_nan`` lower to ``__hmax_nan`` / ``__hmin_nan``: one
        # instruction where the guarded form is a compare, an or and a select. TileLang
        # registers both without a Python wrapper, so they are named through
        # ``tirx.op.Op.get``, as its own ``math_intrinsics`` names ``tl.max2``. For float32
        # both lower to ``fmaxf``/``fminf``, which drop NaN.
        return tirx.call_intrin(a.dtype, tirx.op.Op.get(intrin), a, b)
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
    raises the erf body from 29 to 40 registers and slows GELUAndMul by 2%.
    """
    return bound(x, lambda x: tirx.Select(_is_nan(x), x, fn(x)))


def ne_unordered(a, b):
    """``a != b``, true where either operand is NaN, as torch.ne answers."""
    if str(a.dtype) in ("float16", "bfloat16"):
        return T.call_extern("bool", "tileops::ne_unordered", a, b)
    return a != b


def lane_compare(name):
    """A body's form over two 16-bit lanes of a word: ``half_compare.h``'s ``<name>_lanes``.

    The form maps two uint32 words of *dtype* lanes to a mask, 0xffff a lane where the
    body holds. ``None`` for a dtype without a 16-bit float pair type.
    """

    # The CUDA pair type of a 16-bit float, as ``half_compare.h``'s lane compares take it.
    pair_types = {"float16": "__half2", "bfloat16": "__nv_bfloat162"}

    def form(dtype):
        pair = pair_types.get(str(dtype))
        if pair is None:
            return None
        return lambda a, b: T.call_extern("uint32", f"tileops::{name}_lanes<{pair}>", a, b)

    return form
