"""Arithmetic binary kernels, plus the two lerp forms."""

import functools

import tilelang
import tilelang.language as T
import torch
import tvm.tirx as tirx

from tileops.kernels.elementwise._base import (
    _FLOAT_DTYPES,
    BinaryKernel,
    MultiInputElementwiseKernel,
    _AlphaScaledBinaryKernel,
)
from tileops.kernels.elementwise._dtype import _BINARY_FULL_DTYPES, _BINARY_NO_BOOL_DTYPES
from tileops.kernels.elementwise._nan import _bound, nan_max, nan_min

__all__ = [
    "AddFwdKernel",
    "DivFwdKernel",
    "DivTruncFwdKernel",
    "FloorDivideFwdKernel",
    "LerpFwdKernel",
    "LerpTensorFwdKernel",
    "MaximumFwdKernel",
    "MinimumFwdKernel",
    "MulFwdKernel",
    "PowFwdKernel",
    "RemainderFwdKernel",
    "SubFwdKernel",
]


class AddFwdKernel(_AlphaScaledBinaryKernel):
    """Element-wise addition with scalar alpha: y = a + alpha * b."""

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES

    @staticmethod
    def _combine(a, scaled_b):
        return a + scaled_b


class SubFwdKernel(_AlphaScaledBinaryKernel):
    """Element-wise subtraction with scalar alpha: y = a - alpha * b."""

    SUPPORTED_DTYPES = _BINARY_NO_BOOL_DTYPES

    @staticmethod
    def _combine(a, scaled_b):
        return a - scaled_b


class MulFwdKernel(BinaryKernel):
    """Element-wise multiplication: y = a * b.

    Supports the manifest dtype union (bool / unsigned / signed integer /
    half / single precision floats). Bool multiplication is logical AND
    (PyTorch semantics).
    """

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES

    @staticmethod
    def op_func(a, b):
        return a * b


def _approx_fdiv(num, den):
    """CUDA's ``__fdividef``: one ``div.approx.f32``, two ulp, defined for a
    divisor in ``[2**-126, 2**126]``.

    float16 spans ``[2**-24, 2**16]`` and rounds its result to eleven bits, so
    it meets both bounds. bfloat16 carries float32's exponent and a float32
    result keeps all 24 bits.
    """
    return T.call_extern("float32", "__fdividef", num, den)


def _full_range_fdiv(num, den):
    """PTX's ``div.full.f32``: two ulp for every float32 divisor.

    It runs ``_approx_fdiv`` on both operands scaled by a power of two that brings
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
        return _bound(scale, lambda s: _approx_fdiv(num * s, den * s))

    return _bound(T.abs(den), scaled)


class DivFwdKernel(BinaryKernel):
    """Element-wise division: y = a / b.

    Divides in float32 and rounds once at the store, which is what torch does.
    float16 takes ``_approx_fdiv`` and bfloat16 ``_full_range_fdiv``.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The extern call scalarises the copies, so keep them off its loop."""
        return self.dtype in (torch.float16, torch.bfloat16) or super().stage_broadcast

    @staticmethod
    def op_func(a, b):
        num, den = T.Cast("float32", a), T.Cast("float32", b)
        if str(a.dtype) == "float16":
            return T.Cast(a.dtype, _approx_fdiv(num, den))
        if str(a.dtype) == "bfloat16":
            return T.Cast(a.dtype, _full_range_fdiv(num, den))
        return T.Cast(a.dtype, num / den)


def _ieee_fdiv(num, den):
    """A float32 divide rounded to nearest, which fast math leaves alone."""
    return T.call_extern("float32", "__fdiv_rn", num, den)


class DivTruncFwdKernel(BinaryKernel):
    """Element-wise truncated division: y = trunc(a / b), as torch computes it.

    torch rounds the quotient to the input dtype before truncating it, so a
    float16 ``299.9`` is ``300`` and truncates to ``300``. A float32 divide is IEEE:
    fast math's would leave an exact whole quotient one ulp short of it. A 16-bit
    one takes ``_full_range_fdiv``: truncated, it matches torch for all 2**32
    operand pairs of either dtype.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The extern calls scalarise the copies, so keep them off their loop."""
        return True

    @staticmethod
    def op_func(a, b):
        num, den = T.Cast("float32", a), T.Cast("float32", b)
        divide = _ieee_fdiv if str(a.dtype) == "float32" else _full_range_fdiv
        quotient = T.Cast(a.dtype, divide(num, den))
        return T.Cast(a.dtype, T.trunc(T.Cast("float32", quotient)))


# Below this quotient the whole quotient and the one above it are exact in float32.
_FAST_QUOTIENT = float(1 << 22)


def _floored_quotient(num, den, limit, fast_body):
    """``(fast_body(k, q), holds)``, with ``k = floor(a / b)`` wherever ``holds``.

    ``q`` is the IEEE quotient. Rounded to nearest, a quotient below a whole number
    lands on it at most, so ``t = floor(q)`` is k or k + 1. It is k + 1 exactly where
    ``a - t * b`` has the other sign from b, the sign of ``copysign(a, q) - t * |b|``.
    ``fma`` rounds that once, and a nonzero multiple of the smallest subnormal keeps
    its sign. ``holds`` fails on a NaN quotient or one of *limit* or more, which a zero
    b and an infinite or NaN operand give, and on an infinite b, whose quotient is zero.
    """
    zero = T.cast(0.0, "float32")
    one = T.cast(1.0, "float32")
    magnitude = T.abs(den)
    quotient = _ieee_fdiv(num, den)

    def value(q):
        def pick(t):
            over = T.call_extern("float32", "__fmaf_rn", -t, magnitude, T.copysign(num, q)) < zero
            return fast_body(tirx.Select(over, t - one, t), q)

        return _bound(T.floor(q), pick)

    inf = T.cast(float("inf"), "float32")
    holds = _bound(quotient, lambda q: T.And(T.abs(q) < T.cast(limit, "float32"), magnitude < inf))
    return _bound(quotient, value), holds


# Below this quotient of two operands of the dtype, ``_nudged_floor`` needs no residual.
_NUDGED_QUOTIENT = {"float16": float(1 << 8), "bfloat16": float(1 << 11)}
# Twice the relative error of ``_approx_fdiv``.
_NUDGE = 2.0**-21


def _nudged_floor(num, den, dtype, fast_body):
    """``(fast_body(k, q), holds)``, with ``k = floor(a / b)`` wherever ``holds``.

    For two operands of 16-bit *dtype*, with ``q`` the quotient ``a * (1 / b)``.
    Operands of p significant bits leave a quotient that is not whole at least
    ``2**-p`` of itself, or ``2**-p``, from every whole number, so below the limit it
    is further from one than the error of ``q`` plus the nudge. The nudge lifts an
    exact whole quotient ``q`` left short back onto it, so the floor of the nudged
    quotient is k, and a nonzero ``q`` carries the sign of ``a / b``. ``holds`` fails
    on a zero, NaN or infinite ``q`` and on one past the limit: every zero, infinite
    or NaN operand, every quotient that underflows and every divisor ``_approx_fdiv``
    does not cover.
    """
    limit = T.cast(_NUDGED_QUOTIENT[str(dtype)], "float32")
    # One reciprocal serves every element that shares b.
    quotient = num * _approx_fdiv(T.cast(1.0, "float32"), den)

    def value(q):
        nudged = T.call_extern("float32", "__fmaf_rn", T.abs(q), T.cast(_NUDGE, "float32"), q)
        return fast_body(T.floor(nudged), q)

    holds = _bound(quotient, lambda q: T.And(T.abs(q) < limit, q != T.cast(0.0, "float32")))
    return _bound(quotient, value), holds


def _floored_tiers(num, den, dtype, limit, fast_body):
    """``fast_body(k, q)`` as ``(value, holds)`` pairs, cheapest first.

    A float32 operand has the IEEE quotient alone; a 16-bit one tries the nudged
    quotient first.
    """
    tiers = [_floored_quotient(num, den, limit, fast_body)]
    if str(dtype) in _NUDGED_QUOTIENT:
        tiers.insert(0, _nudged_floor(num, den, dtype, fast_body))
    return tiers


def _first_holding(tiers, last):
    """The value of the first pair in *tiers* whose guard holds, else ``last``."""
    out = last
    for value, holds in reversed(tiers):
        out = T.if_then_else(holds, value, out)
    return out


def _on_float32(a, b, body):
    """``body(num, den)`` on the two operands widened to float32, each bound once."""
    return _bound(
        T.Cast("float32", a),
        lambda num: _bound(T.Cast("float32", b), lambda den: body(num, den)),
    )


class RemainderFwdKernel(BinaryKernel):
    """Element-wise remainder: y = a % b, with the sign of b.

    torch's CUDA kernel takes ``fmod`` in fp32, which is exact, and adds b when the
    result and b differ in sign: ``a - floor(a / b) * b`` rounded once. Here that is
    one ``fma`` from the exact floored quotient; ``fmodf`` serves the rest.
    """

    @staticmethod
    def _remainder(num, den, dtype):
        """``(tiers, slow())`` for ``a % b``: see ``_floored_tiers`` and ``RemainderFwdKernel``."""
        zero = T.cast(0.0, "float32")

        def from_quotient(k, q):
            r = T.call_extern("float32", "__fmaf_rn", -k, den, num)
            # A zero remainder is fmod's, which keeps the dividend's sign.
            return _bound(r, lambda r: tirx.Select(r == zero, T.copysign(zero, num), r))

        def signed(mod):
            flip = T.And(mod != zero, (den < zero) != (mod < zero))
            return tirx.Select(flip, mod + den, mod)

        tiers = _floored_tiers(num, den, dtype, _FAST_QUOTIENT, from_quotient)
        return tiers, _bound(T.fmod(num, den), signed)

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The extern calls scalarise the copies, so keep them off their loop."""
        return True

    @staticmethod
    def op_func(a, b):
        def body(num, den):
            tiers, slow = RemainderFwdKernel._remainder(num, den, a.dtype)
            return T.Cast(a.dtype, _first_holding(tiers, slow))

        return _on_float32(a, b, body)

    @staticmethod
    def fast_func(a, b):
        """The cheapest form of ``op_func``, and where it holds."""
        num, den = T.Cast("float32", a), T.Cast("float32", b)
        value, holds = RemainderFwdKernel._remainder(num, den, a.dtype)[0][0]
        return T.Cast(a.dtype, value), holds


class PowFwdKernel(BinaryKernel):
    """Element-wise power: y = a ** b.

    Computed as ``exp2(b * log2|a|)`` with the sign of a negative base
    restored. Error scales with ``|b * log2(a)|``; ``PowFwdOp`` tabulates it.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES
    REGISTER_COPY_NUM_PER_THREAD = 4

    @staticmethod
    def op_func(a, b):
        base = T.Cast("float32", a)
        expo = T.Cast("float32", b)
        zero = T.cast(0.0, "float32")
        one = T.cast(1.0, "float32")
        # |a| == 1 is the exception: log2 is zero, and 0 * inf is NaN where
        # pow answers one.
        magnitude = T.if_then_else(T.abs(base) == one, one, T.exp2(expo * T.log2(T.abs(base))))
        whole = T.round(expo)
        # A negative base flips the sign on an odd whole exponent.
        signed = T.if_then_else(
            T.fmod(T.abs(whole), T.cast(2.0, "float32")) == one, -magnitude, magnitude
        )
        # Only a finite nonzero negative base is NaN on a fractional exponent.
        defined = T.Or(whole == expo, T.Or(T.isinf(base), base == zero))
        # Sign bit, not ``base < 0``: pow carries the sign of -0.0, and
        # ``-0.0 < 0`` is false.
        negative = T.copysign(one, base) < zero
        out = T.if_then_else(
            negative,
            T.if_then_else(defined, signed, T.cast(float("nan"), "float32")),
            magnitude,
        )
        # x ** 0 is one; the magnitude gives NaN at base zero.
        return T.Cast(a.dtype, T.if_then_else(expo == zero, one, out))


# Below this quotient ``a - fmod(a, b)`` is a whole multiple of b exact in float32,
# so torch's divide returns the whole quotient itself: its significand and b's fit
# 24 bits together.
_EXACT_MULTIPLE_QUOTIENT = {"float16": float(1 << 13), "bfloat16": float(1 << 16)}


class FloorDivideFwdKernel(BinaryKernel):
    """Element-wise floor division: y = floor(a / b), as torch defines it.

    Follows torch's CUDA kernel: ``(a - fmod(a, b)) / b`` in fp32, one less when the
    remainder and b differ in sign, rounded to a whole number held in the input
    dtype. ``floor(a / b)`` differs where ``a / b`` rounds up to a whole number
    (``1.0 // 0.1`` is 9) and at an infinite b.
    """

    @staticmethod
    def _floor_divide(num, den, dtype):
        """``(tiers, slow())`` for torch's ``div_floor_floating`` on two float32 values.

        torch divides ``a - fmod(a, b)``, a multiple of b, by b, floors the quotient into
        *dtype* and rounds it up once where that dropped more than a half; a zero
        quotient keeps the sign of ``a / b`` and a zero divisor returns ``a / b`` itself.
        Where the multiple is exact the divide returns ``floor(a / b)``, and rounding
        that whole number into *dtype* never drops more than a half past the
        representable value below it without the next one up rounding back to it.
        Past it the result depends on how torch's divide rounds, and it is computed as
        torch does.
        """
        zero = T.cast(0.0, "float32")
        one = T.cast(1.0, "float32")
        # The sign ``a / b`` gives a zero, without dividing.
        signed_zero = T.copysign(zero, num) * T.copysign(one, den)

        def rounded(div):
            def bump(floored):
                up = T.Cast("float32", T.Cast(dtype, floored + one))
                return tirx.Select(div - floored > T.cast(0.5, "float32"), up, floored)

            near = _bound(T.Cast("float32", T.Cast(dtype, T.floor(div))), bump)
            return tirx.Select(div != zero, near, signed_zero)

        def slow():
            def from_mod(mod):
                flip = T.And(mod != zero, (den < zero) != (mod < zero))
                div = _ieee_fdiv(num - mod, den)
                return _bound(tirx.Select(flip, div - one, div), rounded)

            general = _bound(T.fmod(num, den), from_mod)
            return T.if_then_else(den == zero, _ieee_fdiv(num, den), general)

        def whole(k, q):
            # ``floor(a / b)`` has the sign of ``a / b``, which ``q`` carries, zero included.
            return T.copysign(k, q)

        limit = _EXACT_MULTIPLE_QUOTIENT.get(str(dtype), _FAST_QUOTIENT)
        return _floored_tiers(num, den, dtype, limit, whole), slow()

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The extern calls scalarise the copies, so keep them off their loop."""
        return True

    @staticmethod
    def op_func(a, b):
        def body(num, den):
            tiers, slow = FloorDivideFwdKernel._floor_divide(num, den, a.dtype)
            return T.Cast(a.dtype, _first_holding(tiers, slow))

        return _on_float32(a, b, body)

    @staticmethod
    def fast_func(a, b):
        """The cheapest form of ``op_func``, and where it holds."""
        num, den = T.Cast("float32", a), T.Cast("float32", b)
        value, holds = FloorDivideFwdKernel._floor_divide(num, den, a.dtype)[0][0]
        return T.Cast(a.dtype, value), holds


class LerpFwdKernel(BinaryKernel):
    """Element-wise lerp: y = a + weight * (b - a).

    PyTorch lerp is ternary (a, b, weight). Here weight is a compile-time
    constant bound at kernel construction, keeping the binary kernel template.

    Args:
        weight: Scalar interpolation weight (default 0.5). Keyword-only so the
            positional ``(dtype, config, tune)`` tail stays uniform.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        raise NotImplementedError(
            "LerpFwdKernel builds its body from weight; use the kernel through "
            "__init__ instead of calling op_func."
        )

    def __init__(self, a_shape, b_shape, dtype, config=None, tune=False, *, weight=0.5):
        self._weight = weight
        super().__init__(a_shape, b_shape, dtype, config=config, tune=tune)

    def _get_effective_op_func(self):
        weight = self._weight

        def lerp_func(a, b):
            return a + T.cast(weight, a.dtype) * (b - a)

        return f"{self._op_func_name()}|weight={weight!r}", lerp_func


class MaximumFwdKernel(BinaryKernel):
    """Element-wise maximum: y = max(a, b).

    For float dtypes, matches torch.maximum semantics:
    - If either operand is NaN, the result is NaN.
    - maximum(+0.0, -0.0) = +0.0 (IEEE 754 signed-zero).

    For integer / bool dtypes (no NaN representation), uses ``T.max``
    directly without the NaN guards.

    Performance (float path): uses T.max for the fast path (correct
    signed-zero on CUDA -- fmaxf returns +0 for max(+0,-0)) plus one
    isnan guard for NaN propagation.
    """

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The float body scalarises the copies, so keep them off its loop.

        Both float forms do it: fp32's NaN select, and the half formats' NaN
        builtin, which the vectoriser cannot see through either. Staged, the
        builtin costs what a plain ``T.max`` costs.

        Integer and bool dtypes take ``T.min``/``T.max`` directly, with nothing
        to scalarise them, and follow the base default.
        """
        return self.dtype.is_floating_point or super().stage_broadcast

    @staticmethod
    def op_func(a, b):
        return nan_max(a, b)


class MinimumFwdKernel(BinaryKernel):
    """Element-wise minimum: y = min(a, b).

    For float dtypes, matches torch.minimum semantics:
    - If either operand is NaN, the result is NaN.
    - minimum(-0.0, +0.0) = -0.0 (IEEE 754 signed-zero).

    For integer / bool dtypes (no NaN representation), uses ``T.min``
    directly without the NaN guards.

    Performance (float path): uses T.min for the fast path (correct
    signed-zero on CUDA -- fminf returns -0 for min(-0,+0)) plus one
    isnan guard for NaN propagation. See MaximumFwdKernel for full
    rationale.
    """

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The float body scalarises the copies, so keep them off its loop.

        Both float forms do it: fp32's NaN select, and the half formats' NaN
        builtin, which the vectoriser cannot see through either. Staged, the
        builtin costs what a plain ``T.max`` costs.

        Integer and bool dtypes take ``T.min``/``T.max`` directly, with nothing
        to scalarise them, and follow the base default.
        """
        return self.dtype.is_floating_point or super().stage_broadcast

    @staticmethod
    def op_func(a, b):
        return nan_min(a, b)


@functools.lru_cache(maxsize=32)
def _make_lerp_tensor_kernel(N, dtype, threads=256, npt=8):
    """Build Tensor-weight lerp kernel: out = a + weight * (b - a).

    ``LerpTensorFwdKernel.forward`` broadcasts ``input`` / ``end`` / ``weight``
    to the output shape and flattens them, so this PrimFunc sees three
    contiguous 1-D tensors of size ``N``, and computes in the input dtype.

    All three inputs move through register fragments so they share one
    vectorized access path; the result is written back into ``a``'s fragment.
    """

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),
            b: T.Tensor((N,), dtype),
            w: T.Tensor((N,), dtype),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                a_reg = T.alloc_fragment((block_size,), dtype)
                b_reg = T.alloc_fragment((block_size,), dtype)
                w_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(a[bx * block_size : (bx + 1) * block_size], a_reg)
                T.copy(b[bx * block_size : (bx + 1) * block_size], b_reg)
                T.copy(w[bx * block_size : (bx + 1) * block_size], w_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    a_reg[k] = a_reg[k] + w_reg[k] * (b_reg[k] - a_reg[k])
                T.copy(a_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class LerpTensorFwdKernel(MultiInputElementwiseKernel):
    """Tensor-weight lerp: out = input + weight * (end - input).

    Implements the Tensor-weight overload of ``torch.lerp`` --
    ``torch.lerp(input, end, weight: Tensor)`` -- where all three operands are
    float tensors of the same dtype, broadcast together and flattened by
    ``forward``.
    """

    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    DEFAULT_THREADS = 512
    INPUTS = (("a", "tile"), ("b", "tile"), ("w", "tile"))

    @staticmethod
    def _builder_fn():
        return _make_lerp_tensor_kernel

    def forward(self, a, b, w):
        return self._run(a=a, b=b, w=w)
