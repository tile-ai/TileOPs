"""Arithmetic binary kernels, plus the two lerp forms."""

import functools

import tilelang
import tilelang.language as T
import torch
import tvm.tirx as tirx

from ._base import (
    _FLOAT_DTYPES,
    BinaryKernel,
    MultiInputElementwiseKernel,
    _AlphaScaledBinaryKernel,
)
from ._dtype import _BINARY_FULL_DTYPES, _BINARY_NO_BOOL_DTYPES
from ._nan import _bound, nan_max, nan_min

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


class DivFwdKernel(BinaryKernel):
    """Element-wise division: y = a / b.

    Divides in float32 and rounds once at the store, which is what torch does.
    float16 takes ``_approx_fdiv``.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The extern call scalarises the copies, so keep them off its loop."""
        return self.dtype == torch.float16 or super().stage_broadcast

    @staticmethod
    def op_func(a, b):
        num, den = T.Cast("float32", a), T.Cast("float32", b)
        if str(a.dtype) == "float16":
            return T.Cast(a.dtype, _approx_fdiv(num, den))
        return T.Cast(a.dtype, num / den)


def _ieee_fdiv(num, den):
    """A float32 divide rounded to nearest, which fast math leaves alone."""
    return T.call_extern("float32", "__fdiv_rn", num, den)


class DivTruncFwdKernel(BinaryKernel):
    """Element-wise truncated division: y = trunc(a / b).

    Matches ``torch.div(a, b, rounding_mode="trunc")`` semantics: rounds
    the quotient toward zero. Division and ``trunc`` are computed in fp32
    to avoid two sources of error: (1) ``htrunc`` is not available for
    ``cutlass::half_t`` in CUDA, and (2) fp16 division rounds the
    quotient before ``trunc`` sees it.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        a_f32 = T.cast(a, "float32")
        b_f32 = T.cast(b, "float32")
        return T.Cast(a.dtype, T.trunc(a_f32 / b_f32))


# The divisors ``__fdividef`` is defined for.
_FDIVIDEF_MIN, _FDIVIDEF_MAX = 2.0**-126, 2.0**126
# Below this quotient a two-ulp divide lands within one of the whole quotient, and
# the whole quotient is exact in float32.
_FAST_QUOTIENT = float(1 << 22)


def _floored_quotient(num, den, dtype, fast_body, slow, limit=_FAST_QUOTIENT):
    """``fast_body(k, u)`` with ``k = floor(a / b)`` exactly, or ``slow()`` off its range.

    ``u = a * sign(b)`` turns the tests on the residual ``a - t * b`` into signs of
    ``u - t * |b|``. ``t = floor(a / b)`` from a fast divide is off ``k`` by at most
    one: it overshoots where ``u - t * |b|`` is negative and falls short where
    ``u - (t + 1) * |b|`` is not. Each is rounded once by ``fma``, which keeps a sign
    and a zero, so both tests decide right. A quotient of *limit* or more, or a
    divisor ``__fdividef`` does not cover, takes ``slow``; a float16 divisor is
    always in range unless it is infinite.
    """
    zero = T.cast(0.0, "float32")
    one = T.cast(1.0, "float32")
    magnitude = T.abs(den)

    def residue(u, t):
        return T.call_extern("float32", "__fmaf_rn", -t, magnitude, u)

    def floored(u, t):
        over = residue(u, t) < zero
        short = residue(u, t + one) >= zero
        k = t + tirx.Select(over, -one, tirx.Select(short, one, zero))
        return _bound(k, lambda k: fast_body(k, u))

    def pick(quotient):
        in_range = magnitude <= T.cast(_FDIVIDEF_MAX, "float32")
        if str(dtype) != "float16":
            in_range = T.And(in_range, magnitude >= T.cast(_FDIVIDEF_MIN, "float32"))
        fast = T.And(T.abs(quotient) < T.cast(limit, "float32"), in_range)
        u = num * T.copysign(one, den)
        body = _bound(u, lambda u: _bound(T.floor(quotient), lambda t: floored(u, t)))
        return T.if_then_else(fast, body, slow())

    return _bound(_approx_fdiv(num, den), pick)


class RemainderFwdKernel(BinaryKernel):
    """Element-wise remainder: y = a % b, with the sign of b.

    torch's CUDA kernel takes ``fmod`` in fp32, which is exact, and adds b when the
    result and b differ in sign: ``a - floor(a / b) * b`` rounded once. Here that is
    one ``fma`` from the exact floored quotient; ``fmodf`` serves the rest.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The extern calls scalarise the copies, so keep them off their loop."""
        return True

    @staticmethod
    def op_func(a, b):
        zero = T.cast(0.0, "float32")

        def body(num, den):
            def fast(k, u):
                r = T.call_extern("float32", "__fmaf_rn", -k, den, num)
                # A zero remainder is fmod's, which keeps the dividend's sign.
                return _bound(r, lambda r: tirx.Select(r == zero, T.copysign(zero, num), r))

            def slow():
                def signed(mod):
                    flip = T.And(mod != zero, (den < zero) != (mod < zero))
                    return tirx.Select(flip, mod + den, mod)

                return _bound(T.fmod(num, den), signed)

            return T.Cast(a.dtype, _floored_quotient(num, den, a.dtype, fast, slow))

        return _bound(
            T.Cast("float32", a),
            lambda num: _bound(T.Cast("float32", b), lambda den: body(num, den)),
        )


class PowFwdKernel(BinaryKernel):
    """Element-wise power: y = a ** b.

    Computed as ``exp2(b * log2|a|)`` with the sign of a negative base
    restored. Error scales with ``|b * log2(a)|``; ``PowFwdOp`` tabulates it.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

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
# The largest whole number below which every whole number is exact in the dtype.
_EXACT_WHOLE = {"float16": float(1 << 11), "bfloat16": float(1 << 8)}


def _floor_divide(num, den, dtype):
    """torch's ``div_floor_floating`` on two float32 values, returning *dtype*.

    torch divides ``a - fmod(a, b)``, a multiple of b, by b, floors the quotient into
    *dtype* and rounds it up once where that dropped more than a half; a zero
    quotient keeps the sign of ``a / b`` and a zero divisor returns ``a / b`` itself.
    Where the multiple is exact the divide returns ``floor(a / b)``, and the dtype
    rounding applies to that. Past it the result depends on how torch's divide
    rounds, and it is computed as torch does.
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

    exact_whole = T.cast(_EXACT_WHOLE.get(str(dtype), _FAST_QUOTIENT), "float32")

    def whole(k, u):
        # ``floor(a / b)`` has the sign of ``a / b``, which ``u`` carries, zero included.
        exact = T.copysign(k, u)
        # A quotient the dtype holds needs no rounding.
        return T.if_then_else(T.abs(k) <= exact_whole, exact, rounded(k))

    limit = _EXACT_MULTIPLE_QUOTIENT.get(str(dtype), _FAST_QUOTIENT)
    return T.Cast(dtype, _floored_quotient(num, den, dtype, whole, slow, limit))


class FloorDivideFwdKernel(BinaryKernel):
    """Element-wise floor division: y = floor(a / b), as torch defines it.

    Follows torch's CUDA kernel: ``(a - fmod(a, b)) / b`` in fp32, one less when the
    remainder and b differ in sign, rounded to a whole number held in the input
    dtype. ``floor(a / b)`` differs where ``a / b`` rounds up to a whole number
    (``1.0 // 0.1`` is 9) and at an infinite b.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @property
    def stage_broadcast(self) -> bool:
        """The extern calls scalarise the copies, so keep them off their loop."""
        return True

    @staticmethod
    def op_func(a, b):
        return _bound(
            T.Cast("float32", a),
            lambda num: _bound(T.Cast("float32", b), lambda den: _floor_divide(num, den, a.dtype)),
        )


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
