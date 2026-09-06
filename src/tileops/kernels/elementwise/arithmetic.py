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


class RemainderFwdKernel(BinaryKernel):
    """Element-wise remainder: y = a - floor(a / b) * b.

    Matches PyTorch remainder semantics for floating-point inputs.
    Uses floor-based formula since T.FloorMod requires integer types.

    Division and floor are computed in fp32 to avoid two sources of error:
    (1) ``hfloor`` is not available for ``cutlass::half_t`` in CUDA, and
    (2) fp16 division rounds the quotient before floor sees it (e.g.
    2.999... rounds to 3.0 in fp16).  The floored quotient is then cast
    back to native dtype so the final ``a - floored * b`` matches PyTorch
    semantics for the multiply-subtract step.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        a_f32 = T.cast(a, "float32")
        b_f32 = T.cast(b, "float32")
        floored = T.Cast(a.dtype, T.floor(a_f32 / b_f32))
        return a - floored * b


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


class FloorDivideFwdKernel(BinaryKernel):
    """Element-wise floor division: y = floor(a / b).

    Division and floor are computed in fp32 to avoid two sources of error:
    (1) ``hfloor`` is not available for ``cutlass::half_t`` in CUDA, and
    (2) fp16 division rounds the quotient before floor sees it (e.g.
    2.999... rounds to 3.0 in fp16, giving floor=3 instead of 2).
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        a_f32 = T.cast(a, "float32")
        b_f32 = T.cast(b, "float32")
        return T.Cast(a.dtype, T.floor(a_f32 / b_f32))


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


def _is_float_dtype_str(dtype_str: str) -> bool:
    """Return True for floating-point TileLang dtype strings.

    TileLang IR exposes operand dtypes only as strings (``"float16"``,
    ``"bfloat16"``, ``"float32"``), so prefix matching is the established
    convention for float detection inside ``op_func`` kernel bodies. All
    TileLang float dtype names start with ``"float"`` or ``"bfloat"``;
    integer / bool dtype names (``"int*"``, ``"uint*"``, ``"bool"``) do not.
    """
    return dtype_str.startswith(("float", "bfloat"))


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
    family. A ``T.max_nan`` upstream would replace this whole helper, and the
    two call sites would not change.
    """
    return tirx.call_intrin(a.dtype, tirx.op.Op.get(name), a, b)


def _nan_max(a, b):
    return _nan_intrin("tl.max_nan", a, b)


def _nan_min(a, b):
    return _nan_intrin("tl.min_nan", a, b)


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
        if not _is_float_dtype_str(str(a.dtype)):
            # Integer / bool: no NaN representation, T.max is sufficient.
            return T.max(a, b)
        if str(a.dtype) in ("float16", "bfloat16"):
            return _nan_max(a, b)
        return _propagate_nan(a, b, T.max(a, b))


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
        if not _is_float_dtype_str(str(a.dtype)):
            return T.min(a, b)
        if str(a.dtype) in ("float16", "bfloat16"):
            return _nan_min(a, b)
        return _propagate_nan(a, b, T.min(a, b))


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
