"""Activation kernels, parameter-free and parametric, plus the fused gated forms."""

import functools

import tilelang
import tilelang.language as T

from tileops.kernels.constants import GELU_TANH_COEFF, INV_SQRT2, LOG2E, SQRT_2_OVER_PI

from ._base import (
    _FLOAT_DTYPES,
    FloatUnaryKernel,
    FusedGatedKernel,
    ParametricUnaryKernel,
)
from ._dtype import _fp8_accum_dtype_str, log_for_output_precision
from ._erf import erf

__all__ = [
    "EluFwdKernel",
    "GeluAndMulFwdKernel",
    "GeluFwdKernel",
    "GeluTanhAndMulFwdKernel",
    "GeluTanhFwdKernel",
    "HardsigmoidFwdKernel",
    "HardswishFwdKernel",
    "HardtanhFwdKernel",
    "LeakyReluFwdKernel",
    "MishFwdKernel",
    "ReluFwdKernel",
    "SeluFwdKernel",
    "SigmoidFwdKernel",
    "SiluAndMulFwdKernel",
    "SiluFwdKernel",
    "SoftplusFwdKernel",
    "TanhFwdKernel",
]


class ReluFwdKernel(FloatUnaryKernel):
    """ReLU: y = max(x, 0)."""

    @staticmethod
    def op_func(x):
        return T.if_then_else(x > T.cast(0, x.dtype), x, T.cast(0, x.dtype))


class GeluFwdKernel(FloatUnaryKernel):
    """Element-wise GELU using the standard erf formulation."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        inv_sqrt_2 = T.cast(INV_SQRT2, "float32")
        half = T.cast(0.5, "float32")
        one = T.cast(1.0, "float32")
        wide = T.cast(x, "float32")
        return half * wide * (one + erf(wide * inv_sqrt_2, x.dtype))


class GeluTanhFwdKernel(FloatUnaryKernel):
    """Element-wise GELU using the tanh approximation.

    Computes ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``,
    matching ``torch.nn.functional.gelu(x, approximate='tanh')``.
    """

    @staticmethod
    def op_func(x):
        sqrt_2_over_pi = T.cast(SQRT_2_OVER_PI, "float32")
        coeff = T.cast(GELU_TANH_COEFF, "float32")
        half = T.cast(0.5, "float32")
        one = T.cast(1.0, "float32")
        x_f32 = T.cast(x, "float32")
        inner = sqrt_2_over_pi * (x_f32 + coeff * x_f32 * x_f32 * x_f32)
        return half * x_f32 * (one + T.tanh(inner))


class SiluFwdKernel(FloatUnaryKernel):
    """Element-wise SiLU (Swish): x * sigmoid(x)."""

    @staticmethod
    def op_func(x):
        """x / (1 + exp2(-x * log2 e)), in fp32.

        The exponential runs in fp32 for both speed and fp32-reference error, and
        as exp2, which lowers to one MUFU.EX2 against expf's multi-op sequence.
        ``SiluAndMulFwdKernel`` computes the same form.
        """
        one = T.cast(1.0, "float32")
        wide = T.cast(x, "float32")
        return wide / (one + T.exp2(-wide * T.cast(LOG2E, "float32")))


class SigmoidFwdKernel(FloatUnaryKernel):
    """Element-wise sigmoid(x)."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        """1 / (1 + exp2(-x * log2 e)), in fp32.

        The exponential runs in fp32 for both speed and fp32-reference error, and
        as exp2, which lowers to one MUFU.EX2 against expf's multi-op sequence.
        """
        one = T.cast(1.0, "float32")
        wide = T.cast(x, "float32")
        return one / (one + T.exp2(-wide * T.cast(LOG2E, "float32")))


class TanhFwdKernel(FloatUnaryKernel):
    """Element-wise tanh(x)."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        return T.tanh(T.cast(x, "float32"))


class HardswishFwdKernel(FloatUnaryKernel):
    """Element-wise HardSwish: x * clamp(x + 3, 0, 6) * (1 / 6).

    Scaling by the reciprocal is what ``torch.nn.functional.hardswish`` does, so
    finite inputs come out bit-identical to it; dividing by 6 lowers to
    ``div.rn.f32``.
    """

    @staticmethod
    def op_func(x):
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        one_sixth = T.cast(1.0 / 6.0, "float32")
        clamped = T.min(T.max(x + three, zero), six)
        return x * clamped * one_sixth


class HardsigmoidFwdKernel(FloatUnaryKernel):
    """Element-wise HardSigmoid: clamp(x + 3, 0, 6) * (1 / 6).

    Scaling by the reciprocal is what ``torch.nn.functional.hardsigmoid`` does, so
    finite inputs come out bit-identical to it; dividing by 6 lowers to
    ``div.rn.f32``. A NaN input reads back as 0, since the clamp lowers to
    ``fminf``/``fmaxf``, which return their non-NaN operand.
    """

    @staticmethod
    def op_func(x):
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        one_sixth = T.cast(1.0 / 6.0, "float32")
        return T.min(T.max(x + three, zero), six) * one_sixth


class MishFwdKernel(FloatUnaryKernel):
    """Element-wise Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))."""

    # Where Mish's tanh factor reaches 1 in fp32, and below where ``e**2`` overflows.
    _SATURATION: float = 20.0

    @staticmethod
    def op_func(x):
        """One transcendental where the definition spells out three.

        With ``e = exp(x)``, ``tanh(log(1 + e))`` is exactly ``(e^2 + 2e)/(e^2 + 2e + 2)``:
        this avoids extra transcendentals and keeps the small values ``log(1 + e)``
        rounds away. Past ``_SATURATION`` the ratio is 1 to every bit fp32 carries,
        so clamping the exponent's argument there returns ``x`` on its own and keeps
        ``e^2`` finite. Clamping rather than selecting on it also keeps the element
        loop vectorised, which ``T.if_then_else`` does not.
        """
        two = T.cast(2.0, "float32")
        wide = T.cast(x, "float32")
        e = T.exp(T.min(wide, T.cast(MishFwdKernel._SATURATION, "float32")))
        saturated = e * e + two * e
        return wide * saturated / (saturated + two)


class SeluFwdKernel(FloatUnaryKernel):
    """Element-wise SELU: scale * (max(0,x) + min(0, alpha*(exp(x)-1))).

    alpha = 1.6732632423543772, scale = 1.0507009873554805
    """

    @staticmethod
    def op_func(x):
        alpha = T.cast(1.6732632423543772, "float32")
        scale = T.cast(1.0507009873554805, "float32")
        one = T.cast(1.0, "float32")
        zero = T.cast(0.0, "float32")
        x32 = T.cast(x, "float32")
        return scale * T.if_then_else(x32 > zero, x32, alpha * (T.exp(x32) - one))


@functools.lru_cache(maxsize=32)
def _make_leaky_relu_kernel(
    N, dtype, negative_slope, output_dtype=None, is_fp8=False, threads=256, npt=8
):
    """Build leaky_relu kernel: y = x if x > 0 else negative_slope * x.

    For non-fp8 dtypes, uses register_copy strategy: fragment load -> compute
    -> fragment store for coalesced memory access.

    For fp8 dtypes, uses explicit_parallel with fp16 accumulation (register_copy
    is unreliable for 8-bit fragments).
    """
    out_dtype = output_dtype or dtype

    if is_fp8:
        accum = _fp8_accum_dtype_str()

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            val = x[idx]
                            v = T.cast(val, accum)
                            zero = T.cast(0, accum)
                            slope = T.cast(negative_slope, accum)
                            result = T.if_then_else(v > zero, v, slope * v)
                            y[idx] = T.Cast(out_dtype, result)

            return main
    else:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        val = x_reg[i * npt_arg + j]
                        zero = T.cast(0, val.dtype)
                        slope = T.cast(negative_slope, val.dtype)
                        y_reg[i * npt_arg + j] = T.if_then_else(val > zero, val, slope * val)
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

    return kernel


class LeakyReluFwdKernel(ParametricUnaryKernel):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x."""

    def __init__(self, N_total, dtype, negative_slope=0.01, config=None, tune=False):
        self.negative_slope = negative_slope
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_leaky_relu_kernel

    def _builder_args(self):
        return (self.negative_slope,)


@functools.lru_cache(maxsize=32)
def _make_elu_kernel(N, dtype, alpha, output_dtype=None, is_fp8=False, threads=256, npt=8):
    """Build ELU kernel: y = x if x > 0 else alpha * (exp(x) - 1)."""
    out_dtype = output_dtype or dtype

    if is_fp8:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            val = x[idx]
                            zero = T.cast(0, "float32")
                            a = T.cast(alpha, "float32")
                            one = T.cast(1.0, "float32")
                            v32 = T.cast(val, "float32")
                            y[idx] = T.if_then_else(
                                v32 > zero,
                                T.Cast(out_dtype, v32),
                                T.Cast(out_dtype, a * (T.exp(v32) - one)),
                            )

            return main
    else:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        val = x_reg[i * npt_arg + j]
                        zero = T.cast(0, "float32")
                        a = T.cast(alpha, "float32")
                        one = T.cast(1.0, "float32")
                        v32 = T.cast(val, "float32")
                        y_reg[i * npt_arg + j] = T.if_then_else(
                            v32 > zero,
                            val,
                            T.Cast(val.dtype, a * (T.exp(v32) - one)),
                        )
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

    return kernel


class EluFwdKernel(ParametricUnaryKernel):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1)."""

    def __init__(self, N_total, dtype, alpha=1.0, config=None, tune=False):
        self.alpha = alpha
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_elu_kernel

    def _builder_args(self):
        return (self.alpha,)


@functools.lru_cache(maxsize=32)
def _make_hardtanh_kernel(
    N, dtype, min_val, max_val, output_dtype=None, is_fp8=False, threads=256, npt=8
):
    """Build hardtanh kernel: y = clamp(x, min_val, max_val)."""
    out_dtype = output_dtype or dtype

    if is_fp8:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            val = x[idx]
                            lo = T.cast(min_val, "float32")
                            hi = T.cast(max_val, "float32")
                            v32 = T.cast(val, "float32")
                            y[idx] = T.Cast(out_dtype, T.min(T.max(v32, lo), hi))

            return main
    else:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        val = x_reg[i * npt_arg + j]
                        lo = T.cast(min_val, "float32")
                        hi = T.cast(max_val, "float32")
                        v32 = T.cast(val, "float32")
                        y_reg[i * npt_arg + j] = T.Cast(val.dtype, T.min(T.max(v32, lo), hi))
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

    return kernel


class HardtanhFwdKernel(ParametricUnaryKernel):
    """Hardtanh: y = clamp(x, min_val, max_val)."""

    def __init__(self, N_total, dtype, min_val=-1.0, max_val=1.0, config=None, tune=False):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_hardtanh_kernel

    def _builder_args(self):
        return (self.min_val, self.max_val)


@functools.lru_cache(maxsize=32)
def _make_softplus_kernel(
    N, dtype, beta, threshold, output_dtype=None, is_fp8=False, threads=256, npt=8
):
    """Build softplus kernel: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x."""
    out_dtype = output_dtype or dtype

    if is_fp8:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            val = x[idx]
                            v32 = T.cast(val, "float32")
                            b = T.cast(beta, "float32")
                            t = T.cast(threshold, "float32")
                            one = T.cast(1.0, "float32")
                            scaled = v32 * b
                            sp = log_for_output_precision(val, one + T.exp(scaled)) / b
                            y[idx] = T.if_then_else(
                                scaled > t, T.Cast(out_dtype, v32), T.Cast(out_dtype, sp)
                            )

            return main
    else:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        val = x_reg[i * npt_arg + j]
                        v32 = T.cast(val, "float32")
                        b = T.cast(beta, "float32")
                        t = T.cast(threshold, "float32")
                        one = T.cast(1.0, "float32")
                        scaled = v32 * b
                        sp = log_for_output_precision(val, one + T.exp(scaled)) / b
                        y_reg[i * npt_arg + j] = T.if_then_else(
                            scaled > t,
                            val,
                            T.Cast(val.dtype, sp),
                        )
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

    return kernel


class SoftplusFwdKernel(ParametricUnaryKernel):
    """Softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x."""

    def __init__(self, N_total, dtype, beta=1.0, threshold=20.0, config=None, tune=False):
        self.beta = beta
        self.threshold = threshold
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_softplus_kernel

    def _builder_args(self):
        return (self.beta, self.threshold)


class SiluAndMulFwdKernel(FusedGatedKernel):
    """SiLU-and-Mul: y = silu(gate) * value = (gate * sigmoid(gate)) * value."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def activation_func(x):
        # exp2 form (fp32): exp2 lowers to one MUFU.EX2 vs expf's multi-op sequence.
        g = T.Cast("float32", x)
        one = T.cast(1.0, "float32")
        log2e = T.cast(LOG2E, "float32")
        return g / (one + T.exp2(-g * log2e))


class GeluAndMulFwdKernel(FusedGatedKernel):
    """GELU-and-Mul: y = gelu(gate) * value.

    Uses exact GELU: gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def activation_func(x):
        inv_sqrt2 = T.cast(INV_SQRT2, "float32")  # 1/sqrt(2)
        half = T.cast(0.5, x.dtype)
        one = T.cast(1.0, x.dtype)
        x_f32 = T.Cast("float32", x)
        erf_val = T.Cast(x.dtype, erf(x_f32 * inv_sqrt2, x.dtype))
        return x * half * (one + erf_val)


class GeluTanhAndMulFwdKernel(FusedGatedKernel):
    """GELU-Tanh-and-Mul: y = gelu_tanh(gate) * value.

    Uses tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def activation_func(x):
        sqrt_2_over_pi = T.cast(SQRT_2_OVER_PI, "float32")  # sqrt(2/pi)
        coeff = T.cast(GELU_TANH_COEFF, "float32")  # GELU tanh approx coefficient
        half = T.cast(0.5, x.dtype)
        one = T.cast(1.0, x.dtype)
        x_f32 = T.Cast("float32", x)
        inner = sqrt_2_over_pi * (x_f32 + coeff * x_f32 * x_f32 * x_f32)
        tanh_val = T.Cast(x.dtype, T.tanh(inner))
        return half * x * (one + tanh_val)
