"""Activation kernels, parameter-free and parametric, plus the fused gated forms."""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import GELU_TANH_COEFF, LOG2E, SQRT_2_OVER_PI
from tileops._kernels.elementwise._base import (
    FloatUnaryKernel,
    FusedGatedKernel,
    MultiInputElementwiseKernel,
    ScalarParamUnaryKernel,
)
from tileops._kernels.elementwise._division import (
    approx_fdiv,
    quartered_fdiv,
    refined_reciprocal,
)
from tileops._kernels.elementwise._dtype import FLOAT_DTYPES, log_for_output_precision
from tileops._kernels.elementwise._erf import gelu
from tileops._kernels.elementwise._nan import nan_max, nan_min
from tileops._kernels.elementwise.call_spec import (
    BoundedUnaryFwdInterface,
    BoundsCall,
    ELUCall,
    ELUFwdInterface,
    GELUAndMulCall,
    GELUCall,
    LeakyReLUCall,
    LeakyReLUFwdInterface,
    SoftplusCall,
    SoftplusFwdInterface,
    UnaryElementwiseFwdInterface,
)
from tileops._kernels.kernel_base import Entry

__all__ = [
    "ELUFwdKernel",
    "GELUAndMulFwdKernel",
    "GELUFwdKernel",
    "GELUTanhAndMulFwdKernel",
    "GELUTanhFwdKernel",
    "HardsigmoidFwdKernel",
    "HardswishFwdKernel",
    "HardtanhFwdKernel",
    "LeakyReLUFwdKernel",
    "MishFwdKernel",
    "ReLUFwdKernel",
    "SELUFwdKernel",
    "SiLUAndMulFwdKernel",
    "SiLUFwdKernel",
    "SigmoidFwdKernel",
    "SoftplusFwdKernel",
    "TanhFwdKernel",
]


class ReLUFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """ReLU: y = max(x, 0)."""

    @staticmethod
    def op_func(x):
        return T.if_then_else(x > T.cast(0, x.dtype), x, T.cast(0, x.dtype))


class GELUFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise GELU using the standard erf formulation."""

    BYTES_PER_THREAD = 32
    # The erf body costs more than its access, so a row the shrink reaches keeps two
    # elements a thread instead of four: 1 x 14336 bfloat16 runs 56 blocks instead of
    # 28, as torch-compile does, at 1.344 us against 1.440 us. One element a thread
    # measured no faster in 16 bits and slower in float32.
    MIN_NUM_PER_THREAD = 2

    @classmethod
    def refusal(cls, call: GELUCall) -> "str | None":
        if call.approximate != "none":
            return f"evaluates the erf form, not approximate={call.approximate!r}"
        return super().refusal(call)

    @staticmethod
    def op_func(x):
        return gelu(x, x.dtype)


class GELUTanhFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise GELU using the tanh approximation.

    Computes ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``,
    matching ``torch.nn.functional.gelu(x, approximate='tanh')``.
    """

    @classmethod
    def refusal(cls, call: GELUCall) -> "str | None":
        if call.approximate != "tanh":
            return f"evaluates the tanh approximation, not approximate={call.approximate!r}"
        return super().refusal(call)

    @staticmethod
    def op_func(x):
        sqrt_2_over_pi = T.cast(SQRT_2_OVER_PI, "float32")
        coeff = T.cast(GELU_TANH_COEFF, "float32")
        half = T.cast(0.5, "float32")
        one = T.cast(1.0, "float32")
        x_f32 = T.cast(x, "float32")
        inner = sqrt_2_over_pi * (x_f32 + coeff * x_f32 * x_f32 * x_f32)
        return half * x_f32 * (one + T.tanh(inner))


class SiLUFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise SiLU (Swish): x * sigmoid(x)."""

    CSRC_HEADERS = ("approx_math.h",)

    def __init__(self, N_total, dtype, config=None):
        # The float32 body's IEEE divide costs more than its loads; more blocks hide it.
        if dtype == torch.float32:
            self.MIN_NUM_PER_THREAD = 2
        super().__init__(N_total, dtype, config=config)

    @staticmethod
    def op_func(x):
        """x / (1 + exp2(-x * log2 e)), in fp32.

        The exponential runs in fp32 for both speed and fp32-reference error, and
        as exp2, which lowers to one MUFU.EX2 against expf's multi-op sequence.

        16-bit results skip the IEEE divide and ``exp2f``'s subnormal guard: a flushed
        subnormal vanishes when added to 1. float16 uses ``approx_fdiv``, whose zero
        past a divisor of ``2**126`` is the float16 result there; bfloat16 uses
        ``quartered_fdiv``.
        """
        one = T.cast(1.0, "float32")
        wide = T.cast(x, "float32")
        exponent = -wide * T.cast(LOG2E, "float32")
        if str(x.dtype) == "float32":
            return wide / (one + T.exp2(exponent))
        den = one + T.call_extern("float32", "tileops::approx_exp2", exponent)
        if str(x.dtype) == "bfloat16":
            return quartered_fdiv(wide, den)
        return approx_fdiv(wide, den)


class SigmoidFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise sigmoid(x)."""

    CSRC_HEADERS = ("approx_math.h",)

    def __init__(self, N_total, dtype, config=None):
        # A 16-bit thread holds two 16-byte vectors. float32 holds one once its input
        # exceeds L2. Why: a smaller input stays in L2 between calls, where two run faster.
        l2_bytes = torch.cuda.get_device_properties(None).L2_cache_size
        one = dtype == torch.float32 and N_total * dtype.itemsize > l2_bytes
        self.BYTES_PER_THREAD = 16 if one else 32
        super().__init__(N_total, dtype, config=config)

    @staticmethod
    def op_func(x):
        """1 / (1 + exp2(-x * log2 e)), in fp32.

        The exponential runs in fp32 for both speed and fp32-reference error, and
        as exp2, which lowers to one MUFU.EX2 against expf's multi-op sequence.
        bfloat16 and float32 skip ``exp2f``'s subnormal guard, since a flushed
        subnormal vanishes when added to 1, and divide with ``quartered_fdiv``; it
        rounds to torch's value at every bfloat16 input. float16 takes
        ``refined_reciprocal``, which rounds to torch's value at every float16 input;
        ``approx_fdiv``'s two ulp alone move two of the 65536 float16 results by one ulp.
        """
        one = T.cast(1.0, "float32")
        exponent = -T.cast(x, "float32") * T.cast(LOG2E, "float32")
        if str(x.dtype) == "float16":
            return refined_reciprocal(one + T.exp2(exponent))
        return quartered_fdiv(one, one + T.call_extern("float32", "tileops::approx_exp2", exponent))


class TanhFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise tanh(x)."""

    BYTES_PER_THREAD = 32

    @staticmethod
    def op_func(x):
        return T.tanh(T.cast(x, "float32"))


class HardswishFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
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


class HardsigmoidFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise HardSigmoid: clamp(x + 3, 0, 6) * (1 / 6).

    Scaling by the reciprocal is what ``torch.nn.functional.hardsigmoid`` does, so
    finite inputs come out bit-identical to it; dividing by 6 lowers to
    ``div.rn.f32``. A NaN input gives NaN, as in torch.
    """

    @staticmethod
    def op_func(x):
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        one_sixth = T.cast(1.0 / 6.0, "float32")
        return nan_min(nan_max(x + three, zero), six) * one_sixth


class MishFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
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


class SELUFwdKernel(FloatUnaryKernel, UnaryElementwiseFwdInterface):
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


class LeakyReLUFwdKernel(ScalarParamUnaryKernel, LeakyReLUFwdInterface):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x."""

    @classmethod
    def entry_for(cls, call: LeakyReLUCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype, call.negative_slope)

    def __init__(self, N_total, dtype, negative_slope=0.01, config=None):
        self.negative_slope = negative_slope
        super().__init__(N_total, dtype, config=config)

    def _param_key(self):
        return f"negative_slope={self.negative_slope!r}"

    def _make_op_func(self):
        negative_slope = self.negative_slope

        # The slope multiplies in fp32: in a narrow dtype a large slope would overflow
        # before the product does, and a small one would lose precision.
        def op_func(x):
            zero = T.cast(0, "float32")
            slope = T.cast(negative_slope, "float32")
            wide = T.cast(x, "float32")
            return T.if_then_else(wide > zero, x, T.Cast(x.dtype, slope * wide))

        return op_func


class ELUFwdKernel(ScalarParamUnaryKernel, ELUFwdInterface):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1)."""

    @classmethod
    def entry_for(cls, call: ELUCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype, call.alpha)

    def __init__(self, N_total, dtype, alpha=1.0, config=None):
        self.alpha = alpha
        super().__init__(N_total, dtype, config=config)

    def _param_key(self):
        return f"alpha={self.alpha!r}"

    def _make_op_func(self):
        alpha = self.alpha

        def op_func(x):
            zero = T.cast(0, "float32")
            one = T.cast(1.0, "float32")
            a = T.cast(alpha, "float32")
            wide = T.cast(x, "float32")
            return T.if_then_else(wide > zero, x, T.Cast(x.dtype, a * (T.exp(wide) - one)))

        return op_func


class HardtanhFwdKernel(ScalarParamUnaryKernel, BoundedUnaryFwdInterface):
    """Hardtanh: y = clamp(x, min_val, max_val)."""

    @classmethod
    def entry_for(cls, call: BoundsCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype, call.min_val, call.max_val)

    def __init__(self, N_total, dtype, min_val=-1.0, max_val=1.0, config=None):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(N_total, dtype, config=config)

    def _param_key(self):
        return f"min_val={self.min_val!r}|max_val={self.max_val!r}"

    def _make_op_func(self):
        min_val, max_val = self.min_val, self.max_val

        def op_func(x):
            lo = T.cast(min_val, "float32")
            hi = T.cast(max_val, "float32")
            wide = T.cast(x, "float32")
            return T.Cast(x.dtype, nan_min(nan_max(wide, lo), hi))

        return op_func


@target_cache(maxsize=32)
def _make_softplus_kernel(N, dtype, beta, threshold, threads=256, npt=8):
    """Build softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x.

    Binding ``softplus`` as a statement lifts the exponential and the logarithm
    out of the guard. float16 rows are 1.2% slower with them inside it.
    """

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
                    softplus = log_for_output_precision(val, one + T.exp(scaled)) / b
                    y_reg[i * npt_arg + j] = T.if_then_else(
                        scaled > t,
                        val,
                        T.Cast(val.dtype, softplus),
                    )
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class SoftplusFwdKernel(MultiInputElementwiseKernel, SoftplusFwdInterface):
    """Softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x."""

    INPUTS = (("x", "tile"),)

    @classmethod
    def entry_for(cls, call: SoftplusCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype, call.beta, call.threshold)

    def __init__(self, N_total, dtype, beta=1.0, threshold=20.0, config=None):
        self.beta = beta
        self.threshold = threshold
        super().__init__(N_total, dtype, config=config)

    @staticmethod
    def _builder_fn():
        return _make_softplus_kernel

    def _builder_args(self):
        return (self.beta, self.threshold)

    def forward(self, x):
        return self._run(x=x)


class SiLUAndMulFwdKernel(FusedGatedKernel):
    """SiLU-and-Mul: y = silu(gate) * value = (gate * sigmoid(gate)) * value."""

    SUPPORTED_DTYPES = FLOAT_DTYPES

    @staticmethod
    def activation_func(x):
        """gate / (1 + exp2(-gate * log2 e)), in fp32.

        exp2 lowers to one MUFU.EX2 against expf's multi-op sequence. A 16-bit gate
        skips ``exp2f``'s subnormal guard, since a flushed subnormal vanishes when added
        to 1, and divides with ``quartered_fdiv``: the product with an infinite value
        needs the activation's sign past a divisor of ``2**126``, where ``approx_fdiv``
        returns zero.
        """
        g = T.Cast("float32", x)
        one = T.cast(1.0, "float32")
        exponent = -g * T.cast(LOG2E, "float32")
        if str(x.dtype) == "float32":
            return g / (one + T.exp2(exponent))
        return quartered_fdiv(g, one + T.call_extern("float32", "tileops::approx_exp2", exponent))


class GELUAndMulFwdKernel(FusedGatedKernel):
    """GELU-and-Mul: y = gelu(gate) * value, in float32 and rounded once.

    Uses exact GELU, gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2))), evaluated by the same
    ``gelu`` as ``GELUFwdKernel``.
    """

    SUPPORTED_DTYPES = FLOAT_DTYPES

    @classmethod
    def refusal(cls, call: GELUAndMulCall) -> "str | None":
        if call.approximate != "none":
            return f"evaluates the erf form, not approximate={call.approximate!r}"
        return super().refusal(call)

    @staticmethod
    def activation_func(x):
        return gelu(x, x.dtype)


class GELUTanhAndMulFwdKernel(FusedGatedKernel):
    """GELU-Tanh-and-Mul: y = gelu_tanh(gate) * value.

    Uses tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """

    SUPPORTED_DTYPES = FLOAT_DTYPES

    @classmethod
    def refusal(cls, call: GELUAndMulCall) -> "str | None":
        if call.approximate != "tanh":
            return f"evaluates the tanh approximation, not approximate={call.approximate!r}"
        return super().refusal(call)

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
