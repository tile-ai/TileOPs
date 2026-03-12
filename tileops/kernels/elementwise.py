"""Elementwise kernel templates and strategy factories.

Three template base classes for 66 elementwise ops:
- UnaryKernel: 1-input → 1-output (relu, sigmoid, abs, ...)
- BinaryKernel: 2-input → 1-output with N-dim broadcast (add, mul, ...)
- FusedGatedKernel: fused gate+activation (silu_and_mul, gelu_and_mul, ...)

Each kernel uses one of three strategies (no shared memory):
  Global → Register → Compute → Register → Global

Strategies:
- direct: 1 element per thread, simplest codegen
- explicit_parallel: N elements per thread via T.Parallel(threads, npt)
- register_copy: fragment load → compute → fragment store (unary only)

Binary register_copy is NOT supported (incompatible with stride-based access).
Boundary checks handled by TileLang LegalizeSafeMemoryAccess.
"""

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = [
    # --- base classes ---
    "BinaryKernel",
    "FusedGatedKernel",
    "UnaryKernel",
    # --- concrete: existing ---
    "AddKernel",
    "ReluKernel",
    "SiluAndMulKernel",
    # --- concrete: unary math (17) ---
    "AbsKernel",
    "CeilKernel",
    "CosKernel",
    "ErfKernel",
    "ExpKernel",
    "Expm1Kernel",
    "FloorKernel",
    "Log1pKernel",
    "LogKernel",
    "NegKernel",
    "ReciprocalKernel",
    "RoundKernel",
    "RsqrtKernel",
    "SignKernel",
    "SinKernel",
    "SqrtKernel",
    "TruncKernel",
    # --- concrete: activations (8) ---
    "GeluKernel",
    "HardsigmoidKernel",
    "HardswishKernel",
    "MishKernel",
    "SeluKernel",
    "SigmoidKernel",
    "SiluKernel",
    "TanhKernel",
    # --- concrete: logical / bitwise (2) ---
    "BitwiseNotKernel",
    "LogicalNotKernel",
    # --- concrete: special predicates (3) ---
    "IsfiniteKernel",
    "IsinfKernel",
    "IsnanKernel",
]

# ---------------------------------------------------------------------------
# Strategy factory: Unary
# ---------------------------------------------------------------------------


def _make_unary_direct(N, dtype, op_func, threads=256):
    """Strategy 1: 1 element per thread."""

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, threads_arg), threads=threads_arg) as bx:
                for i in T.Parallel(threads_arg):
                    idx = bx * threads_arg + i
                    y[idx] = op_func(x[idx])

        return main

    return kernel


def _make_unary_explicit(N, dtype, op_func, threads=256, num_per_thread=8):
    """Strategy 2: N elements per thread via T.Parallel(threads, npt)."""
    block_size = threads * num_per_thread

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    y[idx] = op_func(x[idx])

        return main

    return kernel


def _make_unary_regcopy(N, dtype, op_func, threads=256, num_per_thread=8):
    """Strategy 3: fragment load → compute → fragment store."""
    block_size = threads * num_per_thread

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    y_reg[i * npt_arg + j] = op_func(x_reg[i * npt_arg + j])
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Strategy factory: Binary
# ---------------------------------------------------------------------------


def _compute_broadcast_offsets(flat_idx, ndim, divisors, a_strides, b_strides):
    """Compute a_off and b_off from flat_idx using compile-time unrolled divmod chain.

    All arguments except flat_idx are Python-level constants, so the loop
    unrolls at kernel build time.
    """
    a_off = 0
    b_off = 0
    remaining = flat_idx
    for d in range(ndim - 1):
        coord = remaining // divisors[d]
        remaining = remaining % divisors[d]
        a_off = a_off + coord * a_strides[d]
        b_off = b_off + coord * b_strides[d]
    a_off = a_off + remaining * a_strides[ndim - 1]
    b_off = b_off + remaining * b_strides[ndim - 1]
    return a_off, b_off


def _make_binary_direct(
    N_total, dtype, op_func, coalesced_shape, a_strides, b_strides,
    a_numel, b_numel, output_dtype=None, threads=256,
):
    """Binary direct: 1 element per thread with stride-based broadcast."""
    ndim = len(coalesced_shape)
    out_dtype = output_dtype or dtype
    divisors = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        divisors[i] = divisors[i + 1] * coalesced_shape[i + 1]

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg):
        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N_total, threads_arg), threads=threads_arg) as bx:
                for i in T.Parallel(threads_arg):
                    flat_idx = bx * threads_arg + i
                    a_off, b_off = _compute_broadcast_offsets(
                        flat_idx, ndim, divisors, a_strides, b_strides,
                    )
                    y[flat_idx] = op_func(a[a_off], b[b_off])

        return main

    return kernel


def _make_binary_explicit(
    N_total, dtype, op_func, coalesced_shape, a_strides, b_strides,
    a_numel, b_numel, output_dtype=None, threads=256, num_per_thread=8,
):
    """Binary explicit_parallel: N elements per thread with stride-based broadcast."""
    block_size = threads * num_per_thread
    ndim = len(coalesced_shape)
    out_dtype = output_dtype or dtype
    divisors = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        divisors[i] = divisors[i + 1] * coalesced_shape[i + 1]

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat_idx = (bx * threads_arg + i) * npt_arg + j
                    a_off, b_off = _compute_broadcast_offsets(
                        flat_idx, ndim, divisors, a_strides, b_strides,
                    )
                    y[flat_idx] = op_func(a[a_off], b[b_off])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Strategy factory: FusedGated
# ---------------------------------------------------------------------------


def _make_fused_gated_kernel(M, N, dtype, activation_func, threads=256, num_per_thread=8):
    """FusedGated: x[:, :N] is gate, x[:, N:] is value. y = activation(gate) * value."""
    block_N = threads * num_per_thread

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, block_N), M, threads=threads_arg) as (bx, by):
                for i, j in T.Parallel(threads_arg, npt_arg):
                    col = (bx * threads_arg + i) * npt_arg + j
                    gate = x[by, col]
                    value = x[by, N + col]
                    y[by, col] = activation_func(gate) * value

        return main

    return kernel


# ---------------------------------------------------------------------------
# Template base classes
# ---------------------------------------------------------------------------


class UnaryKernel(Kernel):
    """Template base class for unary elementwise kernels.

    Subclass must override ``op_func`` with a static method implementing
    the pointwise operation (e.g., relu, sigmoid).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype for input/output.
        strategy: One of "direct", "explicit_parallel", "register_copy".
        config: Optional dict with "threads" and "num_per_thread".
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
    # Benchmark (H200, 4M fp16): register_copy 2.40 TB/s > direct 1.04 > explicit 0.86
    DEFAULT_STRATEGY = "register_copy"

    @staticmethod
    def op_func(x):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, N_total, dtype, strategy=None, config=None, tune=False):
        super().__init__()
        self.N_total = N_total
        self.dtype = dtype
        self.strategy = strategy or self.DEFAULT_STRATEGY
        assert self.strategy in self.STRATEGIES, (
            f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
        )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _build_kernel(self, strategy):
        cfg = self.default_config
        if strategy == "direct":
            return _make_unary_direct(
                self.N_total, self.dtype_str, self.op_func, cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_unary_explicit(
                self.N_total, self.dtype_str, self.op_func,
                cfg["threads"], cfg["num_per_thread"],
            )
        elif strategy == "register_copy":
            return _make_unary_regcopy(
                self.N_total, self.dtype_str, self.op_func,
                cfg["threads"], cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        if self.strategy == "direct":
            return self.kernel(cfg["threads"])(x)
        else:
            return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)


class BinaryKernel(Kernel):
    """Template base class for binary elementwise kernels with N-dim broadcast.

    Subclass must override ``op_func`` with a static method implementing
    the pointwise operation (e.g., add, mul).

    Args:
        N_total: Total output elements.
        dtype: Torch dtype for input.
        coalesced_shape: Coalesced broadcast dimensions.
        a_strides: Strides for input a (0 means broadcast).
        b_strides: Strides for input b (0 means broadcast).
        a_numel: Number of elements in a.
        b_numel: Number of elements in b.
        strategy: One of "direct", "explicit_parallel".
        config: Optional dict with "threads" and "num_per_thread".
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel"]
    DEFAULT_STRATEGY = "explicit_parallel"
    OUTPUT_DTYPE = None  # Subclass override for output dtype (e.g., "bool")

    @staticmethod
    def op_func(a, b):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(
        self, N_total, dtype, coalesced_shape, a_strides, b_strides,
        a_numel, b_numel, strategy=None, config=None, tune=False,
    ):
        super().__init__()
        self.N_total = N_total
        self.dtype = dtype
        self.coalesced_shape = coalesced_shape
        self.a_strides = a_strides
        self.b_strides = b_strides
        self.a_numel = a_numel
        self.b_numel = b_numel
        self.strategy = strategy or self.DEFAULT_STRATEGY
        assert self.strategy in self.STRATEGIES, (
            f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
        )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _build_kernel(self, strategy):
        cfg = self.default_config
        if strategy == "direct":
            return _make_binary_direct(
                self.N_total, self.dtype_str, self.op_func,
                self.coalesced_shape, self.a_strides, self.b_strides,
                self.a_numel, self.b_numel,
                output_dtype=self.OUTPUT_DTYPE, threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_binary_explicit(
                self.N_total, self.dtype_str, self.op_func,
                self.coalesced_shape, self.a_strides, self.b_strides,
                self.a_numel, self.b_numel,
                output_dtype=self.OUTPUT_DTYPE,
                threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, a, b):
        cfg = self.config
        if self.strategy == "direct":
            return self.kernel(cfg["threads"])(a, b)
        else:
            return self.kernel(cfg["threads"], cfg["num_per_thread"])(a, b)


class FusedGatedKernel(Kernel):
    """Template base class for fused gated elementwise kernels.

    Input layout: x has shape (M, 2*N) where x[:, :N] is the gate
    and x[:, N:] is the value. Output: y = activation(gate) * value.

    Subclass must override ``activation_func`` with a static method.

    Args:
        M: Number of rows.
        N: Half the column dimension (output width).
        dtype: Torch dtype.
        config: Optional dict with "threads" and "num_per_thread".
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @staticmethod
    def activation_func(x):
        """Activation function. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, M, N, dtype, config=None, tune=False):
        super().__init__()
        self.M = M
        self.N = N
        self.dtype = dtype
        cfg = self.default_config
        self.kernel = _make_fused_gated_kernel(
            M, N, self.dtype_str, self.activation_func,
            cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)


# ---------------------------------------------------------------------------
# Concrete kernel subclasses
# ---------------------------------------------------------------------------


class ReluKernel(UnaryKernel):
    """ReLU: y = max(x, 0)."""

    @staticmethod
    def op_func(x):
        return T.if_then_else(x > T.cast(0, x.dtype), x, T.cast(0, x.dtype))


class AddKernel(BinaryKernel):
    """Element-wise addition: y = a + b."""

    @staticmethod
    def op_func(a, b):
        return a + b


class SiluAndMulKernel(FusedGatedKernel):
    """SiLU-and-Mul: y = silu(gate) * value = (gate * sigmoid(gate)) * value."""

    @staticmethod
    def activation_func(x):
        return x * T.sigmoid(x)


# ---------------------------------------------------------------------------
# Concrete unary kernel subclasses -- math (17)
# ---------------------------------------------------------------------------


class ExpKernel(UnaryKernel):
    """Element-wise exp(x)."""

    @staticmethod
    def op_func(x):
        return T.exp(x)


class LogKernel(UnaryKernel):
    """Element-wise log(x)."""

    @staticmethod
    def op_func(x):
        return T.log(x)


class SqrtKernel(UnaryKernel):
    """Element-wise sqrt(x)."""

    @staticmethod
    def op_func(x):
        return T.sqrt(x)


class RsqrtKernel(UnaryKernel):
    """Element-wise 1/sqrt(x)."""

    @staticmethod
    def op_func(x):
        return T.rsqrt(x)


class AbsKernel(UnaryKernel):
    """Element-wise |x|."""

    @staticmethod
    def op_func(x):
        return T.abs(x)


class NegKernel(UnaryKernel):
    """Element-wise -x."""

    @staticmethod
    def op_func(x):
        return -x


class ReciprocalKernel(UnaryKernel):
    """Element-wise 1/x."""

    @staticmethod
    def op_func(x):
        return T.cast(1.0, "float32") / x


class SignKernel(UnaryKernel):
    """Element-wise sign(x): -1, 0, or +1."""

    @staticmethod
    def op_func(x):
        zero = T.cast(0.0, x.dtype)
        one = T.cast(1.0, x.dtype)
        neg_one = T.cast(-1.0, x.dtype)
        return T.if_then_else(
            x > zero,
            one,
            T.if_then_else(x < zero, neg_one, zero),
        )


class SinKernel(UnaryKernel):
    """Element-wise sin(x)."""

    @staticmethod
    def op_func(x):
        return T.sin(x)


class CosKernel(UnaryKernel):
    """Element-wise cos(x)."""

    @staticmethod
    def op_func(x):
        return T.cos(x)


class FloorKernel(UnaryKernel):
    """Element-wise floor(x).

    Casts to fp32 before calling ``T.floor`` because ``hfloor`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.floor(T.cast(x, "float32"))


class CeilKernel(UnaryKernel):
    """Element-wise ceil(x).

    Casts to fp32 before calling ``T.ceil`` because ``hceil`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.ceil(T.cast(x, "float32"))


class RoundKernel(UnaryKernel):
    """Element-wise round(x) with banker's rounding (round-to-nearest-even).

    Uses ``T.nearbyint`` (maps to ``nearbyintf`` in CUDA) to match
    PyTorch's ``torch.round`` semantics. Casts to fp32 because
    ``hnearbyint`` is not available for ``cutlass::half_t``.
    """

    @staticmethod
    def op_func(x):
        return T.nearbyint(T.cast(x, "float32"))


class TruncKernel(UnaryKernel):
    """Element-wise trunc(x) -- integer part toward zero.

    Casts to fp32 before calling ``T.trunc`` because ``htrunc`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.trunc(T.cast(x, "float32"))


class ErfKernel(UnaryKernel):
    """Element-wise erf(x).

    Casts to fp32 before calling ``T.erf`` because the half-precision
    intrinsic ``herf`` is not a valid CUDA built-in.
    """

    @staticmethod
    def op_func(x):
        return T.erf(T.cast(x, "float32"))


class Log1pKernel(UnaryKernel):
    """Element-wise log(1 + x).

    Uses composite ``log(1 + x)`` because ``T.log1p`` is not lowered
    by the TileLang compiler.
    """

    @staticmethod
    def op_func(x):
        return T.log(T.cast(1.0, "float32") + x)


class Expm1Kernel(UnaryKernel):
    """Element-wise exp(x) - 1."""

    @staticmethod
    def op_func(x):
        return T.exp(x) - T.cast(1.0, "float32")


# ---------------------------------------------------------------------------
# Concrete unary kernel subclasses -- activations (8)
# ---------------------------------------------------------------------------


class GeluKernel(UnaryKernel):
    """Element-wise GELU (tanh approximation).

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    @staticmethod
    def op_func(x):
        sqrt_2_over_pi = T.cast(0.7978845608, "float32")
        coeff = T.cast(0.044715, "float32")
        half = T.cast(0.5, "float32")
        one = T.cast(1.0, "float32")
        return half * x * (one + T.tanh(sqrt_2_over_pi * (x + coeff * x * x * x)))


class SiluKernel(UnaryKernel):
    """Element-wise SiLU (Swish): x * sigmoid(x)."""

    @staticmethod
    def op_func(x):
        return x * T.sigmoid(x)


class SigmoidKernel(UnaryKernel):
    """Element-wise sigmoid(x)."""

    @staticmethod
    def op_func(x):
        return T.sigmoid(x)


class TanhKernel(UnaryKernel):
    """Element-wise tanh(x)."""

    @staticmethod
    def op_func(x):
        return T.tanh(x)


class HardswishKernel(UnaryKernel):
    """Element-wise HardSwish: x * clamp(x + 3, 0, 6) / 6."""

    @staticmethod
    def op_func(x):
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        clamped = T.min(T.max(x + three, zero), six)
        return x * clamped / six


class HardsigmoidKernel(UnaryKernel):
    """Element-wise HardSigmoid: clamp(x + 3, 0, 6) / 6."""

    @staticmethod
    def op_func(x):
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        return T.min(T.max(x + three, zero), six) / six


class MishKernel(UnaryKernel):
    """Element-wise Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))."""

    @staticmethod
    def op_func(x):
        one = T.cast(1.0, "float32")
        return x * T.tanh(T.log(one + T.exp(x)))


class SeluKernel(UnaryKernel):
    """Element-wise SELU: scale * (max(0,x) + min(0, alpha*(exp(x)-1))).

    alpha = 1.6732632423543772, scale = 1.0507009873554805
    """

    @staticmethod
    def op_func(x):
        alpha = T.cast(1.6732632423543772, "float32")
        scale = T.cast(1.0507009873554805, "float32")
        one = T.cast(1.0, "float32")
        zero = T.cast(0.0, "float32")
        return scale * T.if_then_else(x > zero, x, alpha * (T.exp(x) - one))


# ---------------------------------------------------------------------------
# Concrete unary kernel subclasses -- logical / bitwise (2)
# ---------------------------------------------------------------------------


class LogicalNotKernel(UnaryKernel):
    """Element-wise logical NOT: returns 1.0 where x == 0, else 0.0."""

    @staticmethod
    def op_func(x):
        zero = T.cast(0.0, x.dtype)
        one = T.cast(1.0, x.dtype)
        return T.if_then_else(x == zero, one, zero)


class BitwiseNotKernel(UnaryKernel):
    """Element-wise bitwise NOT (~x) for integer types.

    Uses XOR with ``-1`` (all-ones) because ``T.bitwise_not`` fails on
    vectorized ``int4`` CUDA types.
    """

    @staticmethod
    def op_func(x):
        return T.bitwise_xor(x, T.cast(-1, "int32"))


# ---------------------------------------------------------------------------
# Concrete unary kernel subclasses -- special predicates (3)
# ---------------------------------------------------------------------------


class IsnanKernel(UnaryKernel):
    """Element-wise isnan: returns 1.0 for NaN, 0.0 otherwise."""

    @staticmethod
    def op_func(x):
        zero = T.cast(0.0, x.dtype)
        one = T.cast(1.0, x.dtype)
        return T.if_then_else(T.isnan(x), one, zero)


class IsinfKernel(UnaryKernel):
    """Element-wise isinf: returns 1.0 for +/-inf, 0.0 otherwise."""

    @staticmethod
    def op_func(x):
        zero = T.cast(0.0, x.dtype)
        one = T.cast(1.0, x.dtype)
        return T.if_then_else(T.isinf(x), one, zero)


class IsfiniteKernel(UnaryKernel):
    """Element-wise isfinite: returns 1.0 for finite values, 0.0 otherwise."""

    @staticmethod
    def op_func(x):
        zero = T.cast(0.0, x.dtype)
        one = T.cast(1.0, x.dtype)
        return T.if_then_else(T.isfinite(x), one, zero)
