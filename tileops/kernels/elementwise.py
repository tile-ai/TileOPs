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
    # --- unary: existing ---
    "ReluKernel",
    # --- unary: math (17) ---
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
    # --- unary: activations (8) ---
    "GeluKernel",
    "HardsigmoidKernel",
    "HardswishKernel",
    "MishKernel",
    "SeluKernel",
    "SigmoidKernel",
    "SiluKernel",
    "TanhKernel",
    # --- unary: logical / bitwise (2) ---
    "BitwiseNotKernel",
    "LogicalNotKernel",
    # --- unary: special predicates (3) ---
    "IsfiniteKernel",
    "IsinfKernel",
    "IsnanKernel",
    # --- binary arithmetic ---
    "AddKernel",
    "SubKernel",
    "MulKernel",
    "DivKernel",
    "RemainderKernel",
    "PowKernel",
    "FloorDivideKernel",
    "LerpKernel",
    "MaximumKernel",
    "MinimumKernel",
    # --- comparison (OUTPUT_DTYPE = "int8", cast to bool by Op layer) ---
    "EqKernel",
    "NeKernel",
    "GtKernel",
    "LtKernel",
    "GeKernel",
    "LeKernel",
    # --- logical (OUTPUT_DTYPE = "int8", cast to bool by Op layer) ---
    "LogicalAndKernel",
    "LogicalOrKernel",
    # --- bitwise ---
    "BitwiseAndKernel",
    "BitwiseOrKernel",
    "BitwiseXorKernel",
    # --- fused gated ---
    "SiluAndMulKernel",
    "GeluAndMulKernel",
    "GeluTanhAndMulKernel",
]

_BITWISE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

_LOGICAL_DTYPES = _BITWISE_DTYPES + _FLOAT_DTYPES

# ---------------------------------------------------------------------------
# Strategy factory: Unary
# ---------------------------------------------------------------------------


def _make_unary_direct(N, dtype, op_func, output_dtype=None, threads=256):
    """Strategy 1: 1 element per thread."""
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
            with T.Kernel(T.ceildiv(N, threads_arg), threads=threads_arg) as bx:
                for i in T.Parallel(threads_arg):
                    idx = bx * threads_arg + i
                    y[idx] = op_func(x[idx])

        return main

    return kernel


def _make_unary_explicit(N, dtype, op_func, output_dtype=None, threads=256, num_per_thread=8):
    """Strategy 2: N elements per thread via T.Parallel(threads, npt)."""
    block_size = threads * num_per_thread
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    y[idx] = op_func(x[idx])

        return main

    return kernel


def _make_unary_regcopy(N, dtype, op_func, output_dtype=None, threads=256, num_per_thread=8):
    """Strategy 3: fragment load → compute → fragment store."""
    block_size = threads * num_per_thread
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), out_dtype)
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
        dtype: Torch dtype for input.
        strategy: One of "direct", "explicit_parallel", "register_copy".
        config: Optional dict with "threads" and "num_per_thread".
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
    # Benchmark (H200, 4M fp16): register_copy 2.40 TB/s > direct 1.04 > explicit 0.86
    DEFAULT_STRATEGY = "register_copy"
    OUTPUT_DTYPE = None
    SUPPORTED_DTYPES = None

    @staticmethod
    def op_func(x):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, N_total, dtype, strategy=None, config=None, tune=False):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = self.OUTPUT_DTYPE or dtype
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
                self.N_total, self.dtype_str, self.op_func,
                output_dtype=self.output_dtype_str, threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_unary_explicit(
                self.N_total, self.dtype_str, self.op_func,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
            )
        elif strategy == "register_copy":
            return _make_unary_regcopy(
                self.N_total, self.dtype_str, self.op_func,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def output_dtype_str(self) -> str:
        return self.dtype_to_str(self.output_dtype)

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
    OUTPUT_DTYPE = None  # Subclass override for output dtype (e.g., "int8")
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes

    @staticmethod
    def op_func(a, b):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(
        self, N_total, dtype, coalesced_shape, a_strides, b_strides,
        a_numel, b_numel, strategy=None, config=None, tune=False,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
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
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes

    @staticmethod
    def activation_func(x):
        """Activation function. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, M, N, dtype, config=None, tune=False):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
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


class FloatUnaryKernel(UnaryKernel):
    """Unary kernel base for float-only elementwise ops."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES


class FloatPredicateKernel(FloatUnaryKernel):
    """Unary kernel base for float predicates with bool output."""

    DEFAULT_STRATEGY = "direct"
    OUTPUT_DTYPE = torch.bool


class LogicalUnaryKernel(UnaryKernel):
    """Unary kernel base for logical predicates with bool output."""

    DEFAULT_STRATEGY = "direct"
    SUPPORTED_DTYPES = _LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool


class ReluKernel(FloatUnaryKernel):
    """ReLU: y = max(x, 0)."""

    @staticmethod
    def op_func(x):
        return T.if_then_else(x > T.cast(0, x.dtype), x, T.cast(0, x.dtype))


class AddKernel(BinaryKernel):
    """Element-wise addition: y = a + b."""

    @staticmethod
    def op_func(a, b):
        return a + b


class SubKernel(BinaryKernel):
    """Element-wise subtraction: y = a - b."""

    @staticmethod
    def op_func(a, b):
        return a - b


class MulKernel(BinaryKernel):
    """Element-wise multiplication: y = a * b."""

    @staticmethod
    def op_func(a, b):
        return a * b


class DivKernel(BinaryKernel):
    """Element-wise division: y = a / b."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        return a / b


class RemainderKernel(BinaryKernel):
    """Element-wise remainder: y = a - floor(a / b) * b.

    Matches PyTorch remainder semantics for floating-point inputs.
    Uses floor-based formula since T.FloorMod requires integer types.
    Casts to fp32 before ``T.floor`` because ``hfloor`` is not available
    for ``cutlass::half_t`` in CUDA.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        quotient_f32 = T.cast(a, "float32") / T.cast(b, "float32")
        floored = T.Cast(a.dtype, T.floor(quotient_f32))
        return a - floored * b


class PowKernel(BinaryKernel):
    """Element-wise power: y = a ** b."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        a_f32 = T.Cast("float32", a)
        b_f32 = T.Cast("float32", b)
        return T.Cast(a.dtype, T.pow(a_f32, b_f32))


class FloorDivideKernel(BinaryKernel):
    """Element-wise floor division: y = floor(a / b).

    Casts to fp32 before ``T.floor`` because ``hfloor`` is not available
    for ``cutlass::half_t`` in CUDA.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        quotient_f32 = T.cast(a, "float32") / T.cast(b, "float32")
        return T.Cast(a.dtype, T.floor(quotient_f32))


class LerpKernel(BinaryKernel):
    """Element-wise lerp: y = a + weight * (b - a).

    PyTorch lerp is ternary (a, b, weight). Here weight is a compile-time
    constant passed at kernel construction, keeping the binary kernel template.

    Args:
        weight: Scalar interpolation weight (default 0.5).
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        raise NotImplementedError("Use _make_lerp_op_func(weight) instead")

    def __init__(
        self, N_total, dtype, coalesced_shape, a_strides, b_strides,
        a_numel, b_numel, strategy=None, config=None, tune=False, weight=0.5,
    ):
        self._weight = weight
        super().__init__(
            N_total, dtype, coalesced_shape, a_strides, b_strides,
            a_numel, b_numel, strategy=strategy, config=config, tune=tune,
        )

    def _build_kernel(self, strategy):
        """Override to inject compile-time weight into op_func."""
        w = self._weight

        def lerp_func(a, b):
            return a + T.cast(w, a.dtype) * (b - a)

        cfg = self.default_config
        if strategy == "direct":
            return _make_binary_direct(
                self.N_total, self.dtype_str, lerp_func,
                self.coalesced_shape, self.a_strides, self.b_strides,
                self.a_numel, self.b_numel,
                output_dtype=self.OUTPUT_DTYPE, threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_binary_explicit(
                self.N_total, self.dtype_str, lerp_func,
                self.coalesced_shape, self.a_strides, self.b_strides,
                self.a_numel, self.b_numel,
                output_dtype=self.OUTPUT_DTYPE,
                threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class MaximumKernel(BinaryKernel):
    """Element-wise maximum: y = max(a, b) with NaN propagation.

    Matches torch.maximum: if either operand is NaN, the result is NaN.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        # NaN propagation: if a is NaN return a (NaN), if b is NaN return b (NaN),
        # otherwise return the larger value. Signed-zero tie semantics are
        # tracked separately in issue #469.
        a_is_nan = T.isnan(T.Cast("float32", a))
        b_is_nan = T.isnan(T.Cast("float32", b))
        ordered_max = T.if_then_else(a > b, a, b)
        nan_result = T.if_then_else(a_is_nan, a, T.if_then_else(b_is_nan, b, ordered_max))
        return nan_result


class MinimumKernel(BinaryKernel):
    """Element-wise minimum: y = min(a, b) with NaN propagation.

    Matches torch.minimum: if either operand is NaN, the result is NaN.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        # NaN propagation: if a is NaN return a (NaN), if b is NaN return b (NaN),
        # otherwise return the smaller value. Signed-zero tie semantics are
        # tracked separately in issue #469.
        a_is_nan = T.isnan(T.Cast("float32", a))
        b_is_nan = T.isnan(T.Cast("float32", b))
        ordered_min = T.if_then_else(a < b, a, b)
        nan_result = T.if_then_else(a_is_nan, a, T.if_then_else(b_is_nan, b, ordered_min))
        return nan_result


# ---------------------------------------------------------------------------
# Comparison kernel subclasses (output int8: 1/0 flags, cast to bool in Op layer)
# ---------------------------------------------------------------------------


class EqKernel(BinaryKernel):
    """Element-wise equality: y = (a == b), stored as int8 (1/0)."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES
    OUTPUT_DTYPE = "int8"

    @staticmethod
    def op_func(a, b):
        one = T.IntImm("int8", 1)
        zero = T.IntImm("int8", 0)
        return T.if_then_else(a == b, one, zero)


class NeKernel(BinaryKernel):
    """Element-wise not-equal: y = (a != b), stored as int8 (1/0)."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES
    OUTPUT_DTYPE = "int8"

    @staticmethod
    def op_func(a, b):
        one = T.IntImm("int8", 1)
        zero = T.IntImm("int8", 0)
        return T.if_then_else(a != b, one, zero)


class GtKernel(BinaryKernel):
    """Element-wise greater-than: y = (a > b), stored as int8 (1/0)."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES
    OUTPUT_DTYPE = "int8"

    @staticmethod
    def op_func(a, b):
        one = T.IntImm("int8", 1)
        zero = T.IntImm("int8", 0)
        return T.if_then_else(a > b, one, zero)


class LtKernel(BinaryKernel):
    """Element-wise less-than: y = (a < b), stored as int8 (1/0)."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES
    OUTPUT_DTYPE = "int8"

    @staticmethod
    def op_func(a, b):
        one = T.IntImm("int8", 1)
        zero = T.IntImm("int8", 0)
        return T.if_then_else(a < b, one, zero)


class GeKernel(BinaryKernel):
    """Element-wise greater-equal: y = (a >= b), stored as int8 (1/0)."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES
    OUTPUT_DTYPE = "int8"

    @staticmethod
    def op_func(a, b):
        one = T.IntImm("int8", 1)
        zero = T.IntImm("int8", 0)
        return T.if_then_else(a >= b, one, zero)


class LeKernel(BinaryKernel):
    """Element-wise less-equal: y = (a <= b), stored as int8 (1/0)."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES
    OUTPUT_DTYPE = "int8"

    @staticmethod
    def op_func(a, b):
        one = T.IntImm("int8", 1)
        zero = T.IntImm("int8", 0)
        return T.if_then_else(a <= b, one, zero)


# ---------------------------------------------------------------------------
# Logical kernel subclasses (output int8: 1/0-encoded logical values)
# ---------------------------------------------------------------------------


class LogicalAndKernel(BinaryKernel):
    """Element-wise logical AND with non-zero truthiness, stored as int8."""

    SUPPORTED_DTYPES = _LOGICAL_DTYPES
    OUTPUT_DTYPE = "int8"

    @staticmethod
    def op_func(a, b):
        one = T.IntImm("int8", 1)
        zero = T.IntImm("int8", 0)
        a_nonzero = a != T.cast(0, a.dtype)
        b_nonzero = b != T.cast(0, b.dtype)
        return T.if_then_else(a_nonzero & b_nonzero, one, zero)


class LogicalOrKernel(BinaryKernel):
    """Element-wise logical OR with non-zero truthiness, stored as int8."""

    SUPPORTED_DTYPES = _LOGICAL_DTYPES
    OUTPUT_DTYPE = "int8"

    @staticmethod
    def op_func(a, b):
        one = T.IntImm("int8", 1)
        zero = T.IntImm("int8", 0)
        a_nonzero = a != T.cast(0, a.dtype)
        b_nonzero = b != T.cast(0, b.dtype)
        return T.if_then_else(a_nonzero | b_nonzero, one, zero)


# ---------------------------------------------------------------------------
# Bitwise kernel subclasses
# ---------------------------------------------------------------------------


class BitwiseAndKernel(BinaryKernel):
    """Element-wise bitwise AND: y = a & b (integer inputs)."""

    SUPPORTED_DTYPES = _BITWISE_DTYPES

    @staticmethod
    def op_func(a, b):
        return a & b


class BitwiseOrKernel(BinaryKernel):
    """Element-wise bitwise OR: y = a | b (integer inputs)."""

    SUPPORTED_DTYPES = _BITWISE_DTYPES

    @staticmethod
    def op_func(a, b):
        return a | b


class BitwiseXorKernel(BinaryKernel):
    """Element-wise bitwise XOR: y = a ^ b (integer inputs)."""

    SUPPORTED_DTYPES = _BITWISE_DTYPES

    @staticmethod
    def op_func(a, b):
        return a ^ b


# ---------------------------------------------------------------------------
# Fused gated kernel subclasses
# ---------------------------------------------------------------------------


class SiluAndMulKernel(FusedGatedKernel):
    """SiLU-and-Mul: y = silu(gate) * value = (gate * sigmoid(gate)) * value."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def activation_func(x):
        return x * T.sigmoid(x)


class GeluAndMulKernel(FusedGatedKernel):
    """GELU-and-Mul: y = gelu(gate) * value.

    Uses exact GELU: gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
    erf is computed in float32 to avoid missing half-precision intrinsic.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def activation_func(x):
        inv_sqrt2 = T.cast(0.7071067811865476, "float32")  # 1/sqrt(2)
        half = T.cast(0.5, x.dtype)
        one = T.cast(1.0, x.dtype)
        x_f32 = T.Cast("float32", x)
        erf_val = T.Cast(x.dtype, T.erf(x_f32 * inv_sqrt2))
        return x * half * (one + erf_val)


class GeluTanhAndMulKernel(FusedGatedKernel):
    """GELU-Tanh-and-Mul: y = gelu_tanh(gate) * value.

    Uses tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def activation_func(x):
        sqrt_2_over_pi = T.cast(0.7978845608028654, "float32")  # sqrt(2/pi)
        coeff = T.cast(0.044715, "float32")  # GELU tanh approx coefficient
        half = T.cast(0.5, x.dtype)
        one = T.cast(1.0, x.dtype)
        x_f32 = T.Cast("float32", x)
        inner = sqrt_2_over_pi * (x_f32 + coeff * x_f32 * x_f32 * x_f32)
        tanh_val = T.Cast(x.dtype, T.tanh(inner))
        return half * x * (one + tanh_val)


# ---------------------------------------------------------------------------
# Concrete unary kernel subclasses -- math (17)
# ---------------------------------------------------------------------------


class ExpKernel(FloatUnaryKernel):
    """Element-wise exp(x)."""

    @staticmethod
    def op_func(x):
        return T.exp(T.cast(x, "float32"))


class LogKernel(FloatUnaryKernel):
    """Element-wise log(x)."""

    @staticmethod
    def op_func(x):
        return T.log(T.cast(x, "float32"))


class SqrtKernel(FloatUnaryKernel):
    """Element-wise sqrt(x)."""

    @staticmethod
    def op_func(x):
        return T.sqrt(T.cast(x, "float32"))


class RsqrtKernel(FloatUnaryKernel):
    """Element-wise 1/sqrt(x)."""

    @staticmethod
    def op_func(x):
        return T.rsqrt(T.cast(x, "float32"))


class AbsKernel(FloatUnaryKernel):
    """Element-wise |x|."""

    @staticmethod
    def op_func(x):
        return T.abs(x)


class NegKernel(FloatUnaryKernel):
    """Element-wise -x."""

    @staticmethod
    def op_func(x):
        return -x


class ReciprocalKernel(FloatUnaryKernel):
    """Element-wise 1/x."""

    @staticmethod
    def op_func(x):
        return T.cast(1.0, "float32") / x


class SignKernel(FloatUnaryKernel):
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


class SinKernel(FloatUnaryKernel):
    """Element-wise sin(x)."""

    @staticmethod
    def op_func(x):
        return T.sin(T.cast(x, "float32"))


class CosKernel(FloatUnaryKernel):
    """Element-wise cos(x)."""

    @staticmethod
    def op_func(x):
        return T.cos(T.cast(x, "float32"))


class FloorKernel(FloatUnaryKernel):
    """Element-wise floor(x).

    Casts to fp32 before calling ``T.floor`` because ``hfloor`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.floor(T.cast(x, "float32"))


class CeilKernel(FloatUnaryKernel):
    """Element-wise ceil(x).

    Casts to fp32 before calling ``T.ceil`` because ``hceil`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.ceil(T.cast(x, "float32"))


class RoundKernel(FloatUnaryKernel):
    """Element-wise round(x) with banker's rounding (round-to-nearest-even).

    Uses ``T.nearbyint`` (maps to ``nearbyintf`` in CUDA) to match
    PyTorch's ``torch.round`` semantics. Casts to fp32 because
    ``hnearbyint`` is not available for ``cutlass::half_t``.
    """

    @staticmethod
    def op_func(x):
        return T.nearbyint(T.cast(x, "float32"))


class TruncKernel(FloatUnaryKernel):
    """Element-wise trunc(x) -- integer part toward zero.

    Casts to fp32 before calling ``T.trunc`` because ``htrunc`` is not
    available for ``cutlass::half_t`` in CUDA.
    """

    @staticmethod
    def op_func(x):
        return T.trunc(T.cast(x, "float32"))


class ErfKernel(FloatUnaryKernel):
    """Element-wise erf(x).

    Casts to fp32 before calling ``T.erf`` because the half-precision
    intrinsic ``herf`` is not a valid CUDA built-in.
    """

    @staticmethod
    def op_func(x):
        return T.erf(T.cast(x, "float32"))


class Log1pKernel(FloatUnaryKernel):
    """Element-wise log(1 + x).

    Uses composite ``log(1 + x)`` because ``T.log1p`` is not lowered
    by the TileLang compiler.
    """

    @staticmethod
    def op_func(x):
        return T.log(T.cast(1.0, "float32") + x)


class Expm1Kernel(FloatUnaryKernel):
    """Element-wise exp(x) - 1."""

    @staticmethod
    def op_func(x):
        return T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")


# ---------------------------------------------------------------------------
# Concrete unary kernel subclasses -- activations (8)
# ---------------------------------------------------------------------------


class GeluKernel(FloatUnaryKernel):
    """Element-wise GELU using the standard erf formulation."""

    @staticmethod
    def op_func(x):
        inv_sqrt_2 = T.cast(0.7071067811865476, "float32")
        half = T.cast(0.5, "float32")
        one = T.cast(1.0, "float32")
        return half * x * (one + T.erf(T.cast(x, "float32") * inv_sqrt_2))


class SiluKernel(FloatUnaryKernel):
    """Element-wise SiLU (Swish): x * sigmoid(x)."""

    @staticmethod
    def op_func(x):
        return x * T.sigmoid(x)


class SigmoidKernel(FloatUnaryKernel):
    """Element-wise sigmoid(x)."""

    @staticmethod
    def op_func(x):
        return T.sigmoid(x)


class TanhKernel(FloatUnaryKernel):
    """Element-wise tanh(x)."""

    @staticmethod
    def op_func(x):
        return T.tanh(T.cast(x, "float32"))


class HardswishKernel(FloatUnaryKernel):
    """Element-wise HardSwish: x * clamp(x + 3, 0, 6) / 6."""

    @staticmethod
    def op_func(x):
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        clamped = T.min(T.max(x + three, zero), six)
        return x * clamped / six


class HardsigmoidKernel(FloatUnaryKernel):
    """Element-wise HardSigmoid: clamp(x + 3, 0, 6) / 6."""

    @staticmethod
    def op_func(x):
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        return T.min(T.max(x + three, zero), six) / six


class MishKernel(FloatUnaryKernel):
    """Element-wise Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))."""

    @staticmethod
    def op_func(x):
        one = T.cast(1.0, "float32")
        return x * T.tanh(T.log(one + T.exp(x)))


class SeluKernel(FloatUnaryKernel):
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


# ---------------------------------------------------------------------------
# Concrete unary kernel subclasses -- logical / bitwise (2)
# ---------------------------------------------------------------------------


class LogicalNotKernel(LogicalUnaryKernel):
    """Element-wise logical NOT with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return x == T.cast(0, x.dtype)


class BitwiseNotKernel(UnaryKernel):
    """Element-wise bitwise NOT (~x) for bool/integer inputs.

    Uses XOR with ``-1`` (all-ones) because ``T.bitwise_not`` fails on
    vectorized ``int4`` CUDA types.
    """

    DEFAULT_STRATEGY = "direct"
    SUPPORTED_DTYPES = _BITWISE_DTYPES

    @staticmethod
    def op_func(x):
        if x.dtype == "bool":
            return x == T.cast(0, "bool")
        if x.dtype == "uint8":
            return T.bitwise_xor(x, T.cast(255, "uint8"))
        return T.bitwise_xor(x, T.cast(-1, x.dtype))


# ---------------------------------------------------------------------------
# Concrete unary kernel subclasses -- special predicates (3)
# ---------------------------------------------------------------------------


class IsnanKernel(FloatPredicateKernel):
    """Element-wise isnan with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isnan(T.cast(x, "float32"))


class IsinfKernel(FloatPredicateKernel):
    """Element-wise isinf with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isinf(T.cast(x, "float32"))


class IsfiniteKernel(FloatPredicateKernel):
    """Element-wise isfinite with torch-style bool output."""

    @staticmethod
    def op_func(x):
        return T.isfinite(T.cast(x, "float32"))
