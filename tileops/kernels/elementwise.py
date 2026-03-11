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
    "UnaryKernel",
    "BinaryKernel",
    "FusedGatedKernel",
    "ReluKernel",
    "AddKernel",
    "SiluAndMulKernel",
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
