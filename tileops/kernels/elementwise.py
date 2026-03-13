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

fp8 dtype support (e4m3fn, e5m2):
  Accumulation strategy: fp8 input → cast to fp16 → compute → cast back to fp8.
  Direct fp8 arithmetic loses too much precision for non-trivial ops (sigmoid,
  exp, etc.), so all computation is performed in fp16 as the accumulation dtype.
  Default num_per_thread=16 for fp8 (1 byte × 16 = 128-bit memory alignment).
  Default strategy is explicit_parallel (register_copy is unreliable for fp8).

  Saturation semantics (matches NVIDIA spec):
  - e4m3fn: no Inf/NaN representation, kernel uses T.Cast (saturating)
    which clamps overflow to ±448.0 -- correct for this format.
  - e5m2: has Inf/NaN representation, kernel produces fp16 output to
    preserve non-finite values (Inf, NaN). The Op layer performs the final
    non-saturating cast to e5m2 via PyTorch's .to() which preserves Inf/NaN.
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
    # --- independent (custom-signature) ---
    "LeakyReluKernel",
    "EluKernel",
    "HardtanhKernel",
    "SoftplusKernel",
    "PreluKernel",
    "WhereKernel",
    "ClampKernel",
    "MaskedFillKernel",
    "NanToNumKernel",
    "AlibiKernel",
    "SinusoidalKernel",
]

_BITWISE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)

_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)

_LOGICAL_DTYPES = _BITWISE_DTYPES + _FLOAT_DTYPES


def _is_fp8(dtype: torch.dtype) -> bool:
    """Check if a torch dtype is an fp8 variant."""
    return dtype in _FP8_DTYPES


def _fp8_needs_nonsaturating_cast(dtype: torch.dtype) -> bool:
    """Return True if the fp8 format supports Inf/NaN and needs non-saturating output.

    e5m2 has Inf/NaN representation -- TileLang's T.Cast uses saturating conversion
    which incorrectly clamps Inf to max-finite.  For e5m2, the kernel must produce
    fp16 output and let PyTorch do the final non-saturating cast.

    e4m3fn has no Inf representation, so saturating T.Cast is correct.
    """
    return dtype == torch.float8_e5m2


def _fp8_accum_dtype_str() -> str:
    """Return the TileLang dtype string used for fp8 intermediate accumulation."""
    return "float16"


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


def _make_fused_gated_direct(M, N, dtype, activation_func, threads=256):
    """FusedGated direct: 1 element per thread. x[:, :N] is gate, x[:, N:] is value."""

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg):
        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, threads_arg), M, threads=threads_arg) as (bx, by):
                for i in T.Parallel(threads_arg):
                    col = bx * threads_arg + i
                    gate = x[by, col]
                    value = x[by, N + col]
                    y[by, col] = activation_func(gate) * value

        return main

    return kernel


def _make_fused_gated_kernel(M, N, dtype, activation_func, threads=256, num_per_thread=8,
                             fp8_accum=None, output_dtype=None):
    """FusedGated explicit_parallel: N elements per thread. x[:, :N] is gate, x[:, N:] is value.

    Args:
        fp8_accum: "saturating" for e4m3fn (T.Cast back to fp8),
                   "nonsaturating" for e5m2 (leave as fp16, Op casts later).
                   None for non-fp8 dtypes.
        output_dtype: TileLang dtype string for the output tensor. Defaults to dtype.
    """
    block_N = threads * num_per_thread
    out_dtype = output_dtype or dtype

    if fp8_accum == "saturating":
        accum_dtype = _fp8_accum_dtype_str()

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_N), M, threads=threads_arg) as (bx, by):
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        col = (bx * threads_arg + i) * npt_arg + j
                        gate = T.cast(x[by, col], accum_dtype)
                        value = T.cast(x[by, N + col], accum_dtype)
                        y[by, col] = T.Cast(out_dtype, activation_func(gate) * value)

            return main

        return kernel

    if fp8_accum == "nonsaturating":
        accum_dtype = _fp8_accum_dtype_str()

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_N), M, threads=threads_arg) as (bx, by):
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        col = (bx * threads_arg + i) * npt_arg + j
                        gate = T.cast(x[by, col], accum_dtype)
                        value = T.cast(x[by, N + col], accum_dtype)
                        y[by, col] = activation_func(gate) * value

            return main

        return kernel

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), out_dtype)):
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
    # Benchmark (H200): register_copy wins for fp16/bf16 across all tested shapes;
    # fp32 small shapes show variance between register_copy and explicit_parallel.
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
        # For e5m2: kernel produces fp16 to preserve Inf/NaN; Op layer
        # performs the final non-saturating cast to e5m2 via PyTorch.
        # For e4m3fn: kernel produces e4m3fn via saturating T.Cast (correct,
        # since e4m3fn has no Inf representation).
        self._fp8_output_dtype = None
        if _is_fp8(dtype) and self.OUTPUT_DTYPE is None and _fp8_needs_nonsaturating_cast(dtype):
            self._fp8_output_dtype = dtype
            self.output_dtype = torch.float16
        else:
            self.output_dtype = self.OUTPUT_DTYPE or dtype
        # fp8: register_copy may not reliably handle 8-bit fragments;
        # default to explicit_parallel for fp8 dtypes
        if strategy is None and _is_fp8(dtype):
            self.strategy = "explicit_parallel"
        else:
            self.strategy = strategy or self.DEFAULT_STRATEGY
        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
            )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """Return op_func wrapped with fp8->fp16 accumulation if needed.

        fp8 accumulation strategy:
        - e4m3fn: cast to fp16, compute, T.Cast back to e4m3fn (saturating).
          e4m3fn has no Inf, so saturation is correct.
        - e5m2: cast to fp16, compute, leave as fp16 (kernel output is fp16).
          The Op layer does the final non-saturating cast to e5m2 via PyTorch
          to preserve Inf/NaN.
        """
        if _is_fp8(self.dtype) and self.OUTPUT_DTYPE is None:
            base_op = self.op_func
            if _fp8_needs_nonsaturating_cast(self.dtype):
                # e5m2: compute in fp16, leave result as fp16
                def fp8_accum_op(x):
                    x_fp16 = T.cast(x, _fp8_accum_dtype_str())
                    return base_op(x_fp16)

                return fp8_accum_op
            else:
                # e4m3fn: compute in fp16, saturating cast back to e4m3fn
                out_dtype_str = self.dtype_str

                def fp8_accum_op(x):
                    x_fp16 = T.cast(x, _fp8_accum_dtype_str())
                    result = base_op(x_fp16)
                    return T.Cast(out_dtype_str, result)

                return fp8_accum_op
        return self.op_func

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = self._get_effective_op_func()
        if strategy == "direct":
            return _make_unary_direct(
                self.N_total, self.dtype_str, effective_op,
                output_dtype=self.output_dtype_str, threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_unary_explicit(
                self.N_total, self.dtype_str, effective_op,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
            )
        elif strategy == "register_copy":
            return _make_unary_regcopy(
                self.N_total, self.dtype_str, effective_op,
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
        if _is_fp8(self.dtype):
            # fp8: 1 byte per element, 16 elements = 128-bit alignment
            return {"threads": 256, "num_per_thread": 16}
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        if self.strategy == "direct":
            result = self.kernel(cfg["threads"])(x)
        else:
            result = self.kernel(cfg["threads"], cfg["num_per_thread"])(x)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result


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
        self._fp8_output_dtype = None
        if _is_fp8(dtype) and self.OUTPUT_DTYPE is None and _fp8_needs_nonsaturating_cast(dtype):
            self._fp8_output_dtype = dtype
        self.coalesced_shape = coalesced_shape
        self.a_strides = a_strides
        self.b_strides = b_strides
        self.a_numel = a_numel
        self.b_numel = b_numel
        self.strategy = strategy or self.DEFAULT_STRATEGY
        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
            )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """Return op_func wrapped with fp8->fp16 accumulation if needed.

        See UnaryKernel._get_effective_op_func for the e4m3fn vs e5m2 rationale.
        """
        if _is_fp8(self.dtype) and self.OUTPUT_DTYPE is None:
            base_op = self.op_func
            if _fp8_needs_nonsaturating_cast(self.dtype):
                # e5m2: compute in fp16, leave result as fp16
                def fp8_accum_op(a, b):
                    a_fp16 = T.cast(a, _fp8_accum_dtype_str())
                    b_fp16 = T.cast(b, _fp8_accum_dtype_str())
                    return base_op(a_fp16, b_fp16)

                return fp8_accum_op
            else:
                # e4m3fn: compute in fp16, saturating cast back
                out_dtype_str = self.dtype_str

                def fp8_accum_op(a, b):
                    a_fp16 = T.cast(a, _fp8_accum_dtype_str())
                    b_fp16 = T.cast(b, _fp8_accum_dtype_str())
                    result = base_op(a_fp16, b_fp16)
                    return T.Cast(out_dtype_str, result)

                return fp8_accum_op
        return self.op_func

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = self._get_effective_op_func()
        # For e5m2: kernel output is fp16 (non-saturating path)
        kernel_output_dtype = self.OUTPUT_DTYPE
        if self._fp8_output_dtype is not None:
            kernel_output_dtype = _fp8_accum_dtype_str()
        if strategy == "direct":
            return _make_binary_direct(
                self.N_total, self.dtype_str, effective_op,
                self.coalesced_shape, self.a_strides, self.b_strides,
                self.a_numel, self.b_numel,
                output_dtype=kernel_output_dtype, threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_binary_explicit(
                self.N_total, self.dtype_str, effective_op,
                self.coalesced_shape, self.a_strides, self.b_strides,
                self.a_numel, self.b_numel,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        if _is_fp8(self.dtype):
            return {"threads": 256, "num_per_thread": 16}
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, a, b):
        cfg = self.config
        if self.strategy == "direct":
            result = self.kernel(cfg["threads"])(a, b)
        else:
            result = self.kernel(cfg["threads"], cfg["num_per_thread"])(a, b)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result


class FusedGatedKernel(Kernel):
    """Template base class for fused gated elementwise kernels.

    Input layout: x has shape (M, 2*N) where x[:, :N] is the gate
    and x[:, N:] is the value. Output: y = activation(gate) * value.

    Subclass must override ``activation_func`` with a static method.

    Args:
        M: Number of rows.
        N: Half the column dimension (output width).
        dtype: Torch dtype.
        strategy: One of "direct", "explicit_parallel".
        config: Optional dict with "threads" and "num_per_thread".
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel"]
    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes

    @staticmethod
    def activation_func(x):
        """Activation function. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, M, N, dtype, strategy=None, config=None, tune=False):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.dtype = dtype
        self._fp8_output_dtype = None
        # Determine fp8 accumulation mode and output dtype
        self._fp8_accum = None
        self._kernel_output_dtype = None
        if _is_fp8(dtype):
            if _fp8_needs_nonsaturating_cast(dtype):
                self._fp8_accum = "nonsaturating"
                self._kernel_output_dtype = _fp8_accum_dtype_str()
                self._fp8_output_dtype = dtype
            else:
                self._fp8_accum = "saturating"
        self.strategy = strategy or self.DEFAULT_STRATEGY
        assert self.strategy in self.STRATEGIES, (
            f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
        )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _build_kernel(self, strategy):
        cfg = self.default_config
        if strategy == "direct":
            return _make_fused_gated_direct(
                self.M, self.N, self.dtype_str, self.activation_func,
                threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_fused_gated_kernel(
                self.M, self.N, self.dtype_str, self.activation_func,
                cfg["threads"], cfg["num_per_thread"],
                fp8_accum=self._fp8_accum, output_dtype=self._kernel_output_dtype,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        if _is_fp8(self.dtype):
            return {"threads": 256, "num_per_thread": 16}
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        if self.strategy == "direct":
            result = self.kernel(cfg["threads"])(x)
        else:
            result = self.kernel(cfg["threads"], cfg["num_per_thread"])(x)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result


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

        # Wrap with fp8 accumulation if needed
        if _is_fp8(self.dtype) and self.OUTPUT_DTYPE is None:
            if _fp8_needs_nonsaturating_cast(self.dtype):
                # e5m2: compute in fp16, leave as fp16
                def effective_op(a, b):
                    a_fp16 = T.cast(a, _fp8_accum_dtype_str())
                    b_fp16 = T.cast(b, _fp8_accum_dtype_str())
                    return lerp_func(a_fp16, b_fp16)
            else:
                # e4m3fn: compute in fp16, saturating cast back
                out_dtype_str = self.dtype_str

                def effective_op(a, b):
                    a_fp16 = T.cast(a, _fp8_accum_dtype_str())
                    b_fp16 = T.cast(b, _fp8_accum_dtype_str())
                    result = lerp_func(a_fp16, b_fp16)
                    return T.Cast(out_dtype_str, result)
        else:
            effective_op = lerp_func

        # For e5m2: kernel output is fp16 (non-saturating path)
        kernel_output_dtype = self.OUTPUT_DTYPE
        if self._fp8_output_dtype is not None:
            kernel_output_dtype = _fp8_accum_dtype_str()

        cfg = self.default_config
        if strategy == "direct":
            return _make_binary_direct(
                self.N_total, self.dtype_str, effective_op,
                self.coalesced_shape, self.a_strides, self.b_strides,
                self.a_numel, self.b_numel,
                output_dtype=kernel_output_dtype, threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_binary_explicit(
                self.N_total, self.dtype_str, effective_op,
                self.coalesced_shape, self.a_strides, self.b_strides,
                self.a_numel, self.b_numel,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class MaximumKernel(BinaryKernel):
    """Element-wise maximum: y = max(a, b) with NaN propagation.

    Matches torch.maximum semantics:
    - If either operand is NaN, the result is NaN.
    - maximum(+0.0, -0.0) = +0.0 (IEEE 754 signed-zero).
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        a_f32 = T.Cast("float32", a)
        b_f32 = T.Cast("float32", b)
        a_is_nan = T.isnan(a_f32)
        b_is_nan = T.isnan(b_f32)
        # Signed-zero tie-break: when a == b, prefer the operand whose sign
        # is non-negative (+0 over -0). copysign(1, x) is +1 for +0 and -1
        # for -0, so a_sign >= b_sign selects the positive-signed operand.
        one_f32 = T.cast(1.0, "float32")
        a_sign = T.copysign(one_f32, a_f32)
        b_sign = T.copysign(one_f32, b_f32)
        ordered_max = T.if_then_else(
            a > b, a, T.if_then_else(b > a, b,
                                     T.if_then_else(a_sign >= b_sign, a, b)))
        return T.if_then_else(a_is_nan, a, T.if_then_else(b_is_nan, b, ordered_max))


class MinimumKernel(BinaryKernel):
    """Element-wise minimum: y = min(a, b) with NaN propagation.

    Matches torch.minimum semantics:
    - If either operand is NaN, the result is NaN.
    - minimum(-0.0, +0.0) = -0.0 (IEEE 754 signed-zero).
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        a_f32 = T.Cast("float32", a)
        b_f32 = T.Cast("float32", b)
        a_is_nan = T.isnan(a_f32)
        b_is_nan = T.isnan(b_f32)
        # Signed-zero tie-break: when a == b, prefer the operand whose sign
        # is negative (-0 over +0). copysign(1, x) is -1 for -0 and +1 for
        # +0, so a_sign <= b_sign selects the negative-signed operand.
        one_f32 = T.cast(1.0, "float32")
        a_sign = T.copysign(one_f32, a_f32)
        b_sign = T.copysign(one_f32, b_f32)
        ordered_min = T.if_then_else(
            a < b, a, T.if_then_else(b < a, b,
                                     T.if_then_else(a_sign <= b_sign, a, b)))
        return T.if_then_else(a_is_nan, a, T.if_then_else(b_is_nan, b, ordered_min))


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


# ---------------------------------------------------------------------------
# Independent (custom-signature) kernel classes (11)
# ---------------------------------------------------------------------------


def _make_leaky_relu_kernel(N, dtype, negative_slope, threads=256, npt=8):
    """Build leaky_relu kernel: y = x if x > 0 else negative_slope * x.

    Uses register_copy strategy: fragment load -> compute -> fragment store
    for coalesced memory access and reduced per-element instruction overhead.
    """
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
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


class LeakyReluKernel(Kernel):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x."""

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, negative_slope=0.01, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.negative_slope = negative_slope
        cfg = self.default_config
        self.kernel = _make_leaky_relu_kernel(
            N_total, self.dtype_str, negative_slope, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 16
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)


def _make_elu_kernel(N, dtype, alpha, threads=256, npt=8):
    """Build ELU kernel: y = x if x > 0 else alpha * (exp(x) - 1)."""
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    val = x[idx]
                    zero = T.cast(0, "float32")
                    a = T.cast(alpha, "float32")
                    one = T.cast(1.0, "float32")
                    v32 = T.cast(val, "float32")
                    y[idx] = T.if_then_else(v32 > zero, val, T.Cast(val.dtype, a * (T.exp(v32) - one)))

        return main

    return kernel


class EluKernel(Kernel):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1)."""

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, alpha=1.0, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.alpha = alpha
        cfg = self.default_config
        self.kernel = _make_elu_kernel(
            N_total, self.dtype_str, alpha, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)


def _make_hardtanh_kernel(N, dtype, min_val, max_val, threads=256, npt=8):
    """Build hardtanh kernel: y = clamp(x, min_val, max_val)."""
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    val = x[idx]
                    lo = T.cast(min_val, "float32")
                    hi = T.cast(max_val, "float32")
                    v32 = T.cast(val, "float32")
                    y[idx] = T.Cast(val.dtype, T.min(T.max(v32, lo), hi))

        return main

    return kernel


class HardtanhKernel(Kernel):
    """Hardtanh: y = clamp(x, min_val, max_val)."""

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, min_val=-1.0, max_val=1.0, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        cfg = self.default_config
        self.kernel = _make_hardtanh_kernel(
            N_total, self.dtype_str, min_val, max_val, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)


def _make_softplus_kernel(N, dtype, beta, threshold, threads=256, npt=8):
    """Build softplus kernel: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x."""
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    val = x[idx]
                    v32 = T.cast(val, "float32")
                    b = T.cast(beta, "float32")
                    t = T.cast(threshold, "float32")
                    one = T.cast(1.0, "float32")
                    scaled = v32 * b
                    sp = T.log(one + T.exp(scaled)) / b
                    y[idx] = T.if_then_else(scaled > t, val, T.Cast(val.dtype, sp))

        return main

    return kernel


class SoftplusKernel(Kernel):
    """Softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x."""

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, beta=1.0, threshold=20.0, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.beta = beta
        self.threshold = threshold
        cfg = self.default_config
        self.kernel = _make_softplus_kernel(
            N_total, self.dtype_str, beta, threshold, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)


def _make_prelu_kernel(N, C, inner_size, dtype, threads=256, npt=8):
    """Build PReLU kernel: y = x if x > 0 else weight[channel] * x.

    Weight is per-channel. Channel index follows PyTorch convention:
    for flat index ``idx``, channel = (idx // inner_size) % C, where
    ``inner_size`` is the product of all dimensions after the channel dim.

    Uses register_copy strategy for input/output to improve memory
    coalescing for the main data path.
    """
    block_size = threads * npt

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            weight: T.Tensor((C,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    idx = bx * block_size + k
                    val = x_reg[k]
                    ch = (idx // inner_size) % C
                    w = weight[ch]
                    zero = T.cast(0, val.dtype)
                    y_reg[k] = T.if_then_else(val > zero, val, w * val)
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class PreluKernel(Kernel):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel index follows PyTorch convention: channels live at
    dimension 1 (or dimension 0 for 1-D inputs). The ``inner_size``
    parameter is the product of all dimensions after the channel
    dimension.

    Args:
        N_total: Total number of elements (flattened).
        C: Number of channels (weight length).
        inner_size: Product of dimensions after the channel dim.
        dtype: Torch dtype.
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, C, inner_size, dtype, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.C = C
        self.inner_size = inner_size
        self.dtype = dtype
        cfg = self.default_config
        self.kernel = _make_prelu_kernel(
            N_total, C, inner_size, self.dtype_str, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 16
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x, weight):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x, weight)


def _make_where_kernel(N, dtype, threads=256, npt=8):
    """Build where kernel: out = cond ? x : y.

    Accepts bool condition tensor directly to avoid dtype conversion
    overhead in the Op layer. Uses register_copy for the data path
    (x, y, out) with element-wise bool cond access, since T.copy
    does not support bool vectorization.
    """
    block_size = threads * npt

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            cond: T.Tensor((N,), "bool"),
            x: T.Tensor((N,), dtype),
            y_in: T.Tensor((N,), dtype),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                o_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                T.copy(y_in[bx * block_size : (bx + 1) * block_size], y_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    idx = bx * block_size + k
                    o_reg[k] = T.if_then_else(cond[idx], x_reg[k], y_reg[k])
                T.copy(o_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class WhereKernel(Kernel):
    """Where: out = cond ? x : y.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype for x and y.
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        cfg = self.default_config
        self.kernel = _make_where_kernel(
            N_total, self.dtype_str, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 8 if self.dtype == torch.float32 else 32
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, cond, x, y):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(cond, x, y)


def _make_clamp_kernel(N, dtype, has_min, has_max, min_val, max_val, threads=256, npt=8):
    """Build clamp kernel: y = clamp(x, min_val, max_val) with optional bounds."""
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    v32 = T.cast(x[idx], "float32")
                    if has_min:
                        lo = T.cast(min_val, "float32")
                        v32 = T.max(v32, lo)
                    if has_max:
                        hi = T.cast(max_val, "float32")
                        v32 = T.min(v32, hi)
                    y[idx] = T.Cast(dtype, v32)

        return main

    return kernel


class ClampKernel(Kernel):
    """Clamp: y = clamp(x, min, max) with optional bounds.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        min_val: Lower bound (None = no lower bound).
        max_val: Upper bound (None = no upper bound).
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, min_val=None, max_val=None, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        cfg = self.default_config
        self.kernel = _make_clamp_kernel(
            N_total, self.dtype_str,
            has_min=min_val is not None,
            has_max=max_val is not None,
            min_val=min_val if min_val is not None else 0.0,
            max_val=max_val if max_val is not None else 0.0,
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)


def _make_masked_fill_kernel(N, dtype, fill_value, threads=256, npt=8):
    """Build masked_fill kernel: out = mask ? fill_value : x.

    Accepts bool mask tensor directly to avoid dtype conversion
    overhead in the Op layer. Uses register_copy for the data path
    (x and out) with element-wise bool mask access, since T.copy
    does not support bool vectorization.
    """
    block_size = threads * npt

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            mask: T.Tensor((N,), "bool"),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                o_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    idx = bx * block_size + k
                    fv = T.cast(fill_value, dtype)
                    o_reg[k] = T.if_then_else(mask[idx], fv, x_reg[k])
                T.copy(o_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class MaskedFillKernel(Kernel):
    """MaskedFill: out = mask ? fill_value : x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        fill_value: Scalar value to fill where mask is True.
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, fill_value, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.fill_value = fill_value
        cfg = self.default_config
        self.kernel = _make_masked_fill_kernel(
            N_total, self.dtype_str, fill_value, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 8 if self.dtype == torch.float32 else 16
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x, mask):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x, mask)


def _make_nan_to_num_kernel(N, dtype, nan_val, posinf_val, neginf_val, threads=256, npt=8):
    """Build nan_to_num kernel: replace NaN, +Inf, -Inf with given values.

    Uses register_copy strategy: fragment load -> compute -> fragment store
    for coalesced memory access.
    """
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    val = x_reg[k]
                    v32 = T.cast(val, "float32")
                    nan_r = T.cast(nan_val, val.dtype)
                    pos_r = T.cast(posinf_val, val.dtype)
                    neg_r = T.cast(neginf_val, val.dtype)
                    result = T.if_then_else(
                        T.isnan(v32),
                        nan_r,
                        T.if_then_else(
                            T.isinf(v32),
                            T.if_then_else(v32 > T.cast(0, "float32"), pos_r, neg_r),
                            val,
                        ),
                    )
                    y_reg[k] = result
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class NanToNumKernel(Kernel):
    """NanToNum: replace NaN, +Inf, -Inf with specified values.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        nan_val: Replacement for NaN (default 0.0).
        posinf_val: Replacement for +Inf (default 1e4).
        neginf_val: Replacement for -Inf (default -1e4).
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, nan_val=0.0, posinf_val=1e4, neginf_val=-1e4,
                 config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.nan_val = nan_val
        self.posinf_val = posinf_val
        self.neginf_val = neginf_val
        cfg = self.default_config
        self.kernel = _make_nan_to_num_kernel(
            N_total, self.dtype_str, nan_val, posinf_val, neginf_val,
            cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 16
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)


def _make_alibi_kernel(seq_len, num_heads, dtype, threads=256, npt=8):
    """Build ALiBi kernel: bias[h, i, j] = -slope_h * |i - j|.

    Slopes: slope_h = 2^(-8*h/H) for head h in [0, H).
    Output shape: (num_heads, seq_len, seq_len).
    Total elements: num_heads * seq_len * seq_len.
    """
    N_total = num_heads * seq_len * seq_len
    block_size = threads * npt
    S2 = seq_len * seq_len

    @tilelang.jit(out_idx=[0])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(out: T.Tensor((N_total,), dtype)):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat = (bx * threads_arg + i) * npt_arg + j
                    h = flat // S2
                    rem = flat % S2
                    row = rem // seq_len
                    col = rem % seq_len
                    # slope = 2^(-8 * (h+1) / num_heads)
                    exp_val = T.cast(-8.0, "float32") * T.cast(h + 1, "float32") / T.cast(num_heads, "float32")
                    slope = T.exp2(exp_val)
                    dist = T.cast(row - col, "float32")
                    # Use abs via if_then_else since T.abs may not handle int
                    abs_dist = T.if_then_else(dist > T.cast(0, "float32"), dist, -dist)
                    out[flat] = T.Cast(dtype, -slope * abs_dist)

        return main

    return kernel


class AlibiKernel(Kernel):
    """ALiBi position encoding: bias[h, i, j] = -slope_h * |i - j|.

    Generates the full (num_heads, seq_len, seq_len) bias tensor.
    Slopes follow the ALiBi paper: slope_h = 2^(-8*(h+1)/H).

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        dtype: Torch dtype.
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, seq_len, num_heads, dtype, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dtype = dtype
        cfg = self.default_config
        self.kernel = _make_alibi_kernel(
            seq_len, num_heads, self.dtype_str, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])()


def _make_sinusoidal_kernel(seq_len, d_model, dtype, threads=256, npt=8):
    """Build sinusoidal positional encoding kernel.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Output shape: (seq_len, d_model).
    """
    N_total = seq_len * d_model
    block_size = threads * npt

    @tilelang.jit(out_idx=[0])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(out: T.Tensor((N_total,), dtype)):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat = (bx * threads_arg + i) * npt_arg + j
                    pos = flat // d_model
                    dim = flat % d_model
                    # dim_pair = dim // 2 (the "i" in the formula)
                    dim_pair = dim // 2
                    # angle = pos / 10000^(2*dim_pair / d_model)
                    base = T.cast(10000.0, "float32")
                    exp_frac = T.cast(dim_pair, "float32") * T.cast(2.0, "float32") / T.cast(d_model, "float32")
                    divisor = T.pow(base, exp_frac)
                    angle = T.cast(pos, "float32") / divisor
                    # Even dim -> sin, odd dim -> cos
                    is_even = dim % 2 == 0
                    result = T.if_then_else(is_even, T.sin(angle), T.cos(angle))
                    out[flat] = T.Cast(dtype, result)

        return main

    return kernel


class SinusoidalKernel(Kernel):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length.
        d_model: Model dimension (must be even).
        dtype: Torch dtype.
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, seq_len, d_model, dtype, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        cfg = self.default_config
        self.kernel = _make_sinusoidal_kernel(
            seq_len, d_model, self.dtype_str, cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])()
