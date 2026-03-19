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

import math

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


def _strategy_npt(strategy: str, dtype: torch.dtype) -> int:
    """Return the default num_per_thread for a strategy + dtype pair.

    Strategy-aware heuristic (H200 benchmarks, issue 553):
    - explicit_parallel: npt=4 for fp16/bf16 (42% bandwidth gain vs npt=8)
    - register_copy: npt=8 for fp16/bf16 (vectorized 128-bit loads)
    - fp32: npt=4 for all strategies (4 bytes x 4 = 128-bit alignment)
    - fp8: handled separately by callers (npt=16)
    """
    if dtype == torch.float32:
        return 4
    # fp16 / bf16: strategy-dependent
    if strategy == "explicit_parallel" and dtype in (torch.float16, torch.bfloat16):
        return 4
    return 8


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


def _get_fp8_output_dtypes(dtype: torch.dtype):
    """Return (fp8_output_dtype, kernel_output_dtype) for fp8 handling.

    For e5m2: kernel produces fp16 to preserve Inf/NaN; Op layer does the
    final non-saturating cast to e5m2 via PyTorch.
    For e4m3fn or non-fp8: kernel outputs directly in the input dtype.

    Returns:
        Tuple of (_fp8_output_dtype, output_dtype).  _fp8_output_dtype is
        the original fp8 dtype when a post-cast is needed, else None.
    """
    if _is_fp8(dtype) and _fp8_needs_nonsaturating_cast(dtype):
        return dtype, torch.float16
    return None, dtype


def _clamp_to_dtype_range(value: float, dtype: torch.dtype) -> float:
    """Clamp *value* to the finite representable range of *dtype*.

    Prevents TVM FloatImm range-check failures when a scalar literal exceeds
    the target dtype's maximum (e.g. 1e4 into float8_e4m3fn whose max is 448).
    NaN values are passed through unchanged (NaN is a valid sentinel, not an
    out-of-range finite value).  Infinities are mapped to the dtype's
    max/min finite value.
    """
    if math.isnan(value):
        return value
    finfo = torch.finfo(dtype)
    if math.isinf(value):
        return finfo.max if value > 0 else finfo.min
    return max(finfo.min, min(finfo.max, float(value)))


def _wrap_fp8_accumulation(base_op, dtype, dtype_str, arity=1):
    """Wrap an op function with fp8 accumulation logic if *dtype* is fp8.

    This shared helper eliminates duplicated fp8 cast-in / cast-out logic
    across UnaryKernel, BinaryKernel, and FusedGatedKernel.

    fp8 accumulation strategy:
    - e4m3fn (saturating): cast inputs to fp16, compute, T.Cast result back
      to e4m3fn.  e4m3fn has no Inf representation so saturation is correct.
    - e5m2 (non-saturating): cast inputs to fp16, compute, leave result as
      fp16.  The Op layer does the final non-saturating cast to e5m2 via
      PyTorch's ``.to()`` which preserves Inf/NaN.

    For non-fp8 dtypes the original *base_op* is returned unchanged.

    Args:
        base_op: The element-wise callable (unary or binary).
        dtype: ``torch.dtype`` of the kernel input.
        dtype_str: TileLang dtype string (e.g. ``"float8_e4m3fn"``).
        arity: Number of input operands (1 for unary, 2 for binary).

    Returns:
        A callable with the same arity that handles fp8 accumulation, or
        *base_op* itself when no wrapping is needed.
    """
    if not _is_fp8(dtype):
        return base_op

    accum = _fp8_accum_dtype_str()

    if _fp8_needs_nonsaturating_cast(dtype):
        # e5m2: compute in fp16, leave result as fp16
        if arity == 1:
            def fp8_accum_op(x):
                return base_op(T.cast(x, accum))
        else:
            def fp8_accum_op(a, b):
                return base_op(T.cast(a, accum), T.cast(b, accum))
        return fp8_accum_op

    # e4m3fn: compute in fp16, saturating cast back
    if arity == 1:
        def fp8_accum_op(x):
            return T.Cast(dtype_str, base_op(T.cast(x, accum)))
    else:
        def fp8_accum_op(a, b):
            return T.Cast(dtype_str, base_op(T.cast(a, accum), T.cast(b, accum)))
    return fp8_accum_op


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


def _is_contiguous_same_shape(coalesced_shape, a_strides, b_strides):
    """Return True when both inputs are contiguous with the same shape (no broadcast)."""
    return (
        len(coalesced_shape) == 1
        and all(s == 1 for s in a_strides)
        and all(s == 1 for s in b_strides)
    )


def _make_binary_register_copy(
    N_total, dtype, op_func, output_dtype=None, threads=256, num_per_thread=8,
):
    """Binary register_copy: fragment load -> compute -> fragment store.

    Only available for same-shape contiguous inputs (no broadcast).
    Uses T.alloc_fragment + T.copy for vectorized 128-bit memory access,
    giving ~2-3x bandwidth vs scalar access for complex op_funcs that
    prevent TVM's auto-vectorizer from kicking in.
    """
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[2])
    def kernel(threads, num_per_thread):
        block_size = threads * num_per_thread

        @T.prim_func
        def main(
            a: T.Tensor((N_total,), dtype),
            b: T.Tensor((N_total,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads) as bx:
                a_reg = T.alloc_fragment((block_size,), dtype)
                b_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), out_dtype)
                T.copy(a[bx * block_size:(bx + 1) * block_size], a_reg)
                T.copy(b[bx * block_size:(bx + 1) * block_size], b_reg)
                for i, j in T.Parallel(threads, num_per_thread):
                    idx = i * num_per_thread + j
                    y_reg[idx] = op_func(a_reg[idx], b_reg[idx])
                T.copy(y_reg, y[bx * block_size:(bx + 1) * block_size])

        return main

    return kernel


def _make_binary_direct(
    N_total, dtype, op_func, coalesced_shape, a_strides, b_strides,
    a_numel, b_numel, output_dtype=None, threads=256,
):
    """Binary direct: 1 element per thread with stride-based broadcast."""
    out_dtype = output_dtype or dtype

    # Fast path: same-shape contiguous inputs -- skip broadcast machinery
    if _is_contiguous_same_shape(coalesced_shape, a_strides, b_strides):
        @tilelang.jit(out_idx=[2])
        def kernel(threads):
            @T.prim_func
            def main(
                a: T.Tensor((N_total,), dtype),
                b: T.Tensor((N_total,), dtype),
                y: T.Tensor((N_total,), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N_total, threads), threads=threads) as bx:
                    for i in T.Parallel(threads):
                        idx = bx * threads + i
                        y[idx] = op_func(a[idx], b[idx])

            return main

        return kernel

    ndim = len(coalesced_shape)
    divisors = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        divisors[i] = divisors[i + 1] * coalesced_shape[i + 1]

    @tilelang.jit(out_idx=[2])
    def kernel(threads):
        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N_total, threads), threads=threads) as bx:
                for i in T.Parallel(threads):
                    flat_idx = bx * threads + i
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
    out_dtype = output_dtype or dtype

    # Fast path: same-shape contiguous inputs -- skip broadcast machinery
    if _is_contiguous_same_shape(coalesced_shape, a_strides, b_strides):
        @tilelang.jit(out_idx=[2])
        def kernel(threads, num_per_thread):
            block_size = threads * num_per_thread

            @T.prim_func
            def main(
                a: T.Tensor((N_total,), dtype),
                b: T.Tensor((N_total,), dtype),
                y: T.Tensor((N_total,), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N_total, block_size), threads=threads) as bx:
                    for i, j in T.Parallel(threads, num_per_thread):
                        idx = (bx * threads + i) * num_per_thread + j
                        y[idx] = op_func(a[idx], b[idx])

            return main

        return kernel

    ndim = len(coalesced_shape)
    divisors = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        divisors[i] = divisors[i + 1] * coalesced_shape[i + 1]

    @tilelang.jit(out_idx=[2])
    def kernel(threads, num_per_thread):
        block_size = threads * num_per_thread

        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads) as bx:
                for i, j in T.Parallel(threads, num_per_thread):
                    flat_idx = (bx * threads + i) * num_per_thread + j
                    a_off, b_off = _compute_broadcast_offsets(
                        flat_idx, ndim, divisors, a_strides, b_strides,
                    )
                    y[flat_idx] = op_func(a[a_off], b[b_off])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Strategy factory: FusedGated
# ---------------------------------------------------------------------------


def _make_fused_gated_direct(M, N, dtype, op_func, threads=256, output_dtype=None):
    """FusedGated direct: 1 element per thread. x[:, :N] is gate, x[:, N:] is value.

    ``op_func(gate, value)`` is the compound operation that applies the
    activation to *gate* and multiplies by *value*.  For fp8 dtypes the
    caller wraps it via ``_wrap_fp8_accumulation`` so this factory stays
    fp8-agnostic.

    Args:
        output_dtype: TileLang dtype string for the output tensor. Defaults to dtype.
    """
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg):
        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), out_dtype)):
            with T.Kernel(T.ceildiv(N, threads_arg), M, threads=threads_arg) as (bx, by):
                for i in T.Parallel(threads_arg):
                    col = bx * threads_arg + i
                    gate = x[by, col]
                    value = x[by, N + col]
                    y[by, col] = op_func(gate, value)

        return main

    return kernel


def _make_fused_gated_explicit(M, N, dtype, op_func, threads=256, num_per_thread=8,
                               output_dtype=None):
    """FusedGated explicit_parallel: N elements per thread.

    ``op_func(gate, value)`` is the compound operation (see
    ``_make_fused_gated_direct``).  fp8 accumulation is handled by the
    caller wrapping ``op_func`` via ``_wrap_fp8_accumulation``, so this
    factory no longer needs an ``fp8_accum`` parameter.

    Args:
        output_dtype: TileLang dtype string for the output tensor. Defaults to dtype.
    """
    block_N = threads * num_per_thread
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), out_dtype)):
            with T.Kernel(T.ceildiv(N, block_N), M, threads=threads_arg) as (bx, by):
                for i, j in T.Parallel(threads_arg, npt_arg):
                    col = (bx * threads_arg + i) * npt_arg + j
                    gate = x[by, col]
                    value = x[by, N + col]
                    y[by, col] = op_func(gate, value)

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

        Delegates to the shared ``_wrap_fp8_accumulation`` helper.
        When ``OUTPUT_DTYPE`` is set (e.g. bool-output ops) fp8 wrapping is
        skipped because the kernel already outputs a non-fp8 type.
        """
        if self.OUTPUT_DTYPE is not None:
            return self.op_func
        return _wrap_fp8_accumulation(self.op_func, self.dtype, self.dtype_str, arity=1)

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
        npt = _strategy_npt(self.strategy, self.dtype)
        return {"threads": 256, "num_per_thread": npt}

    @property
    def autotune_configs(self) -> list[dict]:
        """Search space: threads in {128, 256, 512} x num_per_thread in {2, 4, 8}.

        Covers a range of occupancy/register-pressure tradeoffs for
        bandwidth-bound unary elementwise kernels.
        """
        if _is_fp8(self.dtype):
            # fp8 needs 128-bit alignment: npt >= 16 for 1-byte elements
            threads_opts = [128, 256, 512]
            npt_opts = [16, 32]
        else:
            # fp16 / bf16 / fp32
            threads_opts = [128, 256, 512]
            npt_opts = [2, 4, 8]
        return [
            {"threads": t, "num_per_thread": n}
            for t in threads_opts
            for n in npt_opts
        ]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Override to handle serialization failures in the TileLang autotuner.

        UnaryKernel JIT functions capture op_func closures that the autotuner
        subprocess cannot serialize.  Catch the error and fall back to the
        default config so that ``tune=True`` never crashes.
        """
        import warnings

        try:
            super().autotune(warmup=warmup, rep=rep)
        except (AssertionError, Exception) as exc:
            if "not serializable" in str(exc) or "pickle" in str(exc).lower():
                warnings.warn(
                    f"{self.__class__.__name__} autotuning failed "
                    f"(op_func is not serializable); falling back to "
                    f"default_config.",
                    stacklevel=2,
                )
                self.config = dict(self.default_config)
            else:
                raise

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        result = self._compiled_fn(x)
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
        strategy: One of "direct", "explicit_parallel", "register_copy".
            If "register_copy" is requested but inputs require broadcast,
            silently downgrades to "explicit_parallel".
        config: Optional dict with "threads" and "num_per_thread".
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
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
        self._same_shape = _is_contiguous_same_shape(
            coalesced_shape, a_strides, b_strides,
        )
        if strategy is not None:
            # register_copy requires same-shape contiguous inputs (no
            # broadcast); silently downgrade to explicit_parallel when
            # the caller requests register_copy on broadcast shapes.
            if strategy == "register_copy" and not self._same_shape:
                self.strategy = "explicit_parallel"
            else:
                self.strategy = strategy
        elif self._same_shape:
            # register_copy gives vectorized 128-bit loads, ~2-3x faster
            # for complex op_funcs that block TVM's auto-vectorizer.
            self.strategy = "register_copy"
        else:
            self.strategy = self.DEFAULT_STRATEGY
        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
            )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """Return op_func wrapped with fp8->fp16 accumulation if needed.

        Delegates to the shared ``_wrap_fp8_accumulation`` helper (arity=2).
        When ``OUTPUT_DTYPE`` is set (e.g. comparison/logical ops) fp8 wrapping
        is skipped because the kernel already outputs a non-fp8 type.
        """
        if self.OUTPUT_DTYPE is not None:
            return self.op_func
        return _wrap_fp8_accumulation(self.op_func, self.dtype, self.dtype_str, arity=2)

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
        elif strategy == "register_copy":
            return _make_binary_register_copy(
                self.N_total, self.dtype_str, effective_op,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        if _is_fp8(self.dtype):
            return {"threads": 256, "num_per_thread": 16}
        npt = _strategy_npt(self.strategy, self.dtype)
        return {"threads": 256, "num_per_thread": npt}

    @property
    def autotune_configs(self) -> list[dict]:
        """Search space: threads in {128, 256, 512} x num_per_thread in {2, 4, 8}.

        Covers a range of occupancy/register-pressure tradeoffs for
        bandwidth-bound binary elementwise kernels.
        """
        if _is_fp8(self.dtype):
            # fp8 needs 128-bit alignment: npt >= 16 for 1-byte elements
            threads_opts = [128, 256, 512]
            npt_opts = [16, 32]
        else:
            # fp16 / bf16 / fp32
            threads_opts = [128, 256, 512]
            npt_opts = [2, 4, 8]
        return [
            {"threads": t, "num_per_thread": n}
            for t in threads_opts
            for n in npt_opts
        ]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Override to handle serialization failures in the TileLang autotuner.

        BinaryKernel JIT functions capture op_func closures that the autotuner
        subprocess cannot serialize.  Catch the error and fall back to the
        default config so that ``tune=True`` never crashes.
        """
        import warnings

        try:
            super().autotune(warmup=warmup, rep=rep)
        except (AssertionError, Exception) as exc:
            if "not serializable" in str(exc) or "pickle" in str(exc).lower():
                warnings.warn(  # noqa: B028
                    f"{self.__class__.__name__} autotuning failed "
                    f"(op_func is not serializable); falling back to "
                    f"default_config.")
                self.config = dict(self.default_config)
            else:
                raise

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, a, b):
        result = self._compiled_fn(a, b)
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
    # Benchmark (H200, 4096x4096 fp16): explicit_parallel ~2x faster than direct
    #   silu_and_mul:       3.04 TB/s explicit vs 1.50 TB/s direct
    #   gelu_and_mul:       2.72 TB/s explicit vs 1.47 TB/s direct
    #   gelu_tanh_and_mul:  3.38 TB/s explicit vs 1.51 TB/s direct
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
        self._kernel_output_dtype = None
        if _is_fp8(dtype) and _fp8_needs_nonsaturating_cast(dtype):
            self._kernel_output_dtype = _fp8_accum_dtype_str()
            self._fp8_output_dtype = dtype
        self.strategy = strategy or self.DEFAULT_STRATEGY
        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
            )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """Return compound op ``(gate, value) -> activation(gate) * value``.

        Delegates to the shared ``_wrap_fp8_accumulation`` helper (arity=2)
        so that fp8 cast-in / cast-out logic is centralised.
        """
        act = self.activation_func

        def fused_op(gate, value):
            return act(gate) * value

        return _wrap_fp8_accumulation(fused_op, self.dtype, self.dtype_str, arity=2)

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = self._get_effective_op_func()
        if strategy == "direct":
            return _make_fused_gated_direct(
                self.M, self.N, self.dtype_str, effective_op,
                threads=cfg["threads"],
                output_dtype=self._kernel_output_dtype,
            )
        elif strategy == "explicit_parallel":
            return _make_fused_gated_explicit(
                self.M, self.N, self.dtype_str, effective_op,
                cfg["threads"], cfg["num_per_thread"],
                output_dtype=self._kernel_output_dtype,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        if _is_fp8(self.dtype):
            return {"threads": 256, "num_per_thread": 16}
        npt = _strategy_npt(self.strategy, self.dtype)
        return {"threads": 256, "num_per_thread": npt}

    @property
    def autotune_configs(self) -> list[dict]:
        """Search space: threads in {128, 256, 512} x num_per_thread in {2, 4, 8}.

        Covers a range of occupancy/register-pressure tradeoffs for
        bandwidth-bound fused gated elementwise kernels.
        """
        if _is_fp8(self.dtype):
            # fp8 needs 128-bit alignment: npt >= 16 for 1-byte elements
            threads_opts = [128, 256, 512]
            npt_opts = [16, 32]
        else:
            # fp16 / bf16 / fp32
            threads_opts = [128, 256, 512]
            npt_opts = [2, 4, 8]
        return [
            {"threads": t, "num_per_thread": n}
            for t in threads_opts
            for n in npt_opts
        ]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Override to handle serialization failures in the TileLang autotuner.

        FusedGatedKernel JIT functions capture activation_func closures that
        the autotuner subprocess cannot serialize.  Catch the error and fall
        back to the default config so that ``tune=True`` never crashes.
        """
        import warnings

        try:
            super().autotune(warmup=warmup, rep=rep)
        except (AssertionError, Exception) as exc:
            if "not serializable" in str(exc) or "pickle" in str(exc).lower():
                warnings.warn(
                    f"{self.__class__.__name__} autotuning failed "
                    f"(activation_func is not serializable); falling back to "
                    f"default_config.",
                    stacklevel=2,
                )
                self.config = dict(self.default_config)
            else:
                raise

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        result = self._compiled_fn(x)
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

        # Wrap with fp8 accumulation via shared helper
        effective_op = _wrap_fp8_accumulation(
            lerp_func, self.dtype, self.dtype_str, arity=2,
        )

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
        elif strategy == "register_copy":
            return _make_binary_register_copy(
                self.N_total, self.dtype_str, effective_op,
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

    Performance: uses T.max for the fast path (correct signed-zero on CUDA
    -- fmaxf returns +0 for max(+0,-0)) plus two isnan guards for NaN
    propagation. Total IR: 1 max + 2 fp32 casts + 2 isnan + 2 select.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        # T.max handles signed-zero correctly (returns +0 for max(+0,-0))
        # but does NOT propagate NaN -- it returns the non-NaN operand.
        result = T.max(a, b)
        # NaN propagation: cast to fp32 for isnan (bfloat16 lacks native isnan).
        a_is_nan = T.isnan(T.Cast("float32", a))
        b_is_nan = T.isnan(T.Cast("float32", b))
        result = T.if_then_else(b_is_nan, b, result)
        result = T.if_then_else(a_is_nan, a, result)
        return result


class MinimumKernel(BinaryKernel):
    """Element-wise minimum: y = min(a, b) with NaN propagation.

    Matches torch.minimum semantics:
    - If either operand is NaN, the result is NaN.
    - minimum(-0.0, +0.0) = -0.0 (IEEE 754 signed-zero).

    Performance: uses T.min for the fast path (correct signed-zero on CUDA
    -- fminf returns -0 for min(-0,+0)) plus two isnan guards for NaN
    propagation. See MaximumKernel for full rationale.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        # T.min handles signed-zero correctly (returns -0 for min(+0,-0))
        # but does NOT propagate NaN -- it returns the non-NaN operand.
        result = T.min(a, b)
        # NaN propagation: cast to fp32 for isnan (bfloat16 lacks native isnan).
        a_is_nan = T.isnan(T.Cast("float32", a))
        b_is_nan = T.isnan(T.Cast("float32", b))
        result = T.if_then_else(b_is_nan, b, result)
        result = T.if_then_else(a_is_nan, a, result)
        return result


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


def _make_leaky_relu_kernel(N, dtype, negative_slope, output_dtype=None,
                            is_fp8=False, threads=256, npt=8):
    """Build leaky_relu kernel: y = x if x > 0 else negative_slope * x.

    For non-fp8 dtypes, uses register_copy strategy: fragment load -> compute
    -> fragment store for coalesced memory access.

    For fp8 dtypes, uses explicit_parallel with fp16 accumulation (register_copy
    is unreliable for 8-bit fragments).
    """
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    if is_fp8:
        accum = _fp8_accum_dtype_str()

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_leaky_relu_kernel(
            N_total, self.dtype_str, negative_slope,
            output_dtype=self.dtype_to_str(self.output_dtype),
            is_fp8=_is_fp8(dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 16
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        return self._compiled_fn(x)


def _make_elu_kernel(N, dtype, alpha, output_dtype=None, threads=256, npt=8):
    """Build ELU kernel: y = x if x > 0 else alpha * (exp(x) - 1)."""
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
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
                        y[idx] = T.if_then_else(v32 > zero, T.Cast(out_dtype, v32), T.Cast(out_dtype, a * (T.exp(v32) - one)))

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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_elu_kernel(
            N_total, self.dtype_str, alpha,
            output_dtype=self.dtype_to_str(self.output_dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        return self._compiled_fn(x)


def _make_hardtanh_kernel(N, dtype, min_val, max_val, output_dtype=None, threads=256, npt=8):
    """Build hardtanh kernel: y = clamp(x, min_val, max_val)."""
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_hardtanh_kernel(
            N_total, self.dtype_str, min_val, max_val,
            output_dtype=self.dtype_to_str(self.output_dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        return self._compiled_fn(x)


def _make_softplus_kernel(N, dtype, beta, threshold, output_dtype=None, threads=256, npt=8):
    """Build softplus kernel: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x."""
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
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
                        sp = T.log(one + T.exp(scaled)) / b
                        y[idx] = T.if_then_else(scaled > t, T.Cast(out_dtype, v32), T.Cast(out_dtype, sp))

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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_softplus_kernel(
            N_total, self.dtype_str, beta, threshold,
            output_dtype=self.dtype_to_str(self.output_dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        return self._compiled_fn(x)


def _make_prelu_kernel(N, C, inner_size, dtype, output_dtype=None,
                       is_fp8=False, threads=256, npt=8):
    """Build PReLU kernel: y = x if x > 0 else weight[channel] * x.

    Weight is per-channel. Channel index follows PyTorch convention:
    for flat index ``idx``, channel = (idx // inner_size) % C, where
    ``inner_size`` is the product of all dimensions after the channel dim.

    For non-fp8 dtypes, uses register_copy strategy for input/output to
    improve memory coalescing for the main data path.

    For fp8 dtypes, uses explicit_parallel with fp16 accumulation
    (register_copy is unreliable for 8-bit fragments).
    """
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    if is_fp8:
        accum = _fp8_accum_dtype_str()

        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                weight: T.Tensor((C,), dtype),
                y: T.Tensor((N,), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        idx = bx * block_size + k
                        if idx < N:
                            val = x[idx]
                            ch = (idx // inner_size) % C
                            w = weight[ch]
                            v = T.cast(val, accum)
                            wf = T.cast(w, accum)
                            zero = T.cast(0, accum)
                            y[idx] = T.if_then_else(v > zero, T.Cast(out_dtype, v), T.Cast(out_dtype, wf * v))

            return main
    else:

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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_prelu_kernel(
            N_total, C, inner_size, self.dtype_str,
            output_dtype=self.dtype_to_str(self.output_dtype),
            is_fp8=_is_fp8(dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 16
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x, weight):
        return self._compiled_fn(x, weight)


def _make_where_kernel(N, dtype, is_fp8=False, threads=256, npt=8):
    """Build where kernel: out = cond ? x : y.

    The Op layer packs the bool condition as uint8 so that T.copy can
    perform vectorized loads (TileLang does not vectorize bool tensors).
    Each uint8 element is 0 or 1; the kernel loads it into a register
    fragment and unpacks per-element with a != 0 comparison.

    For non-fp8 dtypes, writes the result back into the x register fragment
    (in-place) to reduce register pressure and avoid a fourth data-typed
    fragment allocation.

    For fp8 dtypes, uses explicit_parallel (direct element access) since
    register_copy is unreliable for 8-bit fragments.
    """
    block_size = threads * npt

    if is_fp8:

        @tilelang.jit(out_idx=[3])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(
                cond: T.Tensor((N,), "uint8"),
                x: T.Tensor((N,), dtype),
                y_in: T.Tensor((N,), dtype),
                out: T.Tensor((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            out[idx] = T.if_then_else(
                                cond[idx] != T.cast(0, "uint8"), x[idx], y_in[idx],
                            )

            return main
    else:

        @tilelang.jit(out_idx=[3])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(
                cond: T.Tensor((N,), "uint8"),
                x: T.Tensor((N,), dtype),
                y_in: T.Tensor((N,), dtype),
                out: T.Tensor((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    c_reg = T.alloc_fragment((block_size,), "uint8")
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(cond[bx * block_size : (bx + 1) * block_size], c_reg)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    T.copy(y_in[bx * block_size : (bx + 1) * block_size], y_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        x_reg[k] = T.if_then_else(
                            c_reg[k] != T.cast(0, "uint8"), x_reg[k], y_reg[k],
                        )
                    T.copy(x_reg, out[bx * block_size : (bx + 1) * block_size])

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
        # Where is a pure selection op (out = cond ? x : y): no arithmetic,
        # no type conversion needed.  _fp8_output_dtype is explicitly None
        # so _apply_fp8_post_cast is a no-op.
        self._fp8_output_dtype = None
        cfg = self.default_config
        self.kernel = _make_where_kernel(
            N_total, self.dtype_str, is_fp8=_is_fp8(dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 512, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, cond, x, y):
        return self._compiled_fn(cond, x, y)


def _make_clamp_kernel(N, dtype, has_min, has_max, min_val, max_val,
                       output_dtype=None, threads=256, npt=8):
    """Build clamp kernel: y = clamp(x, min_val, max_val) with optional bounds."""
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    if idx < N:
                        v32 = T.cast(x[idx], "float32")
                        if has_min:
                            lo = T.cast(min_val, "float32")
                            v32 = T.max(v32, lo)
                        if has_max:
                            hi = T.cast(max_val, "float32")
                            v32 = T.min(v32, hi)
                        y[idx] = T.Cast(out_dtype, v32)

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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_clamp_kernel(
            N_total, self.dtype_str,
            has_min=min_val is not None,
            has_max=max_val is not None,
            min_val=min_val if min_val is not None else 0.0,
            max_val=max_val if max_val is not None else 0.0,
            output_dtype=self.dtype_to_str(self.output_dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        return self._compiled_fn(x)


def _make_masked_fill_kernel(N, dtype, fill_value, output_dtype=None,
                             is_fp8=False, threads=256, npt=8):
    """Build masked_fill kernel: out = mask ? fill_value : x.

    The Op layer packs the bool mask as uint8 so that T.copy can
    perform vectorized loads (TileLang does not vectorize bool tensors).
    Each uint8 element is 0 or 1; the kernel loads it into a register
    fragment and unpacks per-element with a != 0 comparison.

    For non-fp8 dtypes, writes the result back into the x register fragment
    (in-place) to reduce register pressure and avoid a third data-typed
    fragment allocation.

    For fp8 dtypes, uses explicit_parallel (direct element access) since
    register_copy is unreliable for 8-bit fragments.  For e5m2, the kernel
    outputs fp16 so the Op layer can do a non-saturating cast to e5m2.
    """
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    if is_fp8:

        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                mask: T.Tensor((N,), "uint8"),
                out: T.Tensor((N,), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            fv = T.cast(fill_value, out_dtype)
                            x_val = T.Cast(out_dtype, x[idx])
                            out[idx] = T.if_then_else(
                                mask[idx] != T.cast(0, "uint8"), fv, x_val,
                            )

            return main
    else:

        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                mask: T.Tensor((N,), "uint8"),
                out: T.Tensor((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    m_reg = T.alloc_fragment((block_size,), "uint8")
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(mask[bx * block_size : (bx + 1) * block_size], m_reg)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        fv = T.cast(fill_value, dtype)
                        x_reg[k] = T.if_then_else(
                            m_reg[k] != T.cast(0, "uint8"), fv, x_reg[k],
                        )
                    T.copy(x_reg, out[bx * block_size : (bx + 1) * block_size])

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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        self.fill_value = _clamp_to_dtype_range(fill_value, self.output_dtype)
        cfg = self.default_config
        self.kernel = _make_masked_fill_kernel(
            N_total, self.dtype_str, self.fill_value,
            output_dtype=self.dtype_to_str(self.output_dtype),
            is_fp8=_is_fp8(dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 512, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x, mask):
        return self._compiled_fn(x, mask)


def _make_nan_to_num_kernel(N, dtype, nan_val, posinf_val, neginf_val,
                            output_dtype=None, is_fp8=False, threads=256, npt=8):
    """Build nan_to_num kernel: replace NaN, +Inf, -Inf with given values.

    For non-fp8 dtypes, uses register_copy strategy: fragment load -> compute
    -> fragment store for coalesced memory access.

    For fp8 dtypes, uses explicit_parallel (direct element access) since
    register_copy is unreliable for 8-bit fragments.
    """
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    if is_fp8:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            val = x[idx]
                            v32 = T.cast(val, "float32")
                            nan_r = T.cast(nan_val, out_dtype)
                            pos_r = T.cast(posinf_val, out_dtype)
                            neg_r = T.cast(neginf_val, out_dtype)
                            pass_through = T.Cast(out_dtype, v32)
                            result = T.if_then_else(
                                T.isnan(v32),
                                nan_r,
                                T.if_then_else(
                                    T.isinf(v32),
                                    T.if_then_else(v32 > T.cast(0, "float32"), pos_r, neg_r),
                                    pass_through,
                                ),
                            )
                            y[idx] = result

            return main
    else:

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

    Note: For e4m3fn this is effectively a no-op because e4m3fn has no Inf
    or NaN representation — all values are finite by construction.  The
    kernel still runs but no replacements will occur.

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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        self.nan_val = _clamp_to_dtype_range(nan_val, self.output_dtype)
        self.posinf_val = _clamp_to_dtype_range(posinf_val, self.output_dtype)
        self.neginf_val = _clamp_to_dtype_range(neginf_val, self.output_dtype)
        cfg = self.default_config
        self.kernel = _make_nan_to_num_kernel(
            N_total, self.dtype_str, self.nan_val, self.posinf_val, self.neginf_val,
            output_dtype=self.dtype_to_str(self.output_dtype),
            is_fp8=_is_fp8(dtype),
            threads=cfg["threads"], npt=cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else 16
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        return self._compiled_fn(x)


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
                    if flat < N_total:
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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_alibi_kernel(
            seq_len, num_heads, self.dtype_to_str(self.output_dtype),
            cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self):
        return self._compiled_fn()


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
                    if flat < N_total:
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
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_sinusoidal_kernel(
            seq_len, d_model, self.dtype_to_str(self.output_dtype),
            cfg["threads"], cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self):
        return self._compiled_fn()
