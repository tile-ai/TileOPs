"""Elementwise kernel templates and strategy factories.

Template base classes, one per input shape:
- UnaryKernel: 1-input → 1-output (relu, sigmoid, abs, ...)
- BinaryKernel: 2-input → 1-output with N-dim broadcast (add, mul, ...)
- FusedGatedKernel: fused gate+activation (silu_and_mul, gelu_and_mul, ...)
- ParametricUnaryKernel: 1-input plus values baked in at build time, and the
  multi-input kernels built on it (where, clamp, masked_fill, prelu, lerp)

An op hands over the shapes its manifest entry declares; flattening, broadcasting and
restoring the output shape are each kernel's ``forward``. No kernel here uses shared
memory: Global → Register → Compute → Register → Global.

Three strategies pick the loop body at build time:
- direct: 1 element per thread
- explicit_parallel: N elements per thread via T.Parallel(threads, npt)
- register_copy: fragment load → compute → fragment store

Boundary checks are TileLang's, via LegalizeSafeMemoryAccess.

fp8 (e4m3fn, e5m2) accumulates in fp16 — direct fp8 arithmetic loses too much
precision for sigmoid/exp and friends. Defaults: num_per_thread=16 (128-bit
alignment) and explicit_parallel (register_copy is unreliable for fp8). Saturation
follows from the format: e4m3fn has no Inf, so a saturating ``T.Cast`` clamping to
±448.0 is what it can represent; e5m2 does have Inf, so the PrimFunc emits fp16 and
``forward`` casts to e5m2 without saturating, which keeps Inf and NaN."""

import functools
import math
import warnings

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = [
    "BinaryKernel",
    "FloatPredicateKernel",
    "FloatUnaryKernel",
    "FusedGatedKernel",
    "LogicalUnaryKernel",
    "ParametricUnaryKernel",
    "UnaryKernel",
    "coalesce_broadcast_dims",
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
)


_LOGICAL_DTYPES = _BITWISE_DTYPES + _FLOAT_DTYPES


_BINARY_FULL_DTYPES = _BITWISE_DTYPES + (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


_BINARY_NO_BOOL_DTYPES = tuple(dt for dt in _BINARY_FULL_DTYPES if dt is not torch.bool)


def _is_fp8(dtype: torch.dtype) -> bool:
    """Check if a torch dtype is an fp8 variant."""
    return dtype in _FP8_DTYPES


def _strategy_npt(strategy: str, dtype: torch.dtype) -> int:
    """Return the default num_per_thread for a strategy + dtype pair.

    Strategy-aware heuristic (from H200 benchmarks):
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

    For e5m2: the PrimFunc produces fp16 to preserve Inf/NaN, and ``forward`` casts
    to e5m2 without saturating.
    For e4m3fn or non-fp8: the PrimFunc outputs the input dtype directly.

    Returns:
        Tuple of (_fp8_output_dtype, output_dtype).  _fp8_output_dtype is
        the original fp8 dtype when a post-cast is needed, else None.
    """
    if _is_fp8(dtype) and _fp8_needs_nonsaturating_cast(dtype):
        return dtype, torch.float16
    return None, dtype


def _clamp_to_dtype_range(value, dtype: torch.dtype):
    """Normalize *value* into the storage representation of *dtype*.

    Mirrors PyTorch ``Tensor.masked_fill`` scalar coercion so the literal lands
    as the same bit pattern PyTorch would write:

    - bool: non-zero → ``1``, else ``0``.
    - Signed int: truncate toward zero; ``+/-Inf`` maps to ``iinfo.max/min``
      so a bypassed validator cannot raise ``OverflowError`` on ``int(inf)``.
    - ``uint8``: negatives wrap via ``& 0xFF``, non-negatives truncate.
    - ``fp16/bf16/fp32`` and ``fp8_e5m2``: ``NaN`` / ``+-Inf`` pass through,
      finite values clamp to ``finfo``.
    - ``fp8_e4m3fn`` has no Inf, so ``+-Inf`` saturates to ``finfo.max/min``
      to avoid a TVM ``FloatImm`` overflow.
    """
    if dtype == torch.bool:
        return 1 if bool(value) else 0
    if dtype in _BITWISE_DTYPES:
        if isinstance(value, float) and math.isinf(value):
            iinfo = torch.iinfo(dtype)
            return iinfo.max if value > 0 else iinfo.min
        if (
            dtype == torch.uint8
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value < 0
        ):
            return value & 0xFF
        return int(value)
    fvalue = float(value)
    if math.isnan(fvalue):
        return fvalue
    finfo = torch.finfo(dtype)
    if math.isinf(fvalue):
        if dtype in _FP8_DTYPES and not _fp8_needs_nonsaturating_cast(dtype):
            return finfo.max if fvalue > 0 else finfo.min
        return fvalue
    return max(finfo.min, min(finfo.max, fvalue))


def _wrap_fp8_accumulation(base_op, dtype, dtype_str, arity=1):
    """Wrap an op function with fp8 accumulation logic if *dtype* is fp8.

    Both fp8 dtypes cast inputs to fp16 and compute there. e4m3fn casts the
    result back via saturating ``T.Cast`` (correct — it has no Inf); e5m2
    leaves the result in fp16 and ``forward`` does the final non-saturating
    cast, which preserves Inf/NaN.

    Non-fp8 dtypes get *base_op* back unchanged.
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


@functools.lru_cache(maxsize=32)
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


@functools.lru_cache(maxsize=32)
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


@functools.lru_cache(maxsize=32)
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


def _flat(t):
    """The flat view every PrimFunc here takes.

    The Op layer normalizes contiguity and hands over the manifest-declared
    shape, so reducing it to one dimension is this backend's business.
    """
    return t.reshape(-1)


def _broadcast_target(*tensors):
    """The output shape a multi-operand kernel writes, from the operands it got.

    What ``torch.broadcast_shapes`` says, which is what the manifest's shape rules say:
    all-0-dim operands give ``()``. The PrimFuncs below work on a flat buffer either
    way — a 0-dim tensor holds one element — so restoring that shape is this wrapper's
    last step.
    """
    return torch.broadcast_shapes(*(tuple(t.shape) for t in tensors if t is not None))


def _expand_flat(t, shape):
    """Broadcast *t* to *shape*, then flatten it.

    Materializes a copy when *t* is genuinely broadcast: the kernels below index
    every operand at the output element, so they need one element per output.
    """
    if tuple(t.shape) != tuple(shape):
        t = t.expand(shape)
    return t.contiguous().reshape(-1)


def coalesce_broadcast_dims(a_shape, b_shape):
    """Coalesce N-dim broadcast into minimal effective dimensions.

    Merges adjacent dimensions that have the same broadcast behaviour
    (both real or both broadcast) to minimise the number of divmod
    operations inside the kernel loop.

    This is a lowering decision, not part of the op's contract: the Op layer
    hands ``BinaryKernel`` the two operand shapes and the kernel picks its own
    index representation.

    Args:
        a_shape: Shape tuple of input a.
        b_shape: Shape tuple of input b.

    Returns:
        Tuple of (out_shape, coalesced_shape, a_strides, b_strides) where
        strides use 0 for broadcast dimensions.
    """
    # Normalise scalar (0-dim) inputs to 1-dim with size 1
    if len(a_shape) == 0:
        a_shape = (1,)
    if len(b_shape) == 0:
        b_shape = (1,)

    out_shape = torch.broadcast_shapes(a_shape, b_shape)
    ndim = len(out_shape)
    a_pad = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b_pad = (1,) * (ndim - len(b_shape)) + tuple(b_shape)

    def _make_strides(padded_shape):
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * padded_shape[i + 1]
        # Only zero strides for genuinely broadcast dims (size-1 expanded to >1)
        return [0 if padded_shape[i] == 1 and out_shape[i] > 1 else strides[i] for i in range(ndim)]

    a_raw = _make_strides(a_pad)
    b_raw = _make_strides(b_pad)

    # Coalesce adjacent dims with compatible broadcast patterns
    groups = [(out_shape[0], a_raw[0], b_raw[0])]
    for i in range(1, ndim):
        prev_out, prev_as, prev_bs = groups[-1]
        a_can = (a_raw[i] == 0 and prev_as == 0) or (
            a_raw[i] != 0 and prev_as == a_raw[i] * out_shape[i]
        )
        b_can = (b_raw[i] == 0 and prev_bs == 0) or (
            b_raw[i] != 0 and prev_bs == b_raw[i] * out_shape[i]
        )
        if a_can and b_can:
            groups[-1] = (prev_out * out_shape[i], a_raw[i], b_raw[i])
        else:
            groups.append((out_shape[i], a_raw[i], b_raw[i]))

    # Remove trivial size-1 groups (unless all trivial)
    groups = [g for g in groups if g[0] > 1] or [(1, 0, 0)]
    coalesced_shape = tuple(g[0] for g in groups)
    a_strides = tuple(g[1] for g in groups)
    b_strides = tuple(g[2] for g in groups)
    return out_shape, coalesced_shape, a_strides, b_strides


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


@functools.lru_cache(maxsize=32)
def _make_binary_register_copy(
    N_total,
    dtype,
    op_func,
    output_dtype=None,
    threads=256,
    num_per_thread=8,
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
                T.copy(a[bx * block_size : (bx + 1) * block_size], a_reg)
                T.copy(b[bx * block_size : (bx + 1) * block_size], b_reg)
                for i, j in T.Parallel(threads, num_per_thread):
                    idx = i * num_per_thread + j
                    y_reg[idx] = op_func(a_reg[idx], b_reg[idx])
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_binary_direct(
    N_total,
    dtype,
    op_func,
    coalesced_shape,
    a_strides,
    b_strides,
    a_numel,
    b_numel,
    output_dtype=None,
    threads=256,
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
                        flat_idx,
                        ndim,
                        divisors,
                        a_strides,
                        b_strides,
                    )
                    y[flat_idx] = op_func(a[a_off], b[b_off])

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_binary_explicit(
    N_total,
    dtype,
    op_func,
    coalesced_shape,
    a_strides,
    b_strides,
    a_numel,
    b_numel,
    output_dtype=None,
    threads=256,
    num_per_thread=8,
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
                        flat_idx,
                        ndim,
                        divisors,
                        a_strides,
                        b_strides,
                    )
                    y[flat_idx] = op_func(a[a_off], b[b_off])

        return main

    return kernel


@functools.lru_cache(maxsize=32)
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


@functools.lru_cache(maxsize=32)
def _make_fused_gated_explicit(
    M, N, dtype, op_func, threads=256, num_per_thread=8, output_dtype=None
):
    """FusedGated explicit_parallel: N elements per thread.

    ``op_func(gate, value)`` is the compound operation (see
    ``_make_fused_gated_direct``).  fp8 accumulation belongs to the caller, which
    wraps ``op_func`` via ``_wrap_fp8_accumulation``.

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


class UnaryKernel(Kernel):
    """Template base class for unary elementwise kernels.

    Subclass must override ``op_func`` with a static method implementing
    the pointwise operation (e.g., relu, sigmoid).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype for input.
        config: Optional dict with "strategy", "threads" and "num_per_thread".
            "strategy" is one of "direct", "explicit_parallel",
            "register_copy"; it selects the kernel body at build time.
        tune: Whether to autotune (sweeps "threads" / "num_per_thread"
            within the resolved strategy).
    """

    #: Elementwise work is sized by the tensor shape: an integer operand is
    #: data, and a ``uint8`` ``cond``/``mask`` only selects a result.
    autotune_accepts_random_int_inputs: bool = True

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

    def __init__(self, N_total, dtype, config=None, tune=False):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        # Which fp8 dtype needs a post-cast, and why: module docstring.
        self._fp8_output_dtype = None
        if _is_fp8(dtype) and self.OUTPUT_DTYPE is None and _fp8_needs_nonsaturating_cast(dtype):
            self._fp8_output_dtype = dtype
            self.output_dtype = torch.float16
        else:
            self.output_dtype = self.OUTPUT_DTYPE or dtype
        # Validate a config-requested strategy up front so typos raise the
        # same ValueError regardless of dtype (the bool coercion below would
        # otherwise silently accept an unknown strategy for bool inputs).
        requested = (config or {}).get("strategy")
        if requested is not None and requested not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{requested}', expected one of {self.STRATEGIES}")
        # torch.bool maps to TileLang ``boolx<N>`` for vectorised loads, which
        # the CUDA codegen cannot lower. Keep bool inputs on the scalar path.
        bool_output = torch.bool == self.OUTPUT_DTYPE
        bool_output_needs_scalar = bool_output and dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        if dtype == torch.bool:
            if requested is not None and requested != "direct":
                warnings.warn(
                    f"UnaryKernel: dtype=torch.bool requires strategy="
                    f"'direct' (TileLang cannot lower vectorised boolx<N> "
                    f"loads); overriding requested strategy={requested!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.strategy = "direct"
        elif bool_output_needs_scalar:
            if requested is not None and requested != "direct":
                warnings.warn(
                    f"UnaryKernel: dtype={dtype} with torch.bool output "
                    f"requires strategy='direct' (TileLang cannot lower "
                    f"vectorised boolx<N> stores for sub-32-bit integer "
                    f"inputs); overriding requested strategy={requested!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.strategy = "direct"
        # fp8: register_copy may not reliably handle 8-bit fragments;
        # default to explicit_parallel for fp8 dtypes
        elif requested is None and _is_fp8(dtype):
            self.strategy = "explicit_parallel"
        else:
            self.strategy = requested or self.DEFAULT_STRATEGY
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
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_unary_explicit(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"],
                num_per_thread=cfg["num_per_thread"],
            )
        elif strategy == "register_copy":
            return _make_unary_regcopy(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"],
                num_per_thread=cfg["num_per_thread"],
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
            return {"strategy": self.strategy, "threads": 256, "num_per_thread": 16}
        npt = _strategy_npt(self.strategy, self.dtype)
        return {"strategy": self.strategy, "threads": 256, "num_per_thread": npt}

    @property
    def autotune_configs(self) -> list[dict]:
        """Search space: threads in {128, 256, 512} x num_per_thread in {2, 4, 8}.

        Covers a range of occupancy/register-pressure tradeoffs for
        bandwidth-bound unary elementwise kernels. "strategy" is a
        build-time config key (it selects the kernel body, not a JIT
        parameter), so it is excluded from the sweep.
        """
        if _is_fp8(self.dtype):
            # fp8 needs 128-bit alignment: npt >= 16 for 1-byte elements
            threads_opts = [128, 256, 512]
            npt_opts = [16, 32]
        else:
            # fp16 / bf16 / fp32
            threads_opts = [128, 256, 512]
            npt_opts = [2, 4, 8]
        return [{"threads": t, "num_per_thread": n} for t in threads_opts for n in npt_opts]

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
        # Record the resolved strategy so ``self.config`` is the single source of
        # truth for it, whether the request was coerced, downgraded or autotuned.
        self.config["strategy"] = self.strategy
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(_flat(x))
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result.reshape(x.shape)


class BinaryKernel(Kernel):
    """Template base class for binary elementwise kernels with N-dim broadcast.

    Subclass must override ``op_func`` with a static method implementing
    the pointwise operation (e.g., add, mul).

    Args:
        a_shape: Shape of input a.
        b_shape: Shape of input b. Broadcasts against *a_shape* under the
            PyTorch broadcasting rules.
        dtype: Torch dtype for input.
        config: Optional dict with "strategy", "threads" and "num_per_thread".
            "strategy" is one of "direct", "explicit_parallel",
            "register_copy". If "register_copy" is requested but inputs
            require broadcast, silently downgrades to "explicit_parallel".
        tune: Whether to autotune (sweeps "threads" / "num_per_thread"
            within the resolved strategy).

    Attributes:
        out_shape: Broadcast output shape.
        N_total: Total output elements.
    """

    #: Elementwise work is sized by the tensor shape: an integer operand is
    #: data, and a ``uint8`` ``cond``/``mask`` only selects a result.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
    DEFAULT_STRATEGY = "explicit_parallel"
    OUTPUT_DTYPE = None  # Subclass override for output dtype (e.g., torch.int8)
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes

    @staticmethod
    def op_func(a, b):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, a_shape, b_shape, dtype, config=None, tune=False):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        # Index representation is the kernel's own choice: collapse the
        # broadcast into the fewest dims the divmod chain has to walk.
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
            self.a_shape,
            self.b_shape,
        )
        self.out_shape = out_shape
        # What a caller gets back. ``out_shape`` is the coalesced index space, which
        # normalizes a 0-dim operand to one axis; two 0-dim operands broadcast to ``()``.
        self.result_shape = tuple(torch.broadcast_shapes(self.a_shape, self.b_shape))
        self.N_total = math.prod(out_shape)
        self.a_numel = math.prod(self.a_shape)
        self.b_numel = math.prod(self.b_shape)
        self.dtype = dtype
        self._fp8_output_dtype = None
        if _is_fp8(dtype) and self.OUTPUT_DTYPE is None and _fp8_needs_nonsaturating_cast(dtype):
            self._fp8_output_dtype = dtype
            self.output_dtype = torch.float16
        else:
            self.output_dtype = self.OUTPUT_DTYPE or dtype
        self.coalesced_shape = coalesced_shape
        self.a_strides = a_strides
        self.b_strides = b_strides
        self._same_shape = _is_contiguous_same_shape(
            coalesced_shape,
            a_strides,
            b_strides,
        )
        # Validate a config-requested strategy up front so typos raise the
        # same ValueError regardless of dtype (the bool override below
        # otherwise silently accepts an unknown strategy for bool inputs).
        requested = (config or {}).get("strategy")
        if requested is not None and requested not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{requested}', expected one of {self.STRATEGIES}")
        # torch.bool maps to TileLang ``boolx<N>`` for vectorised loads /
        # stores, which the CUDA codegen cannot lower. Force the scalar
        # ``direct`` strategy for bool inputs regardless of caller request.
        bool_input = dtype == torch.bool
        bool_output = torch.bool == self.OUTPUT_DTYPE
        bool_output_needs_scalar = bool_output and dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        if bool_input:
            if requested is not None and requested != "direct":
                warnings.warn(
                    f"BinaryKernel: dtype=torch.bool requires strategy="
                    f"'direct' (TileLang cannot lower vectorised boolx<N> "
                    f"loads); overriding requested strategy={requested!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.strategy = "direct"
        elif bool_output_needs_scalar:
            if requested is not None and requested != "direct":
                warnings.warn(
                    f"BinaryKernel: dtype={dtype} with torch.bool output "
                    f"requires strategy='direct' (TileLang cannot lower "
                    f"vectorised boolx<N> stores for sub-32-bit integer "
                    f"inputs); overriding requested strategy={requested!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.strategy = "direct"
        elif requested is not None:
            # register_copy requires same-shape contiguous inputs (no
            # broadcast); silently downgrade to explicit_parallel when
            # the caller requests register_copy on broadcast shapes.
            if requested == "register_copy" and (not self._same_shape or bool_output):
                self.strategy = "explicit_parallel"
            else:
                self.strategy = requested
        elif self._same_shape and not bool_output:
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
        kernel_output_dtype = (
            self.dtype_to_str(self.OUTPUT_DTYPE) if self.OUTPUT_DTYPE is not None else None
        )
        if self._fp8_output_dtype is not None:
            kernel_output_dtype = _fp8_accum_dtype_str()
        if strategy == "direct":
            return _make_binary_direct(
                self.N_total,
                self.dtype_str,
                effective_op,
                self.coalesced_shape,
                self.a_strides,
                self.b_strides,
                self.a_numel,
                self.b_numel,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_binary_explicit(
                self.N_total,
                self.dtype_str,
                effective_op,
                self.coalesced_shape,
                self.a_strides,
                self.b_strides,
                self.a_numel,
                self.b_numel,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
                num_per_thread=cfg["num_per_thread"],
            )
        elif strategy == "register_copy":
            return _make_binary_register_copy(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
                num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        if _is_fp8(self.dtype):
            return {"strategy": self.strategy, "threads": 256, "num_per_thread": 16}
        npt = _strategy_npt(self.strategy, self.dtype)
        return {"strategy": self.strategy, "threads": 256, "num_per_thread": npt}

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
        return [{"threads": t, "num_per_thread": n} for t in threads_opts for n in npt_opts]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Override to handle known TileLang autotuner fallback failures.

        BinaryKernel JIT functions capture op_func closures that the autotuner
        subprocess cannot serialize.  Newer TileLang binders can also reject
        the autotune wrapper signature.  Catch these errors and fall back to
        the default config so that ``tune=True`` never crashes.
        """
        import warnings

        try:
            super().autotune(warmup=warmup, rep=rep)
        except (AssertionError, Exception) as exc:
            message = str(exc)
            if (
                "not serializable" in message
                or "pickle" in message.lower()
                or "missing a required argument" in message
            ):
                warnings.warn(  # noqa: B028
                    f"{self.__class__.__name__} autotuning failed "
                    f"({message}); falling back to default_config."
                )
                self.config = dict(self.default_config)
            else:
                raise

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        self.config["strategy"] = self.strategy
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, a, b):
        self._require_cuda(a=a, b=b)
        result = self._compiled_fn(_flat(a), _flat(b))
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result.reshape(self.result_shape)


class FusedGatedKernel(Kernel):
    """Template base class for fused gated elementwise kernels.

    Input layout: x has shape (M, 2*N) where x[:, :N] is the gate
    and x[:, N:] is the value. Output: y = activation(gate) * value.

    Subclass must override ``activation_func`` with a static method.

    Args:
        M: Number of rows.
        N: Half the column dimension (output width).
        dtype: Torch dtype.
        config: Optional dict with "strategy", "threads" and "num_per_thread".
            "strategy" is one of "direct", "explicit_parallel"; it selects
            the kernel body at build time.
        tune: Whether to autotune (sweeps "threads" / "num_per_thread"
            within the resolved strategy).
    """

    #: Elementwise work is sized by the tensor shape: an integer operand is
    #: data, and a ``uint8`` ``cond``/``mask`` only selects a result.
    autotune_accepts_random_int_inputs: bool = True

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
        self._fp8_output_dtype = None
        self._kernel_output_dtype = None
        if _is_fp8(dtype) and _fp8_needs_nonsaturating_cast(dtype):
            self._kernel_output_dtype = _fp8_accum_dtype_str()
            self._fp8_output_dtype = dtype
            self.output_dtype = torch.float16
        else:
            self.output_dtype = dtype
        self.strategy = (config or {}).get("strategy") or self.DEFAULT_STRATEGY
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
                self.M,
                self.N,
                self.dtype_str,
                effective_op,
                threads=cfg["threads"],
                output_dtype=self._kernel_output_dtype,
            )
        elif strategy == "explicit_parallel":
            return _make_fused_gated_explicit(
                self.M,
                self.N,
                self.dtype_str,
                effective_op,
                cfg["threads"],
                cfg["num_per_thread"],
                output_dtype=self._kernel_output_dtype,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        if _is_fp8(self.dtype):
            return {"strategy": self.strategy, "threads": 256, "num_per_thread": 16}
        if self.strategy == "explicit_parallel" and self.dtype in (torch.float16, torch.bfloat16):
            # 128x8 keeps block_N=1024 but widens loads to 128-bit and lifts occupancy.
            # Only fp16/bf16 gain the width: fp32 npt=4 already saturates LDG.128.
            return {"strategy": self.strategy, "threads": 128, "num_per_thread": 8}
        npt = _strategy_npt(self.strategy, self.dtype)
        return {"strategy": self.strategy, "threads": 256, "num_per_thread": npt}

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
        return [{"threads": t, "num_per_thread": n} for t in threads_opts for n in npt_opts]

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
        self.config["strategy"] = self.strategy
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(x)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result


class FloatUnaryKernel(UnaryKernel):
    """Unary kernel base for float-only elementwise ops."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES


class FloatPredicateKernel(FloatUnaryKernel):
    """Unary kernel base for float predicates with bool output."""

    DEFAULT_STRATEGY = "explicit_parallel"
    OUTPUT_DTYPE = torch.bool


class LogicalUnaryKernel(UnaryKernel):
    """Unary kernel base for logical predicates with bool output."""

    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = _LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool


class _Uint8StorageUnaryKernel(UnaryKernel):
    """Unary kernel that computes on uint8 but accepts and returns bool.

    Reinterpreting bool storage as uint8 is this backend's requirement, not part
    of the op's semantics, so the reinterpretation happens here.
    """

    DEFAULT_STRATEGY = "register_copy"
    SUPPORTED_DTYPES = (torch.uint8,)

    @property
    def default_config(self) -> dict:
        return {"strategy": self.strategy, "threads": 256, "num_per_thread": 16}

    def forward(self, x):
        as_bool = x.dtype == torch.bool
        if as_bool:
            x = x.view(torch.uint8)
        result = super().forward(x)
        return result.view(torch.bool) if as_bool else result


class _Uint8StorageBinaryKernel(BinaryKernel):
    """Binary kernel that computes on uint8 but accepts and returns bool.

    Reinterpreting bool storage as uint8 is this backend's requirement, not part
    of the op's semantics, so the reinterpretation happens here. Callers pass
    bool tensors and get a bool result; a caller that already holds uint8 is
    passed through unchanged.
    """

    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = (torch.uint8,)

    @property
    def default_config(self) -> dict:
        return {"strategy": self.strategy, "threads": 256, "num_per_thread": 16}

    def forward(self, a, b):
        as_bool = a.dtype == torch.bool
        if as_bool:
            a = a.view(torch.uint8)
        if b.dtype == torch.bool:
            b = b.view(torch.uint8)
        result = super().forward(a, b)
        return result.view(torch.bool) if as_bool else result


class _AlphaScaledBinaryKernel(BinaryKernel):
    """Shared base for ``y = a (op) alpha * b`` kernels.

    Subclasses set ``_combine`` to either addition or subtraction. ``alpha``
    is baked in at kernel construction time (one specialization per distinct
    ``alpha`` value, matching the lru_cache key shape used by the binary
    builders) so the kernel surface stays scalar-free. It is keyword-only so
    the positional ``(dtype, config, tune)`` tail stays uniform.
    """

    @staticmethod
    def _combine(a_scaled, b_scaled):
        raise NotImplementedError

    @staticmethod
    def op_func(a, b):
        raise NotImplementedError(
            "_AlphaScaledBinaryKernel uses a per-instance op_func built from "
            "alpha; use the kernel via __init__ instead of calling op_func."
        )

    def __init__(self, a_shape, b_shape, dtype, config=None, tune=False, *, alpha=1):
        # PyTorch rejects a floating alpha on an integral input; mirror that so
        # the kernel cannot silently truncate alpha through an fp32 cast.
        # Out-of-range integer alphas are NOT rejected — PyTorch wraps them via
        # the input dtype (uint8 alpha=-1 → 255), which T.cast reproduces.
        if dtype in _BITWISE_DTYPES and float(alpha) != float(int(alpha)):
            raise ValueError("alpha must be an integer when input dtype is integral")
        self._alpha = alpha
        super().__init__(a_shape, b_shape, dtype, config=config, tune=tune)

    def _alpha_op_func(self):
        """Build a binary op_func with ``alpha`` baked in.

        Floating inputs route the scalar multiply through fp32 to dodge
        narrow-type literal issues for fp16 / bf16; integer/bool inputs
        keep native integer arithmetic. Following PyTorch, the integral
        alpha is coerced via the input dtype, so out-of-range values
        wrap silently (uint8 alpha=-1 -> 255; bool alpha=2 -> low-bit).
        """
        alpha = self._alpha
        combine = type(self)._combine

        if alpha == 1:
            # Identity multiplier: skip the scalar multiply so the kernel
            # stays byte-identical to the pre-alpha fast path.
            def op_func(a, b):
                return combine(a, b)

            return op_func

        if self.dtype in _BITWISE_DTYPES:
            # Native integer arithmetic. Coerce alpha into the input dtype's
            # representable range in Python before T.cast: TVM rejects a
            # negative literal cast to an unsigned dtype, so reproduce
            # PyTorch's "scalar wraps via the input dtype" semantics here.
            if self.dtype is torch.bool:
                int_alpha = int(bool(alpha))
            else:
                info = torch.iinfo(self.dtype)
                width = info.max - info.min + 1
                int_alpha = int(alpha)
                if int_alpha < info.min or int_alpha > info.max:
                    int_alpha = ((int_alpha - info.min) % width) + info.min

            def op_func(a, b):
                scaled_b = T.cast(int_alpha, a.dtype) * b
                return combine(a, scaled_b)

            return op_func

        def op_func(a, b):
            scaled_b = T.cast(T.cast(alpha, "float32") * T.cast(b, "float32"), a.dtype)
            return combine(a, scaled_b)

        return op_func

    def _get_effective_op_func(self):
        """Inject the alpha-baked op_func into the parent build pipeline."""
        op_func = self._alpha_op_func()
        if self.OUTPUT_DTYPE is not None:
            return op_func
        return _wrap_fp8_accumulation(op_func, self.dtype, self.dtype_str, arity=2)


class ParametricUnaryKernel(Kernel):
    """Shared base for independent parametric elementwise kernels.

    Subclasses must define:
    - ``_builder_fn``: a ``@staticmethod`` returning the ``@lru_cache``-d
      builder function (e.g. ``_make_leaky_relu_kernel``).
    - ``_builder_args(self) -> tuple``: positional args for the builder
      *between* ``N_total`` and the common ``output_dtype, is_fp8, threads,
      npt`` suffix.

    Optional overrides:
    - ``_DEFAULT_THREADS``: class-level default thread count (default 256).
    - ``_NPT_FP8``: npt when dtype is fp8 but not fp32 (default 16).
    - ``_NPT_NON_FP32``: npt for non-fp32, non-fp8 (default 8).
    - ``_skip_fp8_output``: set to ``True`` if the kernel should *not*
      use ``_get_fp8_output_dtypes`` (e.g. Where, which is a pure selection
      op). When True, ``_fp8_output_dtype`` is ``None``.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    _DEFAULT_THREADS: int = 256
    _NPT_FP8: int = 16
    _NPT_NON_FP32: int = 8
    _skip_fp8_output: bool = False

    def __init__(self, N_total, dtype, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        # fp8 output handling
        if self._skip_fp8_output:
            self._fp8_output_dtype = None
        else:
            self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        # Post-fp8 parameter processing (e.g. clamping scalars to output dtype range)
        self._post_init_params()
        # Build the kernel via the subclass-provided builder
        cfg = self.default_config
        builder_kwargs = {
            "is_fp8": _is_fp8(dtype),
            "threads": cfg["threads"],
            "npt": cfg["num_per_thread"],
        }
        if not self._skip_fp8_output:
            builder_kwargs["output_dtype"] = self.dtype_to_str(self.output_dtype)
        self.kernel = self._builder_fn()(
            *self._builder_positional_args(),
            **builder_kwargs,
        )
        self.init_config(config, tune)

    @staticmethod
    def _builder_fn():
        """Return the @lru_cache builder function for this kernel."""
        raise NotImplementedError

    def _builder_positional_args(self) -> tuple:
        """Return all positional args for the builder function.

        Default: ``(N_total, dtype_str, *_builder_args())``.
        Override if the builder has a different parameter order (e.g. PReLU).
        """
        return (self.N_total, self.dtype_str, *self._builder_args())

    def _builder_args(self) -> tuple:
        """Return op-specific positional args (after N_total, dtype_str)."""
        return ()

    def _post_init_params(self):
        """Hook called after fp8 output dtypes are set, before kernel build.

        Override to clamp scalar parameters to the output dtype range (e.g.
        MaskedFill, NanToNum).
        """

    @property
    def default_config(self):
        if self.dtype == torch.float32:
            npt = 4
        elif _is_fp8(self.dtype):
            npt = self._NPT_FP8
        else:
            npt = self._NPT_NON_FP32
        return {"threads": self._DEFAULT_THREADS, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(_flat(x))
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result.reshape(x.shape)
