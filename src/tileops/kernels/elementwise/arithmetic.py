"""Arithmetic binary kernels, plus the two lerp forms."""

import functools

import tilelang
import tilelang.language as T
import torch

from ._base import (
    _BINARY_FULL_DTYPES,
    _BINARY_NO_BOOL_DTYPES,
    _FLOAT_DTYPES,
    BinaryKernel,
    ParametricUnaryKernel,
    _AlphaScaledBinaryKernel,
    _broadcast_target,
    _expand_flat,
    _fp8_accum_dtype_str,
    _make_binary_direct,
    _make_binary_explicit,
    _make_binary_register_copy,
    _wrap_fp8_accumulation,
)

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


class DivFwdKernel(BinaryKernel):
    """Element-wise division: y = a / b."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        return a / b


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
    """Element-wise power: y = a ** b."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        a_f32 = T.Cast("float32", a)
        b_f32 = T.Cast("float32", b)
        return T.Cast(a.dtype, T.pow(a_f32, b_f32))


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
    constant passed at kernel construction, keeping the binary kernel template.

    Args:
        weight: Scalar interpolation weight (default 0.5). Keyword-only so the
            positional ``(dtype, config, tune)`` tail stays uniform.
    """

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @staticmethod
    def op_func(a, b):
        raise NotImplementedError("Use _make_lerp_op_func(weight) instead")

    def __init__(self, a_shape, b_shape, dtype, config=None, tune=False, *, weight=0.5):
        self._weight = weight
        super().__init__(a_shape, b_shape, dtype, config=config, tune=tune)

    def _build_kernel(self, strategy):
        """Override to inject compile-time weight into op_func."""
        w = self._weight

        def lerp_func(a, b):
            return a + T.cast(w, a.dtype) * (b - a)

        # Wrap with fp8 accumulation via shared helper
        effective_op = _wrap_fp8_accumulation(
            lerp_func,
            self.dtype,
            self.dtype_str,
            arity=2,
        )

        # For e5m2: kernel output is fp16 (non-saturating path)
        kernel_output_dtype = (
            self.dtype_to_str(self.OUTPUT_DTYPE) if self.OUTPUT_DTYPE is not None else None
        )
        if self._fp8_output_dtype is not None:
            kernel_output_dtype = _fp8_accum_dtype_str()

        cfg = self.default_config
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


def _is_float_dtype_str(dtype_str: str) -> bool:
    """Return True for floating-point TileLang dtype strings.

    TileLang IR exposes operand dtypes only as strings (``"float16"``,
    ``"bfloat16"``, ``"float32"``, ``"float8_e4m3fn"`` ...), so prefix
    matching is the established convention for float detection inside
    ``op_func`` kernel bodies. All TileLang float dtype names start
    with ``"float"`` or ``"bfloat"``; integer / bool dtype names
    (``"int*"``, ``"uint*"``, ``"bool"``) do not.
    """
    return dtype_str.startswith(("float", "bfloat"))


class MaximumFwdKernel(BinaryKernel):
    """Element-wise maximum: y = max(a, b).

    For float dtypes, matches torch.maximum semantics:
    - If either operand is NaN, the result is NaN.
    - maximum(+0.0, -0.0) = +0.0 (IEEE 754 signed-zero).

    For integer / bool dtypes (no NaN representation), uses ``T.max``
    directly without the NaN guards.

    Performance (float path): uses T.max for the fast path (correct
    signed-zero on CUDA -- fmaxf returns +0 for max(+0,-0)) plus two
    isnan guards for NaN propagation. Total IR: 1 max + 2 fp32 casts +
    2 isnan + 2 select.
    """

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES

    @staticmethod
    def op_func(a, b):
        result = T.max(a, b)
        if not _is_float_dtype_str(str(a.dtype)):
            # Integer / bool: no NaN representation, T.max is sufficient.
            return result
        # Float path: T.max handles signed-zero correctly but does NOT
        # propagate NaN -- it returns the non-NaN operand. Cast to fp32
        # for isnan (bfloat16 lacks native isnan).
        a_is_nan = T.isnan(T.Cast("float32", a))
        b_is_nan = T.isnan(T.Cast("float32", b))
        result = T.if_then_else(b_is_nan, b, result)
        result = T.if_then_else(a_is_nan, a, result)
        return result


class MinimumFwdKernel(BinaryKernel):
    """Element-wise minimum: y = min(a, b).

    For float dtypes, matches torch.minimum semantics:
    - If either operand is NaN, the result is NaN.
    - minimum(-0.0, +0.0) = -0.0 (IEEE 754 signed-zero).

    For integer / bool dtypes (no NaN representation), uses ``T.min``
    directly without the NaN guards.

    Performance (float path): uses T.min for the fast path (correct
    signed-zero on CUDA -- fminf returns -0 for min(-0,+0)) plus two
    isnan guards for NaN propagation. See MaximumFwdKernel for full
    rationale.
    """

    SUPPORTED_DTYPES = _BINARY_FULL_DTYPES

    @staticmethod
    def op_func(a, b):
        result = T.min(a, b)
        if not _is_float_dtype_str(str(a.dtype)):
            return result
        a_is_nan = T.isnan(T.Cast("float32", a))
        b_is_nan = T.isnan(T.Cast("float32", b))
        result = T.if_then_else(b_is_nan, b, result)
        result = T.if_then_else(a_is_nan, a, result)
        return result


@functools.lru_cache(maxsize=32)
def _make_lerp_tensor_kernel(N, dtype, output_dtype=None, is_fp8=False, threads=256, npt=8):
    """Build Tensor-weight lerp kernel: out = a + weight * (b - a).

    ``LerpTensorFwdKernel.forward`` broadcasts ``input`` / ``end`` / ``weight``
        to the output shape and flattens them, so this PrimFunc sees three
        contiguous 1-D tensors of size ``N``. Computation is performed in the input dtype for fp16 /
        bfloat16 / float32 (the only dtypes the manifest declares); the fp8
        path is unreachable here because the kernel's ``SUPPORTED_DTYPES``
        excludes fp8.

        Uses the register-fragment load -> compute -> fragment store strategy
        (matches the non-fp8 ``_make_where_kernel`` layout) so all three
        inputs and the output share the same vectorized memory access path.
    """
    del is_fp8  # fp8 is not in the manifest contract for this op
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),
            b: T.Tensor((N,), dtype),
            w: T.Tensor((N,), dtype),
            out: T.Tensor((N,), out_dtype),
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


class LerpTensorFwdKernel(ParametricUnaryKernel):
    """Tensor-weight lerp: out = input + weight * (end - input).

        Implements the Tensor-weight overload of ``torch.lerp`` —
        ``torch.lerp(input, end, weight: Tensor)`` — where all three operands
    are float tensors of the same dtype, broadcast together and flattened
        by ``forward``.

        Manifest declares ``float16 | bfloat16 | float32``; fp8 is rejected
        at construction. ``forward`` takes the manifest shapes and broadcasts
        them to ``N_total`` itself.
    """

    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    _DEFAULT_THREADS = 512
    _skip_fp8_output = True

    @staticmethod
    def _builder_fn():
        return _make_lerp_tensor_kernel

    def forward(self, a, b, w):
        self._require_cuda(a=a, b=b, w=w)
        out_shape = _broadcast_target(a, b, w)
        result = self._compiled_fn(
            _expand_flat(a, out_shape),
            _expand_flat(b, out_shape),
            _expand_flat(w, out_shape),
        )
        return result.reshape(out_shape)
