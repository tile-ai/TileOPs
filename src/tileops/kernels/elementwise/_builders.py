"""TileLang JIT factories for elementwise kernels."""

import functools

import tilelang
import tilelang.language as T

from ._broadcast import _compute_broadcast_offsets, _is_contiguous_same_shape, broadcast_plan_for
from ._op_body import op_func_for


def _broadcast_index_terms(plan_name):
    """Return ``(ndim, divisors, a_strides, b_strides)`` for one broadcast plan.

    Called inside a builder, where these lists and tuples may not be closed over.
    """
    plan = broadcast_plan_for(plan_name)
    ndim = len(plan.coalesced_shape)
    divisors = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        divisors[i] = divisors[i + 1] * plan.coalesced_shape[i + 1]
    return ndim, divisors, plan.a_strides, plan.b_strides


@functools.lru_cache(maxsize=32)
def _make_unary_direct(N, dtype, op_name, output_dtype=None, threads=256):
    """Strategy 1: 1 element per thread."""
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg):
        op_func = op_func_for(op_name)

        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
            with T.Kernel(T.ceildiv(N, threads_arg), threads=threads_arg) as bx:
                for i in T.Parallel(threads_arg):
                    idx = bx * threads_arg + i
                    y[idx] = op_func(x[idx])

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_unary_explicit(N, dtype, op_name, output_dtype=None, threads=256, num_per_thread=8):
    """Strategy 2: N elements per thread via T.Parallel(threads, npt)."""
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        op_func = op_func_for(op_name)
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    y[idx] = op_func(x[idx])

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_unary_regcopy(N, dtype, op_name, output_dtype=None, threads=256, num_per_thread=8):
    """Strategy 3: fragment load -> compute -> fragment store."""
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        op_func = op_func_for(op_name)
        block_size = threads_arg * npt_arg

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


@functools.lru_cache(maxsize=32)
def _make_binary_register_copy(
    N_total,
    dtype,
    op_name,
    output_dtype=None,
    threads=256,
    num_per_thread=8,
):
    """Binary register_copy: fragment load -> compute -> fragment store."""
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[2])
    def kernel(threads, num_per_thread):
        op_func = op_func_for(op_name)
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
    op_name,
    plan_name,
    a_numel,
    b_numel,
    output_dtype=None,
    threads=256,
):
    """Binary direct: 1 element per thread with stride-based broadcast."""
    out_dtype = output_dtype or dtype
    plan = broadcast_plan_for(plan_name)

    if _is_contiguous_same_shape(plan.coalesced_shape, plan.a_strides, plan.b_strides):

        @tilelang.jit(out_idx=[2])
        def kernel(threads):
            op_func = op_func_for(op_name)

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

    @tilelang.jit(out_idx=[2])
    def kernel(threads):
        op_func = op_func_for(op_name)
        ndim, divisors, a_strides, b_strides = _broadcast_index_terms(plan_name)

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
    op_name,
    plan_name,
    a_numel,
    b_numel,
    output_dtype=None,
    threads=256,
    num_per_thread=8,
):
    """Binary explicit_parallel: N elements per thread with stride-based broadcast."""
    out_dtype = output_dtype or dtype
    plan = broadcast_plan_for(plan_name)

    if _is_contiguous_same_shape(plan.coalesced_shape, plan.a_strides, plan.b_strides):

        @tilelang.jit(out_idx=[2])
        def kernel(threads, num_per_thread):
            op_func = op_func_for(op_name)
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

    @tilelang.jit(out_idx=[2])
    def kernel(threads, num_per_thread):
        op_func = op_func_for(op_name)
        ndim, divisors, a_strides, b_strides = _broadcast_index_terms(plan_name)
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
def _make_fused_gated_direct(M, N, dtype, op_name, threads=256, output_dtype=None):
    """FusedGated direct: 1 element per thread."""
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg):
        op_func = op_func_for(op_name)

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
    M, N, dtype, op_name, threads=256, num_per_thread=8, output_dtype=None
):
    """FusedGated explicit_parallel: N elements per thread."""
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        op_func = op_func_for(op_name)
        block_N = threads_arg * npt_arg

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
