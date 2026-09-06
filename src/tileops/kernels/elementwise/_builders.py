"""TileLang JIT factories for elementwise kernels."""

import functools

import tilelang
import tilelang.language as T

from ._broadcast import (
    _compute_broadcast_offsets,
    _is_contiguous_same_shape,
    broadcast_plan_for,
    row_broadcast_split,
    row_tile_leaves_tail,
)
from ._op_body import op_func_for

# Packing the leftover T columns of a W-wide block saves rows * (W - T) idle lane
# slots and spends rows * T index chains, so it pays while T / (W - T) stays under
# the lane slots one chain is worth. At W // 4 that ratio is one chain per three
# idle slots.
_TAIL_PACK_RATIO = 4


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


def _row_broadcast_prim(
    N_total,
    dtype,
    out_dtype,
    op_name,
    plan_name,
    a_numel,
    b_numel,
    threads,
    num_per_thread,
    stage=False,
):
    """PrimFunc for a broadcast whose innermost coalesced dim reads at stride 0 or 1.

    One grid row per output row: the divmod chain that locates each operand
    runs once per block, and the inner loop indexes affinely, so it vectorizes.
    The generic path pays that chain per element and every access is a gather.
    A ragged inner extent splits at trace time. Full blocks always run unguarded.
    Where *num_per_thread* divides the row, the grid indexes vectors and leaves no
    columns over. Otherwise they go one of two ways, scalar either way: packed end
    to end across rows into blocks of their own, or left as one guarded block per row.

    *stage* moves the full blocks through fragments, which keeps the copies wide
    when the body cannot vectorize. It is opt-in because a body that does
    vectorize is slower for the round trip.
    """
    op_func = op_func_for(op_name)
    ndim, divisors, a_strides, b_strides = _broadcast_index_terms(plan_name)
    plan = broadcast_plan_for(plan_name)
    inner = plan.coalesced_shape[-1]
    rows = N_total // inner
    a_inner = plan.a_strides[-1]
    b_inner = plan.b_strides[-1]
    block_cols = threads * num_per_thread
    full_blocks = inner // block_cols
    exact = full_blocks * block_cols == inner
    tail_cols = inner - full_blocks * block_cols
    tail_slots = rows * tail_cols
    body_blocks = full_blocks * rows
    tail_blocks = -(-tail_slots // block_cols)
    pack_tail = not exact and full_blocks > 0 and tail_cols <= block_cols // _TAIL_PACK_RATIO
    # Columns past the last full block go one element per thread. Indexing
    # vectors rather than rows leaves none, where a vector stays inside a row.
    vecs_per_row = inner // num_per_thread if inner % num_per_thread == 0 else 0
    total_vecs = rows * vecs_per_row
    grid_vecs = -(-total_vecs // threads)
    flat_grid = (
        not exact
        and num_per_thread > 1
        and not row_tile_leaves_tail(inner, rows, threads, num_per_thread, stage)
    )

    @T.macro
    def write_col(a, b, y, a_base, b_base, by, col):
        y[by * inner + col] = op_func(a[a_base + col * a_inner], b[b_base + col * b_inner])

    @T.macro
    def write_block_staged(a, b, y, a_base, b_base, by, bx):
        """One full block, read and written a fragment at a time.

        An operand at stride 0 is one value for the whole row, so it is read
        into a register rather than staged across a fragment it would fill with
        copies of itself. Reading it at the use site instead leaves the load in
        the element loop, and with it whatever the body wraps around it -- for
        maximum and minimum that is a NaN test, which is the row's property and
        not the element's.
        """
        y_reg = T.alloc_fragment((block_cols,), out_dtype)
        col0 = bx * block_cols
        if a_inner:
            a_reg = T.alloc_fragment((block_cols,), dtype)
            T.copy(a[a_base + col0 : a_base + col0 + block_cols], a_reg)
        else:
            a_held = T.alloc_local((1,), dtype)
            a_held[0] = a[a_base]
        if b_inner:
            b_reg = T.alloc_fragment((block_cols,), dtype)
            T.copy(b[b_base + col0 : b_base + col0 + block_cols], b_reg)
        else:
            b_held = T.alloc_local((1,), dtype)
            b_held[0] = b[b_base]
        for i, j in T.Parallel(threads, num_per_thread):
            idx = i * num_per_thread + j
            y_reg[idx] = op_func(
                a_reg[idx] if a_inner else a_held[0],
                b_reg[idx] if b_inner else b_held[0],
            )
        T.copy(y_reg, y[by * inner + col0 : by * inner + col0 + block_cols])

    @T.macro
    def write_held_block(a, b, y, a_base, b_base, by, bxc):
        """One full block with the row's stride-0 operand read into a register.

        An operand at stride 0 is one value for the whole row. Left as a
        subscript it is read again at every element, and so is the arithmetic
        the body wraps around it -- for maximum and minimum that is a NaN test,
        which is the row's and not the element's.
        """
        held = T.alloc_local((1,), dtype)
        held[0] = a[a_base] if not a_inner else b[b_base]
        for i, j in T.Parallel(threads, num_per_thread):
            col = (bxc * threads + i) * num_per_thread + j
            if a_inner:
                y[by * inner + col] = op_func(a[a_base + col * a_inner], held[0])
            else:
                y[by * inner + col] = op_func(held[0], b[b_base + col * b_inner])

    @T.macro
    def write_full_block(a, b, y, a_base, b_base, by, bxc):
        if stage:
            write_block_staged(a, b, y, a_base, b_base, by, bxc)
        elif a_inner and b_inner:
            for i, j in T.Parallel(threads, num_per_thread):
                write_col(a, b, y, a_base, b_base, by, (bxc * threads + i) * num_per_thread + j)
        else:
            write_held_block(a, b, y, a_base, b_base, by, bxc)

    if flat_grid:
        exact_grid = grid_vecs * threads == total_vecs

        @T.macro
        def write_flat_vec(a, b, y, v, j):
            """The *j*-th column of vector *v*, wherever in the grid that lands."""
            row = v // vecs_per_row
            col = (v % vecs_per_row) * num_per_thread + j
            a_base, b_base = _compute_broadcast_offsets(
                row * inner, ndim, divisors, a_strides, b_strides
            )
            write_col(a, b, y, a_base, b_base, row, col)

        @T.macro
        def write_flat_staged(a, b, y, bx):
            """One block of vectors, read and written a fragment at a time.

            The copy loops carry no body, so they keep the full width where the
            body would have scalarised them. A block spans rows, so the operand
            a row broadcasts is one register per thread, not one per block.
            """
            y_reg = T.alloc_fragment((block_cols,), out_dtype)
            if a_inner:
                a_reg = T.alloc_fragment((block_cols,), dtype)
            else:
                a_held = T.alloc_fragment((threads,), dtype)
            if b_inner:
                b_reg = T.alloc_fragment((block_cols,), dtype)
            else:
                b_held = T.alloc_fragment((threads,), dtype)
            for i, j in T.Parallel(threads, num_per_thread):
                v = bx * threads + i
                col = (v % vecs_per_row) * num_per_thread + j
                a_base, b_base = _compute_broadcast_offsets(
                    (v // vecs_per_row) * inner, ndim, divisors, a_strides, b_strides
                )
                if a_inner:
                    a_reg[i * num_per_thread + j] = a[a_base + col * a_inner]
                if b_inner:
                    b_reg[i * num_per_thread + j] = b[b_base + col * b_inner]
            for i in T.Parallel(threads):
                a_base, b_base = _compute_broadcast_offsets(
                    ((bx * threads + i) // vecs_per_row) * inner,
                    ndim,
                    divisors,
                    a_strides,
                    b_strides,
                )
                if not a_inner:
                    a_held[i] = a[a_base]
                if not b_inner:
                    b_held[i] = b[b_base]
            for i, j in T.Parallel(threads, num_per_thread):
                k = i * num_per_thread + j
                y_reg[k] = op_func(
                    a_reg[k] if a_inner else a_held[i],
                    b_reg[k] if b_inner else b_held[i],
                )
            for i, j in T.Parallel(threads, num_per_thread):
                v = bx * threads + i
                y[(v // vecs_per_row) * inner + (v % vecs_per_row) * num_per_thread + j] = y_reg[
                    i * num_per_thread + j
                ]

        @T.prim_func
        def flat_main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(grid_vecs, threads=threads) as bx:
                if stage:
                    write_flat_staged(a, b, y, bx)
                else:
                    for i, j in T.Parallel(threads, num_per_thread):
                        v = bx * threads + i
                        if exact_grid:
                            write_flat_vec(a, b, y, v, j)
                        else:
                            with T.If(v < total_vecs):  # noqa: SIM117
                                with T.Then():
                                    write_flat_vec(a, b, y, v, j)

        return flat_main

    if pack_tail:

        @T.prim_func
        def packed_main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(body_blocks + tail_blocks, threads=threads) as bx:  # noqa: SIM117
                with T.If(bx < body_blocks):
                    with T.Then():
                        by = bx // full_blocks
                        bxc = bx % full_blocks
                        a_base, b_base = _compute_broadcast_offsets(
                            by * inner, ndim, divisors, a_strides, b_strides
                        )
                        write_full_block(a, b, y, a_base, b_base, by, bxc)
                    with T.Else():
                        for i, j in T.Parallel(threads, num_per_thread):
                            slot = (bx - body_blocks) * block_cols + i * num_per_thread + j
                            with T.If(slot < tail_slots):  # noqa: SIM117
                                with T.Then():
                                    tail_row = slot // tail_cols
                                    ta_base, tb_base = _compute_broadcast_offsets(
                                        tail_row * inner, ndim, divisors, a_strides, b_strides
                                    )
                                    write_col(
                                        a,
                                        b,
                                        y,
                                        ta_base,
                                        tb_base,
                                        tail_row,
                                        full_blocks * block_cols + slot % tail_cols,
                                    )

        return packed_main

    @T.prim_func
    def main(
        a: T.Tensor((a_numel,), dtype),
        b: T.Tensor((b_numel,), dtype),
        y: T.Tensor((N_total,), out_dtype),
    ):
        with T.Kernel(T.ceildiv(inner, block_cols), rows, threads=threads) as (bx, by):
            a_base, b_base = _compute_broadcast_offsets(
                by * inner, ndim, divisors, a_strides, b_strides
            )
            if exact:
                write_full_block(a, b, y, a_base, b_base, by, bx)
            else:
                with T.If(bx < full_blocks):
                    with T.Then():
                        write_full_block(a, b, y, a_base, b_base, by, bx)
                    with T.Else():
                        for i, j in T.Parallel(threads, num_per_thread):
                            col = (bx * threads + i) * num_per_thread + j
                            with T.If(col < inner):  # noqa: SIM117
                                with T.Then():
                                    write_col(a, b, y, a_base, b_base, by, col)

    return main


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

    if row_broadcast_split(plan.coalesced_shape, plan.a_strides, plan.b_strides):

        @tilelang.jit(out_idx=[2])
        def kernel(threads):
            return _row_broadcast_prim(
                N_total, dtype, out_dtype, op_name, plan_name, a_numel, b_numel, threads, 1
            )

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
    stage=False,
):
    """Binary explicit_parallel: N elements per thread with stride-based broadcast."""
    out_dtype = output_dtype or dtype
    plan = broadcast_plan_for(plan_name)

    if row_broadcast_split(plan.coalesced_shape, plan.a_strides, plan.b_strides):

        @tilelang.jit(out_idx=[2])
        def kernel(threads, num_per_thread):
            return _row_broadcast_prim(
                N_total,
                dtype,
                out_dtype,
                op_name,
                plan_name,
                a_numel,
                b_numel,
                threads,
                num_per_thread,
                stage=stage,
            )

        return kernel

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
def _make_fused_gated_direct(M, N, dtype, op_name, threads=256):
    """FusedGated direct: 1 element per thread."""

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg):
        op_func = op_func_for(op_name)

        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, threads_arg), M, threads=threads_arg) as (bx, by):
                for i in T.Parallel(threads_arg):
                    col = bx * threads_arg + i
                    gate = x[by, col]
                    value = x[by, N + col]
                    y[by, col] = op_func(gate, value)

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_fused_gated_explicit(M, N, dtype, op_name, threads=256, num_per_thread=8):
    """FusedGated explicit_parallel: N elements per thread."""

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        op_func = op_func_for(op_name)
        block_N = threads_arg * npt_arg

        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, block_N), M, threads=threads_arg) as (bx, by):
                for i, j in T.Parallel(threads_arg, npt_arg):
                    col = (bx * threads_arg + i) * npt_arg + j
                    gate = x[by, col]
                    value = x[by, N + col]
                    y[by, col] = op_func(gate, value)

        return main

    return kernel
