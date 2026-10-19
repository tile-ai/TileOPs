"""INT8 NT GEMM with per-token / per-channel dequantization in the epilogue, for SM89 / SM90.

``d = (a @ b^T) * scale_a * scale_b + bias``: ``a`` is ``[m, k]`` and ``b`` is ``[n, k]``,
both int8; tensor-core instructions accumulate the product in int32. ``scale_a`` is one scale for the
tensor or one per row of ``a``, ``scale_b`` one for the tensor or one per row of ``b``. The
epilogue converts the int32 sum to fp32, multiplies by the row's and the column's scale,
adds the bias in fp32 and rounds once to the output dtype.
"""

import functools
from abc import abstractmethod
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops.kernels.gemm.call_spec import GemmINT8W8A8Call, GemmINT8W8A8FwdInterface
from tileops.kernels.gemm.heuristics import int8_w8a8_config
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.utils import get_sm_count

__all__ = ["GemmINT8W8A8Kernel", "GemmINT8W8A8PingPongKernel", "GemmINT8W8A8SimpleKernel"]


@T.macro
def _tile_id(flat_id, mt, nt, *, group_size_m: int, num_pid_m: int, num_pid_n: int):
    """Write the grouped-rasterization (m, n) tile coordinates of a flat tile id."""
    gin = T.int32(group_size_m * num_pid_n)
    gid = flat_id // gin
    first_m = gid * T.int32(group_size_m)
    gsize = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
    mt[0] = first_m + (flat_id % gin) % gsize
    nt[0] = (flat_id % gin) // gsize


@T.macro
def _load_stage(
    a,
    b,
    a_top,
    a_bot,
    b_smem,
    ab_full,
    slot,
    kb,
    m_start,
    n_start,
    *,
    top_rows: int,
    bot_rows: int,
    half_m: int,
    block_n: int,
    block_k: int,
):
    """One K step of the producer: the step's ``a`` boxes and its ``b`` box.

    ``top_rows`` and ``bot_rows`` are the rows of ``a`` each consumer's box loads, 0 for
    a consumer that does not exist. A box past the end of a tensor is zero-filled by TMA,
    so the N and K tails need no predicate: a zero K tail adds nothing to the int32 sum.
    """
    ks = kb * block_k
    T.tma_copy(
        a[m_start : m_start + top_rows, ks : ks + block_k],
        a_top[slot, 0:top_rows, :],
        barrier=ab_full[slot],
    )
    if bot_rows > 0:
        T.tma_copy(
            a[m_start + half_m : m_start + half_m + bot_rows, ks : ks + block_k],
            a_bot[slot, 0:bot_rows, :],
            barrier=ab_full[slot],
        )
    T.tma_copy(
        b[n_start : n_start + block_n, ks : ks + block_k], b_smem[slot, :, :], barrier=ab_full[slot]
    )
    T.barrier_arrive(ab_full[slot])


@T.macro
def _mma_step(a_smem, b_smem, ab_full, ab_empty, acc, prev, slot, phase, ki):
    """One K step of a consumer: WGMMA accumulates, the previous slot is released.

    The release trails by one step because step ``ki``'s WGMMA still reads slot ``ki``;
    ``prev`` carries the slot the next release belongs to.
    """
    T.barrier_wait(ab_full[slot], phase)
    T.wgmma_gemm(
        a_smem[slot, :, :],
        b_smem[slot, :, :],
        acc,
        transpose_B=True,
        policy=T.GemmWarpPolicy.FullRow,
        clear_accum=(ki == 0),
    )
    if ki > 0:
        T.wait_wgmma(1)
        T.barrier_arrive(ab_empty[prev[0]])
    prev[0] = slot


@T.macro
def _mma_drain(ab_empty, acc, prev, *, num_regs: int):
    """Wait for a consumer's last WGMMA and release the slot it read."""
    T.wait_wgmma(0)
    T.barrier_arrive(ab_empty[prev[0]])
    T.warpgroup_fence_operand(acc, num_regs=num_regs)


@T.macro
def _load_scales(
    scale_a,
    scale_b,
    bias,
    sa_f,
    sb_f,
    bias_f,
    m_start,
    n_start,
    *,
    m: int,
    n: int,
    half_m: int,
    block_n: int,
    per_token: bool,
    per_channel: bool,
    has_bias: bool,
):
    """Read a tile's row scales, column scales and bias into registers, the bias in fp32,
    once a tile rather than once an element."""
    for i in T.Parallel(half_m):
        sa_f[i] = scale_a[T.min(m_start + i, m - 1), 0] if per_token else scale_a[0, 0]
    for j in T.Parallel(block_n):
        sb_f[j] = scale_b[0, T.min(n_start + j, n - 1)] if per_channel else scale_b[0, 0]
    if has_bias:
        for j in T.Parallel(block_n):
            bias_f[j] = T.cast(bias[T.min(n_start + j, n - 1)], "float32")


@T.macro
def _dequantize(
    acc, out, sa_f, sb_f, bias_f, *, half_m: int, block_n: int, has_bias: bool, out_dtype: str
):
    """``out = cast(float(acc) * scale_a[row] * scale_b[col] + bias[col])``, in fp32."""
    if has_bias:
        for i, j in T.Parallel(half_m, block_n):
            out[i, j] = T.cast(
                T.cast(acc[i, j], "float32") * sa_f[i] * sb_f[j] + bias_f[j], out_dtype
            )
    else:
        for i, j in T.Parallel(half_m, block_n):
            out[i, j] = T.cast(T.cast(acc[i, j], "float32") * sa_f[i] * sb_f[j], out_dtype)


@T.macro
def _store_tile(
    c,
    out,
    c_smem,
    m_start,
    n_start,
    rows,
    cols,
    *,
    bar: int,
    half_m: int,
    block_n: int,
    stage_store: bool,
):
    """Store one consumer's cast tile: one TMA box when full, elements otherwise.

    ``stage_store`` is False where no row tile can be full -- ``m`` below one consumer's
    rows -- and the staging tile is then not allocated.
    """
    if not stage_store:
        if rows > T.int32(0):
            for i, j in T.Parallel(half_m, block_n):
                if i < rows and j < cols:
                    c[m_start + i, n_start + j] = out[i, j]
    elif rows == T.int32(half_m) and cols == T.int32(block_n):
        T.sync_threads(barrier_id=bar, arrive_count=128)
        T.copy(out, c_smem)
        T.fence_proxy_async()
        T.sync_threads(barrier_id=bar, arrive_count=128)
        T.copy(c_smem, c[m_start, n_start])
    elif rows > T.int32(0):
        for i, j in T.Parallel(half_m, block_n):
            if i < rows and j < cols:
                c[m_start + i, n_start + j] = out[i, j]


@functools.lru_cache(maxsize=32)
def _gemm_int8_ws_kernel(
    m: int,
    n: int,
    k: int,
    out_dtype: str,
    per_token: bool,
    per_channel: bool,
    has_bias: bool,
    *,
    sm_count: int,
) -> Callable:
    """Warp-specialized persistent INT8 GEMM over 64-row tiles: one producer, one consumer.

    The producer warpgroup issues the TMA loads into a ``num_stages`` ring; the consumer
    warpgroup runs WGMMA over it, accumulating the whole K axis in int32, and the
    persistent grid sweeps a grouped tile order for L2 reuse. The scales and the bias land
    in the epilogue, read before the tile's mainloop.

    Where ``a`` has fewer rows than a tile, its box loads only the rows ``a`` has. TMA
    fills the rest of a full box with zeros at a cost a real row does not have (measured:
    34.0 us at ``m = 1`` against 22.8 us at ``m = 64``, 4096 x 14336, one tile shape); the
    shared rows a box leaves unwritten feed only accumulator rows past ``m``, which are
    never stored, and int8 has no value that could poison the rows that are.

    Args:
        m: Rows of ``a`` and of the output.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim, a multiple of 16 (TMA's 16-byte unit).
        out_dtype: ``"bfloat16"`` or ``"float16"``; the bias carries it too.
        per_token: ``scale_a`` is ``[m, 1]``; otherwise ``[1, 1]``.
        per_channel: ``scale_b`` is ``[1, n]``; otherwise ``[1, 1]``.
        has_bias: Whether the compiled function reads a ``[n]`` bias.
        sm_count: Persistent grid width, the device SM count; part of the cache key so a
            kernel built for one GPU is never reused on another.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, num_stages, group_size_m,
        block_k)`` returns the compiled ``prim_func``.
    """
    # WGMMA's M extent, one consumer warpgroup's rows.
    block_m = 64

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_int8_ws_func(
        block_n: int = 64,
        num_stages: int = 8,
        group_size_m: int = 8,
        block_k: int = 128,
    ) -> Callable:
        # The consumer's 64 x block_n int32 accumulator, in registers per thread.
        nr = (block_m * block_n) // 128
        a_rows = min(m, block_m)
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        grid = min(sm_count, total_tiles)
        max_waves = -(-total_tiles // grid)
        k_iters = -(-k // block_k)
        sa_shape = (m, 1) if per_token else (1, 1)
        sb_shape = (1, n) if per_channel else (1, 1)
        bias_shape = (n,) if has_bias else (1,)
        stage_store = m >= block_m
        store_rows = block_m if stage_store else 1
        # The consumer's epilogue staging barrier; ids below 8 belong to TileLang.
        bar = 8

        @T.prim_func
        def _gemm_int8_ws_main(
            a: T.Tensor((m, k), "int8"),  # type: ignore
            b: T.Tensor((n, k), "int8"),  # type: ignore
            scale_a: T.Tensor(sa_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(sb_shape, "float32"),  # type: ignore
            bias: T.Tensor(bias_shape, out_dtype),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(grid, threads=256) as (pid,):
                a_smem = T.alloc_shared((num_stages, block_m, block_k), "int8")
                b_smem = T.alloc_shared((num_stages, block_n, block_k), "int8")
                c_smem = T.alloc_shared((store_rows, block_n), out_dtype)
                acc = T.alloc_fragment((block_m, block_n), "int32")
                out = T.alloc_fragment((block_m, block_n), out_dtype)
                sa_f = T.alloc_fragment((block_m,), "float32")
                sb_f = T.alloc_fragment((block_n,), "float32")
                bias_f = T.alloc_fragment((block_n,), "float32")

                layouts = {
                    a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                    b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                }
                if stage_store:
                    layouts[c_smem] = tilelang.layout.make_swizzled_layout(c_smem)
                T.annotate_layout(layouts)

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons = T.alloc_var("int32", init=0)
                prev = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid) * w + pid
                        if flat_id < total_tiles:
                            _tile_id(
                                flat_id,
                                mt,
                                nt,
                                group_size_m=group_size_m,
                                num_pid_m=num_pid_m,
                                num_pid_n=num_pid_n,
                            )
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                _load_stage(
                                    a,
                                    b,
                                    a_smem,
                                    a_smem,
                                    b_smem,
                                    ab_full,
                                    slot,
                                    ki,
                                    m_start,
                                    n_start,
                                    top_rows=a_rows,
                                    bot_rows=0,
                                    half_m=block_m,
                                    block_n=block_n,
                                    block_k=block_k,
                                )
                                gi_prod = gi_prod + 1
                else:
                    T.inc_max_nreg(232)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid) * w + pid
                        if flat_id < total_tiles:
                            _tile_id(
                                flat_id,
                                mt,
                                nt,
                                group_size_m=group_size_m,
                                num_pid_m=num_pid_m,
                                num_pid_n=num_pid_n,
                            )
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            _load_scales(
                                scale_a,
                                scale_b,
                                bias,
                                sa_f,
                                sb_f,
                                bias_f,
                                m_start,
                                n_start,
                                m=m,
                                n=n,
                                half_m=block_m,
                                block_n=block_n,
                                per_token=per_token,
                                per_channel=per_channel,
                                has_bias=has_bias,
                            )
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons % num_stages
                                _mma_step(
                                    a_smem,
                                    b_smem,
                                    ab_full,
                                    ab_empty,
                                    acc,
                                    prev,
                                    slot,
                                    (gi_cons // num_stages) & 1,
                                    ki,
                                )
                                gi_cons = gi_cons + 1
                            _mma_drain(ab_empty, acc, prev, num_regs=nr)
                            _dequantize(
                                acc,
                                out,
                                sa_f,
                                sb_f,
                                bias_f,
                                half_m=block_m,
                                block_n=block_n,
                                has_bias=has_bias,
                                out_dtype=out_dtype,
                            )
                            _store_tile(
                                c,
                                out,
                                c_smem,
                                m_start,
                                n_start,
                                T.min(T.int32(block_m), T.int32(m) - m_start),
                                T.min(T.int32(block_n), T.int32(n) - n_start),
                                bar=bar,
                                half_m=block_m,
                                block_n=block_n,
                                stage_store=stage_store,
                            )

        return _gemm_int8_ws_main

    return _gemm_int8_ws_func


@functools.lru_cache(maxsize=32)
def _gemm_int8_pingpong_kernel(
    m: int,
    n: int,
    k: int,
    out_dtype: str,
    per_token: bool,
    per_channel: bool,
    has_bias: bool,
    *,
    sm_count: int,
) -> Callable:
    """Persistent ping-pong INT8 GEMM: two consumer warpgroups take whole tiles in turn.

    One producer warpgroup fills one TMA ring in tile order; consumer ``w % 2`` takes a
    CTA's ``w``-th tile, a whole 128 x 128 tile held as two 64-row int32 accumulators. The consumers' mainloops alternate on the tensor cores -- a consumer
    waits on its turn barrier before its first WGMMA and, after issuing its last, arrives
    on the other's -- so one consumer's epilogue runs under the other's mainloop. That is
    what a short K axis needs, where an epilogue is a large share of a tile.

    The epilogue holds no output fragment: it stages the tile's column scales and bias in
    shared memory and dequantizes each 64-row half straight into the staging tile the TMA
    store reads (measured: 151.3 us against 155.2 us with an output fragment and the column
    vectors in registers, 4096 x 6144 x 4096). Every tile leaves through the TMA store,
    which drops what an M / N tail puts past the output; K tails are TMA zero-filled.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(num_stages, group_size_m)`` returns
        the compiled ``prim_func``.
    """
    block_m = 128
    half_m = 64
    # Wider tiles measured no faster: TileLang splits a 192-wide WGMMA into three n64 ones,
    # and through a native-width helper 160-192 came within run-to-run noise of 128.
    block_n = 128
    block_k = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_int8_pingpong_func(num_stages: int = 4, group_size_m: int = 16) -> Callable:
        nr = (half_m * block_n) // 128
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        max_waves = -(-total_tiles // sm_count)
        # Each consumer takes every other wave.
        consumer_waves = -(-max_waves // 2)
        k_iters = -(-k // block_k)
        sa_shape = (m, 1) if per_token else (1, 1)
        sb_shape = (1, n) if per_channel else (1, 1)
        bias_shape = (n,) if has_bias else (1,)
        # One named barrier per consumer for its epilogue staging; ids below 8 belong to
        # TileLang.
        bar_0, bar_1 = 8, 9

        @T.macro
        def half_mma(a_ring, b_ring, slot, acc, ki):
            T.wgmma_gemm(
                a_ring[slot, :, :],
                b_ring[slot, :, :],
                acc,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=(ki == 0),
            )

        @T.macro
        def store_half(c, acc, sa_f, sb_s, bias_s, c_smem, row0, n_start, bar):
            """Dequantize one 64-row half into c_smem and store it."""
            # The previous store may still read c_smem; waiting here lets the first one
            # overlap the mainloop that ran since.
            T.tma_store_wait(0)
            T.sync_threads(barrier_id=bar, arrive_count=128)
            if has_bias:
                for i, j in T.Parallel(half_m, block_n):
                    c_smem[i, j] = T.cast(
                        T.cast(acc[i, j], "float32") * sa_f[i] * sb_s[j] + bias_s[j], out_dtype
                    )
            else:
                for i, j in T.Parallel(half_m, block_n):
                    c_smem[i, j] = T.cast(
                        T.cast(acc[i, j], "float32") * sa_f[i] * sb_s[j], out_dtype
                    )
            T.fence_proxy_async()
            T.sync_threads(barrier_id=bar, arrive_count=128)
            T.tma_copy(c_smem, c[row0 : row0 + half_m, n_start : n_start + block_n])

        @T.macro
        def consumer(
            wg,
            pid,
            c,
            scale_a,
            scale_b,
            bias,
            a_top,
            a_bot,
            b_smem,
            ab_full,
            ab_empty,
            turn,
            acc_top,
            acc_bot,
            sa_top,
            sa_bot,
            sb_s,
            bias_s,
            c_smem,
            mt,
            nt,
            prev,
            bar,
        ):
            for j in T.serial(consumer_waves):
                w = 2 * j + wg
                flat_id = T.int32(sm_count) * w + pid
                if flat_id < total_tiles:
                    _tile_id(
                        flat_id,
                        mt,
                        nt,
                        group_size_m=group_size_m,
                        num_pid_m=num_pid_m,
                        num_pid_n=num_pid_n,
                    )
                    m_start = mt[0] * block_m
                    n_start = nt[0] * block_n
                    # This consumer's j-th turn on the tensor cores.
                    T.barrier_wait(turn[wg], j & 1)
                    for ki in T.Pipelined(k_iters, num_stages=0):
                        stage = w * k_iters + ki
                        slot = stage % num_stages
                        T.barrier_wait(ab_full[slot], (stage // num_stages) & 1)
                        half_mma(a_top, b_smem, slot, acc_top, ki)
                        half_mma(a_bot, b_smem, slot, acc_bot, ki)
                        if ki > 0:
                            T.wait_wgmma(2)
                            T.barrier_arrive(ab_empty[prev[0]])
                        prev[0] = slot
                    # The other consumer's next tile may start its WGMMAs; they queue behind
                    # this tile's last.
                    if flat_id + T.int32(sm_count) < total_tiles:
                        T.barrier_arrive(turn[1 - wg])
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[prev[0]])
                    T.warpgroup_fence_operand(acc_top, num_regs=nr)
                    T.warpgroup_fence_operand(acc_bot, num_regs=nr)
                    for i in T.Parallel(half_m):
                        sa_top[i] = (
                            scale_a[T.min(m_start + i, m - 1), 0] if per_token else scale_a[0, 0]
                        )
                        sa_bot[i] = (
                            scale_a[T.min(m_start + half_m + i, m - 1), 0]
                            if per_token
                            else scale_a[0, 0]
                        )
                    # The tile's column scales and bias, staged in shared memory: fragments
                    # indexed by column would hold block_n / 4 registers each beside the
                    # accumulators. The last tile's dequantization is done with them once
                    # this barrier passes.
                    T.sync_threads(barrier_id=bar, arrive_count=128)
                    for jj in T.Parallel(block_n):
                        sb_s[jj] = (
                            scale_b[0, T.min(n_start + jj, n - 1)] if per_channel else scale_b[0, 0]
                        )
                        if has_bias:
                            bias_s[jj] = T.cast(bias[T.min(n_start + jj, n - 1)], "float32")
                    store_half(c, acc_top, sa_top, sb_s, bias_s, c_smem, m_start, n_start, bar)
                    store_half(
                        c, acc_bot, sa_bot, sb_s, bias_s, c_smem, m_start + half_m, n_start, bar
                    )
            # Drain the last store before the CTA's shared memory goes.
            T.tma_store_wait(0)

        @T.prim_func
        def _gemm_int8_pingpong_main(
            a: T.Tensor((m, k), "int8"),  # type: ignore
            b: T.Tensor((n, k), "int8"),  # type: ignore
            scale_a: T.Tensor(sa_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(sb_shape, "float32"),  # type: ignore
            bias: T.Tensor(bias_shape, out_dtype),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_top = T.alloc_shared((num_stages, half_m, block_k), "int8")
                a_bot = T.alloc_shared((num_stages, half_m, block_k), "int8")
                b_smem = T.alloc_shared((num_stages, block_n, block_k), "int8")
                acc_top_0 = T.alloc_fragment((half_m, block_n), "int32")
                acc_bot_0 = T.alloc_fragment((half_m, block_n), "int32")
                acc_top_1 = T.alloc_fragment((half_m, block_n), "int32")
                acc_bot_1 = T.alloc_fragment((half_m, block_n), "int32")
                sa_top_0 = T.alloc_fragment((half_m,), "float32")
                sa_bot_0 = T.alloc_fragment((half_m,), "float32")
                sa_top_1 = T.alloc_fragment((half_m,), "float32")
                sa_bot_1 = T.alloc_fragment((half_m,), "float32")
                sb_s_0 = T.alloc_shared((block_n,), "float32")
                sb_s_1 = T.alloc_shared((block_n,), "float32")
                bias_s_0 = T.alloc_shared((block_n,), "float32")
                bias_s_1 = T.alloc_shared((block_n,), "float32")
                c_smem_0 = T.alloc_shared((half_m, block_n), out_dtype)
                c_smem_1 = T.alloc_shared((half_m, block_n), out_dtype)
                T.annotate_layout(
                    {
                        a_top: tilelang.layout.make_swizzled_layout(a_top),
                        a_bot: tilelang.layout.make_swizzled_layout(a_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)
                # One per consumer: completes a phase when the other consumer hands it the
                # tensor cores.
                turn = T.alloc_barrier([128, 128])
                gi_prod = T.alloc_var("int32", init=0)
                prev_0 = T.alloc_local((1,), "int32")
                prev_1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")
                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            _tile_id(
                                flat_id,
                                mt,
                                nt,
                                group_size_m=group_size_m,
                                num_pid_m=num_pid_m,
                                num_pid_n=num_pid_n,
                            )
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                _load_stage(
                                    a,
                                    b,
                                    a_top,
                                    a_bot,
                                    b_smem,
                                    ab_full,
                                    slot,
                                    ki,
                                    m_start,
                                    n_start,
                                    top_rows=half_m,
                                    bot_rows=half_m,
                                    half_m=half_m,
                                    block_n=block_n,
                                    block_k=block_k,
                                )
                                gi_prod = gi_prod + 1
                elif tx < 256:
                    T.inc_max_nreg(240)
                    consumer(
                        0,
                        pid,
                        c,
                        scale_a,
                        scale_b,
                        bias,
                        a_top,
                        a_bot,
                        b_smem,
                        ab_full,
                        ab_empty,
                        turn,
                        acc_top_0,
                        acc_bot_0,
                        sa_top_0,
                        sa_bot_0,
                        sb_s_0,
                        bias_s_0,
                        c_smem_0,
                        mt,
                        nt,
                        prev_0,
                        bar_0,
                    )
                else:
                    T.inc_max_nreg(240)
                    # The first consumer's first tile goes first.
                    if pid < total_tiles:
                        T.barrier_arrive(turn[0])
                    consumer(
                        1,
                        pid,
                        c,
                        scale_a,
                        scale_b,
                        bias,
                        a_top,
                        a_bot,
                        b_smem,
                        ab_full,
                        ab_empty,
                        turn,
                        acc_top_1,
                        acc_bot_1,
                        sa_top_1,
                        sa_bot_1,
                        sb_s_1,
                        bias_s_1,
                        c_smem_1,
                        mt,
                        nt,
                        prev_1,
                        bar_1,
                    )

        return _gemm_int8_pingpong_main

    return _gemm_int8_pingpong_func


@functools.lru_cache(maxsize=32)
def _gemm_int8_simple_kernel(
    m: int,
    n: int,
    k: int,
    out_dtype: str,
    per_token: bool,
    per_channel: bool,
    has_bias: bool,
) -> Callable:
    """INT8 GEMM on a plain software pipeline, for shapes TMA cannot address.

    One output tile a CTA; the K loop copies each step's tiles into shared memory through
    predicated copies, which read no element past the end of a row, and ``T.gemm``
    accumulates in int32. The epilogue is the warp-specialized kernel's.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n, block_k,
        num_stages, threads)`` returns the compiled ``prim_func``.
    """
    sa_shape = (m, 1) if per_token else (1, 1)
    sb_shape = (1, n) if per_channel else (1, 1)
    bias_shape = (n,) if has_bias else (1,)

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_int8_simple_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        threads: int = 256,
    ) -> Callable:
        @T.prim_func
        def _gemm_int8_simple_main(
            a: T.Tensor((m, k), "int8"),  # type: ignore
            b: T.Tensor((n, k), "int8"),  # type: ignore
            scale_a: T.Tensor(sa_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(sb_shape, "float32"),  # type: ignore
            bias: T.Tensor(bias_shape, out_dtype),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_smem = T.alloc_shared((block_m, block_k), "int8")
                b_smem = T.alloc_shared((block_n, block_k), "int8")
                acc = T.alloc_fragment((block_m, block_n), "int32")
                out = T.alloc_fragment((block_m, block_n), out_dtype)
                sa_f = T.alloc_fragment((block_m,), "float32")
                sb_f = T.alloc_fragment((block_n,), "float32")
                bias_f = T.alloc_fragment((block_n,), "float32")
                m_start = by * block_m
                n_start = bx * block_n
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(a[m_start, ki * block_k], a_smem)
                    T.copy(b[n_start, ki * block_k], b_smem)
                    T.gemm(a_smem, b_smem, acc, transpose_B=True)
                _load_scales(
                    scale_a,
                    scale_b,
                    bias,
                    sa_f,
                    sb_f,
                    bias_f,
                    m_start,
                    n_start,
                    m=m,
                    n=n,
                    half_m=block_m,
                    block_n=block_n,
                    per_token=per_token,
                    per_channel=per_channel,
                    has_bias=has_bias,
                )
                _dequantize(
                    acc,
                    out,
                    sa_f,
                    sb_f,
                    bias_f,
                    half_m=block_m,
                    block_n=block_n,
                    has_bias=has_bias,
                    out_dtype=out_dtype,
                )
                T.copy(out, c[m_start, n_start])

        return _gemm_int8_simple_main

    return _gemm_int8_simple_func


def _int8_entry(cls: type, call: GemmINT8W8A8Call) -> Entry:
    """The entry for a kernel taking every fact of the call; the device picks the config."""
    index = call.device.index if call.device is not None else None
    identity = (
        call.m,
        call.n,
        call.k,
        call.per_token,
        call.per_channel,
        call.out_dtype,
        call.has_bias,
        index,
    )
    return identity, lambda: cls(
        call.m,
        call.n,
        call.k,
        call.per_token,
        call.per_channel,
        call.out_dtype,
        call.has_bias,
        device_index=index,
    )


class _GemmINT8Kernel(Kernel, GemmINT8W8A8FwdInterface):
    """The facts both INT8 kernels hold, and the stand-in for an absent bias."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        per_token: bool,
        per_channel: bool,
        out_dtype: torch.dtype,
        has_bias: bool,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.m = m
        self.n = n
        self.k = k
        # A one-row ``scale_a`` and a one-column ``scale_b`` are one scale either way.
        self.per_token = per_token and m > 1
        self.per_channel = per_channel and n > 1
        self.out_dtype = out_dtype
        self.has_bias = has_bias
        self.init_config(config)
        self._unused_bias: Optional[torch.Tensor] = None

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    def _bias_operand(self, bias: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        """The bias, or a one-element stand-in the no-bias program never reads."""
        if bias is not None:
            return bias
        if self._unused_bias is None or self._unused_bias.device != like.device:
            self._unused_bias = torch.zeros(1, dtype=self.out_dtype, device=like.device)
        return self._unused_bias


class _GemmINT8TMAKernel(_GemmINT8Kernel):
    """The alignment both TMA kernels need, and the call they take."""

    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: GemmINT8W8A8Call) -> "str | None":
        if call.k == 0 or call.k % 16:
            return (
                f"k={call.k}: TMA addresses a row of a and b in 16-byte units, so k must be a "
                f"positive multiple of 16"
            )
        if call.n % 8:
            return (
                f"n={call.n}: the output tile is stored through TMA, whose 16-byte row pitch "
                f"needs n a multiple of 8 in a 16-bit dtype"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GemmINT8W8A8Call) -> Entry:
        return _int8_entry(cls, call)

    @abstractmethod
    def _builder(self) -> Callable:
        """The kernel's ``@tilelang.jit`` factory, which ``self.config`` parameterizes."""

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        program = self._builder()(**self.config)
        # A TMA descriptor needs its tensor to start on a 16-byte boundary, which a
        # contiguous view into another allocation need not.
        a, b = vector_aligned(a), vector_aligned(b)
        return program(a, b, scale_a, scale_b, self._bias_operand(bias, a))


class GemmINT8W8A8Kernel(_GemmINT8TMAKernel):
    """INT8 NT GEMM over 64-row tiles: one producer and one consumer warpgroup, persistent.

    The tile, ring depth and K step are :func:`int8_w8a8_config`'s; see
    :func:`_gemm_int8_ws_kernel`.
    """

    @property
    def default_config(self) -> dict:
        return int8_w8a8_config(self.m, self.n, self.k, get_sm_count(self.device_index))

    def _builder(self) -> Callable:
        return _gemm_int8_ws_kernel(
            self.m,
            self.n,
            self.k,
            self.out_dtype_str,
            self.per_token,
            self.per_channel,
            self.has_bias,
            sm_count=get_sm_count(self.device_index),
        )


class GemmINT8W8A8PingPongKernel(_GemmINT8TMAKernel):
    """INT8 NT GEMM over 128 x 128 tiles, two consumer warpgroups in turn; see
    :func:`_gemm_int8_pingpong_kernel`.

    It serves ``m`` over one 128-row tile. At or below that the 128-row grid holds half
    the tiles of the 64-row kernel's, leaves SMs idle on every decode shape measured, and
    the 64-row kernel serves the call instead.
    """

    preferred_over = frozenset({"gemm_int8_w8a8"})

    # Rows of a tile; the call must have more.
    _BLOCK_M = 128

    @classmethod
    def refusal(cls, call: GemmINT8W8A8Call) -> "str | None":
        if call.m <= cls._BLOCK_M:
            return f"m={call.m} fits one {cls._BLOCK_M}-row tile; the 64-row kernel serves it"
        return super().refusal(call)

    @property
    def default_config(self) -> dict:
        # A ring stage: both consumers' 64 A rows and the 128 B rows of one 128-wide K step,
        # and its full and empty mbarriers.
        stage = (self._BLOCK_M + 128) * 128 + 16
        # Each consumer's 64 x 128 output staging tile and its staged column scales and bias.
        epilogue = 2 * (64 * 128 * 2 + 2 * 128 * 4)
        num_stages = (BLOCK_SHARED_BYTES_OPT_IN[90] - epilogue) // stage
        return {"num_stages": num_stages, "group_size_m": 16}

    def _builder(self) -> Callable:
        return _gemm_int8_pingpong_kernel(
            self.m,
            self.n,
            self.k,
            self.out_dtype_str,
            self.per_token,
            self.per_channel,
            self.has_bias,
            sm_count=get_sm_count(self.device_index),
        )


class GemmINT8W8A8SimpleKernel(_GemmINT8Kernel):
    """INT8 NT GEMM on SM89, and for shapes the SM90 TMA kernels refuse.

    Its loads are predicated copies, so ``k`` and ``n`` need no alignment.
    """

    supported_archs: list[int] = [89, 90]
    general = True

    @classmethod
    def entry_for(cls, call: GemmINT8W8A8Call) -> Entry:
        return _int8_entry(cls, call)

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "block_n": 128, "block_k": 64, "num_stages": 3, "threads": 256}

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        program = _gemm_int8_simple_kernel(
            self.m,
            self.n,
            self.k,
            self.out_dtype_str,
            self.per_token,
            self.per_channel,
            self.has_bias,
        )(**self.config)
        # A contiguous view need not start on a 16-byte boundary, which SM90's TMA
        # descriptors and SM89's vectorized loads both require.
        a, b = vector_aligned(a), vector_aligned(b)
        return program(a, b, scale_a, scale_b, self._bias_operand(bias, a))
