import functools
from typing import Callable, Optional

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops.kernels.constants import BF16_SPLIT_EXP2_SPAN, LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GLAChunkCall,
    GLAChunkFwdInterface,
    head_count_refusal,
)
from tileops.kernels.linear_attention.v_tile import GEMM_MIN_N, min_gemm_n
from tileops.utils import get_shared_memory_optin, get_sm_count


@functools.lru_cache(maxsize=32)
def gla_precompute_g_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    dtype: str,
    output_dtype: str = "float32",
) -> Callable:
    """Pre-compute intra-chunk cumulative sum of g.

    Parallel over (batch, heads, chunks): B*H*NC thread blocks.
    Each block computes cumsum for one chunk independently.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _fn(num_stages, threads=128):
        g_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(
            g: T.Tensor(g_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, output_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                cs = i_c * chunk_size

                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_out_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)

                T.copy(g[i_b, cs : cs + chunk_size, i_h, :], g_s, disable_tma=True)

                for i_k in T.Parallel(dim_k):
                    g_out_s[0, i_k] = T.cast(g_s[0, i_k], accum_dtype)
                for i_t in T.Serial(1, chunk_size):
                    for i_k in T.Parallel(dim_k):
                        g_out_s[i_t, i_k] = g_out_s[i_t - 1, i_k] + T.cast(
                            g_s[i_t, i_k], accum_dtype
                        )

                T.copy(g_out_s, g_cumsum[i_b, cs : cs + chunk_size, i_h, :])

        return _main

    return _fn


@functools.lru_cache(maxsize=32)
def gla_fwd_h_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    num_v_partitions: int = 1,
    num_k_partitions: int = 1,
    has_initial_state: bool = True,
) -> Callable:
    """Compute per-chunk hidden states h in forward order.

    Sequential over chunks (inter-chunk recurrence); the pipelined loop prefetches k, v and
    g_cumsum.

    KV-partition parallelism: splits K and V dimensions across thread blocks
    for higher SM utilization and more square GEMM shapes.
    Grid: B * H * num_k_partitions * num_v_partitions blocks.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    if dim_v % num_v_partitions:
        raise ValueError(
            f"dim_v ({dim_v}) is not divisible by num_v_partitions ({num_v_partitions})"
        )
    dim_v_part = dim_v // num_v_partitions
    if dim_v_part < GEMM_MIN_N:
        raise ValueError(
            f"dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) gives a "
            f"{dim_v_part}-column T.gemm B operand, below the minimum N extent ({GEMM_MIN_N})"
        )
    dim_k_part = dim_k // num_k_partitions
    num_kv = num_k_partitions * num_v_partitions

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _h_func(num_stages, threads=128):
        if dim_v_part < min_gemm_n(threads):
            raise ValueError(
                f"dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) "
                f"gives a {dim_v_part}-column T.gemm B operand, below the minimum N "
                f"extent ({min_gemm_n(threads)}) at {threads} threads"
            )
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        init_state_shape = [batch, heads, dim_k, dim_v] if has_initial_state else [1]
        h_out_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            initial_state: T.Tensor(init_state_shape, accum_dtype),
            h_out: T.Tensor(h_out_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_kv, threads=threads) as bx:
                i_b = bx // (heads * num_kv)
                i_h = (bx // num_kv) % heads
                i_kv = bx % num_kv
                i_kp = i_kv // num_v_partitions
                i_vp = i_kv % num_v_partitions
                k_offset = i_kp * dim_k_part
                v_offset = i_vp * dim_v_part

                h_s = T.alloc_shared([dim_k_part, dim_v_part], accum_dtype)
                k_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k_part], accum_dtype)

                if has_initial_state:
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = initial_state[i_b, i_h, k_offset + i_k, v_offset + i_v]
                else:
                    T.clear(h_s)

                for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(
                        k[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        k_s,
                        disable_tma=True,
                    )
                    T.copy(
                        v[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            v_offset : v_offset + dim_v_part,
                        ],
                        v_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        g_cumsum_s,
                        disable_tma=True,
                    )

                    # h_out[i_c] is the state entering chunk i_c.
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_out[i_b, i_c, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]

                    g_last = T.alloc_fragment([dim_k_part], accum_dtype)
                    for i_k in T.Parallel(dim_k_part):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] * T.exp2(g_last[i_k] * LOG2E)

                    # k_adj in fragment (RS GEMM: A=register, B=shared)
                    k_adj_f = T.alloc_fragment([chunk_size, dim_k_part], dtype)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj_f[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2E),
                            dtype,
                        )

                    delta_h = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                    T.fill(delta_h, 0.0)
                    T.gemm(k_adj_f, v_s, delta_h, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] + delta_h[i_k, i_v]

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_out[i_b, num_chunks, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]

        return _main

    return _h_func


@T.macro
def chunk_gate_cumsum(g_s, g_cumsum_s, chunk_size: int, dim_k: int, threads: int, dtype: str):
    """The in-chunk cumulative sum of the gate into ``g_cumsum_s``.

    A 16-bit gate is summed on tensor cores as the lower-triangular ones times g, exact
    products into a float32 accumulator. A float32 gate would round to TF32 there, so it is
    summed in order, one key channel a thread.
    """
    if dtype == "float32":
        tx = T.get_thread_binding()
        running = T.alloc_local([1], "float32")
        T.sync_threads()
        for i_r in T.serial(T.ceildiv(dim_k, threads)):
            i_k = i_r * threads + tx
            if i_k < dim_k:
                running[0] = 0.0
                for i_t in T.serial(chunk_size):
                    running[0] = running[0] + g_s[i_t, i_k]
                    g_cumsum_s[i_t, i_k] = running[0]
        T.sync_threads()
    else:
        lower_s = T.alloc_shared([chunk_size, chunk_size], dtype)
        for i_t, i_j in T.Parallel(chunk_size, chunk_size):
            lower_s[i_t, i_j] = T.cast(T.if_then_else(i_j <= i_t, 1.0, 0.0), dtype)
        running = T.alloc_fragment([chunk_size, dim_k], "float32")
        T.clear(running)
        T.gemm(lower_s, g_s, running)
        T.copy(running, g_cumsum_s)


@functools.lru_cache(maxsize=32)
def _gla_fwd_update_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
) -> Callable:
    """Each chunk's contribution to the state, every chunk in parallel.

    One block a (batch, head, chunk) takes the gate cumsum g, writes its last row g_last, and
    writes U = (k * exp(g_last - g))^T v, so that the state the chunk leaves is
    exp(g_last) * S + U for the state S it is handed.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _update_func(threads=128):
        @T.prim_func
        def _main(
            k: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            v: T.Tensor([batch, seq_len, heads, dim_v], dtype),
            g: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            update: T.Tensor([batch, num_chunks, heads, dim_k, dim_v], accum_dtype),
            g_last: T.Tensor([batch, num_chunks, heads, dim_k], accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                start = i_c * chunk_size

                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                k_adj_s = T.alloc_shared([chunk_size, dim_k], dtype)

                T.copy(g[i_b, start : start + chunk_size, i_h, :], g_s, disable_tma=True)
                T.copy(v[i_b, start : start + chunk_size, i_h, :], v_s, disable_tma=True)
                chunk_gate_cumsum(g_s, g_cumsum_s, chunk_size, dim_k, threads, dtype)

                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    k_adj_s[i_t, i_k] = T.cast(
                        T.cast(k[i_b, start + i_t, i_h, i_k], accum_dtype)
                        * T.exp2((g_cumsum_s[chunk_size - 1, i_k] - g_cumsum_s[i_t, i_k]) * LOG2E),
                        dtype,
                    )
                acc = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                T.clear(acc)
                T.gemm(k_adj_s, v_s, acc, transpose_A=True)
                T.copy(acc, update[i_b, i_c, i_h, :, :])
                for i_k in T.Parallel(dim_k):
                    g_last[i_b, i_c, i_h, i_k] = g_cumsum_s[chunk_size - 1, i_k]

        return _main

    return _update_func


@functools.lru_cache(maxsize=32)
def gla_state_walk_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    state_rows: int,
    has_initial_state: bool,
    reverse: bool = False,
) -> Callable:
    """A state walked chunk by chunk from per-chunk updates: S <- exp(g_last) * S + U.

    One block a (batch, head, ``state_rows`` state rows) carries its rows of the state in
    registers and per chunk writes them, then applies the chunk's update: no matmul on the
    serial chain. The pipelined loop prefetches the updates.

    Forward, ``out[c]`` is the state entering chunk c and ``out[num_chunks]`` the one the last
    chunk leaves. With *reverse* the walk runs from the last chunk to the first, as the
    gradient of the state does: ``out[c + 1]`` is the state chunk c is handed and ``out[0]``
    the one the first chunk leaves.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    row_tiles = dim_k // state_rows

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _walk_func(num_stages, threads):
        init_state_shape = [batch, heads, dim_k, dim_v] if has_initial_state else [1]

        @T.prim_func
        def _main(
            update: T.Tensor([batch, num_chunks, heads, dim_k, dim_v], accum_dtype),
            g_last: T.Tensor([batch, num_chunks, heads, dim_k], accum_dtype),
            initial_state: T.Tensor(init_state_shape, accum_dtype),
            h_out: T.Tensor([batch, num_chunks + 1, heads, dim_k, dim_v], accum_dtype),
        ):
            with T.Kernel(batch * heads * row_tiles, threads=threads) as bx:
                i_b = bx // (heads * row_tiles)
                i_h = (bx // row_tiles) % heads
                k0 = (bx % row_tiles) * state_rows

                state = T.alloc_fragment([state_rows, dim_v], accum_dtype)
                update_s = T.alloc_shared([state_rows, dim_v], accum_dtype)
                g_last_s = T.alloc_shared([state_rows], accum_dtype)

                if has_initial_state:
                    T.copy(initial_state[i_b, i_h, k0 : k0 + state_rows, :], state)
                else:
                    T.clear(state)

                for i_t in T.Pipelined(num_chunks, num_stages=num_stages):
                    i_c = num_chunks - 1 - i_t if reverse else i_t
                    T.copy(
                        update[i_b, i_c, i_h, k0 : k0 + state_rows, :], update_s, disable_tma=True
                    )
                    T.copy(g_last[i_b, i_c, i_h, k0 : k0 + state_rows], g_last_s, disable_tma=True)
                    T.copy(
                        state, h_out[i_b, i_c + 1 if reverse else i_c, i_h, k0 : k0 + state_rows, :]
                    )
                    for i_k, i_v in T.Parallel(state_rows, dim_v):
                        state[i_k, i_v] = (
                            state[i_k, i_v] * T.exp2(g_last_s[i_k] * LOG2E) + update_s[i_k, i_v]
                        )

                T.copy(
                    state, h_out[i_b, 0 if reverse else num_chunks, i_h, k0 : k0 + state_rows, :]
                )

        return _main

    return _walk_func


def state_walk_plan(batch: int, heads: int, dim_k: int, dim_v: int, stages: int) -> tuple:
    """``(state_rows, num_stages, threads)`` of :func:`gla_state_walk_kernel` on this device.

    The narrowest tile of state rows whose blocks fit one wave, else the widest whose stage
    fits shared memory; a stage holds one tile of the update and its rows' decay, the state
    stays in registers. At most *stages* stages, fewer where shared memory cannot hold them;
    four state elements a thread, clamped to between one warp and two warp groups.
    """
    budget = get_shared_memory_optin()
    stage_bytes = {
        rows: rows * (dim_v + 1) * 4
        for rows in (2**i for i in range(1, dim_k.bit_length()))
        if dim_k % rows == 0 and rows * (dim_v + 1) * 4 <= budget
    }
    rows = next(
        (rows for rows in stage_bytes if batch * heads * (dim_k // rows) <= get_sm_count()),
        max(stage_bytes),
    )
    return rows, min(stages, budget // stage_bytes[rows]), max(32, min(256, rows * dim_v // 4))


@functools.lru_cache(maxsize=32)
def _gla_fwd_output_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    """Every chunk's output in parallel: o = scale * (q * exp(g)) S + A v.

    One block a (batch, head, chunk). The causal attention A[i, j] = scale * sum_d q[i, d]
    k[j, d] exp(g[i, d] - g[j, d]) is a tensor-core product of the factors q * exp(g) and
    k * exp(-g) while the chunk's gate decay keeps both representable; a chunk that decays
    further takes the exact per-pair form. Each factor is held as a bfloat16 pair, its
    leading part and the remainder, and the product sums the three terms that carry
    float32-level precision: bfloat16 has float32's exponent range, which the factors span.
    The factors are formed and multiplied 64 key channels at a time.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    # Key channels a factor tile holds; wider heads take several tiles.
    block_k = min(dim_k, 64)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _output_func(threads=128):
        @T.prim_func
        def _main(
            q: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            k: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            v: T.Tensor([batch, seq_len, heads, dim_v], dtype),
            g: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            h: T.Tensor([batch, num_chunks + 1, heads, dim_k, dim_v], accum_dtype),
            o: T.Tensor([batch, seq_len, heads, dim_v], dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                start = i_c * chunk_size

                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                q_hi_s = T.alloc_shared([chunk_size, block_k], "bfloat16")
                q_lo_s = T.alloc_shared([chunk_size, block_k], "bfloat16")
                k_hi_s = T.alloc_shared([chunk_size, block_k], "bfloat16")
                k_lo_s = T.alloc_shared([chunk_size, block_k], "bfloat16")
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)
                a_s = T.alloc_shared([chunk_size, chunk_size], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                h_s = T.alloc_shared([dim_k, dim_v], dtype)
                factor = T.alloc_fragment([chunk_size, block_k], accum_dtype)
                spans = T.alloc_fragment([dim_k], accum_dtype)
                span = T.alloc_fragment([1], accum_dtype)
                a_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)

                T.copy(g[i_b, start : start + chunk_size, i_h, :], g_s, disable_tma=True)
                chunk_gate_cumsum(g_s, g_cumsum_s, chunk_size, dim_k, threads, dtype)

                # The chunk's whole decay sets the range of exp(g) and exp(-g).
                for i_k in T.Parallel(dim_k):
                    spans[i_k] = -g_cumsum_s[chunk_size - 1, i_k] * LOG2E
                T.reduce_max(spans, span, dim=0)

                T.clear(a_frag)
                if span[0] < BF16_SPLIT_EXP2_SPAN:
                    for i_kb in T.serial(dim_k // block_k):
                        k0 = i_kb * block_k
                        for i_t, i_k in T.Parallel(chunk_size, block_k):
                            factor[i_t, i_k] = T.cast(
                                q[i_b, start + i_t, i_h, k0 + i_k], accum_dtype
                            ) * T.exp2(g_cumsum_s[i_t, k0 + i_k] * LOG2E)
                        for i_t, i_k in T.Parallel(chunk_size, block_k):
                            q_gated_s[i_t, k0 + i_k] = T.cast(factor[i_t, i_k], dtype)
                            q_hi_s[i_t, i_k] = T.cast(factor[i_t, i_k], "bfloat16")
                            q_lo_s[i_t, i_k] = T.cast(
                                factor[i_t, i_k]
                                - T.cast(T.cast(factor[i_t, i_k], "bfloat16"), accum_dtype),
                                "bfloat16",
                            )
                        for i_t, i_k in T.Parallel(chunk_size, block_k):
                            factor[i_t, i_k] = T.cast(
                                k[i_b, start + i_t, i_h, k0 + i_k], accum_dtype
                            ) * T.exp2(-g_cumsum_s[i_t, k0 + i_k] * LOG2E)
                        for i_t, i_k in T.Parallel(chunk_size, block_k):
                            k_hi_s[i_t, i_k] = T.cast(factor[i_t, i_k], "bfloat16")
                            k_lo_s[i_t, i_k] = T.cast(
                                factor[i_t, i_k]
                                - T.cast(T.cast(factor[i_t, i_k], "bfloat16"), accum_dtype),
                                "bfloat16",
                            )
                        T.gemm(q_hi_s, k_hi_s, a_frag, transpose_B=True)
                        T.gemm(q_hi_s, k_lo_s, a_frag, transpose_B=True)
                        T.gemm(q_lo_s, k_hi_s, a_frag, transpose_B=True)
                else:
                    for i_k in T.Serial(dim_k):
                        for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                            a_frag[i_t, i_j] = a_frag[i_t, i_j] + (
                                T.cast(q[i_b, start + i_t, i_h, i_k], accum_dtype)
                                * T.cast(k[i_b, start + i_j, i_h, i_k], accum_dtype)
                                * T.exp2((g_cumsum_s[i_t, i_k] - g_cumsum_s[i_j, i_k]) * LOG2E)
                            )
                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        q_gated_s[i_t, i_k] = T.cast(
                            T.cast(q[i_b, start + i_t, i_h, i_k], accum_dtype)
                            * T.exp2(g_cumsum_s[i_t, i_k] * LOG2E),
                            dtype,
                        )
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    a_s[i_t, i_j] = T.cast(
                        T.if_then_else(i_j <= i_t, a_frag[i_t, i_j] * scale, 0.0), dtype
                    )

                T.copy(v[i_b, start : start + chunk_size, i_h, :], v_s, disable_tma=True)
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)

                acc = T.alloc_fragment([chunk_size, dim_v], accum_dtype)
                T.clear(acc)
                T.gemm(q_gated_s, h_s, acc)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    acc[i_t, i_v] = acc[i_t, i_v] * scale
                T.gemm(a_s, v_s, acc)
                T.copy(acc, o[i_b, start : start + chunk_size, i_h, :])

        return _main

    return _output_func


class GLAChunkedFwdKernel(Kernel):
    """Gated Linear Attention (GLA) chunked forward: what its programs share.

    Holds the call's extents and scale, refuses extents the tiling cannot build, and tunes
    the launches a subclass builds in ``_build_kernels`` and runs in ``forward``. The default
    config is the one of the per-chunk state pass ``gla_fwd_h_kernel``.

    Reference:
        https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        output_final_state: bool = False,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale if scale > 0 else dim_k**-0.5
        self.output_final_state = output_final_state
        self.dtype_name = str(dtype).split(".")[-1]
        reason = self.region_refusal(dim_k, dim_v, chunk_size)
        if reason:
            raise ValueError(f"{type(self).__name__} does not serve this call: {reason}")
        self.init_config(config)
        self._build_kernels(self.config)

    @staticmethod
    def region_refusal(dim_k: int, dim_v: int, chunk_size: int) -> Optional[str]:
        """Why the default tiling cannot build these extents, or ``None`` when it can.

        Its three GEMMs run on two warps: the state update over
        ``(dim_k / 2) x (dim_v / 4)`` and the two chunk outputs over
        ``chunk_size x dim_v``. Each warp takes whole 16 x 8 tiles, so the chunk
        either fills both warps' rows (a multiple of 32) or one warp's (16, the
        warps then splitting the columns); the state update likewise needs
        ``dim_k / 2`` a multiple of 32 or exactly 16.
        """
        if not (chunk_size == 16 or chunk_size % 32 == 0):
            return f"chunk_size={chunk_size} must be 16 or a multiple of 32"
        if not (dim_k % 64 == 0 and dim_v % 32 == 0 or dim_k == 32 and dim_v % 64 == 0):
            return (
                f"dim_k={dim_k}, dim_v={dim_v}: dim_k must be a multiple of 64 with dim_v a "
                "multiple of 32, or 32 with dim_v a multiple of 64"
            )
        return None

    @staticmethod
    def _v_partitions(dim_v: int, threads: int, candidates: list[int]) -> list[int]:
        """Return the candidates the recurrence can build at *threads*, widest first."""
        floor = max(GEMM_MIN_N, min_gemm_n(threads))
        return [n for n in candidates if dim_v % n == 0 and dim_v // n >= floor]

    @staticmethod
    def _state_shared_bytes(c: int, k: int, v: int, elem: int, stages: int) -> int:
        """Shared memory of the state pass over a k x v slice: the fp32 state, and k, v and the
        fp32 gate once per stage."""
        return stages * c * (k * (elem + 4) + v * elem) + k * v * 4

    @property
    def default_config(self) -> dict:
        return self._default_config_for(
            get_shared_memory_optin(),
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            getattr(torch, self.dtype_name).itemsize,
        )

    @classmethod
    def _default_config_for(
        cls, budget: int, chunk_size: int, dim_k: int, dim_v: int, elem: int
    ) -> dict:
        """The config this kernel builds at *budget* bytes of shared memory per block."""
        threads = 64
        num_v_partitions, num_k_partitions = cls._v_partitions(dim_v, threads, [4, 2, 1])[0], 2
        k, v = dim_k // num_k_partitions, dim_v // num_v_partitions
        # Only the state pass pipelines, so its size alone sets the stages.
        stages = next(
            (s for s in (3, 2) if cls._state_shared_bytes(chunk_size, k, v, elem, s) <= budget),
            1,
        )
        return {
            "num_stages": stages,
            "threads_par": threads,
            "threads_seq": threads,
            "num_v_partitions": num_v_partitions,
            "num_k_partitions": num_k_partitions,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for ns in [1, 2, 3]:
            for t_par in [64, 128, 256]:
                for t_seq in [64, 128, 256]:
                    for nvp in self._v_partitions(self.dim_v, t_seq, [2, 4]):
                        for nkp in [1, 2]:
                            configs.append(
                                {
                                    "num_stages": ns,
                                    "threads_par": t_par,
                                    "threads_seq": t_seq,
                                    "num_v_partitions": nvp,
                                    "num_k_partitions": nkp,
                                }
                            )
        return configs

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Custom autotuning for multi-kernel forward pass."""
        if self.autotune_configs is None:
            return
        print(
            f"Start autotuning {self.__class__.__name__} ({len(self.autotune_configs)} configs)..."
        )

        B, T, H, K, V = (self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v)
        dtype_torch = getattr(torch, self.dtype_name)

        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()
        # State-reading specializations need a real input while tuning. Zero-state
        # specializations ignore it; prepare it once, outside the timed calls.
        initial_state = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)

        best_lat = float("inf")
        best_cfg = None

        for cfg in self.tuning_candidates(self.autotune_configs):
            try:
                self._build_kernels(cfg)

                self.forward(q, k, v, g, initial_state)
                torch.cuda.synchronize()

                lat = do_bench(
                    lambda: self.forward(q, k, v, g, initial_state),
                    warmup=warmup,
                    rep=rep,
                )
                print(f"  config={cfg} -> {lat:.3f}ms")
                if lat < best_lat:
                    best_lat = lat
                    best_cfg = cfg
            except Exception as e:
                print(f"  config={cfg} -> FAILED: {e}")
                continue

        if best_cfg is not None:
            self.config = best_cfg
            self._build_kernels(best_cfg)
            print(f"Best config: {best_cfg} ({best_lat:.3f}ms)")
        else:
            print("Autotuning failed, using default config")
            self.config = self.default_config
            self._build_kernels(self.config)


class GLAChunkFwdKernel(GLAChunkedFwdKernel, GLAChunkFwdInterface):
    """The chunked forward as the training op calls it, with its own recurrent state.

    Three launches: every chunk's state update in parallel; a walk of the state across the
    chunks, which writes the state entering each chunk (the backward reads it) and carries no
    matmul on its serial chain; and every chunk's output in parallel from the state it is
    handed. The walk takes the narrowest tile of state rows whose blocks fit one wave, else the
    widest whose stage fits shared memory.
    """

    @staticmethod
    def _shared_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Shared memory the forward's launches need at chunk *c*, head dims *k* / *v*.

        Both chunk-parallel launches first hold the gate, its fp32 cumsum and, for a 16-bit
        gate, the lower-triangular ones; the update also holds v by then. The update then holds
        the cumsum, v and the scaled k. The output holds the cumsum, the four bfloat16 factor
        tiles (64 key channels each) and the gated q while A is formed, then the gated q, A, v
        and the cast state for the two products. The walk needs one stage of its narrowest
        tile.
        """
        cumsum = c * k * (elem + 4) + (c * c * elem if elem < 4 else 0)
        update = max(cumsum + c * v * elem, c * k * (4 + elem) + c * v * elem)
        output = max(
            cumsum,
            c * k * (4 + elem) + 8 * c * min(k, 64),
            (c * k + c * c + c * v + k * v) * elem,
        )
        walk_stage = 2 * (v + 1) * 4
        return max(update, output, walk_stage)

    @classmethod
    def _default_config_for(
        cls, budget: int, chunk_size: int, dim_k: int, dim_v: int, elem: int
    ) -> dict:
        """One warp group for the chunk-parallel launches and the walk's prefetch depth.

        Neither depends on *budget* or the extents: the walk takes fewer stages when building
        on a device that cannot hold them.
        """
        return {"threads": 128, "num_stages": 16}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"threads": threads, "num_stages": stages}
            for threads in (128, 256)
            for stages in (4, 8, 16, 32)
        ]

    def _build_kernels(self, config: dict) -> None:
        """Build the update and output launches; the walk is built for the state it is given."""
        # A 16-token chunk is one warp's rows in every product.
        self._threads = config["threads"] if self.chunk_size >= 32 else 32
        self._walk_rows, self._walk_stages, self._walk_threads = state_walk_plan(
            self.batch, self.heads, self.dim_k, self.dim_v, config["num_stages"]
        )
        self._walk_fns = {}
        self._unread_state = None
        shape = (self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v, self.chunk_size)
        self._update_fn = _gla_fwd_update_kernel(*shape, self.dtype_name)(self._threads)
        self._output_fn = _gla_fwd_output_kernel(*shape, self.scale, self.dtype_name)(self._threads)

    def _walk_fn(self, has_initial_state: bool) -> Callable:
        """The walk that starts from a given state or from zeros, built on first use."""
        if has_initial_state not in self._walk_fns:
            self._walk_fns[has_initial_state] = gla_state_walk_kernel(
                self.batch,
                self.seq_len,
                self.heads,
                self.dim_k,
                self.dim_v,
                self.chunk_size,
                self._walk_rows,
                has_initial_state,
            )(self._walk_stages, self._walk_threads)
        return self._walk_fns[has_initial_state]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = getattr(torch, self.dtype_name)
        q, k, v, g = (t.to(dtype) for t in (q, k, v, g))
        update, g_last = self._update_fn(k, v, g)
        if initial_state is None:
            if self._unread_state is None:
                self._unread_state = torch.empty(1, dtype=torch.float32, device=q.device)
            h = self._walk_fn(False)(update, g_last, self._unread_state)
        else:
            h = self._walk_fn(True)(update, g_last, initial_state.to(torch.float32))
        o = self._output_fn(q, k, v, g, h)
        self._h_out = h
        return o, h[:, -1]

    @classmethod
    def refusal(cls, call: GLAChunkCall) -> Optional[str]:
        """Why no program serves this call, or ``None``.

        Refuses a call whose launches need more shared memory at once than the device gives.
        """
        reason = head_count_refusal(call.heads) or cls.region_refusal(
            call.dim_k, call.dim_v, call.chunk_size
        )
        if reason is not None or not call.smem_budget:
            return reason
        c, k, v = call.chunk_size, call.dim_k, call.dim_v
        need = cls._shared_bytes(c, k, v, call.dtype.itemsize)
        if need <= call.smem_budget:
            return None
        return (
            f"needs at least {need} bytes of shared memory per block at chunk {c}, head dims "
            f"{k} / {v} in {call.dtype}; the device gives {call.smem_budget}"
        )

    @classmethod
    def entry_for(cls, call: GLAChunkCall) -> Entry:
        """The final state is always produced, and the initial one is a launch argument.

        The device index is in the identity because the constructor compiles for the
        architecture it is built on.
        """
        arguments = (
            call.batch,
            call.seq_len,
            call.heads,
            call.dim_k,
            call.dim_v,
            call.chunk_size,
            call.scale,
        )
        index = call.device.index if call.device is not None else None
        return (*arguments, call.dtype, index), lambda: cls(
            *arguments, output_final_state=True, dtype=call.dtype
        )
