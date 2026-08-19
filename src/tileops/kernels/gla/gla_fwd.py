import functools
from typing import Callable, Optional

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.v_tile import GEMM_MIN_N

LOG2_E = 1.44269504


# Pre-compute: g_cumsum per chunk (parallel, B*H*NC thread blocks)


@functools.lru_cache(maxsize=32)
def _gla_precompute_g_kernel(
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


# Pass 1: compute h per chunk (sequential, B*H thread blocks)
# Uses pre-computed g_cumsum — no T.Serial cumsum needed.


@functools.lru_cache(maxsize=32)
def _gla_fwd_h_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    state_dtype: str = "float32",
    num_v_partitions: int = 1,
    num_k_partitions: int = 1,
) -> Callable:
    """Compute per-chunk hidden states h in forward order.

    Sequential over chunks (inter-chunk recurrence).
    Uses T.Pipelined + T.copy for async prefetch of k, v, g_cumsum.
    g_cumsum is pre-computed — no T.Serial cumsum in this kernel.

    KV-partition parallelism: splits K and V dimensions across thread blocks
    for higher SM utilization and more square GEMM shapes.
    Grid: B * H * num_k_partitions * num_v_partitions blocks.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    dim_v_part = dim_v // num_v_partitions
    if dim_v_part < GEMM_MIN_N:
        raise ValueError(
            f"dim_v ({dim_v}) split across num_v_partitions "
            f"({num_v_partitions}) gives a {dim_v_part}-column T.gemm B "
            f"operand, below the minimum N extent ({GEMM_MIN_N})"
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
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        init_state_shape = [batch, heads, dim_k, dim_v]
        h_out_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            initial_state: T.Tensor(init_state_shape, state_dtype),
            h_out: T.Tensor(h_out_shape, state_dtype),
        ):
            with T.Kernel(batch * heads * num_kv, threads=threads) as bx:
                i_b = bx // (heads * num_kv)
                i_h = (bx // num_kv) % heads
                i_kv = bx % num_kv
                i_kp = i_kv // num_v_partitions
                i_vp = i_kv % num_v_partitions
                k_offset = i_kp * dim_k_part
                v_offset = i_vp * dim_v_part

                h_f = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                k_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k_part], accum_dtype)

                # Load initial state KV-slice
                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_f[i_k, i_v] = initial_state[i_b, i_h, k_offset + i_k, v_offset + i_v]

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

                    # Save pre-decay h KV-slice
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_out[i_b, i_c, i_h, k_offset + i_k, v_offset + i_v] = h_f[i_k, i_v]

                    # g_last from pre-computed cumsum
                    g_last = T.alloc_fragment([dim_k_part], accum_dtype)
                    for i_k in T.Parallel(dim_k_part):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    # Decay h
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_f[i_k, i_v] = h_f[i_k, i_v] * T.exp2(g_last[i_k] * LOG2_E)

                    # k_adj in fragment (RS GEMM: A=register, B=shared)
                    k_adj_f = T.alloc_fragment([chunk_size, dim_k_part], dtype)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj_f[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2_E),
                            dtype,
                        )

                    # Accumulate the recurrence in registers across chunks.
                    T.gemm(k_adj_f, v_s, h_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_out[i_b, num_chunks, i_h, k_offset + i_k, v_offset + i_v] = h_f[i_k, i_v]

        return _main

    return _h_func


@functools.lru_cache(maxsize=32)
def _gla_fwd_h_summary_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    partition_chunks: int,
    dtype: str,
    gate_dtype: str,
    num_v_partitions: int = 4,
    num_k_partitions: int = 2,
) -> Callable:
    """Summarise independent chunk partitions from a zero initial state."""
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    if num_chunks % partition_chunks != 0:
        raise ValueError(
            f"num_chunks ({num_chunks}) must be divisible by partition_chunks ({partition_chunks})"
        )
    num_partitions = num_chunks // partition_chunks
    dim_v_part = dim_v // num_v_partitions
    dim_k_part = dim_k // num_k_partitions
    num_kv = num_k_partitions * num_v_partitions

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _summary_func(num_stages=2, threads=128):
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_shape = [batch, seq_len, heads, dim_k]
        summary_shape = [batch, num_partitions, heads, dim_k, dim_v]
        decay_shape = [batch, num_partitions, heads, dim_k]

        @T.prim_func
        def _main(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_shape, gate_dtype),
            summaries: T.Tensor(summary_shape, accum_dtype),
            log_decays: T.Tensor(decay_shape, accum_dtype),
        ):
            with T.Kernel(num_partitions * num_kv, batch, heads, threads=threads) as (
                i_pk,
                i_b,
                i_h,
            ):
                i_p = i_pk // num_kv
                i_kv = i_pk % num_kv
                i_kp = i_kv // num_v_partitions
                i_vp = i_kv % num_v_partitions
                k_offset = i_kp * dim_k_part
                v_offset = i_vp * dim_v_part

                h_f = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                k_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                g_s = T.alloc_shared([chunk_size, dim_k_part], gate_dtype)
                k_adj = T.alloc_fragment([chunk_size, dim_k_part], dtype)
                log_decay = T.alloc_fragment([dim_k_part], accum_dtype)

                T.clear(h_f)
                T.clear(log_decay)
                for i_local in T.Pipelined(partition_chunks, num_stages=num_stages):
                    i_c = i_p * partition_chunks + i_local
                    chunk_start = i_c * chunk_size
                    T.copy(
                        k[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        k_s,
                        disable_tma=True,
                    )
                    T.copy(
                        v[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            v_offset : v_offset + dim_v_part,
                        ],
                        v_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        g_s,
                        disable_tma=True,
                    )

                    for i_k in T.Parallel(dim_k_part):
                        log_decay[i_k] = log_decay[i_k] + T.cast(
                            g_s[chunk_size - 1, i_k], accum_dtype
                        )
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_f[i_k, i_v] = h_f[i_k, i_v] * T.exp2(
                            T.cast(g_s[chunk_size - 1, i_k], accum_dtype) * LOG2_E
                        )
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2(
                                (
                                    T.cast(g_s[chunk_size - 1, i_k], accum_dtype)
                                    - T.cast(g_s[i_t, i_k], accum_dtype)
                                )
                                * LOG2_E
                            ),
                            dtype,
                        )
                    T.gemm(
                        k_adj,
                        v_s,
                        h_f,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    summaries[i_b, i_p, i_h, k_offset + i_k, v_offset + i_v] = h_f[i_k, i_v]
                if i_vp == 0:
                    for i_k in T.Parallel(dim_k_part):
                        log_decays[i_b, i_p, i_h, k_offset + i_k] = log_decay[i_k]

        return _main

    return _summary_func


@functools.lru_cache(maxsize=32)
def _gla_fwd_h0_scan_kernel(
    batch: int,
    heads: int,
    num_partitions: int,
    dim_k: int,
    dim_v: int,
    block_v: int = 32,
) -> Callable:
    """Scan affine partition summaries into the true partition start states."""
    accum_dtype = "float32"
    num_v_tiles = dim_v // block_v

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _scan_func(threads=128):
        summary_shape = [batch, num_partitions, heads, dim_k, dim_v]
        decay_shape = [batch, num_partitions, heads, dim_k]

        @T.prim_func
        def _main(
            summaries: T.Tensor(summary_shape, accum_dtype),
            log_decays: T.Tensor(decay_shape, accum_dtype),
            initial_states: T.Tensor(summary_shape, accum_dtype),
        ):
            with T.Kernel(num_v_tiles, batch, heads, threads=threads) as (i_vt, i_b, i_h):
                v_offset = i_vt * block_v
                h_s = T.alloc_shared([dim_k, block_v], accum_dtype)
                summary_s = T.alloc_shared([dim_k, block_v], accum_dtype)
                T.clear(h_s)

                for i_p in T.Pipelined(num_partitions, num_stages=2):
                    for i_k, i_v in T.Parallel(dim_k, block_v):
                        initial_states[i_b, i_p, i_h, i_k, v_offset + i_v] = h_s[i_k, i_v]
                    T.copy(
                        summaries[i_b, i_p, i_h, :, v_offset : v_offset + block_v],
                        summary_s,
                        disable_tma=True,
                    )
                    for i_k, i_v in T.Parallel(dim_k, block_v):
                        h_s[i_k, i_v] = (
                            h_s[i_k, i_v] * T.exp2(log_decays[i_b, i_p, i_h, i_k] * LOG2_E)
                            + summary_s[i_k, i_v]
                        )

        return _main

    return _scan_func


@functools.lru_cache(maxsize=32)
def _gla_prefill_fused_replay_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    partition_chunks: int,
    scale: float,
    dtype: str,
    gate_dtype: str,
) -> Callable:
    """Replay independent partitions while producing output in the same CTA.

    This is the GLA specialization of the GDN prefill replay skeleton.  GLA
    has no delta-rule correction, so the transform warpgroup only prepares
    gate-adjusted Q/K while the state and output warpgroups run concurrently.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    if num_chunks % partition_chunks != 0:
        raise ValueError(
            f"num_chunks ({num_chunks}) must be divisible by partition_chunks ({partition_chunks})"
        )
    if chunk_size != 64 or dim_k != 128 or dim_v != 128:
        raise ValueError("fused GLA prefill replay requires chunk_size=64 and DK=DV=128")
    num_partitions = num_chunks // partition_chunks

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "-include", "tl_templates/cuda/gemm.h"],
    )
    def _replay_func(threads=512):
        qk_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_shape = [batch, seq_len, heads, dim_k]
        initial_shape = [batch, num_partitions, heads, dim_k, dim_v]
        o_shape = [batch, seq_len, heads, dim_v]
        final_shape = [batch, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(qk_shape, dtype),
            k: T.Tensor(qk_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_shape, gate_dtype),
            initial_states: T.Tensor(initial_shape, accum_dtype),
            o: T.Tensor(o_shape, dtype),
            final_state: T.Tensor(final_shape, dtype),
        ):
            with T.Kernel(num_partitions, batch, heads, threads=threads) as (i_p, i_b, i_h):
                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                k_s = T.alloc_shared([chunk_size, dim_k], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                g_s = T.alloc_shared([chunk_size, dim_k], gate_dtype)
                q_intra_s = T.alloc_shared([chunk_size, dim_k], dtype)
                h_s = T.alloc_shared([dim_k, dim_v], dtype)
                p_s = T.alloc_shared([chunk_size, chunk_size], dtype)

                h_f = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                p_f = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                o_f = T.alloc_fragment([chunk_size, dim_v], accum_dtype)

                tx = T.get_thread_binding()
                if tx < 128:
                    T.set_max_nreg(160, 1)
                    T.copy(initial_states[i_b, i_p, i_h, :, :], h_f)
                elif tx < 256:
                    T.set_max_nreg(64, 1)
                elif tx < 384:
                    T.set_max_nreg(160, 1)
                else:
                    T.set_max_nreg(24, 0)
                T.sync_threads()

                for i_local in T.serial(partition_chunks):
                    i_c = i_p * partition_chunks + i_local
                    chunk_start = i_c * chunk_size

                    if tx >= 384:
                        T.copy(
                            q[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            q_s,
                            disable_tma=True,
                        )
                        T.copy(
                            k[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            k_s,
                            disable_tma=True,
                        )
                        T.copy(
                            v[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            v_s,
                            disable_tma=True,
                        )
                        T.copy(
                            g_cumsum[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            g_s,
                            disable_tma=True,
                        )
                    T.sync_threads()

                    if tx < 128:
                        T.copy(h_f, h_s)
                        for i_k, i_v in T.Parallel(dim_k, dim_v):
                            h_f[i_k, i_v] *= T.exp2(
                                T.cast(g_s[chunk_size - 1, i_k], accum_dtype) * LOG2_E
                            )
                    elif tx < 256:
                        for i_t, i_k in T.Parallel(chunk_size, dim_k):
                            g_last = T.cast(g_s[chunk_size - 1, i_k], accum_dtype)
                            g_value = T.cast(g_s[i_t, i_k], accum_dtype)
                            q_value = T.cast(q_s[i_t, i_k], accum_dtype)
                            k_value = T.cast(k_s[i_t, i_k], accum_dtype)
                            q_intra_s[i_t, i_k] = T.cast(
                                q_value * T.exp2((g_value - g_last) * LOG2_E), dtype
                            )
                            k_s[i_t, i_k] = T.cast(
                                k_value * T.exp2((g_last - g_value) * LOG2_E), dtype
                            )
                            q_s[i_t, i_k] = T.cast(q_value * T.exp2(g_value * LOG2_E), dtype)
                    T.sync_threads()

                    if tx < 128:
                        T.gemm(k_s, v_s, h_f, transpose_A=True, clear_accum=False)
                    elif tx < 384:
                        T.clear(p_f)
                        T.gemm(q_intra_s, k_s, p_f, transpose_B=True)
                        for i_t, i_s in T.Parallel(chunk_size, chunk_size):
                            p_s[i_t, i_s] = T.cast(
                                T.if_then_else(i_s <= i_t, p_f[i_t, i_s] * scale, 0.0),
                                dtype,
                            )
                        T.clear(o_f)
                        T.gemm(q_s, h_s, o_f)
                        for i_t, i_v in T.Parallel(chunk_size, dim_v):
                            o_f[i_t, i_v] *= scale
                        T.gemm(p_s, v_s, o_f, clear_accum=False)
                        for i_t, i_v in T.Parallel(chunk_size, dim_v):
                            o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(o_f[i_t, i_v], dtype)
                    T.sync_threads()

                if tx < 128 and i_p == num_partitions - 1:
                    for i_k, i_v in T.Parallel(dim_k, dim_v):
                        final_state[i_b, i_h, i_k, i_v] = T.cast(h_f[i_k, i_v], dtype)

        return _main

    return _replay_func


# Pass 2a: compute the causal intra-chunk attention matrix.


@functools.lru_cache(maxsize=32)
def _gla_fwd_a_inter_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    """Compute strictly lower GLA attention sub-blocks with tensor cores.

    A 16-token sub-block is used as the unit of parallelism.  q/k are
    normalised around the first gate of the later block, which keeps both
    exponential factors at most one.
    """
    accum_dtype = "float32"
    block_t = 16
    num_sub_blocks = chunk_size // block_t
    num_inter_blocks = num_sub_blocks * (num_sub_blocks - 1) // 2
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _a_func(threads=128):
        qk_shape = [batch, seq_len, heads, dim_k]
        a_shape = [batch, seq_len, heads, chunk_size]

        @T.prim_func
        def _main(
            q: T.Tensor(qk_shape, dtype),
            k: T.Tensor(qk_shape, dtype),
            g_cumsum: T.Tensor(qk_shape, accum_dtype),
            A: T.Tensor(a_shape, dtype),
        ):
            with T.Kernel(num_chunks, num_inter_blocks, batch * heads, threads=threads) as (
                i_c,
                i_pair,
                i_bh,
            ):
                i_b = i_bh // heads
                i_h = i_bh % heads
                i_i = T.if_then_else(
                    i_pair < 1,
                    1,
                    T.if_then_else(i_pair < 3, 2, 3),
                )
                i_j = i_pair - i_i * (i_i - 1) // 2
                chunk_start = i_c * chunk_size
                q_start = chunk_start + i_i * block_t
                k_start = chunk_start + i_j * block_t

                q_s = T.alloc_shared([block_t, dim_k], dtype)
                k_s = T.alloc_shared([block_t, dim_k], dtype)
                gq_s = T.alloc_shared([block_t, dim_k], accum_dtype)
                gk_s = T.alloc_shared([block_t, dim_k], accum_dtype)
                q_adj_s = T.alloc_shared([block_t, dim_k], dtype)
                k_adj_s = T.alloc_shared([block_t, dim_k], dtype)
                a_frag = T.alloc_fragment([block_t, block_t], accum_dtype)

                T.copy(q[i_b, q_start : q_start + block_t, i_h, :], q_s, disable_tma=True)
                T.copy(k[i_b, k_start : k_start + block_t, i_h, :], k_s, disable_tma=True)
                T.copy(g_cumsum[i_b, q_start : q_start + block_t, i_h, :], gq_s, disable_tma=True)
                T.copy(g_cumsum[i_b, k_start : k_start + block_t, i_h, :], gk_s, disable_tma=True)

                for i_t, i_k in T.Parallel(block_t, dim_k):
                    anchor = gq_s[0, i_k]
                    q_adj_s[i_t, i_k] = T.cast(
                        T.cast(q_s[i_t, i_k], accum_dtype)
                        * T.exp2((gq_s[i_t, i_k] - anchor) * LOG2_E),
                        dtype,
                    )
                    k_adj_s[i_t, i_k] = T.cast(
                        T.cast(k_s[i_t, i_k], accum_dtype)
                        * T.exp2((anchor - gk_s[i_t, i_k]) * LOG2_E),
                        dtype,
                    )
                T.clear(a_frag)
                T.gemm(q_adj_s, k_adj_s, a_frag, transpose_B=True)

                for i_t, i_s in T.Parallel(block_t, block_t):
                    A[i_b, q_start + i_t, i_h, i_j * block_t + i_s] = T.cast(
                        a_frag[i_t, i_s] * scale,
                        dtype,
                    )

        return _main

    return _a_func


@functools.lru_cache(maxsize=32)
def _gla_fwd_a_intra_gemm_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    """Compute diagonal 16-token blocks with one fp32/TF32 GEMM."""
    accum_dtype = "float32"
    block_t = 16
    num_sub_blocks = chunk_size // block_t
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _a_func(threads=128):
        qk_shape = [batch, seq_len, heads, dim_k]
        a_shape = [batch, seq_len, heads, chunk_size]

        @T.prim_func
        def _main(
            q: T.Tensor(qk_shape, dtype),
            k: T.Tensor(qk_shape, dtype),
            g_cumsum: T.Tensor(qk_shape, accum_dtype),
            A: T.Tensor(a_shape, dtype),
        ):
            with T.Kernel(num_chunks, num_sub_blocks, batch * heads, threads=threads) as (
                i_c,
                i_i,
                i_bh,
            ):
                i_b = i_bh // heads
                i_h = i_bh % heads
                block_start = i_c * chunk_size + i_i * block_t

                q_s = T.alloc_shared([block_t, dim_k], dtype)
                k_s = T.alloc_shared([block_t, dim_k], dtype)
                g_s = T.alloc_shared([block_t, dim_k], accum_dtype)
                q_adj = T.alloc_shared([block_t, dim_k], accum_dtype)
                k_adj = T.alloc_shared([block_t, dim_k], accum_dtype)
                a_frag = T.alloc_fragment([block_t, block_t], accum_dtype)

                T.copy(q[i_b, block_start : block_start + block_t, i_h, :], q_s, disable_tma=True)
                T.copy(k[i_b, block_start : block_start + block_t, i_h, :], k_s, disable_tma=True)
                T.copy(
                    g_cumsum[i_b, block_start : block_start + block_t, i_h, :],
                    g_s,
                    disable_tma=True,
                )
                for i_t, i_k in T.Parallel(block_t, dim_k):
                    anchor = g_s[block_t - 1, i_k]
                    q_adj[i_t, i_k] = T.cast(q_s[i_t, i_k], accum_dtype) * T.exp2(
                        (g_s[i_t, i_k] - anchor) * LOG2_E
                    )
                    k_adj[i_t, i_k] = T.cast(k_s[i_t, i_k], accum_dtype) * T.exp2(
                        (anchor - g_s[i_t, i_k]) * LOG2_E
                    )
                T.clear(a_frag)
                T.gemm(q_adj, k_adj, a_frag, transpose_B=True)
                for i_t, i_s in T.Parallel(block_t, block_t):
                    if i_s <= i_t:
                        A[
                            i_b,
                            block_start + i_t,
                            i_h,
                            i_i * block_t + i_s,
                        ] = T.cast(a_frag[i_t, i_s] * scale, dtype)

        return _main

    return _a_func


# Pass 2b: compute output per chunk (parallel, B*H*NC thread blocks).
# A is prepared separately so both output terms are tensor-core GEMMs.


@functools.lru_cache(maxsize=32)
def _gla_fwd_o_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    state_dtype: str = "float32",
) -> Callable:
    """Compute output o for each chunk independently.

    Parallel over (batch, heads, chunks): B*H*NC thread blocks.
    Each block reads h[i_c] and g_cumsum from global memory.
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
    def _o_func(num_stages, threads=128):
        q_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        o_shape = [batch, seq_len, heads, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            h: T.Tensor(h_shape, state_dtype),
            A: T.Tensor([batch, seq_len, heads, chunk_size], dtype),
            o: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                chunk_start = i_c * chunk_size

                # h cast to native dtype for tensor core
                h_cast_s = T.alloc_shared([dim_k, dim_v], dtype)

                # Input buffers
                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)

                # Compute buffers
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)
                A_s = T.alloc_shared([chunk_size, chunk_size], dtype)

                # Load inputs via T.copy
                T.copy(
                    q[i_b, chunk_start : chunk_start + chunk_size, i_h, :], q_s, disable_tma=True
                )
                T.copy(
                    v[i_b, chunk_start : chunk_start + chunk_size, i_h, :], v_s, disable_tma=True
                )
                T.copy(
                    g_cumsum[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                    g_cumsum_s,
                    disable_tma=True,
                )
                T.copy(
                    A[i_b, chunk_start : chunk_start + chunk_size, i_h, :], A_s, disable_tma=True
                )
                for i_t, i_s in T.Parallel(chunk_size, chunk_size):
                    if i_s > i_t:
                        A_s[i_t, i_s] = T.cast(0.0, dtype)

                # Load h[i_c] and cast to native dtype
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_cast_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)

                # ---- Gated q (inter-chunk term, exp(g_cumsum) <= 1) ----
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    q_gated_s[i_t, i_k] = T.cast(
                        T.cast(q_s[i_t, i_k], accum_dtype) * T.exp2(g_cumsum_s[i_t, i_k] * LOG2_E),
                        dtype,
                    )

                # ---- o = scale * q_gated @ h + A @ v ----
                acc = T.alloc_fragment([chunk_size, dim_v], accum_dtype)
                T.fill(acc, 0.0)
                T.gemm(q_gated_s, h_cast_s, acc, policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    acc[i_t, i_v] = acc[i_t, i_v] * scale
                T.gemm(A_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)

                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(acc[i_t, i_v], dtype)

        return _main

    return _o_func


class GLAFwdKernel(Kernel):
    """GLA (Gated Linear Attention) forward kernel — three-pass architecture.

    Pass 0 (parallel, B*H*NC blocks): Pre-compute g_cumsum per chunk.
    Pass 1 (sequential, B*H blocks): Compute per-chunk hidden states h.
    Pass 2 (parallel, B*H*NC blocks): Compute output o per chunk independently.

    By pre-computing g_cumsum, the sequential h_kernel is free of T.Serial
    cumsum loops, dramatically reducing its latency.

    h_out is saved for the backward pass (no recomputation needed).

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
        tune: bool = False,
        state_dtype: str = "float32",
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
        self.state_dtype_name = state_dtype
        self.init_config(config, tune)
        if not tune:
            self._build_kernels(self.config)

    @property
    def default_config(self) -> dict:
        return {
            "g_num_stages": 2,
            "g_threads": 128,
            "h_num_stages": 2,
            "h_threads": 128,
            "a_inter_threads": 64,
            "o_num_stages": 2,
            "o_threads": 256,
            "num_v_partitions": 4,
            "num_k_partitions": 2,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for ns in [1, 2, 3]:
            for t_par in [64, 128, 256]:
                for t_seq in [64, 128, 256]:
                    for nvp in [2, 4]:
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

    def _build_kernels(self, config: dict) -> None:
        """Rebuild all sub-kernels from a config dict."""
        ns = config.get("num_stages", 2)
        thr_seq = config.get("threads_seq", config.get("threads", 256))
        thr_par = config.get("threads_par", config.get("threads", 256))
        g_ns = config.get("g_num_stages", ns)
        g_threads = config.get("g_threads", thr_par)
        h_ns = config.get("h_num_stages", ns)
        h_threads = config.get("h_threads", thr_seq)
        a_inter_threads = config.get("a_inter_threads", 64)
        o_ns = config.get("o_num_stages", ns)
        o_threads = config.get("o_threads", thr_par)
        num_vp = config.get("num_v_partitions", 4)
        num_kp = config.get("num_k_partitions", 1)
        self._g_fn = _gla_precompute_g_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.chunk_size,
            self.dtype_name,
        )(g_ns, g_threads)
        self._h_fn = _gla_fwd_h_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.dtype_name,
            self.state_dtype_name,
            num_v_partitions=num_vp,
            num_k_partitions=num_kp,
        )(h_ns, h_threads)
        self._a_inter_fn = _gla_fwd_a_inter_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.chunk_size,
            self.scale,
            self.dtype_name,
        )(a_inter_threads)
        self._a_intra_fn = _gla_fwd_a_intra_gemm_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.chunk_size,
            self.scale,
            self.dtype_name,
        )(32)
        self._o_fn = _gla_fwd_o_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            self.dtype_name,
            self.state_dtype_name,
        )(o_ns, o_threads)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Custom autotuning for multi-kernel forward pass."""
        if self.autotune_configs is None:
            return
        print(
            f"Start autotuning {self.__class__.__name__} ({len(self.autotune_configs)} configs)..."
        )

        B, T, H, K, V = (self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v)
        dtype_torch = getattr(torch, self.dtype_name)

        # Generate representative inputs
        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()

        best_lat = float("inf")
        best_cfg = None

        for cfg in self.autotune_configs:
            try:
                self._build_kernels(cfg)

                # Warmup run
                self.forward(q, k, v, g)
                torch.cuda.synchronize()

                lat = do_bench(
                    lambda: self.forward(q, k, v, g),
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, K, V = self.batch, self.heads, self.dim_k, self.dim_v
        dtype_torch = getattr(torch, self.dtype_name)
        state_dtype_torch = getattr(torch, self.state_dtype_name)

        if initial_state is None:
            init_state = torch.zeros(B, H, K, V, dtype=state_dtype_torch, device=q.device)
        else:
            init_state = initial_state.to(state_dtype_torch)

        # Pass 0: pre-compute g_cumsum (parallel, fast)
        g_cumsum = self._g_fn(g.to(dtype_torch))

        # Pass 1: sequential h computation
        h_out = self._h_fn(
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            init_state,
        )

        # Pass 2a: parallel causal intra-chunk attention computation
        A = self._a_inter_fn(
            q.to(dtype_torch),
            k.to(dtype_torch),
            g_cumsum,
        )
        self._a_intra_fn(
            q.to(dtype_torch),
            k.to(dtype_torch),
            g_cumsum,
            A,
        )

        # Pass 2b: parallel output computation
        o = self._o_fn(
            q.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            h_out,
            A,
        )

        # Store h_out for backward access
        self._h_out = h_out

        final_state = h_out[:, -1] if self.output_final_state else None
        return o, final_state


class GLAPrefillFwdKernel(GLAFwdKernel):
    """Inference-only GLA prefill with a partitioned long-context scan."""

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config["num_v_partitions"] = 1
        config["num_k_partitions"] = 2
        config["partition_chunks"] = 32
        config["partition_min_chunks"] = 512
        config["scan_threads"] = 128
        return config

    def _build_kernels(self, config: dict) -> None:
        num_chunks = self.seq_len // self.chunk_size
        requested_partition_chunks = config.get("partition_chunks", 64)
        self._partition_chunks = 0
        self._summary_fn = None
        self._scan_fn = None
        self._fused_replay_fn = None

        # This is the model-shaped Hopper path inherited from GDN prefill.
        # Short and irregular workloads retain the ordinary recurrence.
        use_partition = (
            self.batch == 1
            and self.heads <= 64
            and self.chunk_size == 64
            and self.dim_k == 128
            and self.dim_v == 128
            and self.dtype_name in ("float16", "bfloat16")
            and num_chunks >= config.get("partition_min_chunks", 512)
            and requested_partition_chunks > 0
            and num_chunks % requested_partition_chunks == 0
        )
        if not use_partition:
            super()._build_kernels(config)
            return

        self._partition_chunks = requested_partition_chunks
        ns = config.get("num_stages", 2)
        thr_par = config.get("threads_par", config.get("threads", 256))
        g_ns = config.get("g_num_stages", ns)
        g_threads = config.get("g_threads", thr_par)
        # Accumulate each chunk in FP32, then keep low-precision prefixes in
        # FP16.  FP16 has finer mantissa precision than BF16 at the gate ranges
        # used here while halving the former FP32 HBM traffic.  FP32 stays on
        # the generic path because this replay exceeds its shared-memory limit.
        gate_dtype = "float16"
        self._g_fn = _gla_precompute_g_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.chunk_size,
            self.dtype_name,
            gate_dtype,
        )(g_ns, g_threads)

        num_vp = config.get("num_v_partitions", 4)
        num_kp = config.get("num_k_partitions", 2)
        h_ns = config.get("h_num_stages", 2)
        h_threads = config.get("h_threads", 128)
        scan_threads = config.get("scan_threads", 128)
        self._summary_fn = _gla_fwd_h_summary_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self._partition_chunks,
            self.dtype_name,
            gate_dtype,
            num_v_partitions=num_vp,
            num_k_partitions=num_kp,
        )(h_ns, h_threads)
        self._scan_fn = _gla_fwd_h0_scan_kernel(
            self.batch,
            self.heads,
            num_chunks // self._partition_chunks,
            self.dim_k,
            self.dim_v,
        )(scan_threads)
        self._fused_replay_fn = _gla_prefill_fused_replay_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self._partition_chunks,
            self.scale,
            self.dtype_name,
            gate_dtype,
        )(512)

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=dim_v,
            chunk_size=chunk_size,
            scale=scale,
            output_final_state=True,
            dtype=dtype,
            config=config,
            tune=tune,
            state_dtype=str(dtype).split(".")[-1],
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._partition_chunks:
            o, final_state = super().forward(q, k, v, g, initial_state=None)
            assert final_state is not None
            return o, final_state

        dtype_torch = getattr(torch, self.dtype_name)
        q = q.to(dtype_torch)
        k = k.to(dtype_torch)
        v = v.to(dtype_torch)
        g_cumsum = self._g_fn(g.to(dtype_torch))

        assert self._summary_fn is not None
        assert self._scan_fn is not None
        assert self._fused_replay_fn is not None
        summaries, log_decays = self._summary_fn(k, v, g_cumsum)
        partition_initial_states = self._scan_fn(summaries, log_decays)
        return self._fused_replay_fn(q, k, v, g_cumsum, partition_initial_states)
