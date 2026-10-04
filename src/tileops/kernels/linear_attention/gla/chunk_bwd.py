"""GLA (Gated Linear Attention) backward kernel — TileLang implementation.

Two-pass architecture:
  Pass 1 (sequential reverse, B*H*Vp*Kp blocks): Accumulate dh per chunk, store dh_out.
  Pass 2 (parallel, B*H*NC blocks): Given h[i_c] and dh[i_c], compute dq,dk,dv,dg.
    A (intra-chunk attention) is recomputed internally — no external input needed.

h is read from the forward pass's h_out (no recomputation).

Reference:
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py
"""

import functools
from typing import Callable, Optional, Tuple

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GLABwdInterface,
    GLAChunkCall,
    head_count_refusal,
)
from tileops.kernels.linear_attention.gla.chunk_fwd import gla_precompute_g_kernel
from tileops.kernels.linear_attention.v_tile import GEMM_MIN_N, min_gemm_n
from tileops.utils import get_shared_memory_optin, get_sm_count, get_sm_version

__all__ = ["GLABwdKernel"]


def _dh_tile_refusal(dim_k_part: int, dim_v_part: int, threads: int) -> Optional[str]:
    """Why the dh gemm does not take a ``dim_k_part`` by ``dim_v_part`` tile, or ``None``.

    The bounds classify every compile probe over 64, 128 and 256 threads with 16, 32, 64
    and 128-wide tiles. Outside those extents the warp mapping decides, so a tile there is
    refused. Re-run the probes on a tilelang bump.
    """
    probed_extents = (16, 32, 64, 128)
    accumulator_per_thread = 4  # one 16x8 tile over a warp's 32 threads

    if dim_k_part not in probed_extents or dim_v_part not in probed_extents:
        return f"a {dim_k_part}x{dim_v_part} tile is outside the extents probes cover"
    floor_n = max(GEMM_MIN_N, min_gemm_n(threads))
    if dim_v_part < floor_n:
        return f"a {dim_v_part}-column B operand is below the {floor_n}-column floor"
    if dim_k_part * dim_v_part < accumulator_per_thread * threads:
        return (
            f"a {dim_k_part}x{dim_v_part} accumulator leaves a thread of {threads} fewer "
            f"than {accumulator_per_thread} elements"
        )
    return None


# Pass 1: compute dh per chunk (reverse order, sequential)


@functools.lru_cache(maxsize=32)
def _gla_bwd_dh_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    has_initial_state: bool,
    dtype: str,
    num_v_partitions: int = 1,
    num_k_partitions: int = 1,
) -> Callable:
    """Accumulate dh in reverse chunk order, store per-chunk dh.

    dh carries from one chunk to the next, so the walk is sequential and the block count
    is the only parallelism: ``batch * heads * Vp * Kp`` blocks, each owning one K slice
    of one V slice. A row of dh decays under its own gate and takes its own gemm row, so
    a K partition changes nothing the kernel computes.
    Stores dh_out[i_c] = dh after adding chunk i_c's contribution, before decay.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    if dim_v % num_v_partitions:
        raise ValueError(
            f"dim_v ({dim_v}) is not divisible by num_v_partitions ({num_v_partitions})"
        )
    if dim_k % num_k_partitions:
        raise ValueError(
            f"dim_k ({dim_k}) is not divisible by num_k_partitions ({num_k_partitions})"
        )
    dim_k_part = dim_k // num_k_partitions
    dim_v_part = dim_v // num_v_partitions
    if dim_v_part < GEMM_MIN_N:
        raise ValueError(
            f"dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) gives a "
            f"{dim_v_part}-column T.gemm B operand, below the minimum N extent ({GEMM_MIN_N})"
        )

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _dh_func(num_stages, threads=128):
        if dim_v_part < min_gemm_n(threads):
            raise ValueError(
                f"dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) "
                f"gives a {dim_v_part}-column T.gemm B operand, below the minimum N "
                f"extent ({min_gemm_n(threads)}) at {threads} threads"
            )
        # The whole tile is exempt: it is what this kernel built before partitioning.
        partitioned = (num_k_partitions, num_v_partitions) != (1, 1)
        refusal = _dh_tile_refusal(dim_k_part, dim_v_part, threads) if partitioned else None
        if refusal is not None:
            raise ValueError(
                f"dim_k ({dim_k}) over {num_k_partitions} partitions and dim_v ({dim_v}) "
                f"over {num_v_partitions} at {threads} threads: {refusal}"
            )
        q_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        do_shape = [batch, seq_len, heads, dim_v]
        dht_shape = [batch, heads, dim_k, dim_v]
        dh_out_shape = [batch, num_chunks, heads, dim_k, dim_v]
        dh0_shape = [batch, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            do: T.Tensor(do_shape, dtype),
            dht: T.Tensor(dht_shape, accum_dtype),
            dh_out: T.Tensor(dh_out_shape, accum_dtype),
            dh0: T.Tensor(dh0_shape, accum_dtype),
        ):
            parts = num_v_partitions * num_k_partitions
            with T.Kernel(batch * heads * parts, threads=threads) as bx:
                i_b = bx // (heads * parts)
                i_h = (bx // parts) % heads
                i_vp = (bx % parts) // num_k_partitions
                i_kp = bx % num_k_partitions
                v_offset = i_vp * dim_v_part
                k_offset = i_kp * dim_k_part

                dh_s = T.alloc_shared([dim_k_part, dim_v_part], accum_dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k_part], accum_dtype)
                q_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                do_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                q_gated_s = T.alloc_shared([chunk_size, dim_k_part], dtype)

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    dh_s[i_k, i_v] = dht[i_b, i_h, k_offset + i_k, v_offset + i_v]

                for t in T.Serial(num_chunks):
                    i_c = num_chunks - 1 - t
                    chunk_start = i_c * chunk_size

                    T.copy(
                        q[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        q_s,
                        disable_tma=True,
                    )
                    T.copy(
                        do[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            v_offset : v_offset + dim_v_part,
                        ],
                        do_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        g_cumsum_s,
                        disable_tma=True,
                    )

                    g_last = T.alloc_fragment([dim_k_part], accum_dtype)
                    for i_k in T.Parallel(dim_k_part):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    # q_gated, rebuilt in every partition that reads this chunk
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        q_gated_s[i_t, i_k] = T.cast(
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.exp2(g_cumsum_s[i_t, i_k] * LOG2E),
                            dtype,
                        )

                    # dh += scale * q_gated^T @ do_slice
                    dh_delta = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                    T.fill(dh_delta, 0.0)
                    T.gemm(
                        q_gated_s, do_s, dh_delta, transpose_A=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        dh_s[i_k, i_v] = dh_s[i_k, i_v] + scale * dh_delta[i_k, i_v]

                    # Store dh BEFORE decay
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        dh_out[i_b, i_c, i_h, k_offset + i_k, v_offset + i_v] = dh_s[i_k, i_v]

                    # Decay for next (earlier) chunk
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        dh_s[i_k, i_v] = dh_s[i_k, i_v] * T.exp2(g_last[i_k] * LOG2E)

                # Write dh0
                if has_initial_state:
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        dh0[i_b, i_h, k_offset + i_k, v_offset + i_v] = dh_s[i_k, i_v]
                else:
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        dh0[i_b, i_h, k_offset + i_k, v_offset + i_v] = 0.0

        return _main

    return _dh_func


# Pass 2: fused intra+inter kernel (eliminates global memory round-trip)


@functools.lru_cache(maxsize=32)
def _gla_bwd_fused_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    sub_chunk_size: int = 16,
    lean: bool = False,
) -> Callable:
    """Fused intra+inter backward kernel.

    Phase A: Compute dq_intra, dk_intra, dv_intra using sub-chunk GEMM tiling
             (kept in registers — no global memory write).
    Phase B: Add inter-chunk contributions via GEMM, compute dg, write final outputs.

    This avoids the global memory round-trip of the split intra/inter approach.

    *lean* forms dA before A and has phase B read v, do, q and k again and h late, so
    neither phase A's operands nor h stay live through ``dv_inter``; same results.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    BT = chunk_size
    BC = sub_chunk_size
    NS = BT // BC

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _fused_func(num_stages, threads=256):
        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        do_shape = [batch, seq_len, heads, dim_v]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        dh_shape = [batch, num_chunks, heads, dim_k, dim_v]
        dq_shape = [batch, seq_len, heads, dim_k]
        dk_shape = [batch, seq_len, heads, dim_k]
        dv_shape = [batch, seq_len, heads, dim_v]
        dg_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            do: T.Tensor(do_shape, dtype),
            h: T.Tensor(h_shape, accum_dtype),
            dh: T.Tensor(dh_shape, accum_dtype),
            dq_out: T.Tensor(dq_shape, accum_dtype),
            dk_out: T.Tensor(dk_shape, accum_dtype),
            dv_out: T.Tensor(dv_shape, accum_dtype),
            dg_out: T.Tensor(dg_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                chunk_start = i_c * BT

                q_s = T.alloc_shared([BT, dim_k], dtype)
                k_s = T.alloc_shared([BT, dim_k], dtype)
                v_s = T.alloc_shared([BT, dim_v], dtype)
                do_s = T.alloc_shared([BT, dim_v], dtype)
                g_cumsum_s = T.alloc_shared([BT, dim_k], accum_dtype)
                A_s = T.alloc_shared([BT, BT], dtype)

                T.copy(q[i_b, chunk_start : chunk_start + BT, i_h, :], q_s, disable_tma=True)
                T.copy(k[i_b, chunk_start : chunk_start + BT, i_h, :], k_s, disable_tma=True)
                T.copy(v[i_b, chunk_start : chunk_start + BT, i_h, :], v_s, disable_tma=True)
                T.copy(do[i_b, chunk_start : chunk_start + BT, i_h, :], do_s, disable_tma=True)
                T.copy(
                    g_cumsum[i_b, chunk_start : chunk_start + BT, i_h, :],
                    g_cumsum_s,
                    disable_tma=True,
                )

                # PHASE A: Intra-chunk (results kept in fragments)

                if lean:
                    dA_frag = T.alloc_fragment([BT, BT], accum_dtype)
                    T.fill(dA_frag, 0.0)
                    T.gemm(do_s, v_s, dA_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # ---- A[i,j] = scale * sum_k q*k*exp(g_i - g_j), causal ----
                A_frag = T.alloc_fragment([BT, BT], accum_dtype)
                T.fill(A_frag, 0.0)
                for i_k in T.Serial(dim_k):
                    for i_t, i_j in T.Parallel(BT, BT):
                        A_frag[i_t, i_j] = A_frag[i_t, i_j] + (
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.cast(k_s[i_j, i_k], accum_dtype)
                            * T.exp2((g_cumsum_s[i_t, i_k] - g_cumsum_s[i_j, i_k]) * LOG2E)
                        )
                for i_t, i_j in T.Parallel(BT, BT):
                    A_s[i_t, i_j] = T.cast(
                        T.if_then_else(i_j <= i_t, A_frag[i_t, i_j] * scale, 0.0), dtype
                    )

                # dv_intra = A^T @ do (keep in fragment for phase B)
                dv_frag = T.alloc_fragment([BT, dim_v], accum_dtype)
                T.fill(dv_frag, 0.0)
                T.gemm(A_s, do_s, dv_frag, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dA = scale * do @ v^T, causal (overwrite A_s)
                if not lean:
                    dA_frag = T.alloc_fragment([BT, BT], accum_dtype)
                    T.fill(dA_frag, 0.0)
                    T.gemm(do_s, v_s, dA_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_j in T.Parallel(BT, BT):
                    A_s[i_t, i_j] = T.cast(
                        T.if_then_else(
                            i_j <= i_t,
                            scale * dA_frag[i_t, i_j],
                            0.0,
                        ),
                        dtype,
                    )

                # Sub-chunk tiled dq_intra (kept in fragment)
                dq_frag = T.alloc_fragment([BT, dim_k], accum_dtype)
                T.fill(dq_frag, 0.0)
                dA_sub = T.alloc_shared([BC, BC], dtype)
                k_shifted_sub = T.alloc_shared([BC, dim_k], dtype)

                for s_i in T.Serial(NS):
                    dq_sub = T.alloc_fragment([BC, dim_k], accum_dtype)
                    T.fill(dq_sub, 0.0)

                    for s_j in T.Serial(NS):
                        if s_j < s_i:
                            for i_t, i_j in T.Parallel(BC, BC):
                                dA_sub[i_t, i_j] = A_s[s_i * BC + i_t, s_j * BC + i_j]
                            for i_t, i_k in T.Parallel(BC, dim_k):
                                k_shifted_sub[i_t, i_k] = T.cast(
                                    T.cast(k_s[s_j * BC + i_t, i_k], accum_dtype)
                                    * T.exp2(
                                        (
                                            g_cumsum_s[s_i * BC, i_k]
                                            - g_cumsum_s[s_j * BC + i_t, i_k]
                                        )
                                        * LOG2E
                                    ),
                                    dtype,
                                )
                            T.gemm(dA_sub, k_shifted_sub, dq_sub, policy=T.GemmWarpPolicy.FullRow)

                    for i_local, i_k in T.Parallel(BC, dim_k):
                        dq_sub[i_local, i_k] = dq_sub[i_local, i_k] * T.exp2(
                            (g_cumsum_s[s_i * BC + i_local, i_k] - g_cumsum_s[s_i * BC, i_k])
                            * LOG2E
                        )

                    # Diagonal: fragment-based to avoid bf16 smem codegen bug
                    for j_local in T.Serial(BC):
                        dA_col = T.alloc_fragment([BC], accum_dtype)
                        k_row = T.alloc_fragment([dim_k], accum_dtype)
                        g_j = T.alloc_fragment([dim_k], accum_dtype)
                        for i_local in T.Parallel(BC):
                            dA_col[i_local] = T.if_then_else(
                                j_local <= i_local,
                                T.cast(A_s[s_i * BC + i_local, s_i * BC + j_local], accum_dtype),
                                0.0,
                            )
                        for i_k in T.Parallel(dim_k):
                            k_row[i_k] = T.cast(k_s[s_i * BC + j_local, i_k], accum_dtype)
                            g_j[i_k] = g_cumsum_s[s_i * BC + j_local, i_k]
                        for i_local, i_k in T.Parallel(BC, dim_k):
                            dq_sub[i_local, i_k] = dq_sub[i_local, i_k] + dA_col[i_local] * k_row[
                                i_k
                            ] * T.exp2((g_cumsum_s[s_i * BC + i_local, i_k] - g_j[i_k]) * LOG2E)

                    for i_local, i_k in T.Parallel(BC, dim_k):
                        dq_frag[s_i * BC + i_local, i_k] = dq_sub[i_local, i_k]

                # Sub-chunk tiled dk_intra (kept in fragment)
                dk_frag = T.alloc_fragment([BT, dim_k], accum_dtype)
                T.fill(dk_frag, 0.0)
                q_shifted_sub = T.alloc_shared([BC, dim_k], dtype)

                for s_j in T.Serial(NS):
                    dk_sub = T.alloc_fragment([BC, dim_k], accum_dtype)
                    T.fill(dk_sub, 0.0)

                    for s_i in T.Serial(NS):
                        if s_i > s_j:
                            for i_i, i_j in T.Parallel(BC, BC):
                                dA_sub[i_i, i_j] = A_s[s_i * BC + i_i, s_j * BC + i_j]
                            for i_t, i_k in T.Parallel(BC, dim_k):
                                q_shifted_sub[i_t, i_k] = T.cast(
                                    T.cast(q_s[s_i * BC + i_t, i_k], accum_dtype)
                                    * T.exp2(
                                        (
                                            g_cumsum_s[s_i * BC + i_t, i_k]
                                            - g_cumsum_s[(s_j + 1) * BC - 1, i_k]
                                        )
                                        * LOG2E
                                    ),
                                    dtype,
                                )
                            T.gemm(
                                dA_sub,
                                q_shifted_sub,
                                dk_sub,
                                transpose_A=True,
                                policy=T.GemmWarpPolicy.FullRow,
                            )

                    for j_local, i_k in T.Parallel(BC, dim_k):
                        dk_sub[j_local, i_k] = dk_sub[j_local, i_k] * T.exp2(
                            (
                                g_cumsum_s[(s_j + 1) * BC - 1, i_k]
                                - g_cumsum_s[s_j * BC + j_local, i_k]
                            )
                            * LOG2E
                        )

                    # Diagonal: fragment-based
                    for i_local in T.Serial(BC):
                        dA_row = T.alloc_fragment([BC], accum_dtype)
                        q_row = T.alloc_fragment([dim_k], accum_dtype)
                        g_i = T.alloc_fragment([dim_k], accum_dtype)
                        for j_local in T.Parallel(BC):
                            dA_row[j_local] = T.if_then_else(
                                j_local <= i_local,
                                T.cast(A_s[s_j * BC + i_local, s_j * BC + j_local], accum_dtype),
                                0.0,
                            )
                        for i_k in T.Parallel(dim_k):
                            q_row[i_k] = T.cast(q_s[s_j * BC + i_local, i_k], accum_dtype)
                            g_i[i_k] = g_cumsum_s[s_j * BC + i_local, i_k]
                        for j_local, i_k in T.Parallel(BC, dim_k):
                            dk_sub[j_local, i_k] = dk_sub[j_local, i_k] + dA_row[j_local] * q_row[
                                i_k
                            ] * T.exp2((g_i[i_k] - g_cumsum_s[s_j * BC + j_local, i_k]) * LOG2E)

                    for j_local, i_k in T.Parallel(BC, dim_k):
                        dk_frag[s_j * BC + j_local, i_k] = dk_sub[j_local, i_k]

                # PHASE B: Inter-chunk gradients + combine + dg

                h_cast_s = T.alloc_shared([dim_k, dim_v], dtype)
                dh_cast_s = T.alloc_shared([dim_k, dim_v], dtype)

                if not lean:
                    for i_k, i_v in T.Parallel(dim_k, dim_v):
                        h_cast_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    dh_cast_s[i_k, i_v] = T.cast(dh[i_b, i_c, i_h, i_k, i_v], dtype)

                g_last = T.alloc_fragment([dim_k], accum_dtype)
                for i_k in T.Parallel(dim_k):
                    g_last[i_k] = g_cumsum_s[BT - 1, i_k]

                # dv_inter = k_adj @ dh
                k_gated_s = T.alloc_shared([BT, dim_k], dtype)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    k_gated_s[i_t, i_k] = T.cast(
                        T.cast(k_s[i_t, i_k], accum_dtype)
                        * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2E),
                        dtype,
                    )

                T.gemm(k_gated_s, dh_cast_s, dv_frag, policy=T.GemmWarpPolicy.FullRow)
                # dv_frag now = dv_intra + dv_inter (accumulated)
                for i_t, i_v in T.Parallel(BT, dim_v):
                    dv_out[i_b, chunk_start + i_t, i_h, i_v] = dv_frag[i_t, i_v]

                # dq_inter = do @ h^T → write to shared to avoid layout conflict
                if lean:
                    do_b = T.alloc_shared([BT, dim_v], dtype)
                    T.copy(do[i_b, chunk_start : chunk_start + BT, i_h, :], do_b, disable_tma=True)
                    for i_k, i_v in T.Parallel(dim_k, dim_v):
                        h_cast_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)
                dq_inter_s = T.alloc_shared([BT, dim_k], accum_dtype)
                dq_inter_frag = T.alloc_fragment([BT, dim_k], accum_dtype)
                T.fill(dq_inter_frag, 0.0)
                T.gemm(
                    do_b if lean else do_s,
                    h_cast_s,
                    dq_inter_frag,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dq_inter_s[i_t, i_k] = dq_inter_frag[i_t, i_k]

                # dq = dq_intra + scale * dq_inter * exp(g_cumsum)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dq_frag[i_t, i_k] = dq_frag[i_t, i_k] + scale * dq_inter_s[i_t, i_k] * T.exp2(
                        g_cumsum_s[i_t, i_k] * LOG2E
                    )
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dq_out[i_b, chunk_start + i_t, i_h, i_k] = dq_frag[i_t, i_k]

                # dk_inter = v @ dh^T → write to shared to avoid layout conflict
                if lean:
                    v_b = T.alloc_shared([BT, dim_v], dtype)
                    T.copy(v[i_b, chunk_start : chunk_start + BT, i_h, :], v_b, disable_tma=True)
                dk_inter_s = T.alloc_shared([BT, dim_k], accum_dtype)
                dk_inter_frag = T.alloc_fragment([BT, dim_k], accum_dtype)
                T.fill(dk_inter_frag, 0.0)
                T.gemm(
                    v_b if lean else v_s,
                    dh_cast_s,
                    dk_inter_frag,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dk_inter_s[i_t, i_k] = dk_inter_frag[i_t, i_k]

                # dk = dk_intra + dk_inter * exp(g_last - g_cumsum)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dk_frag[i_t, i_k] = dk_frag[i_t, i_k] + dk_inter_s[i_t, i_k] * T.exp2(
                        (g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2E
                    )
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dk_out[i_b, chunk_start + i_t, i_h, i_k] = dk_frag[i_t, i_k]

                # ==== dg ====
                if lean:
                    q_b = T.alloc_shared([BT, dim_k], dtype)
                    k_b = T.alloc_shared([BT, dim_k], dtype)
                    T.copy(q[i_b, chunk_start : chunk_start + BT, i_h, :], q_b, disable_tma=True)
                    T.copy(k[i_b, chunk_start : chunk_start + BT, i_h, :], k_b, disable_tma=True)
                dg_inter = T.alloc_shared([dim_k], accum_dtype)
                for i_k in T.Parallel(dim_k):
                    dg_inter[i_k] = 0.0
                for i_v2 in T.Serial(dim_v):
                    for i_k in T.Parallel(dim_k):
                        dg_inter[i_k] = dg_inter[i_k] + (
                            h[i_b, i_c, i_h, i_k, i_v2]
                            * T.cast(dh[i_b, i_c, i_h, i_k, i_v2], accum_dtype)
                        )
                for i_k in T.Parallel(dim_k):
                    dg_inter[i_k] = dg_inter[i_k] * T.exp2(g_last[i_k] * LOG2E)

                # Correction: k * dk_inter_gated
                corr_s = T.alloc_shared([BT, dim_k], accum_dtype)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    corr_s[i_t, i_k] = (
                        T.cast((k_b if lean else k_s)[i_t, i_k], accum_dtype)
                        * dk_inter_s[i_t, i_k]
                        * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2E)
                    )
                for i_t in T.Serial(BT):
                    for i_k in T.Parallel(dim_k):
                        dg_inter[i_k] = dg_inter[i_k] + corr_s[i_t, i_k]

                # dg_local = q * dq - k * dk (using final combined values)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    g_cumsum_s[i_t, i_k] = (
                        T.cast((q_b if lean else q_s)[i_t, i_k], accum_dtype) * dq_frag[i_t, i_k]
                        - T.cast((k_b if lean else k_s)[i_t, i_k], accum_dtype) * dk_frag[i_t, i_k]
                    )

                # Reverse cumsum
                for s in T.Serial(BT - 1):
                    i_t_rev = BT - 2 - s
                    for i_k in T.Parallel(dim_k):
                        g_cumsum_s[i_t_rev, i_k] = (
                            g_cumsum_s[i_t_rev, i_k] + g_cumsum_s[i_t_rev + 1, i_k]
                        )

                for i_t, i_k in T.Parallel(BT, dim_k):
                    dg_out[i_b, chunk_start + i_t, i_h, i_k] = g_cumsum_s[i_t, i_k] + dg_inter[i_k]

        return _main

    return _fused_func


class GLABwdKernel(Kernel, GLABwdInterface):
    """GLA backward kernel — two-pass architecture.

    Pass 1 (sequential reverse, B*H*Vp*Kp blocks): Accumulate dh per chunk.
    Pass 2 (parallel, B*H*NC blocks): Fused intra+inter kernel computes
        dq, dk, dv, dg in a single pass using sub-chunk GEMM tiling.

    h is read from forward's h_out (no recomputation needed).
    """

    supported_archs: list[int] = [80, 89, 90]
    # Threads of the default dh pass: one warp group.
    _THREADS_SEQ = 128

    @classmethod
    def applies(cls, call: GLAChunkCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: GLAChunkCall) -> Optional[str]:
        """Why no program serves this call, or ``None``; reads lower bounds, so a call above
        them that TileLang still cannot place in the device's shared memory is built and
        fails at launch."""
        reason = head_count_refusal(call.heads) or cls.region_refusal(
            call.dim_k, call.dim_v, call.chunk_size, call.dtype, call.arch
        )
        if reason is not None or not call.smem_budget:
            return reason
        c, k, v, elem = call.chunk_size, call.dim_k, call.dim_v, call.dtype.itemsize
        dh = min(
            cls._dh_shared_bytes(c, k // kp, v // vp, elem)
            for vp, kp in cls._partitionings(k, v, cls._THREADS_SEQ) or [(1, 1)]
        )
        need = max(cls._fused_live_bytes(c, k, v, elem), dh)
        if need <= call.smem_budget:
            return None
        return (
            f"needs at least {need} bytes of shared memory per block at chunk {c}, head dims "
            f"{k} / {v} in {call.dtype}; the device gives {call.smem_budget}"
        )

    @classmethod
    def entry_for(cls, call: GLAChunkCall) -> Entry:
        """``has_initial_state`` is a launch argument, so it is out of the identity."""
        index = call.device.index if call.device is not None else None
        identity = (
            call.batch,
            call.seq_len,
            call.heads,
            call.dim_k,
            call.dim_v,
            call.chunk_size,
            call.scale,
            call.dtype,
            index,
        )
        return identity, lambda: cls(*identity[:-1], device_index=index)

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale if scale > 0 else dim_k**-0.5
        self.dtype = dtype
        self.dtype_name = str(dtype).split(".")[-1]
        reason = self.region_refusal(
            dim_k, dim_v, chunk_size, dtype, get_sm_version(self.device_index)
        )
        if reason:
            raise ValueError(f"{type(self).__name__} does not serve this call: {reason}")
        # The default fused pass only where every buffer it allocates fits the device.
        default = self._fused_shared_bytes(chunk_size, dim_k, dim_v, dtype.itemsize)
        self.lean = default > get_shared_memory_optin(self.device_index)
        self.init_config(config, tune)
        if not tune:
            self._build_kernels(self.config)

    @staticmethod
    def region_refusal(
        dim_k: int, dim_v: int, chunk_size: int, dtype: torch.dtype, arch: int
    ) -> Optional[str]:
        """Why the default tiling cannot build these extents, or ``None`` when it can.

        The fused pass runs its chunk GEMMs on four warps, which take whole 16-row
        tiles of ``chunk_size`` rows: 32 splits two by two, and otherwise all four
        split the rows, which takes a multiple of 64. The state gradient runs over
        a partition of ``dim_k x dim_v`` that :meth:`_partitionings` bounds, and the
        output GEMMs put ``dim_k`` on the columns. On SM90 a 16-bit ``dim_k`` of at
        least 64 takes the warp-group
        instruction, which splits four warps over it and admits ``dim_k`` 64 past
        a multiple of 128; the per-warp instruction does not.
        """
        if not (chunk_size == 32 or chunk_size % 64 == 0):
            return f"chunk_size={chunk_size} must be 32 or a multiple of 64"
        warp_group = arch == 90 and torch.finfo(dtype).bits == 16
        served = (
            dim_k == 32
            and dim_v % 128 == 0
            or dim_k == 64
            and dim_v % 64 == 0
            or dim_k % 128 == 0
            and dim_v % 32 == 0
            or warp_group
            and dim_k % 128 == 64
            and dim_v % 64 == 0
        )
        if not served:
            return (
                f"dim_k={dim_k}, dim_v={dim_v}: dim_k must be 32 with dim_v a multiple of "
                "128, 64 with dim_v a multiple of 64, or a multiple of 128 with dim_v a "
                "multiple of 32"
                + (
                    "; or 64 past a multiple of 128 with dim_v a multiple of 64"
                    if warp_group
                    else ""
                )
            )
        return None

    @staticmethod
    def _partitionings(dim_k: int, dim_v: int, threads_seq: int) -> list[tuple[int, int]]:
        """The (V, K) partitionings the dh kernel builds at *threads_seq*, finest first."""
        counts = (1, 2, 4, 8)  # a power of two keeps a probed dimension a probed extent
        pairs = [
            (vp, kp)
            for vp in counts
            for kp in counts
            if dim_v % vp == 0
            and dim_k % kp == 0
            and _dh_tile_refusal(dim_k // kp, dim_v // vp, threads_seq) is None
        ]
        # Ties go to the V split, the minor axis of every tensor a block reads and writes.
        return sorted(pairs, key=lambda pair: (pair[0] * pair[1], pair[0]), reverse=True)

    @staticmethod
    def _dh_shared_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Shared memory of the dh pass over a *k* by *v* slice: its five buffers stay live
        through the chunk walk, so the sum is what TileLang compiles."""
        return k * v * 4 + c * k * (4 + 2 * elem) + c * v * elem

    @staticmethod
    def _fused_shared_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Upper bound on the default fused pass's shared memory: every buffer it allocates,
        with the 16-row sub-chunk tiles."""
        a, b, f = c * k * elem, c * v * elem, c * k * 4
        sub = 16 * 16 * elem + 2 * 16 * k * elem
        return 3 * a + 2 * b + 4 * f + c * c * elem + 2 * k * v * elem + sub + k * 4

    @staticmethod
    def _fused_live_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Lower bound on the lean fused pass's shared memory: the largest set of its buffers
        live at once, the fp32 gate in all."""
        a, b, h, causal, f = c * k * elem, c * v * elem, k * v * elem, c * c * elem, c * k * 4
        sub = 16 * 16 * elem + 16 * k * elem
        dg = 2 * a + 2 * f + k * 4
        return max(2 * a + 2 * b, 2 * a + b + causal, 2 * a + causal + sub, b + 2 * h, dg) + f

    @property
    def default_config(self) -> dict:
        # The finest partitioning whose dh slice fits the device and whose blocks still fit
        # one wave. A wider block splits the gemm's B operand across warp groups and doubles
        # the V tile a partitioning must leave; past the SM count a further split only
        # repeats the gated query.
        threads_seq = self._THREADS_SEQ
        blocks = self.batch * self.heads
        sm_count = get_sm_count(self.device_index)
        budget = get_shared_memory_optin(self.device_index)
        c, elem = self.chunk_size, self.dtype.itemsize
        admitted = [
            (vp, kp)
            for vp, kp in self._partitionings(self.dim_k, self.dim_v, threads_seq)
            if self._dh_shared_bytes(c, self.dim_k // kp, self.dim_v // vp, elem) <= budget
        ] or [(1, 1)]
        fitting = [pair for pair in admitted if blocks * pair[0] * pair[1] <= sm_count]
        num_v_partitions, num_k_partitions = (fitting or admitted[-1:])[0]
        return {
            "num_stages": 1,
            "threads_par": 128,
            "threads_seq": threads_seq,
            "num_v_partitions": num_v_partitions,
            "num_k_partitions": num_k_partitions,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for ns in [1, 2, 3]:
            for t_par in [64, 128, 256]:
                for t_seq in [64, 128, 256]:
                    for nvp, nkp in self._partitionings(self.dim_k, self.dim_v, t_seq):
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
        num_vp = config.get("num_v_partitions", 4)
        num_kp = config.get("num_k_partitions", 1)
        self._g_fn = gla_precompute_g_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.chunk_size,
            self.dtype_name,
        )(ns, thr_par)
        self._dh_fn = _gla_bwd_dh_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            False,
            self.dtype_name,
            num_v_partitions=num_vp,
            num_k_partitions=num_kp,
        )(1, thr_seq)
        self._dh_fn_with_init = _gla_bwd_dh_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            True,
            self.dtype_name,
            num_v_partitions=num_vp,
            num_k_partitions=num_kp,
        )(1, thr_seq)
        self._fused_fn = _gla_bwd_fused_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            self.dtype_name,
            lean=self.lean,
        )(ns, thr_par)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Custom autotuning for multi-kernel backward pass."""
        if self.autotune_configs is None:
            return
        print(
            f"Start autotuning {self.__class__.__name__} ({len(self.autotune_configs)} configs)..."
        )

        B, T, H, K, V = (self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v)
        BT = self.chunk_size
        NT = T // BT
        dtype_torch = self.dtype

        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()
        h = torch.randn(B, NT + 1, H, K, V, device="cuda", dtype=torch.float32) * 0.01
        do = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        dht = torch.zeros(B, H, K, V, dtype=torch.float32, device="cuda")

        best_lat = float("inf")
        best_cfg = None

        for cfg in self.autotune_configs:
            try:
                self._build_kernels(cfg)

                # Warmup run
                self.forward(q, k, v, g, h, do, dht)
                torch.cuda.synchronize()

                lat = do_bench(
                    lambda: self.forward(q, k, v, g, h, do, dht),
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
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        has_initial_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype_torch = self.dtype

        # Pre-compute g_cumsum (parallel, fast)
        g_cumsum = self._g_fn(g.to(dtype_torch))

        # Pass 1: compute dh per chunk (sequential reverse)
        dh_fn = self._dh_fn_with_init if has_initial_state else self._dh_fn
        dh_out, dh0 = dh_fn(
            q.to(dtype_torch),
            g_cumsum,
            do.to(dtype_torch),
            dht,
        )

        # Pass 2: fused intra+inter (dq, dk, dv, dg in one kernel)
        dq, dk, dv, dg = self._fused_fn(
            q.to(dtype_torch),
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            do.to(dtype_torch),
            h.to(torch.float32),
            dh_out,
        )

        return dq, dk, dv, dg
