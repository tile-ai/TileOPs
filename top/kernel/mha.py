import torch
from torch import nn
from torch.nn import functional as F
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import autotune
import itertools
from top.utils import is_hopper


__all__ = ['MHAKernel']


def get_configs():
    block_M = [32, 64, 128]
    block_N = [32, 64, 128]
    num_stages = [1, 2, 3]
    threads = [128, 256]
    _configs = list(itertools.product(block_M, block_N, num_stages, threads))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'num_stages': c[2],
        'threads': c[3]
    } for c in _configs]
    return configs


def _mha_fwd(batch, heads, seq_len, dim, is_causal, dtype: str, tune=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    accum_dtype = "float"

    def _mha_fwd_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA0(
            K: T.Tensor(shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(shape, dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def _mha_fwd_main(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                            logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)  # consider directly LD/ST (bypass smem) for sm80?
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

        @T.prim_func
        def _mha_fwd_main_wgmma_pipelined(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                            logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])
    
        return _mha_fwd_main_wgmma_pipelined if is_hopper() else _mha_fwd_main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tl.jit(
            out_idx=[3, 4], 
            pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True},
            compile_flags=["-O3", "-DENABLE_BF16"])
        def _mha_fwd_kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return _mha_fwd_func(block_M, block_N, num_stages, threads)

        return _mha_fwd_kernel()
    else:

        @tl.jit(
            out_idx=[3, 4], 
            pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True},
            compile_flags=["-O3", "-DENABLE_BF16"])
        def _mha_fwd_kernel(block_M, block_N, num_stages, threads):
            return _mha_fwd_func(block_M, block_N, num_stages, threads)

        return _mha_fwd_kernel


@tl.jit(
    out_idx=[2], 
    pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    compile_flags=["-O3", "-DENABLE_BF16"])
def _mha_bwd_preprocess(batch, heads, seq_len, dim, dtype: str):
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
            O: T.Tensor(shape, dtype),  # type: ignore
            dO: T.Tensor(shape, dtype),  # type: ignore
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o)
                T.copy(dO[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk:(by + 1) * blk])

    return flash_bwd_prep


def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(dQ.shape,
                    lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])


def _mha_bwd(batch, heads, seq_len, dim, is_causal, dtype: str, tune=False):
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    accum_dtype = "float"

    def _mha_bwd_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _mha_bwd_main(
                Q: T.Tensor(shape, dtype),  # type: ignore
                K: T.Tensor(shape, dtype),  # type: ignore
                V: T.Tensor(shape, dtype),  # type: ignore
                dO: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dQ: T.Tensor(shape, accum_dtype),  # type: ignore
                dK: T.Tensor(shape, dtype),  # type: ignore
                dV: T.Tensor(shape, dtype),  # type: ignore
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
                K_shared = T.alloc_shared([block_M, dim], dtype)
                dsT_shared = T.alloc_shared([block_M, block_N], dtype)
                # should not store K to local if dim is large
                # K_local = T.alloc_fragment([block_M, dim], dtype)
                # K_local_T = T.alloc_fragment([block_M, dim], dtype)
                # V_local = T.alloc_fragment([block_M, dim], dtype)
                q = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_M, dim], dtype)
                qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
                dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
                qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
                dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_N], accum_dtype)
                delta = T.alloc_shared([block_N], accum_dtype)
                do = T.alloc_shared([block_N, dim], dtype)
                dv = T.alloc_fragment([block_M, dim], accum_dtype)
                dk = T.alloc_fragment([block_M, dim], accum_dtype)
                dq = T.alloc_fragment([block_N, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_M, dim], dtype)
                dk_shared = T.alloc_shared([block_M, dim], dtype)

                T.annotate_layout({
                    # dQ: make_dq_layout(dQ),
                    K_shared: tl.layout.make_swizzled_layout(K_shared),
                    dv_shared: tl.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tl.layout.make_swizzled_layout(dk_shared),
                })
                T.copy(K[bz, by * block_M:(by + 1) * block_M, bx, :], K_shared)
                T.copy(V[bz, by * block_M:(by + 1) * block_M, bx, :], V_shared)
                T.clear(dv)
                T.clear(dk)
                loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_N)
                for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q)
                    T.clear(qkT)
                    T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j,
                                                       qkT[i, j], 0)
                    T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do)
                    T.clear(dsT)
                    T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta)

                    for i, j in T.Parallel(block_M, block_N):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(dsT_cast, dsT_shared)
                    T.clear(dq)
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                    for i, j in T.Parallel(block_N, dim):
                            T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
                T.copy(dv, dv_shared)
                T.copy(dk, dk_shared)
                T.copy(dv_shared, dV[bz, by * block_M:(by + 1) * block_M, bx, :])
                T.copy(dk_shared, dK[bz, by * block_M:(by + 1) * block_M, bx, :])

        @T.prim_func
        def _mha_bwd_main_wgmma_pipelined(
                Q: T.Tensor(shape, dtype),  # type: ignore
                K: T.Tensor(shape, dtype),  # type: ignore
                V: T.Tensor(shape, dtype),  # type: ignore
                dO: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dQ: T.Tensor(shape, accum_dtype),  # type: ignore
                dK: T.Tensor(shape, dtype),  # type: ignore
                dV: T.Tensor(shape, dtype),  # type: ignore
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
                K_shared = T.alloc_shared([block_M, dim], dtype)
                dsT_shared = T.alloc_shared([block_M, block_N], dtype)
                # should not store K to local if dim is large
                # K_local = T.alloc_fragment([block_M, dim], dtype)
                # K_local_T = T.alloc_fragment([block_M, dim], dtype)
                # V_local = T.alloc_fragment([block_M, dim], dtype)
                q = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_M, dim], dtype)
                qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
                dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
                qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
                dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_N], accum_dtype)
                delta = T.alloc_shared([block_N], accum_dtype)
                do = T.alloc_shared([block_N, dim], dtype)
                dv = T.alloc_fragment([block_M, dim], accum_dtype)
                dk = T.alloc_fragment([block_M, dim], accum_dtype)
                dq = T.alloc_fragment([block_N, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_M, dim], dtype)
                dk_shared = T.alloc_shared([block_M, dim], dtype)

                T.annotate_layout({
                    dQ: make_dq_layout(dQ),
                    K_shared: tl.layout.make_swizzled_layout(K_shared),
                    dv_shared: tl.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tl.layout.make_swizzled_layout(dk_shared),
                })

                T.copy(K[bz, by * block_M:(by + 1) * block_M, bx, :], K_shared)
                T.copy(V[bz, by * block_M:(by + 1) * block_M, bx, :], V_shared)
                T.clear(dv)
                T.clear(dk)
                loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_N)
                for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q)
                    T.clear(qkT)
                    T.gemm(
                        K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, wg_wait=-1)
                    T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do)
                    T.clear(dsT)
                    T.gemm(
                        V_shared,
                        do,
                        dsT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    T.wait_wgmma(1)

                    T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j],
                                                    0)
                    T.wait_wgmma(0)
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow, wg_wait=-1)

                    T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta)

                    for i, j in T.Parallel(block_M, block_N):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow, wg_wait=1)

                    T.copy(dsT_cast, dsT_shared)
                    T.clear(dq)
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True, wg_wait=1)
                    T.wait_wgmma(0)
                    for i, j in T.Parallel(block_N, dim):
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
                T.copy(dv, dv_shared)
                T.copy(dk, dk_shared)
                T.copy(dv_shared, dV[bz, by * block_M:(by + 1) * block_M, bx, :])
                T.copy(dk_shared, dK[bz, by * block_M:(by + 1) * block_M, bx, :])

        return _mha_bwd_main_wgmma_pipelined if is_hopper() else _mha_bwd_main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tl.jit(
            out_idx=[6, 7, 8], 
            pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True},
            compile_flags=["-O3", "-DENABLE_BF16"])
        def _mha_bwd_kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return _mha_bwd_func(block_M, block_N, num_stages, threads)

        return _mha_bwd_kernel()
    else:

        @tl.jit(
            out_idx=[6, 7, 8], 
            pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True},
            compile_flags=["-O3", "-DENABLE_BF16"])
        def _mha_bwd_kernel(block_M, block_N, num_stages, threads):
            return _mha_bwd_func(block_M, block_N, num_stages, threads)

        return _mha_bwd_kernel


@tl.jit(
    out_idx=[1], 
    pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    compile_flags=["-O3", "-DENABLE_BF16"])
def _mha_bwd_postprocess(batch, heads, seq_len, dim, dtype: str):
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
            dQ: T.Tensor(shape, accum_dtype),  # type: ignore
            dQ_out: T.Tensor(shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk:(bx + 1) * blk, by, :],
            )

    return flash_bwd_post


class _MHA_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, config, bwd_config):
        BATCH, N_CTX, H, D_HEAD = q.shape
        dtype = q.dtype
        dtype_str = dtype.__str__().split('.')[-1]
        mod = _mha_fwd(BATCH, H, N_CTX, D_HEAD, causal, dtype_str)(**config)
        o, lse = mod(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.bwd_config = bwd_config
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD = q.shape

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        dtype = q.dtype
        dtype_str = dtype.__str__().split('.')[-1]
        mod_prep = _mha_bwd_preprocess(BATCH, H, N_CTX, D_HEAD, dtype_str)
        mod_post = _mha_bwd_postprocess(BATCH, H, N_CTX, D_HEAD, dtype_str)
        mod = _mha_bwd(BATCH, H, N_CTX, D_HEAD, ctx.causal, dtype_str)(**ctx.bwd_config)
        delta = mod_prep(o, do)
        dq = torch.zeros_like(q, dtype=torch.float, device=q.device, requires_grad=False)
        dk = torch.zeros_like(k, dtype=torch.float, device=k.device, requires_grad=False)
        dv = torch.zeros_like(v, dtype=torch.float, device=v.device, requires_grad=False)
        dq, dk, dv = mod(q, k, v, do, lse, delta)
        dq = mod_post(dq)
        dk = dk.to(torch.float16)
        dv = dv.to(torch.float16)
        return dq, dk, dv, None, None, None


MHA_attention = _MHA_attention.apply


class MHAKernel(nn.Module):

    def __init__(self,
                 batch_size,
                 num_heads,
                 seq_len,
                 head_dim,
                 causal,
                 fwd_block_M=None,
                 fwd_block_N=None,
                 bwd_block_M=None,
                 bwd_block_N=None,
                 fwd_tune=False,
                 bwd_tune=False,
                 num_stages=None,
                 threads=None,
                 dtype: torch.dtype = torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = MHA_attention
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.causal = causal

        # Use heuristics if not specified
        if is_hopper():
            _fwd_block_M = 128
            _fwd_block_N = 128
            _bwd_block_M = 128
            _bwd_block_N = 32
            _threads = 256
            _num_stages = 2
        else:  # Ampere
            _fwd_block_M = 64
            _fwd_block_N = 64 if head_dim <= 128 else 32
            _bwd_block_M = 64
            _bwd_block_N = 64 if head_dim <= 64 else 32
            _threads = 128
            _num_stages = 1

        self.fwd_block_M = fwd_block_M if fwd_block_M is not None else _fwd_block_M
        self.fwd_block_N = fwd_block_N if fwd_block_N is not None else _fwd_block_N
        self.bwd_block_M = bwd_block_M if bwd_block_M is not None else _bwd_block_M
        self.bwd_block_N = bwd_block_N if bwd_block_N is not None else _bwd_block_N
        self.num_stages = num_stages if num_stages is not None else _num_stages
        self.threads = threads if threads is not None else _threads
        
        self.fwd_config = {
            "block_M": self.fwd_block_M,
            "block_N": self.fwd_block_N,
            "num_stages": self.num_stages,
            "threads": self.threads
        }
        print(f'MHAKernel fwd config: {self.fwd_config}')
        self.bwd_config = {
            "block_M": self.bwd_block_M,
            "block_N": self.bwd_block_N,
            "num_stages": self.num_stages,
            "threads": self.threads
        }
        print(f'MHAKernel bwd config: {self.bwd_config}')
        
        self.fwd_tune = fwd_tune
        self.bwd_tune = bwd_tune
        self.fwd_tune_config = None
        self.bwd_tune_config = None

        assert dtype in [torch.float16, torch.bfloat16], f"dtype must be float16 or bfloat16, got {dtype}"
        self.dtype = dtype
        self.dtype_str = dtype.__str__().split('.')[-1]
        self.device = device
        flops_per_matmul = 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim
        self.fwd_flops = 2 * flops_per_matmul
        self.bwd_flops = 5 * flops_per_matmul
        if causal:
            self.fwd_flops *= 0.5
            self.bwd_flops *= 0.5
        # (BATCH, H, N_CTX, D_HEAD, causal)(**config)
        self.fwd_program = _mha_fwd(batch_size, num_heads, seq_len, head_dim,
                                    causal, self.dtype_str)(**self.fwd_config)
        # self.fwd_kernel = tilelang.compile(self.fwd_program, out_idx=[4, 5])
        self.fwd_profiler = self.fwd_program.get_profiler(
            tensor_supply_type=tl.TensorSupplyType.Auto)
        self.bwd_program = _mha_bwd(batch_size, num_heads, seq_len, head_dim,
                                    causal, self.dtype_str)(**self.bwd_config)
        # self.bwd_kernel = tilelang.compile(self.bwd_program)
        self.bwd_profiler = self.bwd_program.get_profiler(
            tensor_supply_type=tl.TensorSupplyType.Randn)

    def forward(self, q, k, v):  # Layout: BSHD
        if self.fwd_tune_config is None and self.fwd_tune:
            self.fwd_autotune()
        if self.bwd_tune_config is None and self.bwd_tune:
            self.bwd_autotune()
        config = self.fwd_tune_config if self.fwd_tune_config else self.fwd_config
        bwd_config = self.bwd_tune_config if self.bwd_tune_config else self.bwd_config
        o = self.attention(q, k, v, self.causal, config, bwd_config)
        return o

    def backward(self, q, k, v, do):
        if self.bwd_tune_config is None and self.bwd_tune:
            self.bwd_autotune()
        o = self.forward(q, k, v)
        o.backward(do, retain_graph=True)
        return o

    def fwd_autotune(self):
        best_result = _mha_fwd(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim, self.causal, tune=True)
        best_latency = best_result.latency
        best_config = best_result.config
        print(f"Best fwd latency: {best_latency}")
        print(f"Best TFlops: {self.fwd_flops / best_latency * 1e-9}")
        print(f"Best fwd config: {best_config}")
        if best_result.config:
            self.fwd_tune_config = dict(
                zip(["block_M", "block_N", "num_stages", "threads"], list(best_config.values())))

    def bwd_autotune(self):
        best_result = _mha_bwd(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim, self.causal, tune=True)
        best_latency = best_result.latency
        best_config = best_result.config
        print(f"Best bwd latency: {best_latency}")
        print(f"Best TFlops: {self.bwd_flops / best_latency * 1e-9}")
        print(f"Best bwd config: {best_config}")
        if best_result.config:
            self.bwd_tune_config = dict(
                zip(["block_M", "block_N", "num_stages", "threads"], list(best_config.values())))

    def ref_program(self, q, k, v):
        dim = q.size(-1)
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        if self.causal:
            seq_len = q.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v)
        return output

    def gen_inputs(self):
        return (torch.randn((self.batch_size, self.seq_len, self.num_heads, self.head_dim),
                            device=self.device,
                            dtype=self.dtype,
                            requires_grad=True) for _ in range(4))

    def check(self):
        rtol, atol = {
            torch.float16: (1e-2, 1e-2),
            torch.bfloat16: (2e-2, 2e-2),
        }[self.dtype]

        q, k, v, do = self.gen_inputs()
        o = self.forward(q, k, v)
        o.backward(do)
        dq, q.grad = q.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dv, v.grad = v.grad.clone(), None
        o_ref = self.ref_program(q, k, v)
        o_ref.backward(do)
        dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

        assert torch.allclose(o, o_ref, rtol=rtol, atol=atol), f"o max err: {(o-o_ref).abs().max()}"
        assert torch.allclose(dq, dq_ref, rtol=rtol, atol=atol), f"dq max err: {(dq-dq_ref).abs().max()}"
        assert torch.allclose(dk, dk_ref, rtol=rtol, atol=atol), f"dk max err: {(dk-dk_ref).abs().max()}"
        assert torch.allclose(dv, dv_ref, rtol=rtol, atol=atol), f"dv max err: {(dv-dv_ref).abs().max()}"
        print("All checks passed! ✅")

    def profile(self, warmup=100):
        # fwd
        with torch.no_grad():
            if self.fwd_tune_config is None and self.fwd_tune:
                self.fwd_autotune()
            if self.fwd_tune_config:
                self.fwd_program = _mha_fwd(self.batch_size, self.num_heads, self.seq_len,
                                            self.head_dim, self.causal, self.dtype)(**self.fwd_config)
                # self.fwd_kernel = tilelang.compile(self.fwd_program, out_idx=[4, 5])
                self.fwd_profiler = self.fwd_program.get_profiler(
                    tensor_supply_type=tl.TensorSupplyType.Auto)
            fwd_latency = self.fwd_profiler.do_bench(warmup=warmup)
            print(f"Fwd latency: {fwd_latency:.2f} ms")
            print(f"Fwd FLOPs: {self.fwd_flops / fwd_latency * 1e-9:.2f} TFLOPs")
            fwd_ref_latency = self.fwd_profiler.do_bench(
                lambda q, k, v: self.ref_program(q, k, v), warmup=warmup)
            print(f"Fwd ref latency: {fwd_ref_latency:.2f} ms")
            print(f"Fwd ref FLOPs: {self.fwd_flops / fwd_ref_latency * 1e-9:.2f} TFLOPs")

        # bwd
        if self.bwd_tune_config is None and self.bwd_tune:
            self.bwd_autotune()
        if self.bwd_tune_config:
            self.bwd_program = _mha_bwd(self.batch_size, self.num_heads, self.seq_len,
                                        self.head_dim, self.causal, self.dtype)(**self.bwd_config)
            self.bwd_profiler = self.bwd_program.get_profiler(
                tensor_supply_type=tl.TensorSupplyType.Auto)
        bwd_latency = self.bwd_profiler.do_bench(warmup=warmup)
        print(f"Bwd latency: {bwd_latency:.2f} ms")
        print(f"Bwd FLOPs: {self.bwd_flops / bwd_latency * 1e-9:.2f} TFLOPs")

        def ref_bwd(q, k, v, do, *others):
            q = q.detach().requires_grad_()
            k = k.detach().requires_grad_()
            v = v.detach().requires_grad_()
            out = self.ref_program(q, k, v)
            out.backward(do, retain_graph=True)

        bwd_ref_latency = self.bwd_profiler.do_bench(ref_bwd, warmup=warmup)
        print(f"Bwd ref latency: {bwd_ref_latency:.2f} ms")
        print(f"Bwd ref FLOPs: {self.bwd_flops / bwd_ref_latency * 1e-9:.2f} TFLOPs")