"""Shared macros for Hopper batch=1 GQA decode kernels."""

import tilelang.language as T

RING_DEPTH = 2

COMPILE_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DNDEBUG",
]


def make_gqa_decode_bs1_consumer(
    block_m: int,
    block_n: int,
    dim: int,
    scale: float,
    kv_group_num: int,
    ring_depth: int,
    accum_dtype: str,
):
    """Create the paging-independent WGMMA consumer macro."""

    @T.macro
    def consumer(
        Q,
        bid,
        hid,
        sid,
        this_len,
        loop_range,
        Qs,
        Ks,
        Vs,
        Ps,
        ready,
        free,
        acc_s,
        acc_o,
        sm,
        smp,
        alpha,
        ss,
        logsum,
        glse,
        Output_partial,
    ):
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(sm, -T.infinity(accum_dtype))
        T.copy(
            Q[bid, hid * kv_group_num : hid * kv_group_num + kv_group_num, :],
            Qs[0:kv_group_num, :],
        )
        for k in T.serial(loop_range):
            T.mbarrier_wait_parity(ready[k % ring_depth], (k // ring_depth) % ring_depth)
            T.wgmma_gemm(
                Qs,
                Ks[k % ring_depth, :, :],
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.if_then_else(
                    k * block_n + j < this_len,
                    acc_s[i, j],
                    -T.infinity(accum_dtype),
                )
            T.copy(sm, smp)
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            for i in T.Parallel(block_m):
                alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
            T.reduce_sum(acc_s, ss, dim=1)
            for i in T.Parallel(block_m):
                logsum[i] = logsum[i] * alpha[i] + ss[i]
            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] *= alpha[i]
            T.copy(acc_s, Ps)
            T.wgmma_gemm(
                Ps,
                Vs[k % ring_depth, :, :],
                acc_o,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=False,
            )
            T.wait_wgmma(0)
            T.mbarrier_arrive(free[k % ring_depth])
        for i, j in T.Parallel(block_m, dim):
            acc_o[i, j] /= logsum[i]
        for i in T.Parallel(block_m):
            if i < kv_group_num:
                glse[bid, hid * kv_group_num + i, sid] = T.log2(logsum[i]) + sm[i] * scale
        for i, j in T.Parallel(block_m, dim):
            if i < kv_group_num:
                Output_partial[bid, hid * kv_group_num + i, sid, j] = acc_o[i, j]

    return consumer


def make_gqa_decode_bs1_combine(
    batch: int,
    heads: int,
    ctx_splits: int,
    dim: int,
    dtype: str,
    accum_dtype: str,
):
    """Create the paging-independent split-output combine macro."""

    @T.macro
    def combine(glse, Output_partial, Output):
        with T.Kernel(heads, batch, threads=128) as (hq, bid):
            lse_vec = T.alloc_fragment([ctx_splits], accum_dtype)
            lse_max = T.alloc_fragment([1], accum_dtype)
            lse_logsum = T.alloc_local([1], accum_dtype)
            o_accum = T.alloc_fragment([dim], accum_dtype)

            for s in T.Parallel(ctx_splits):
                lse_vec[s] = glse[bid, hq, s]
            T.fill(lse_max, -T.infinity(accum_dtype))
            T.reduce_max(lse_vec, lse_max, dim=0, clear=False)
            lse_logsum[0] = 0
            for s in T.serial(ctx_splits):
                lse_logsum[0] += T.exp2(glse[bid, hq, s] - lse_max[0])
            lse_logsum[0] = T.log2(lse_logsum[0]) + lse_max[0]
            T.clear(o_accum)
            for s in T.serial(ctx_splits):
                weight = T.exp2(glse[bid, hq, s] - lse_logsum[0])
                for j in T.Parallel(dim):
                    o_accum[j] += Output_partial[bid, hq, s, j] * weight
            for j in T.Parallel(dim):
                Output[bid, hq, j] = T.cast(o_accum[j], dtype)

    return combine
