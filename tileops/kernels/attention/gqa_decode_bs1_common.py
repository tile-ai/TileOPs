"""Shared macros for Hopper batch=1 GQA decode kernels."""

import tilelang
import tilelang.language as T
import torch

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


class GQADecodeBs1KernelMixin:
    """Shared runtime policy for contiguous and paged Hopper batch=1 decode."""

    _MIN_CTX = 1024
    _CTX_SPLIT_CANDIDATES = (32, 16, 8)

    def _select_tier(self, real_seqlen_kv: int) -> str:
        return "ctx" if real_seqlen_kv >= self._MIN_CTX else "no_split"

    def _ctx_splits_for(self, real_seqlen_kv: int) -> int:
        block_n = self.config["block_N"]
        for ctx_splits in self._CTX_SPLIT_CANDIDATES:
            if real_seqlen_kv % (ctx_splits * block_n) == 0:
                return ctx_splits
        return self._CTX_SPLIT_CANDIDATES[-1]

    def _allocate_partials(
        self,
        q: torch.Tensor,
        ctx_splits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        glse = torch.empty(
            (self.batch, self.heads, ctx_splits),
            dtype=torch.float32,
            device=q.device,
        )
        output_partial = torch.empty(
            (self.batch, self.heads, ctx_splits, self.dim),
            dtype=torch.float32,
            device=q.device,
        )
        return glse, output_partial


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


def make_gqa_decode_bs1_split(
    batch: int,
    groups: int,
    block_m: int,
    block_n: int,
    dim: int,
    dtype: str,
    scale: float,
    kv_group_num: int,
    ctx_splits: int,
    threads: int,
    accum_dtype: str,
    real_seqlen_is_buffer: bool,
    load_kv,
):
    """Create the shared context-split schedule around a layout-specific KV loader."""
    consumer = make_gqa_decode_bs1_consumer(
        block_m,
        block_n,
        dim,
        scale,
        kv_group_num,
        RING_DEPTH,
        accum_dtype,
    )

    @T.macro
    def split(Q, K, V, kv_layout, real_seqlen_kv, glse, output_partial):
        with T.Kernel(batch, groups, ctx_splits, threads=threads) as (bid, hid, sid):
            qs = T.alloc_shared([block_m, dim], dtype)
            ks = T.alloc_shared([RING_DEPTH, block_n, dim], dtype)
            vs = T.alloc_shared([RING_DEPTH, block_n, dim], dtype)
            ps = T.alloc_shared([block_m, block_n], dtype)
            T.annotate_layout(
                {
                    qs: tilelang.layout.make_swizzled_layout(qs),
                    ks: tilelang.layout.make_swizzled_layout(ks),
                    vs: tilelang.layout.make_swizzled_layout(vs),
                    ps: tilelang.layout.make_swizzled_layout(ps),
                }
            )
            ready = T.alloc_barrier([32] * RING_DEPTH)
            free = T.alloc_barrier([128] * RING_DEPTH)
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            sm = T.alloc_fragment([block_m], accum_dtype)
            smp = T.alloc_fragment([block_m], accum_dtype)
            alpha = T.alloc_fragment([block_m], accum_dtype)
            ss = T.alloc_fragment([block_m], accum_dtype)
            logsum = T.alloc_fragment([block_m], accum_dtype)

            seqlen_kv_b = (
                real_seqlen_kv[bid] if real_seqlen_is_buffer else real_seqlen_kv
            )
            base_len = seqlen_kv_b // (ctx_splits * block_n) * block_n
            this_len = T.if_then_else(
                sid == ctx_splits - 1,
                seqlen_kv_b - (ctx_splits - 1) * base_len,
                base_len,
            )
            base = base_len * sid
            loop_range = T.ceildiv(this_len, block_n)
            tx = T.get_thread_binding()

            if tx >= 128:
                for k in T.serial(loop_range):
                    T.mbarrier_wait_parity(
                        free[k % RING_DEPTH],
                        ((k // RING_DEPTH) % RING_DEPTH) ^ 1,
                    )
                    load_kv(
                        K,
                        V,
                        kv_layout,
                        bid,
                        hid,
                        base,
                        k,
                        ks,
                        vs,
                        ready,
                    )
                    T.mbarrier_arrive(ready[k % RING_DEPTH])
            else:
                consumer(
                    Q,
                    bid,
                    hid,
                    sid,
                    this_len,
                    loop_range,
                    qs,
                    ks,
                    vs,
                    ps,
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
                    output_partial,
                )

    return split


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
