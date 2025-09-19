import torch.nn.functional as F
import torch
import tilelang
import tilelang.language as T
import torch.nn as nn
from tilelang.autotuner import *
import itertools
from tilelang.cache import clear_cache


def get_configs():
    num_stages = [1, 2, 3]
    threads = [64, 128, 256]
    _configs = list(itertools.product(num_stages, threads))

    configs = [{'num_stages': c[0], 'threads': c[1]} for c in _configs]
    return configs


def _blocksparse_flashattn_fwd(batch,
                               heads,
                               seq_len,
                               dim_qk,
                               dim_v,
                               is_causal,
                               block_M,
                               block_N,
                               tune=False,
                               groups=1):
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dtype = "float16"
    accum_dtype = "float"
    block_mask_dtype = "bool"

    def _blocksparse_fwd_func(num_stages, threads):
        downsample_q_len = (seq_len + block_M - 1) // block_M
        downsample_kv_len = (seq_len + block_N - 1) // block_N
        block_mask_shape = [batch, heads, downsample_q_len, downsample_kv_len]

        @T.prim_func
        def _blocksparse_fwd_main(
                Q: T.Tensor(q_shape, dtype),  # type: ignore
                K: T.Tensor(k_shape, dtype),  # type: ignore
                V: T.Tensor(v_shape, dtype),  # type: ignore
                BlockSparseMask: T.Tensor(block_mask_shape, block_mask_dtype),  # type: ignore
                Output: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
                K_shared = T.alloc_shared([block_N, dim_qk], dtype)
                V_shared = T.alloc_shared([block_N, dim_v], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                block_mask = T.alloc_local([downsample_kv_len], block_mask_dtype)

                T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for vj in T.serial(downsample_kv_len):
                    block_mask[vj] = BlockSparseMask[bz, by, bx, vj]

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if block_mask[k] != 0:
                        T.copy(K[bz, k * block_N:(k + 1) * block_N, by // groups, :], K_shared)
                        if is_causal:
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                             -T.infinity(acc_s.dtype))
                        else:
                            T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow)
                        T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :], V_shared)
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                                     scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, dim_v):
                            acc_o[i, j] *= scores_scale[i]
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.copy(acc_s, acc_s_cast)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

        return _blocksparse_fwd_main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit(out_idx=[4, 5])
        def _blocksparse_fwd_kernel(num_stages=None, threads=None):
            return _blocksparse_fwd_func(num_stages, threads)

        return _blocksparse_fwd_kernel()
    else:

        @tilelang.jit(out_idx=[4, 5])
        def _blocksparse_fwd_kernel(num_stages, threads):
            return _blocksparse_fwd_func(num_stages, threads)

        return _blocksparse_fwd_kernel


@tilelang.jit(out_idx=[2])
def flashattn_bwd_preprocess(batch, heads, seq_len, dim_v):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_v]
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
            for k in range(T.ceildiv(dim_v, blk)):
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


@tilelang.jit(out_idx=[1])
def flashattn_bwd_postprocess(batch, heads, seq_len, dim_qk):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_qk]
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


def _blocksparse_flashattn_bwd(batch,
                               heads,
                               seq_len,
                               dim_qk,
                               dim_v,
                               is_causal,
                               block_M,
                               block_N,
                               tune=False,
                               groups=1):
    sm_scale = (1.0 / dim_qk)**0.5
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dtype = "float16"
    accum_dtype = "float"
    block_mask_dtype = "bool"

    def _blocksparse_bwd_func(num_stages, threads):
        downsample_q_len = (seq_len + block_M - 1) // block_M
        downsample_kv_len = (seq_len + block_N - 1) // block_N
        block_mask_shape = [batch, heads, downsample_q_len, downsample_kv_len]

        @T.prim_func
        def _blocksparse_flash_bwd(
                Q: T.Tensor(q_shape, dtype),  # type: ignore
                K: T.Tensor(k_shape, dtype),  # type: ignore
                V: T.Tensor(v_shape, dtype),  # type: ignore
                dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
                BlockSparseMask: T.Tensor(block_mask_shape, block_mask_dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dQ: T.Tensor(q_shape, accum_dtype),  # type: ignore
                dK: T.Tensor(k_shape, dtype),  # type: ignore
                dV: T.Tensor(v_shape, dtype),  # type: ignore
        ):
            with T.Kernel(
                    heads, T.ceildiv(seq_len, block_N), batch, threads=threads) as (bx, by, bz):
                K_shared = T.alloc_shared([block_N, dim_qk], dtype)
                dsT_shared = T.alloc_shared([block_N, block_M], dtype)
                q = T.alloc_shared([block_M, dim_qk], dtype)
                V_shared = T.alloc_shared([block_N, dim_v], dtype)
                qkT = T.alloc_fragment([block_N, block_M], accum_dtype)
                dsT = T.alloc_fragment([block_N, block_M], accum_dtype)
                qkT_cast = T.alloc_fragment([block_N, block_M], dtype)
                dsT_cast = T.alloc_fragment([block_N, block_M], dtype)
                lse_shared = T.alloc_shared([block_M], accum_dtype)
                delta = T.alloc_shared([block_M], accum_dtype)
                do = T.alloc_shared([block_M, dim_v], dtype)
                dv = T.alloc_fragment([block_N, dim_v], accum_dtype)
                dk = T.alloc_fragment([block_N, dim_qk], accum_dtype)
                dq = T.alloc_fragment([block_M, dim_qk], accum_dtype)
                dv_shared = T.alloc_shared([block_M, dim_v], dtype)
                dk_shared = T.alloc_shared([block_M, dim_qk], dtype)
                block_mask = T.alloc_local([downsample_q_len], block_mask_dtype)

                T.annotate_layout({
                    dQ: make_dq_layout(dQ),
                    K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })

                T.copy(K[bz, by * block_N:(by + 1) * block_N, bx // groups, :], K_shared)
                T.copy(V[bz, by * block_N:(by + 1) * block_N, bx // groups, :], V_shared)
                T.clear(dv)
                T.clear(dk)

                for vj in T.serial(downsample_q_len):
                    block_mask[vj] = BlockSparseMask[bz, bx, vj, by]

                loop_st = T.floordiv(by * block_N, block_M) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_M)
                for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    if block_mask[k] != 0:
                        T.copy(Q[bz, k * block_M:(k + 1) * block_M, bx, :], q)
                        T.clear(qkT)
                        T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.copy(lse[bz, bx, k * block_M:(k + 1) * block_M], lse_shared)
                        for i, j in T.Parallel(block_N, block_M):
                            qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                        if is_causal:
                            for i, j in T.Parallel(block_N, block_M):
                                qkT[i, j] = T.if_then_else(by * block_N + i <= k * block_M + j,
                                                           qkT[i, j], 0)
                        T.copy(dO[bz, k * block_M:(k + 1) * block_M, bx, :], do)
                        T.clear(dsT)
                        T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.copy(qkT, qkT_cast)
                        T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                        T.copy(Delta[bz, bx, k * block_M:(k + 1) * block_M], delta)

                        for i, j in T.Parallel(block_N, block_M):
                            dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                        T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                        T.copy(dsT_cast, dsT_shared)
                        T.clear(dq)
                        T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                        for i, j in T.Parallel(block_M, dim_qk):
                            if k * block_M + i < seq_len:
                                T.atomic_add(dQ[bz, k * block_M + i, bx, j], dq[i, j])

                for i, j in T.Parallel(block_N, dim_v):
                    T.atomic_add(dV[bz, by * block_N + i, bx // groups, j], dv[i, j])
                for i, j in T.Parallel(block_N, dim_qk):
                    T.atomic_add(dK[bz, by * block_N + i, bx // groups, j], dk[i, j])

        return _blocksparse_flash_bwd

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit()
        def _blocksparse_bwd_kernel(num_stages=None, threads=None):
            return _blocksparse_bwd_func(num_stages, threads)

        return _blocksparse_bwd_kernel()
    else:

        @tilelang.jit
        def _blocksparse_bwd_kernel(num_stages, threads):
            return _blocksparse_bwd_func(num_stages, threads)

        return _blocksparse_bwd_kernel


@torch.compile
class _blocksparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_mask, block_M, block_N, causal, config, bwd_config, groups=1):
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        D_HEAD_V = v.shape[-1]
        assert k.shape == (BATCH, N_CTX, H // groups, D_HEAD_QK), "k shape mismatch"
        assert v.shape == (BATCH, N_CTX, H // groups, D_HEAD_V), "v shape mismatch"
        mod = _blocksparse_flashattn_fwd(
            BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, causal, block_M, block_N, groups=groups)(**config)
        o, lse = mod(q, k, v, block_mask)
        ctx.save_for_backward(q, k, v, block_mask, o, lse)
        ctx.block_M = block_M
        ctx.block_N = block_N
        ctx.causal = causal
        ctx.bwd_config = bwd_config
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, block_mask, o, lse = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        HEAD_KV, D_HEAD_V, = v.shape[-2], v.shape[-1]
        groups = H // HEAD_KV

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        block_M = ctx.block_M
        block_N = ctx.block_N
        mod_prep = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD_V)
        mod_post = flashattn_bwd_postprocess(BATCH, H, N_CTX, D_HEAD_QK)
        delta = mod_prep(o, do)
        kernel = _blocksparse_flashattn_bwd(
            BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, ctx.causal, block_M, block_N,
            groups=groups)(**ctx.bwd_config)
        shape_q = [BATCH, N_CTX, H, D_HEAD_QK]
        shape_k = [BATCH, N_CTX, HEAD_KV, D_HEAD_QK]
        shape_v = [BATCH, N_CTX, HEAD_KV, D_HEAD_V]
        dq = torch.zeros(shape_q, dtype=torch.float32, device=q.device)
        dk = torch.zeros(shape_k, dtype=torch.float16, device=q.device)
        dv = torch.zeros(shape_v, dtype=torch.float16, device=q.device)
        kernel(q, k, v, do, block_mask, lse, delta, dq, dk, dv)
        dq = mod_post(dq)
        return dq, dk, dv, None, None, None, None, None, None, None


blocksparse_attention = _blocksparse_attention.apply


class BlockSparseAttentionKernel(nn.Module):

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim_qk,
                 dim_v,
                 block_M,
                 block_N,
                 causal,
                 block_mask,
                 fwd_tune=False,
                 bwd_tune=False,
                 num_stages=1,
                 threads=128,
                 groups=1,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = blocksparse_attention
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.block_M = block_M
        self.block_N = block_N
        self.causal = causal
        self.block_mask = block_mask
        self.groups = groups
        self.num_stages = num_stages
        self.threads = threads
        self.fwd_config = {"num_stages": self.num_stages, "threads": self.threads}
        self.bwd_config = {"num_stages": self.num_stages, "threads": self.threads}
        self.fwd_tune = fwd_tune
        self.bwd_tune = bwd_tune
        self.fwd_tune_config = None
        self.bwd_tune_config = None
        flops_per_qk = 2.0 * batch * heads * seq_len * seq_len * dim_qk
        flops_per_v = 2.0 * batch * heads * seq_len * seq_len * dim_v
        self.fwd_flops = flops_per_qk + flops_per_v
        self.total_flops = 3 * flops_per_qk + 2 * flops_per_v
        if causal:
            self.fwd_flops *= 0.5
            self.total_flops *= 0.5
        self.fwd_program = _blocksparse_flashattn_fwd(
            batch, heads, seq_len, dim_qk, dim_v, causal, block_M, block_N,
            groups=groups)(**self.fwd_config)
        # self.fwd_kernel = tilelang.compile(self.fwd_program, out_idx=[4, 5])
        self.fwd_profiler = self.fwd_program.get_profiler(
            tensor_supply_type=tilelang.TensorSupplyType.Auto)
        self.bwd_program = _blocksparse_flashattn_bwd(
            batch, heads, seq_len, dim_qk, dim_v, causal, block_M, block_N,
            groups=groups)(**self.bwd_config)
        # self.bwd_kernel = tilelang.compile(self.bwd_program)
        self.bwd_profiler = self.bwd_program.get_profiler(
            tensor_supply_type=tilelang.TensorSupplyType.Randn)

    def forward(self, q, k, v):
        if self.fwd_tune_config is None and self.fwd_tune:
            self.fwd_autotune()
        if self.bwd_tune_config is None and self.bwd_tune:
            self.bwd_autotune()
        config = self.fwd_tune_config if self.fwd_tune_config else self.fwd_config
        bwd_config = self.bwd_tune_config if self.bwd_tune_config else self.bwd_config
        o = self.attention(q, k, v, self.block_mask, self.block_M, self.block_N, self.causal,
                           config, bwd_config, self.groups)
        return o

    def backward(self, q, k, v, do):
        o = self.forward(q, k, v)
        o.backward(do, retain_graph=True)
        return o

    def fwd_autotune(self):
        best_result = _blocksparse_flashattn_fwd(
            self.batch,
            self.heads,
            self.seq_len,
            self.dim_qk,
            self.dim_v,
            self.causal,
            self.block_M,
            self.block_N,
            tune=True,
            groups=self.groups)
        best_latency = best_result.latency
        best_config = best_result.config
        print(f"Best fwd latency: {best_latency}")
        print(f"Best TFlops: {self.fwd_flops / best_latency * 1e-9}")
        print(f"Best fwd config: {best_config}")
        if best_result.config:
            self.fwd_tune_config = dict(zip(["num_stages", "threads"], list(best_config.values())))

    def bwd_autotune(self):
        best_result = _blocksparse_flashattn_bwd(
            self.batch,
            self.heads,
            self.seq_len,
            self.dim_qk,
            self.dim_v,
            self.causal,
            self.block_M,
            self.block_N,
            tune=True,
            groups=self.groups)
        best_latency = best_result.latency
        best_config = best_result.config
        print(f"Best bwd latency: {best_latency}")
        print(f"Best TFlops: {self.total_flops / best_latency * 1e-9}")
        print(f"Best bwd config: {best_config}")
        if best_result.config:
            self.bwd_tune_config = dict(zip(["num_stages", "threads"], list(best_config.values())))

    def ref_program(self, q, k, v):
        B, T, HQ, D_QK = q.shape
        _, _, HK, _ = k.shape
        _, _, HV, D_V = v.shape
        block_M = self.block_M
        block_N = self.block_N
        causal = self.causal
        groups = self.groups

        assert HK * groups == HQ, f"Q heads {HQ} != K heads {HK} * groups {groups}"
        assert HV * groups == HQ, f"Q heads {HQ} != V heads {HV} * groups {groups}"

        k_expanded = k.repeat_interleave(groups, dim=2)  # [B, T, HQ, D_QK]
        v_expanded = v.repeat_interleave(groups, dim=2)  # [B, T, HQ, D_V]

        attn_scores = torch.einsum('bthd,bshd->bhts', q, k_expanded)  # [B, H, T, T]
        attn_scores = attn_scores / torch.sqrt(
            torch.tensor(D_QK, dtype=attn_scores.dtype, device=attn_scores.device))

        # Apply causal mask if needed
        if causal:
            causal_mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Apply block_mask (simulate sparse attention pattern)
        if self.block_mask is not None:
            mask_exp = self.block_mask.repeat_interleave(
                block_M, dim=2).repeat_interleave(
                    block_N, dim=3)
            mask_exp = mask_exp[:, :, :T, :T]  # [B, H, T, T]
            attn_scores = attn_scores.masked_fill(~mask_exp.bool(), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum('bhts,bshd->bthd', attn_weights, v_expanded)  # [B, T, HQ, D_V]

        return output

    def check_forward(self, q, k, v):
        o = self.forward(q, k, v)

        o_ref = self.ref_program(q, k, v)

        assert torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2)
        print("BlockSparseAttention kernel check passed!")

    def check(self, q, k, v, do):

        o = self.forward(q, k, v)
        dq, q.grad = q.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dv, v.grad = v.grad.clone(), None

        o_ref = self.ref_program(q, k, v)
        o_ref.backward(do, retain_graph=True)
        dq_ref, q.grad = q.grad.clone(), None
        dk_ref, k.grad = k.grad.clone(), None
        dv_ref, v.grad = v.grad.clone(), None

        assert torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dv, dv_ref, rtol=1e-2, atol=1e-2)
        assert torch.allclose(dk, dk_ref, rtol=1e-2, atol=1e-2)
        assert torch.allclose(dq, dq_ref, rtol=1e-2, atol=1e-2)
        print("BlockSparseAttention kernel check passed!")

    def profile(self, warmup=100):
        with torch.no_grad():
            if self.fwd_tune_config is None and self.fwd_tune:
                self.fwd_autotune()
            if self.fwd_tune_config:
                self.fwd_program = _blocksparse_flashattn_fwd(
                    self.batch,
                    self.heads,
                    self.seq_len,
                    self.dim_qk,
                    self.dim_v,
                    self.causal,
                    self.block_M,
                    self.block_N,
                    groups=self.groups)(**self.fwd_tune_config)
                self.fwd_profiler = self.fwd_program.get_profiler(
                    tensor_supply_type=tilelang.TensorSupplyType.Auto)
            fwd_latency = self.fwd_profiler.do_bench(warmup=warmup)
            print(f"Fwd latency: {fwd_latency:.2f} ms")
            fwd_ref_latency = self.fwd_profiler.do_bench(
                lambda q, k, v, dO: self.ref_program(q, k, v), warmup=warmup)
            print(f"Fwd ref latency: {fwd_ref_latency:.2f} ms")
        if self.bwd_tune_config is None and self.bwd_tune:
            self.bwd_autotune()
        if self.bwd_tune_config:
            self.bwd_program = _blocksparse_flashattn_bwd(
                self.batch,
                self.heads,
                self.seq_len,
                self.dim_qk,
                self.dim_v,
                self.causal,
                self.block_M,
                self.block_N,
                groups=self.groups)(**self.bwd_tune_config)
            self.bwd_profiler = self.bwd_program.get_profiler(
                tensor_supply_type=tilelang.TensorSupplyType.Auto)
        bwd_latency = self.bwd_profiler.do_bench(warmup=warmup)
        print(f"Bwd latency: {bwd_latency:.2f} ms")

        def ref_bwd(q, k, v, do, *others):
            q = q.detach().requires_grad_()
            k = k.detach().requires_grad_()
            v = v.detach().requires_grad_()
            out = self.ref_program(q, k, v)
            out.backward(do, retain_graph=True)

        bwd_ref_latency = self.bwd_profiler.do_bench(ref_bwd, warmup=warmup)
        print(f"Bwd ref latency: {bwd_ref_latency:.2f} ms")
