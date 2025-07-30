import torch.nn.functional as F
import torch
import tilelang
import tilelang.language as T
import torch.nn as nn
from tilelang.autotuner import *
from einops import rearrange, einsum
import itertools


def get_configs():
    block_M = [32, 64, 128]
    block_N = [32, 64, 128]
    num_stages = [1, 2, 3]
    threads = [128]
    _configs = list(itertools.product(block_M, block_N, num_stages, threads))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'num_stages': c[2],
        'threads': c[3]
    } for c in _configs]
    return configs


def _gqa_fwd(batch, heads, seq_len, dim_qk, dim_v, is_causal, tune=False, groups=1):
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    assert groups <= heads, "groups must <= heads"

    def _gqa_fwd_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _gqa_fwd_main(
                Q: T.Tensor(q_shape, dtype),  # type: ignore
                K: T.Tensor(k_shape, dtype),  # type: ignore
                V: T.Tensor(v_shape, dtype),  # type: ignore
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

                T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
                for k in T.Pipelined(loop_range, num_stages=num_stages):
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
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
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

        return _gqa_fwd_main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @jit(out_idx=[3, 4])
        def _gqa_fwd_kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return _gqa_fwd_func(block_M, block_N, num_stages, threads)

        return _gqa_fwd_kernel()
    else:

        @tilelang.jit(out_idx=[3, 4])
        def _gqa_fwd_kernel(block_M, block_N, num_stages, threads):
            return _gqa_fwd_func(block_M, block_N, num_stages, threads)

        return _gqa_fwd_kernel


@tilelang.jit(out_idx=[2])
def _gqa_bwd_preprocess(batch, heads, seq_len, dim_v):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_v]
    blk = 32

    @T.prim_func
    def _gqa_bwd_prep(
            Out: T.Tensor(shape, dtype),  # type: ignore
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
                T.copy(Out[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o)
                T.copy(dO[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk:(by + 1) * blk])

    return _gqa_bwd_prep


def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(
        dQ.shape, lambda b, ly, h, d: [b, ly // 8, h, d // 8, (d % 2), 4 * (ly % 8) + (d % 8) // 2])


@tilelang.jit(out_idx=[1])
def _gqa_bwd_postprocess(batch, heads, seq_len, dim_qk):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_qk]
    blk = 64

    @T.prim_func
    def _gqa_bwd_post(
            dQ: T.Tensor(shape, accum_dtype),  # type: ignore
            dQ_out: T.Tensor(shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk:(bx + 1) * blk, by, :],
            )

    return _gqa_bwd_post


def _gqa_bwd(batch, heads, seq_len, dim_qk, dim_v, is_causal, tune=False, groups=1):
    sm_scale = (1.0 / dim_qk)**0.5
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dtype = "float16"
    accum_dtype = "float"

    def _gqa_bwd_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _gqa_bwd_main(
                Q: T.Tensor(q_shape, dtype),  # type: ignore
                K: T.Tensor(k_shape, dtype),  # type: ignore
                V: T.Tensor(v_shape, dtype),  # type: ignore
                dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dQ: T.Tensor(q_shape, accum_dtype),  # type: ignore
                dK: T.Tensor(k_shape, dtype),  # type: ignore
                dV: T.Tensor(v_shape, dtype),  # type: ignore
        ):
            with T.Kernel(
                    heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
                K_shared = T.alloc_shared([block_M, dim_qk], dtype)
                dsT_shared = T.alloc_shared([block_M, block_N], dtype)
                q = T.alloc_shared([block_N, dim_qk], dtype)
                V_shared = T.alloc_shared([block_M, dim_v], dtype)
                qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
                dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
                qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
                dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_N], accum_dtype)
                delta = T.alloc_shared([block_N], accum_dtype)
                do = T.alloc_shared([block_N, dim_v], dtype)
                dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
                dk = T.alloc_fragment([block_M, dim_qk], accum_dtype)
                dq = T.alloc_fragment([block_N, dim_qk], accum_dtype)
                dv_shared = T.alloc_shared([block_N, dim_v], dtype)
                dk_shared = T.alloc_shared([block_N, dim_qk], dtype)

                T.annotate_layout({
                    dQ: make_dq_layout(dQ),
                    K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })

                T.copy(K[bz, by * block_M:(by + 1) * block_M, bx // groups, :], K_shared)
                T.copy(V[bz, by * block_M:(by + 1) * block_M, bx // groups, :], V_shared)
                T.clear(dv)
                T.clear(dk)
                loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_N)
                for k in T.Pipelined(loop_st, loop_ed, num_stages):
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
                    for i, j in T.Parallel(block_N, dim_qk):
                        if k * block_N + i < seq_len:
                            T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

                for i, j in T.Parallel(block_M, dim_v):
                    T.atomic_add(dV[bz, by * block_M + i, bx // groups, j], dv[i, j])
                for i, j in T.Parallel(block_M, dim_qk):
                    T.atomic_add(dK[bz, by * block_M + i, bx // groups, j], dk[i, j])

        return _gqa_bwd_main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @jit()
        def _gqa_bwd_kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return _gqa_bwd_func(block_M, block_N, num_stages, threads)

        return _gqa_bwd_kernel()
    else:

        @tilelang.jit
        def _gqa_bwd_kernel(block_M, block_N, num_stages, threads):
            return _gqa_bwd_func(block_M, block_N, num_stages, threads)

        return _gqa_bwd_kernel


@torch.compile
class _GQA_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, config, bwd_config, groups=1):
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        D_HEAD_V = v.shape[-1]
        mod = _gqa_fwd(BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, causal, groups=groups)(**config)
        o, lse = mod(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.bwd_config = bwd_config
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        HEAD_KV, D_HEAD_V, = v.shape[-2], v.shape[-1]
        groups = H // HEAD_KV

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        mod_prep = _gqa_bwd_preprocess(BATCH, H, N_CTX, D_HEAD_V)
        mod_post = _gqa_bwd_postprocess(BATCH, H, N_CTX, D_HEAD_QK)
        delta = mod_prep(o, do)
        kernel = _gqa_bwd(
            BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, ctx.causal, groups=groups)(**ctx.bwd_config)
        shape_q = [BATCH, N_CTX, H, D_HEAD_QK]
        shape_k = [BATCH, N_CTX, HEAD_KV, D_HEAD_QK]
        shape_v = [BATCH, N_CTX, HEAD_KV, D_HEAD_V]
        dq = torch.zeros(shape_q, dtype=torch.float32, device=q.device)
        dk = torch.zeros(shape_k, dtype=torch.float16, device=q.device)
        dv = torch.zeros(shape_v, dtype=torch.float16, device=q.device)
        kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = mod_post(dq)
        return dq, dk, dv, None, None, None, None


GQA_attention = _GQA_attention.apply


class GQAKernel(nn.Module):

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim_qk,
                 dim_v,
                 fwd_block_M,
                 fwd_block_N,
                 bwd_block_M,
                 bwd_block_N,
                 causal,
                 fwd_tune=False,
                 bwd_tune=False,
                 num_stages=1,
                 threads=128,
                 groups=1,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = GQA_attention
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.fwd_block_M = fwd_block_M
        self.fwd_block_N = fwd_block_N
        self.bwd_block_M = bwd_block_M
        self.bwd_block_N = bwd_block_N
        self.causal = causal
        self.groups = groups
        self.num_stages = num_stages
        self.threads = threads
        self.fwd_config = {
            "block_M": self.fwd_block_M,
            "block_N": self.fwd_block_N,
            "num_stages": self.num_stages,
            "threads": self.threads
        }
        self.bwd_config = {
            "block_M": self.bwd_block_M,
            "block_N": self.bwd_block_N,
            "num_stages": self.num_stages,
            "threads": self.threads
        }
        self.fwd_tune = fwd_tune
        self.bwd_tune = bwd_tune
        self.fwd_tune_config = None
        self.bwd_tune_config = None
        flops_per_qk = 2.0 * batch * heads * seq_len * seq_len * dim_qk
        flops_per_v = 2.0 * batch * heads * seq_len * seq_len * dim_v
        self.fwd_flops = flops_per_qk + flops_per_v
        self.total_flops = 3 * flops_per_qk + 2 * flops_per_v
        self.fwd_program = _gqa_fwd(
            batch, heads, seq_len, dim_qk, dim_v, causal, groups=groups)(**self.fwd_config)
        # self.fwd_kernel = tilelang.compile(self.fwd_program, out_idx=[3, 4])
        self.fwd_profiler = self.fwd_program.get_profiler(
            tensor_supply_type=tilelang.TensorSupplyType.Auto)
        self.bwd_program = _gqa_bwd(
            batch, heads, seq_len, dim_qk, dim_v, causal, groups=groups)(**self.bwd_config)
        # self.bwd_kernel = tilelang.compile(self.bwd_program, out_idx=[6, 7, 8])
        self.bwd_profiler = self.bwd_program.get_profiler(
            tensor_supply_type=tilelang.TensorSupplyType.Randn)
        if causal:
            self.fwd_flops *= 0.5
            self.total_flops *= 0.5

    def forward(self, q, k, v):
        if self.fwd_tune_config is None and self.fwd_tune:
            self.fwd_autotune()
        config = self.fwd_tune_config if self.fwd_tune_config else self.fwd_config
        bwd_config = self.bwd_tune_config if self.bwd_tune_config else self.bwd_config
        o = self.attention(q, k, v, self.causal, config, bwd_config, self.groups)
        return o

    def backward(self, q, k, v, do):
        if self.bwd_tune_config is None and self.bwd_tune:
            self.bwd_autotune()
        o = self.forward(q, k, v)
        o.backward(do, retain_graph=True)
        return o

    def profile(self, q, k, v, do, warmup=500):
        with torch.no_grad():
            if self.fwd_tune_config is None and self.fwd_tune:
                self.fwd_autotune()
            if self.fwd_tune_config:
                self.fwd_program = _gqa_fwd(
                    self.batch,
                    self.heads,
                    self.seq_len,
                    self.dim_qk,
                    self.dim_v,
                    self.causal,
                    groups=self.groups)(**self.fwd_tune_config)
                self.fwd_profiler = self.fwd_program.get_profiler(
                    tensor_supply_type=tilelang.TensorSupplyType.Auto)
            fwd_latency = self.fwd_profiler.do_bench(warmup=warmup)
            print(f"Fwd latency: {fwd_latency:.2f} ms")
        if self.bwd_tune_config is None and self.bwd_tune:
            self.bwd_autotune()
        if self.bwd_tune_config:
            self.bwd_program = _gqa_bwd(
                self.batch,
                self.heads,
                self.seq_len,
                self.dim_qk,
                self.dim_v,
                self.causal,
                groups=self.groups)(**self.bwd_tune_config)
            self.bwd_profiler = self.bwd_program.get_profiler(
                tensor_supply_type=tilelang.TensorSupplyType.Auto)
        bwd_latency = self.bwd_profiler.do_bench(warmup=warmup)
        print(f"Bwd latency: {bwd_latency:.2f} ms")
        return fwd_latency, bwd_latency

    def fwd_autotune(self):
        best_result = _gqa_fwd(
            self.batch,
            self.heads,
            self.seq_len,
            self.dim_qk,
            self.dim_v,
            self.causal,
            tune=True,
            groups=self.groups)
        best_latency = best_result.latency
        best_config = best_result.config
        print(f"Best fwd latency: {best_latency}")
        print(f"Best TFlops: {self.fwd_flops / best_latency * 1e-9}")
        print(f"Best fwd config: {best_config}")
        if best_result.config:
            self.fwd_tune_config = dict(
                zip(["block_M", "block_N", "num_stages", "threads"], best_config))

    def bwd_autotune(self):
        best_result = _gqa_bwd(
            self.batch,
            self.heads,
            self.seq_len,
            self.dim_qk,
            self.dim_v,
            self.causal,
            tune=True,
            groups=self.groups)
        best_latency = best_result.latency
        best_config = best_result.config
        print(f"Best bwd latency: {best_latency}")
        print(f"Best TFlops: {self.total_flops / best_latency * 1e-9}")
        print(f"Best bwd config: {best_config}")
        if best_result.config:
            self.bwd_tune_config = dict(
                zip(["block_M", "block_N", "num_stages", "threads"], best_config))

    def ref_program(self, q, k, v):
        # Q: [B, T, HQ, D_QK]
        # K: [B, T, HK, D_QK]
        # V: [B, T, HV, D_V]
        # HQ = HKV * groups
        groups = self.groups
        assert q.size(2) == k.size(
            2) * groups, f"Q.size(2): {q.size(2)}, K.size(2): {k.size(2)}, groups: {groups}"
        assert q.size(2) == v.size(
            2) * groups, f"Q.size(2): {q.size(2)}, V.size(2): {v.size(2)}, groups: {groups}"

        dim_qk = q.size(-1)
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        scores = scores / torch.sqrt(torch.tensor(dim_qk, dtype=scores.dtype))
        if self.causal:
            seq_len = q.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v)
        return output

    def check(self, q, k, v, do):
        o = self.forward(q, k, v)
        o.backward(do, retain_graph=True)
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), do, retain_graph=True)

        o_ref = self.ref_program(q, k, v)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(o_ref, (q, k, v), do, retain_graph=True)

        assert torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dv, dv_ref, rtol=1e-2, atol=1e-2)
        # assert torch.allclose(dV, dV_ref, rtol=1e-2, atol=1e-2)
        assert torch.allclose(dk, dk_ref, rtol=1e-2, atol=1e-2)
        assert torch.allclose(dq, dq_ref, rtol=1e-2, atol=1e-2)
        print("GQA kernel check passed!")


def get_configs_decode():
    block_N = [64, 128]
    block_H = [64]
    num_split = [2, 4, 8]
    num_stages = [1, 2, 3]
    threads = [128]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

    configs = [{
        'block_N': c[0],
        'block_H': c[1],
        'num_split': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def _gqa_decode(batch, heads, seqlen_kv, dim, groups=1, tune=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, groups, dim]
    shape_v = [batch, seqlen_kv, groups, dim]
    shape_o = [batch, heads, dim]
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // groups

    def _gqa_decode_func(block_N, block_H, num_split, num_stages, threads):
        part_shape = [batch, heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)
        valid_block_N = min(block_N, seqlen_kv // num_split)

        @T.macro
        def _gqa_decode_flash_attn(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_shared)
                    T.copy(mask[bid, k * block_N:(k + 1) * block_N, cur_kv_head], mask_local)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                     -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(V[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])

        @T.macro
        def _gqa_decode_flash_attn_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                T.fill(K_shared, 0)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        K[bid, (seqlen_kv // num_split) * sid +
                          k * valid_block_N:(seqlen_kv // num_split) * sid +
                          (k + 1) * valid_block_N, cur_kv_head, :], K_shared)
                    T.copy(
                        mask[bid, (seqlen_kv // num_split) * sid +
                             k * valid_block_N:(seqlen_kv // num_split) * sid +
                             (k + 1) * valid_block_N, cur_kv_head], mask_local)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (mask_local[j] != 0) & (j < seqlen_kv // num_split), acc_s[i, j],
                            -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[bid, (seqlen_kv // num_split) * sid +
                          k * valid_block_N:(seqlen_kv // num_split) * sid +
                          (k + 1) * valid_block_N, cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid, :])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split, threads], dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([threads], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    # lse_local: (local_id, thread_id)
                    lse_local:
                        T.Fragment(lse_local.shape, forward_fn=lambda i, j: (j, i)),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                for k, j in T.Parallel(num_split, threads):
                    lse_local[k, j] = glse[bz, by, k]
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split[0] = glse[bz, by, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def _gqa_decode_flashattn_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_flash_attn_split(Q, K, V, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def _gqa_decode_flashattn_no_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_flash_attn(Q, K, V, mask, Output)

        if num_split > 1:
            return _gqa_decode_flashattn_split
        else:
            return _gqa_decode_flashattn_no_split

    if tune:

        @autotune(configs=get_configs_decode(), warmup=10, rep=10)
        @jit(
            out_idx=[6],
            supply_type=tilelang.TensorSupplyType.Auto,
            ref_prog=gqa_decode_ref_program,
            max_mismatched_ratio=0.05,
            cache_input_tensors=False)
        def _gqa_decode_kernel(block_N=None,
                               block_H=None,
                               num_split=None,
                               num_stages=None,
                               threads=None):
            return _gqa_decode_func(block_N, block_H, num_split, num_stages, threads)

        return _gqa_decode_kernel()
    else:

        def _gqa_decode_kernel(block_N, block_H, num_split, num_stages, threads):
            return _gqa_decode_func(block_N, block_H, num_split, num_stages, threads)

        return _gqa_decode_kernel


def gqa_decode_ref_program(query, key, value, mask, glse, Output_partial):
    """
    Inputs:
    - query (Tensor): [batch, heads, dim]
    - key (Tensor): [batch, seqlen_kv, groups, dim]
    - value (Tensor): [batch, seqlen_kv, groups, dim]
    - mask (Tensor): [batch, seqlen_kv, groups]
    Outputs:
    - output (Tensor): [batch, heads, dim]
    """
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5
    key = rearrange(key, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]
    value = rearrange(value, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

    query = rearrange(
        query, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

    scores = einsum(
        query, key,
        'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]
    if mask is not None:
        mask = rearrange(mask, 'b s h -> b h s')
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention = F.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

    out = einsum(attention, value,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out


def gqa_decode_split_ref(Q, K, V, mask, num_split):
    batch = Q.size(0)
    nheads = Q.size(1)
    groups = K.size(2)
    dim = Q.size(-1)
    block_N = 32
    seqlen_kv = K.size(1)
    num_head_groups = nheads // groups

    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    acc_s = torch.empty((batch, num_head_groups, groups, block_N), device="cuda", dtype=torch.float)
    acc_s_cast = torch.empty((batch, num_head_groups, groups, block_N),
                             device="cuda",
                             dtype=torch.float16)
    acc_o = torch.empty((batch, num_head_groups, groups, dim), device="cuda", dtype=torch.float)
    scores_max = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    scores_max_prev = torch.empty((batch, num_head_groups, groups),
                                  device="cuda",
                                  dtype=torch.float)
    scores_scale = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    scores_sum = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    logsum = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    gacc_o = torch.empty((num_split, batch, nheads, dim), device="cuda", dtype=torch.float)
    glogsum = torch.empty((num_split, batch, nheads), device="cuda", dtype=torch.float)

    Q_ = Q * scale
    Q_ = rearrange(Q_, 'b (h g) d -> b g h d', g=num_head_groups)

    for ks in range(num_split):
        acc_o.fill_(0)
        logsum.fill_(0)
        scores_max.fill_(float('-inf'))
        scores_max_prev.fill_(float('-inf'))
        for i in range(((seqlen_kv // num_split) + block_N - 1) // block_N):
            acc_s.fill_(0)
            acc_s = torch.einsum('bghd,bkhd->bghk', Q_,
                                 K[:, (seqlen_kv // num_split) * ks +
                                   i * block_N:(seqlen_kv // num_split) * ks +
                                   (i + 1) * block_N, :, :])  # [batch, nheads, block_N]
            if mask is not None:
                mask_local = mask[:, (seqlen_kv // num_split) * ks +
                                  i * block_N:(seqlen_kv // num_split) * ks + (i + 1) * block_N, :]
                mask_local = rearrange(mask_local, 'b s h -> b h s')
                mask_local = mask_local.unsqueeze(1)
                acc_s = acc_s.masked_fill(mask_local == 0, float('-inf'))
            scores_max_prev = scores_max
            scores_max = acc_s.max(dim=-1, keepdim=False).values  # [batch, nheads]
            scores_scale = torch.exp2(scores_max_prev - scores_max)  # [batch, nheads]
            acc_o *= scores_scale[:, :, :, None]
            acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
            acc_s_cast = acc_s.to(torch.float16)  # [batch, nheads, block_N]
            acc_o += torch.einsum(
                'bghk,bkhd->bghd', acc_s_cast,
                V[:, (seqlen_kv // num_split) * ks + i * block_N:(seqlen_kv // num_split) * ks +
                  (i + 1) * block_N, :, :])
            scores_sum = acc_s.sum(dim=-1, keepdim=False)
            logsum = logsum * scores_scale + scores_sum
        acc_o_out = rearrange(acc_o, 'b g h d->b (h g) d')
        logsum_out = rearrange(logsum, 'b g h->b (h g)')
        acc_o_out /= logsum_out[:, :, None]
        logsum_out = torch.log2(logsum_out) + rearrange(scores_max, 'b g h->b (h g)')
        gacc_o[ks, :, :, :] = acc_o_out
        glogsum[ks, :, :] = logsum_out

    return glogsum.to(torch.float16).permute(1, 2, 0), gacc_o.to(torch.float16).permute(1, 2, 0, 3)


def gqa_decode_reduce_ref(Q, K, V, mask, glse, Output_partial, num_split):
    o = torch.empty_like(Output_partial[:, :, 0, :]).fill_(0)
    lse_logsum = torch.empty_like(glse[:, :, 0]).fill_(0)  # [batch, heads]
    lse_max = glse.max(dim=2, keepdim=False).values
    for ks in range(num_split):
        lse = glse[:, :, ks]
        lse_logsum += torch.exp2(lse - lse_max)
    lse_logsum = torch.log2(lse_logsum) + lse_max
    for ks in range(num_split):
        lse = glse[:, :, ks]
        scale = torch.exp2(lse - lse_logsum)  # [batch, heads]
        o += Output_partial[:, :, ks, :] * scale[:, :, None]
    return o.to(torch.float16)


class GQADecodeKernel(nn.Module):

    def __init__(self,
                 batch,
                 heads,
                 kv_seqlen,
                 dim,
                 block_N=None,
                 block_H=None,
                 threads=None,
                 num_split=1,
                 groups=1,
                 tune=False,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.batch = batch
        self.kv_seqlen = kv_seqlen
        self.groups = groups
        self.heads = heads
        self.dim = dim
        self.block_N = block_N if block_N else 64
        self.block_H = block_H if block_H else 64
        self.threads = threads if threads else 128
        self.num_split = num_split
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        sm_major, sm_minor = torch.cuda.get_device_capability(device)
        self.sm_version = sm_major * 10 + sm_minor
        print(f"CUDA device capability: {self.sm_version}")
        if self.sm_version == 89:
            self.num_stages = 0
        else:
            self.num_stages = 2
        self.config = {
            "block_N": self.block_N,
            "block_H": self.block_H,
            "num_split": self.num_split,
            "num_stages": self.num_stages,
            "threads": self.threads
        }
        self.tune = tune
        self.tune_config = None

        self.program = _gqa_decode(self.batch, self.heads, self.kv_seqlen, self.dim,
                                   self.groups)(**self.config)
        if self.sm_version == 90:
            self.kernel = tilelang.compile(
                self.program, out_idx=[6], pass_configs={"tl.disable_tma_lower": True})
        else:
            self.kernel = tilelang.compile(self.program, out_idx=[6])
        self.profiler = self.kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        self.intermediate_tensor = {}
        self.intermediate_tensor['mask'] = torch.randint(
            0, 2, (self.batch, self.kv_seqlen, self.groups), device=device, dtype=torch.uint8)
        self.intermediate_tensor['glse'] = torch.empty(
            self.batch, self.heads, self.num_split, device=device, dtype=dtype)
        self.intermediate_tensor['output_partial'] = torch.empty(
            self.batch, self.heads, self.num_split, dim, device=device, dtype=dtype)
        qk_flops = 2 * batch * heads * kv_seqlen * dim
        pv_flops = 2 * batch * heads * kv_seqlen * dim
        self.total_flops = qk_flops + pv_flops

    def decode(self, q, k, v):
        o = self.kernel(q, k, v, self.intermediate_tensor['mask'], self.intermediate_tensor['glse'],
                        self.intermediate_tensor['output_partial'])
        return o

    def autotune(self):
        best_result = _gqa_decode(
            self.batch, self.heads, self.kv_seqlen, self.dim, self.groups, tune=True)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {self.total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")
        if best_result.config:
            self.tune_config = best_result.config
            self.config = dict(
                zip(["block_N", "block_H", "num_split", "num_stages", "threads"],
                    best_result.config))
            self.program = _gqa_decode(self.batch, self.heads, self.kv_seqlen, self.dim,
                                       self.groups)(**self.config)
            if self.sm_version == 90:
                self.kernel = tilelang.compile(
                    self.program, out_idx=[6], pass_configs={"tl.disable_tma_lower": True})
            else:
                self.kernel = tilelang.compile(self.program, out_idx=[6])
            self.profiler = self.kernel.get_profiler(
                tensor_supply_type=tilelang.TensorSupplyType.Auto)

    def profile(self, warmup=500):
        ref_latency = self.profiler.do_bench(gqa_decode_ref_program, warmup=warmup)
        print(f'Reference Latency: {ref_latency:.4f} ms')
        print(f"Reference FLOPs: {self.total_flops / ref_latency * 1e-9:.2f} TFLOPs")

        latency = self.profiler.do_bench(warmup=warmup)
        print(f'Latency: {latency:.4f} ms')
        print(f"FLOPs: {self.total_flops / latency * 1e-9:.2f} TFLOPs")
        return latency

    def ref_program(self, q, k, v):
        ref = gqa_decode_ref_program(q, k, v, self.intermediate_tensor['mask'],
                                     self.intermediate_tensor['glse'],
                                     self.intermediate_tensor['output_partial'])
        return ref

    def ref_program_split(self, Q, K, V):
        glse_, Output_partial_ = gqa_decode_split_ref(Q, K, V, self.intermediate_tensor['mask'],
                                                      self.num_split)
        return gqa_decode_reduce_ref(Q, K, V, self.intermediate_tensor['mask'], glse_,
                                     Output_partial_, self.num_split)

    def check(self, q, k, v):
        o = self.decode(q, k, v)
        o_ref = self.ref_program(q, k, v)
        o_ref_split = self.ref_program_split(q, k, v)

        torch.testing.assert_close(o, o_ref, rtol=0.01, atol=0.01)
        torch.testing.assert_close(o_ref_split, o_ref, rtol=0.01, atol=0.01)

        print("All checks pass.")
