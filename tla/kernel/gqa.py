# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import torch.nn.functional as F
import torch
import tilelang
import tilelang.language as T
import torch.nn as nn


@tilelang.jit(out_idx=[3, 4])
def _gqa_fwd(batch,
             heads,
             seq_len,
             dim_qk,
             dim_v,
             is_causal,
             block_M,
             block_N,
             groups=1):
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    assert groups <= heads, "groups must <= heads"

    @T.prim_func
    def _gqa_fwd_main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(k_shape, dtype),  # type: ignore
            V: T.Tensor(v_shape, dtype),  # type: ignore
            Output: T.Tensor([batch, seq_len, heads, dim_v],
                             dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len],
                          accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch,
                      threads=128) as (bx, by, bz):
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

            T.annotate_layout(
                {Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            loop_range = (T.ceildiv((bx + 1) * block_M, block_N)
                          if is_causal else T.ceildiv(seq_len, block_N))
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N:(k + 1) * block_N, by // groups, :],
                       K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j, 0,
                            -T.infinity(acc_s.dtype))
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared,
                       K_shared,
                       acc_s,
                       transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :],
                       V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                             scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale -
                                         scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast,
                       V_shared,
                       acc_o,
                       policy=T.GemmWarpPolicy.FullRow)
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
            Delta: T.Tensor([batch, heads, seq_len],
                            accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim_v, blk)):
                T.copy(
                    Out[bz, by * blk:(by + 1) * blk, bx,
                        k * blk:(k + 1) * blk], o)
                T.copy(
                    dO[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk],
                    do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk:(by + 1) * blk])

    return _gqa_bwd_prep


def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(
        dQ.shape, lambda b, ly, h, d:
        [b, ly // 8, h, d // 8, (d % 2), 4 * (ly % 8) + (d % 8) // 2])


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
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch,
                      threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk:(bx + 1) * blk, by, :],
            )

    return _gqa_bwd_post


@tilelang.jit
def _gqa_bwd(batch,
             heads,
             seq_len,
             dim_qk,
             dim_v,
             is_causal,
             block_M,
             block_N,
             groups=1):
    sm_scale = (1.0 / dim_qk)**0.5
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def _gqa_bwd_main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(k_shape, dtype),  # type: ignore
            V: T.Tensor(v_shape, dtype),  # type: ignore
            dO: T.Tensor([batch, seq_len, heads, dim_v],
                         dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len],
                          accum_dtype),  # type: ignore
            Delta: T.Tensor([batch, heads, seq_len],
                            accum_dtype),  # type: ignore
            dQ: T.Tensor(q_shape, accum_dtype),  # type: ignore
            dK: T.Tensor(k_shape, dtype),  # type: ignore
            dV: T.Tensor(v_shape, dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch,
                      threads=128) as (bx, by, bz):
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
                dQ:
                make_dq_layout(dQ),
                K_shared:
                tilelang.layout.make_swizzled_layout(K_shared),
                dv_shared:
                tilelang.layout.make_swizzled_layout(dv_shared),
                dk_shared:
                tilelang.layout.make_swizzled_layout(dk_shared),
            })

            T.copy(K[bz, by * block_M:(by + 1) * block_M, bx // groups, :],
                   K_shared)
            T.copy(V[bz, by * block_M:(by + 1) * block_M, bx // groups, :],
                   V_shared)
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(K_shared,
                       q,
                       qkT,
                       transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            by * block_M + i <= k * block_N + j, qkT[i, j], 0)
                T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(V_shared,
                       do,
                       dsT,
                       transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta)

                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i,
                             j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim_qk):
                    if k * block_N + i < seq_len:
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

            for i, j in T.Parallel(block_M, dim_v):
                T.atomic_add(dV[bz, by * block_M + i, bx // groups, j], dv[i,
                                                                           j])
            for i, j in T.Parallel(block_M, dim_qk):
                T.atomic_add(dK[bz, by * block_M + i, bx // groups, j], dk[i,
                                                                           j])

    return _gqa_bwd_main


@torch.compile
class _GQA_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_M, block_N, causal, groups=1):
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        D_HEAD_V = v.shape[-1]
        mod = _gqa_fwd(BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, causal, block_M,
                       block_N, groups)
        o, lse = mod(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
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
        block_M = 64
        block_N = 32
        mod_prep = _gqa_bwd_preprocess(BATCH, H, N_CTX, D_HEAD_V)
        mod_post = _gqa_bwd_postprocess(BATCH, H, N_CTX, D_HEAD_QK)
        delta = mod_prep(o, do)
        kernel = _gqa_bwd(BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, ctx.causal,
                          block_M, block_N, groups)
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


class GQA_kernel(nn.Module):

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim_qk,
                 dim_v,
                 block_M,
                 block_N,
                 causal,
                 groups=1,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = GQA_attention
        self.block_M = block_M
        self.block_N = block_N
        self.causal = causal
        self.groups = groups
        flops_per_qk = 2.0 * batch * heads * seq_len * seq_len * dim_qk
        flops_per_v = 2.0 * batch * heads * seq_len * seq_len * dim_v
        self.total_flops = 3 * flops_per_qk + 2 * flops_per_v
        self.bwd_program = _gqa_bwd(batch, heads, seq_len, dim_qk, dim_v,
                                    causal, block_M, block_N, groups)
        # self.bwd_kernel = tilelang.compile(self.bwd_program, out_idx=[6, 7, 8])
        self.bwd_profiler = self.bwd_program.get_profiler(
            tensor_supply_type=tilelang.TensorSupplyType.Randn)
        if causal:
            self.total_flops *= 0.5

    def forward(self, q, k, v):
        o = self.attention(q, k, v, self.block_M, self.block_N, self.causal,
                           self.groups)
        return o

    def backward(self, q, k, v, do):
        o = self.attention(q, k, v, self.block_M, self.block_N, self.causal,
                           self.groups)
        o.backward(do, retain_graph=True)
        return o

    def profile(self, warmup=500):
        latency = self.bwd_profiler.do_bench(warmup=warmup)
        return latency

    def ref_program(self, q, k, v, causal, groups):
        # Q: [B, T, HQ, D_QK]
        # K: [B, T, HK, D_QK]
        # V: [B, T, HV, D_V]
        # HQ = HKV * groups
        assert q.size(2) == k.size(
            2
        ) * groups, f"Q.size(2): {q.size(2)}, K.size(2): {k.size(2)}, groups: {groups}"
        assert q.size(2) == v.size(
            2
        ) * groups, f"Q.size(2): {q.size(2)}, V.size(2): {v.size(2)}, groups: {groups}"

        dim_qk = q.size(-1)
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        scores = scores / torch.sqrt(torch.tensor(dim_qk, dtype=scores.dtype))
        if causal:
            seq_len = q.size(1)
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v)
        return output

    def check(self, q, k, v, do):
        o = self.attention(q, k, v, self.block_M, self.block_N, self.causal,
                           self.groups)
        dq, q.grad = q.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dv, v.grad = v.grad.clone(), None

        o_ref = self.ref_program(q, k, v, self.causal, self.groups)
        o_ref.backward(do, retain_graph=True)
        dq_ref, q.grad = q.grad.clone(), None
        dk_ref, k.grad = k.grad.clone(), None
        dv_ref, v.grad = v.grad.clone(), None

        assert torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dv, dv_ref, rtol=1e-2, atol=1e-2)
        # assert torch.allclose(dV, dV_ref, rtol=1e-2, atol=1e-2)
        assert torch.allclose(dk, dk_ref, rtol=1e-2, atol=1e-2)
        assert torch.allclose(dq, dq_ref, rtol=1e-2, atol=1e-2)
        print("GQA kernel check passed!")
