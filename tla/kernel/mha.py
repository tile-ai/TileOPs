from sympy import false
import torch
from torch import nn 
from torch.nn import functional as F
import tilelang as tl 
import tilelang.language as T
from tilelang.profiler import do_bench

@tl.jit(out_idx=[3, 4])
def _mha_fwd(batch, heads, seq_len, dim, is_casual, block_M, block_N):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def flash_fwd(
        Q: T.Tensor(shape, dtype),  # type: ignore
        K: T.Tensor(shape, dtype),  # type: ignore
        V: T.Tensor(shape, dtype),  # type: ignore
        Output: T.Tensor(shape, dtype),  # type: ignore
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.clear(acc_o)
            T.clear(logsum)
            T.fill(scores_max, -T.infinity(accum_dtype))
            loop_range = (
                T.ceildiv(
                    (bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N))
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                if is_casual:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                     -T.infinity(acc_s.dtype))
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

    return flash_fwd

@tl.jit(out_idx=[2])
def _mha_bwd_preprocess(batch, heads, seq_len, dim):
    dtype = "float16"
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


@tl.jit(out_idx=[6, 7, 8])
def _mha_bwd(batch, heads, seq_len, dim, is_casual, block_M, block_N):
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def flash_bwd(
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
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=256) as (bx, by, bz):
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
            dv_shared = T.alloc_shared([block_N, dim], dtype)
            dk_shared = T.alloc_shared([block_N, dim], dtype)

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
            loop_st = T.floordiv(by * block_M, block_N) if is_casual else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=0): # On 4090 this will exceed the limit of smem
                T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_casual:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j],
                                                   0)
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
                    if k * block_N + i < seq_len:
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_M:(by + 1) * block_M, bx, :])
            T.copy(dk_shared, dK[bz, by * block_M:(by + 1) * block_M, bx, :])

    return flash_bwd


@tl.jit(out_idx=[1])
def _mha_bwd_postprocess(batch, heads, seq_len, dim):
    dtype = "float16"
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
    def forward(ctx, q, k, v, causal):
        BATCH, N_CTX, H, D_HEAD = q.shape
        block_M = 64
        block_N = 64 if D_HEAD <= 128 else 32
        mod = _mha_fwd(BATCH, H, N_CTX, D_HEAD, causal, block_M, block_N)
        o, lse = mod(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
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
        block_M = 128
        block_N = 128 if D_HEAD <= 64 else 32
        mod_prep = _mha_bwd_preprocess(BATCH, H, N_CTX, D_HEAD)
        mod_post = _mha_bwd_postprocess(BATCH, H, N_CTX, D_HEAD)
        mod = _mha_bwd(BATCH, H, N_CTX, D_HEAD, ctx.causal, block_M, block_N)
        delta = mod_prep(o, do)
        dq, dk, dv = mod(q, k, v, do, lse, delta)
        dq = mod_post(dq)
        return dq, dk, dv, None


MHA_attention = _MHA_attention.apply


class MHA_kernel:
    def __init__(self,
                 batch_size,
                 num_heads,
                 seq_len,
                 head_dim,
                 causal,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = MHA_attention
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.causal = causal
        self.dtype = dtype
        self.device = device
        flops_per_matmul = 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim
        self.fwd_flops = 2 * flops_per_matmul
        self.bwd_flops = 5 * flops_per_matmul 
        if causal:
            self.fwd_flops *= 0.5
            self.bwd_flops *= 0.5

    def forward(self, q, k, v): # Layout: BSHD
        return self.attention(q, k, v, self.causal)
    
    def ref_program(self, q, k, v, is_causal: bool):
        dim = q.size(-1)
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        if is_causal:
            seq_len = q.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v)
        return output
        
    def gen_inputs(self, n: int):
        return (torch.randn(
            (self.batch_size, self.seq_len, self.num_heads, self.head_dim),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True) for _ in range(n))
        
    def check(self):
        q, k, v, do = self.gen_inputs(4)
        o = self.forward(q, k, v)
        o.backward(do)
        dq, q.grad = q.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dv, v.grad = v.grad.clone(), None
        o_ref = self.ref_program(q, k, v, self.causal)
        o_ref.backward(do)
        dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
        assert torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2), "o does not match reference"
        assert torch.allclose(dq, dq_ref, rtol=1e-2, atol=1e-2), "dq does not match reference"
        assert torch.allclose(dk, dk_ref, rtol=1e-2, atol=1e-2), "dk does not match reference"
        assert torch.allclose(dv, dv_ref, rtol=1e-2, atol=1e-2), "dv does not match reference"
        print("All checks passed! ✅")
        
    def profile(self, warmup=500):
        q, k, v, do = self.gen_inputs(4)
        # fwd
        with torch.no_grad():
            fwd_latency = do_bench(lambda: self.forward(q, k, v),
                                   warmup=warmup)
            print(f"Fwd latency: {fwd_latency:.2f} ms")
            print(f"Fwd FLOPs: {self.fwd_flops / fwd_latency * 1e-9:.2f} TFLOPs")
        # bwd
        o = self.forward(q, k, v)
        bwd_latency = do_bench(lambda: o.backward(do, retain_graph=True), warmup=warmup)
        print(f"Bwd latency: {bwd_latency:.2f} ms")
        print(f"Bwd FLOPs: {self.bwd_flops / bwd_latency * 1e-9:.2f} TFLOPs")
        
if __name__ == "__main__":
    kernel = MHA_kernel(8, 32, 1024, 64, False)
    kernel.check()
    kernel.profile()
        