# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import torch
from torch import nn
import tilelang as tl
from tilelang.profiler import do_bench
import tilelang.language as T
import fla.ops.linear_attn  # We compare with Triton implementation in FLA
from top.utils import str2dtype, reduce_on_dim0

__all__ = [
    'LinearAttentionFusedChunkKernel',
    'LinearAttentionFusedRecurrentKernel'
]


@tl.jit(out_idx=[3])
def _fused_chunk_fwd(B, S, H, D, scale=None, dtype='float16', BK=64, BV=64, chunk_size=64):
    accum_dtype = "float"
    NK = D // BK
    NV = D // BV
    NT = S // chunk_size

    if scale is None:
        scale = D**-0.5

    @T.prim_func
    def main(
            Q: T.Tensor([B, S, H, D], dtype),  # type: ignore
            K: T.Tensor([B, S, H, D], dtype),  # type: ignore
            V: T.Tensor([B, S, H, D], dtype),  # type: ignore
            Output: T.Tensor([NK, B, S, H, D], dtype)  # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            h_shared = T.alloc_shared([BK, BV], dtype)
            s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            T.clear(h)

            T.use_swizzle(8)

            for i in T.serial(0, NT):
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, i * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(K[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV], v)

                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    s_shared[row, col] = T.if_then_else(row >= col, s[row, col], 0)

                T.gemm(s_shared, v, o, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(q, h_shared, o)
                T.gemm(k, v, h, transpose_A=True)
                T.copy(
                    o, Output[i_k, i_b, i * chunk_size:(i + 1) * chunk_size, i_h,
                              i_v * BV:(i_v + 1) * BV])

    return main


@tl.jit(out_idx=[4, 5, 6])
def _fused_chunk_bwd(B, S, H, D, scale=None, dtype='float16', BK=64, BV=64, chunk_size=64):
    accum_dtype = "float"
    NK = D // BK
    NV = D // BV
    NT = S // chunk_size

    if scale is None:
        scale = D**-0.5

    @T.prim_func
    def main(
            Q: T.Tensor([B, S, H, D], dtype),  # type: ignore
            K: T.Tensor([B, S, H, D], dtype),  # type: ignore
            V: T.Tensor([B, S, H, D], dtype),  # type: ignore
            dO: T.Tensor([B, S, H, D], dtype),  # type: ignore
            dQ: T.Tensor([NV, B, S, H, D], dtype),  # type: ignore
            dK: T.Tensor([NV, B, S, H, D], dtype),  # type: ignore
            dV: T.Tensor([NK, B, S, H, D], dtype),  # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            ds = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            ds_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            dq = T.alloc_fragment([chunk_size, BK], accum_dtype)
            dk = T.alloc_fragment([chunk_size, BK], accum_dtype)
            dv = T.alloc_fragment([chunk_size, BV], accum_dtype)
            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            do = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BV, BK], accum_dtype)
            h_shared = T.alloc_shared([BV, BK], dtype)
            dh = T.alloc_fragment([BK, BV], accum_dtype)
            dh_shared = T.alloc_shared([BK, BV], dtype)
            T.clear(h)
            T.clear(dh)

            T.use_swizzle(8)

            # Calculate dQ
            for i in T.serial(0, NT):
                T.copy(dO[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV],
                       do)
                T.copy(K[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV], v)

                T.gemm(do, v, ds, transpose_B=True, clear_accum=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row >= col, ds[row, col], 0)

                T.gemm(ds_shared, k, dq, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(do, h_shared, dq)
                if i < NT - 1:
                    T.gemm(v, k, h, transpose_A=True)
                for row, col in T.Parallel(chunk_size, BK):
                    dQ[i_v, i_b, i * chunk_size + row, i_h, i_k * BK + col] = dq[row, col] * scale

            # Calculate dK, dV (reversely)
            for i in T.Pipelined(1, NT + 1, num_stages=1):
                start = NT - i
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, start * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(
                    K[i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                      i_k * BK:(i_k + 1) * BK], k)
                T.copy(
                    V[i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                      i_v * BV:(i_v + 1) * BV], v)
                T.copy(
                    dO[i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                       i_v * BV:(i_v + 1) * BV], do)
                T.copy(dh, dh_shared)

                # Calculate dk
                T.gemm(
                    v, do, ds, transpose_B=True, clear_accum=True
                )  # ds here actually means `s`, but we simply reuse the buffer `ds`
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row <= col, ds[row, col], 0)
                T.gemm(ds_shared, q, dk, clear_accum=True)
                T.gemm(v, dh_shared, dk, transpose_B=True)

                # Calculate dv
                T.gemm(k, q, ds, transpose_B=True, clear_accum=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row <= col, ds[row, col], 0)
                T.gemm(ds_shared, do, dv, clear_accum=True)
                T.gemm(k, dh_shared, dv)

                # Update dh
                if i < NT:
                    T.gemm(q, do, dh, transpose_A=True)

                T.copy(
                    dk, dK[i_v, i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                           i_k * BK:(i_k + 1) * BK])
                T.copy(
                    dv, dV[i_k, i_b, start * chunk_size:(start + 1) * chunk_size, i_h,
                           i_v * BV:(i_v + 1) * BV])

    return main


class _fused_chunk_linear_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale, dtype, BK, BV, chunk_size):
        B, S, H, D = q.shape
        ctx.B, ctx.S, ctx.H, ctx.D, ctx.scale, ctx.dtype, ctx.BK, ctx.BV, ctx.chunk_size = (
            B, S, H, D, scale, dtype, BK, BV, chunk_size)
        ctx.save_for_backward(q, k, v)

        mod = _fused_chunk_fwd(B, S, H, D, scale, dtype, BK, BV, chunk_size)
        return reduce_on_dim0(mod(q, k, v))

    @staticmethod
    def backward(ctx, do):
        B, S, H, D, scale, dtype, BK, BV, chunk_size = (ctx.B, ctx.S, ctx.H, ctx.D, ctx.scale,
                                                        ctx.dtype, ctx.BK, ctx.BV, ctx.chunk_size)
        q, k, v = ctx.saved_tensors

        mod = _fused_chunk_bwd(B, S, H, D, scale, dtype, BK, BV, chunk_size)
        dq, dk, dv = map(reduce_on_dim0, mod(q, k, v, do))
        return dq, dk, dv, None, None, None, None, None


fused_chunk_linear_attention = _fused_chunk_linear_attention.apply


class LinearAttentionFusedChunkKernel(nn.Module):
    '''Calculate the results in one pass without materializing intermediate hidden states.'''

    def __init__(self,
                 batch_size,
                 seq_len,
                 num_heads,
                 head_dim,
                 dtype='float16',
                 block_K=64,
                 block_V=64,
                 chunk_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.torch_dtype = str2dtype[dtype]
        self.block_K = block_K
        self.block_V = block_V
        self.chunk_size = chunk_size

        self.attention = fused_chunk_linear_attention

    def forward(self, q, k, v, scale=None):  # Layout: [B, S, H, D]
        return self.attention(q, k, v, scale, self.dtype, self.block_K, self.block_V,
                              self.chunk_size)

    @classmethod
    def ref_program(cls, q, k, v, scale=None):
        return fla.ops.linear_attn.fused_chunk_linear_attn(q, k, v, scale, normalize=False)

    def gen_inputs(self, n: int = 4):
        return (torch.randn((self.batch_size, self.seq_len, self.num_heads, self.head_dim),
                            device='cuda',
                            dtype=self.torch_dtype,
                            requires_grad=True) for _ in range(n))

    def profile(self, warmup=100):
        q, k, v, do = self.gen_inputs(4)
        # fwd
        with torch.no_grad():
            fwd_latency = do_bench(lambda: self.forward(q, k, v), warmup=warmup)
            print(f"Fwd latency: {fwd_latency:.2f} ms")
            fwd_ref_latency = do_bench(lambda: self.ref_program(q, k, v), warmup=warmup)
            print(f"Fwd ref latency: {fwd_ref_latency:.2f} ms")
        # bwd
        o = self.forward(q, k, v)
        bwd_latency = do_bench(lambda: o.backward(do, retain_graph=True), warmup=warmup)
        print(f"Bwd latency: {bwd_latency:.2f} ms")
        o_ref, _ = self.ref_program(q, k, v)
        bwd_ref_latency = do_bench(lambda: o_ref.backward(do, retain_graph=True), warmup=warmup)
        print(f"Bwd ref latency: {bwd_ref_latency:.2f} ms")

    def check(self):
        q, k, v, do = self.gen_inputs(4)
        o = self.forward(q, k, v)
        o.backward(do)
        dq, q.grad = q.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dv, v.grad = v.grad.clone(), None
        o_ref, _ = self.ref_program(q, k, v)
        o_ref.backward(do)
        dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
        assert torch.allclose(o, o_ref), "o does not match reference"
        assert torch.allclose(dq, dq_ref), "dq does not match reference"
        assert torch.allclose(dk, dk_ref), "dk does not match reference"
        assert torch.allclose(dv, dv_ref), "dv does not match reference"
        print("All checks passed! ✅")


@tl.jit(out_idx=[3])
def _fused_recurrent_fwd(B, H, S, D, scale=None, dtype='float16', BK=32, BV=32):
    accum_dtype = "float"
    NK = D // BK
    NV = D // BV
        
    if scale is None:
        scale = D**-0.5
        
    @T.prim_func
    def main(
        Q: T.Tensor([B, H, S, D], dtype),  # type: ignore
        K: T.Tensor([B, H, S, D], dtype),  # type: ignore
        V: T.Tensor([B, H, S, D], dtype),  # type: ignore
        Output: T.Tensor([NK, B, H, S, D], dtype)  # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            q = T.alloc_shared([BK], accum_dtype)
            k = T.alloc_shared([BK], accum_dtype)
            v = T.alloc_shared([BV], accum_dtype)
            o = T.alloc_fragment([BV, BK], accum_dtype)
            o_sum = T.alloc_fragment([BV], accum_dtype)
            h = T.alloc_fragment([BV, BK], accum_dtype)
            T.clear(h)

            for t in T.Pipelined(0, S):
                T.copy(K[i_b, i_h, t, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i_h, t, i_v * BV:(i_v + 1) * BV], v)
                for i in T.Parallel(BK):
                    q[i] = Q[i_b, i_h, t, i_k * BK + i] * scale

                for i, j in T.Parallel(BV, BK):
                    h[i, j] += k[j] * v[i]
                    o[i, j] = h[i, j] * q[j]
                T.reduce_sum(o, o_sum, dim=1, clear=True)
                    
                T.copy(o_sum, Output[i_k, i_b, i_h, t, i_v * BV:(i_v + 1) * BV])
                
    return main

@tl.jit(out_idx=[4, 5, 6])
def _fused_recurrent_bwd(B, H, S, D, scale=None, dtype='float16', BK=32, BV=32):
    accum_dtype = "float"
    NK = D // BK
    NV = D // BV
        
    if scale is None:
        scale = D**-0.5
        
    @T.prim_func
    def main(
        Q: T.Tensor([B, H, S, D], dtype),  # type: ignore
        K: T.Tensor([B, H, S, D], dtype),  # type: ignore
        V: T.Tensor([B, H, S, D], dtype),  # type: ignore
        dO: T.Tensor([B, H, S, D], dtype),  # type: ignore
        dQ: T.Tensor([NV, B, H, S, D], dtype),  # type: ignore
        dK: T.Tensor([NV, B, H, S, D], dtype),  # type: ignore
        dV: T.Tensor([NK, B, H, S, D], dtype)  # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H
            
            q = T.alloc_shared([BK], accum_dtype)
            k = T.alloc_shared([BK], accum_dtype)
            v = T.alloc_shared([BV], accum_dtype)
            do = T.alloc_shared([BV], accum_dtype)
            dq = T.alloc_fragment([BK, BV], accum_dtype)
            dq_sum = T.alloc_fragment([BK], accum_dtype)
            dk  = T.alloc_fragment([BK, BV], accum_dtype)
            dk_sum = T.alloc_fragment([BK], accum_dtype)
            dv = T.alloc_fragment([BV, BK], accum_dtype)
            dv_sum = T.alloc_fragment([BV], accum_dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            dh = T.alloc_fragment([BK, BV], accum_dtype)
            T.clear(h)
            T.clear(dh)
            
            # 1. Calculate dQ
            for t in T.Pipelined(0, S):
                T.copy(K[i_b, i_h, t, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i_h, t, i_v * BV:(i_v + 1) * BV], v)
                T.copy(dO[i_b, i_h, t, i_v * BV:(i_v + 1) * BV], do)

                for i, j in T.Parallel(BK, BV):
                    h[i, j] += k[i] * v[j]
                    dq[i, j] = h[i, j] * do[j]
                T.reduce_sum(dq, dq_sum, dim=1, clear=True)
                for i in T.Parallel(BK):
                    dq_sum[i] *= scale
                
                T.copy(dq_sum, dQ[i_v, i_b, i_h, t, i_k * BK:(i_k + 1) * BK])
                
            # 2. Calculate dK, dV (reversely)
            for t in T.Pipelined(0, S):
                for i in T.Parallel(BK):
                    q[i] = Q[i_b, i_h, S-t-1, i_k * BK + i] * scale
                T.copy(K[i_b, i_h, S-t-1, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i_h, S-t-1, i_v * BV:(i_v + 1) * BV], v)
                T.copy(dO[i_b, i_h, S-t-1, i_v * BV:(i_v + 1) * BV], do)

                for i, j in T.Parallel(BK, BV):
                    dh[i, j] += q[i] * do[j]
                    dk[i, j] = dh[i, j] * v[j]
                    dv[i, j] = k[i] * dh[i, j]
                T.reduce_sum(dk, dk_sum, dim=1, clear=True)
                T.reduce_sum(dv, dv_sum, dim=0, clear=True)

                T.copy(dk_sum,
                       dK[i_v, i_b, i_h, S-t-1, i_k * BK:(i_k + 1) * BK])
                T.copy(dv_sum,
                       dV[i_k, i_b, i_h, S-t-1, i_v * BV:(i_v + 1) * BV])
                    
    return main


class _fused_recurrent_linear_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale, dtype, BK, BV):
        B, H, S, D = q.shape
        ctx.B, ctx.H, ctx.S, ctx.D, ctx.scale, ctx.dtype, ctx.BK, ctx.BV = (
            B, H, S, D, scale, dtype, BK, BV)
        ctx.save_for_backward(q, k, v)

        mod = _fused_recurrent_fwd(B, H, S, D, scale, dtype, BK, BV)
        return reduce_on_dim0(mod(q, k, v))

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        mod = _fused_recurrent_bwd(ctx.B, ctx.H, ctx.S, ctx.D, ctx.scale, ctx.dtype, ctx.BK, ctx.BV)
        dq, dk, dv = map(reduce_on_dim0, mod(q, k, v, do))
        return dq, dk, dv, None, None, None, None


fused_recurrent_linear_attention = _fused_recurrent_linear_attention.apply


class LinearAttentionFusedRecurrentKernel(nn.Module):
    '''Calculate the results in a recursive way.'''
    
    def __init__(self,
                 batch_size,
                 seq_len,
                 num_heads,
                 head_dim,
                 dtype='float16',
                 block_K=32,
                 block_V=32):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.torch_dtype = str2dtype[dtype]
        self.block_K = block_K
        self.block_V = block_V
        assert self.head_dim % self.block_K == 0, "head_dim must be divisible by block_K"
        assert self.head_dim % self.block_V == 0, "head_dim must be divisible by block_V"

        self.attention = fused_recurrent_linear_attention

    def forward(self, q, k, v, scale=None):  # Attention: Layout is [B, H, S, D]
        return self.attention(q, k, v, scale, self.dtype, self.block_K, self.block_V)

    @classmethod
    def ref_program(cls, q, k, v, scale=None):
        # You may need to comment out the deprecated warning in this function.
        return fla.ops.linear_attn.fused_recurrent_linear_attn(q, k, v, scale, normalize=False, head_first=True)

    def gen_inputs(self, n: int = 4):
        return (torch.randn((self.batch_size, self.num_heads, self.seq_len, self.head_dim),
                            device='cuda',
                            dtype=self.torch_dtype,
                            requires_grad=True) for _ in range(n))

    def profile(self, warmup=100):
        q, k, v, do = self.gen_inputs(4)
        # fwd
        with torch.no_grad():
            fwd_latency = do_bench(lambda: self.forward(q, k, v), warmup=warmup)
            print(f"Fwd latency: {fwd_latency:.2f} ms")
            fwd_ref_latency = do_bench(lambda: self.ref_program(q, k, v), warmup=warmup)
            print(f"Fwd ref latency: {fwd_ref_latency:.2f} ms")
        # bwd
        o = self.forward(q, k, v)
        bwd_latency = do_bench(lambda: o.backward(do, retain_graph=True), warmup=warmup)
        print(f"Bwd latency: {bwd_latency:.2f} ms")
        o_ref, _ = self.ref_program(q, k, v)
        bwd_ref_latency = do_bench(lambda: o_ref.backward(do, retain_graph=True), warmup=warmup)
        print(f"Bwd ref latency: {bwd_ref_latency:.2f} ms")

    def check(self):
        #TODO: This kernel still has some minor numerical differences compared with FLA implementation.
        q, k, v, do = self.gen_inputs(4)
        o = self.forward(q, k, v)
        o.backward(do)
        dq, q.grad = q.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dv, v.grad = v.grad.clone(), None
        o_ref, _ = self.ref_program(q, k, v)
        o_ref.backward(do)
        dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
        assert (o - o_ref).abs().mean() < 1e-4, "o does not match reference"
        assert (dq - dq_ref).abs().mean() < 1e-4, "dq does not match reference"
        assert (dk - dk_ref).abs().mean() < 1e-4, "dk does not match reference"
        assert (dv - dv_ref).abs().mean() < 1e-4, "dv does not match reference"
        print("All checks passed! ✅")
    

