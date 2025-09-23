import torch
from torch import nn
from torch.nn import functional as F
import tilelang as tl
import tilelang.language as T
# from tilelang.profiler import do_bench
from tilelang.autotuner import *
import itertools


__all__ = ['MHAKernel', 'MHADecodeKernel']


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


def _mha_fwd(batch, heads, seq_len, dim, is_causal, tune=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    def _mha_fwd_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _mha_fwd_main(
                Q: T.Tensor(shape, dtype),  # type: ignore
                K: T.Tensor(shape, dtype),  # type: ignore
                V: T.Tensor(shape, dtype),  # type: ignore
                Output: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
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

        return _mha_fwd_main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tl.jit(out_idx=[3, 4])
        def _mha_fwd_kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return _mha_fwd_func(block_M, block_N, num_stages, threads)

        return _mha_fwd_kernel()
    else:

        @tilelang.jit(out_idx=[3, 4])
        def _mha_fwd_kernel(block_M, block_N, num_stages, threads):
            return _mha_fwd_func(block_M, block_N, num_stages, threads)

        return _mha_fwd_kernel


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


def _mha_bwd(batch, heads, seq_len, dim, is_causal, tune=False):
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
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
            with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=128) as (bx, by, bz):
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
                    K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })
                T.copy(K[bz, by * block_M:(by + 1) * block_M, bx, :], K_shared)
                T.copy(V[bz, by * block_M:(by + 1) * block_M, bx, :], V_shared)
                T.clear(dv)
                T.clear(dk)
                loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_N)
                for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
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
                        if k * block_N + i < seq_len:
                            T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
                T.copy(dv, dv_shared)
                T.copy(dk, dk_shared)
                T.copy(dv_shared, dV[bz, by * block_M:(by + 1) * block_M, bx, :])
                T.copy(dk_shared, dK[bz, by * block_M:(by + 1) * block_M, bx, :])

        return _mha_bwd_main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tl.jit(out_idx=[6, 7, 8])
        def _mha_bwd_kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return _mha_bwd_func(block_M, block_N, num_stages, threads)

        return _mha_bwd_kernel()
    else:

        @tilelang.jit(out_idx=[6, 7, 8])
        def _mha_bwd_kernel(block_M, block_N, num_stages, threads):
            return _mha_bwd_func(block_M, block_N, num_stages, threads)

        return _mha_bwd_kernel


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
    def forward(ctx, q, k, v, causal, config, bwd_config):
        BATCH, N_CTX, H, D_HEAD = q.shape
        mod = _mha_fwd(BATCH, H, N_CTX, D_HEAD, causal)(**config)
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
        mod_prep = _mha_bwd_preprocess(BATCH, H, N_CTX, D_HEAD)
        mod_post = _mha_bwd_postprocess(BATCH, H, N_CTX, D_HEAD)
        mod = _mha_bwd(BATCH, H, N_CTX, D_HEAD, ctx.causal)(**ctx.bwd_config)
        delta = mod_prep(o, do)
        dq = torch.zeros_like(q, dtype=torch.float, device=q.device, requires_grad=False)
        dk = torch.zeros_like(k, dtype=torch.float16, device=k.device, requires_grad=False)
        dv = torch.zeros_like(v, dtype=torch.float16, device=v.device, requires_grad=False)
        dq, dk, dv = mod(q, k, v, do, lse, delta)
        dq = mod_post(dq)
        return dq, dk, dv, None, None, None


MHA_attention = _MHA_attention.apply


class MHAKernel:

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
                 num_stages=1,
                 threads=128,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = MHA_attention
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        f_block_M = 64
        f_block_N = 64 if head_dim <= 128 else 32
        self.fwd_block_M = fwd_block_M if fwd_block_M is not None else f_block_M
        self.fwd_block_N = fwd_block_N if fwd_block_N is not None else f_block_N
        b_block_M = 64
        b_block_N = 64 if head_dim <= 64 else 32
        self.bwd_block_M = bwd_block_M if bwd_block_M is not None else b_block_M
        self.bwd_block_N = bwd_block_N if bwd_block_N is not None else b_block_N
        self.causal = causal
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
        self.dtype = dtype
        self.device = device
        flops_per_matmul = 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim
        self.fwd_flops = 2 * flops_per_matmul
        self.bwd_flops = 5 * flops_per_matmul
        if causal:
            self.fwd_flops *= 0.5
            self.bwd_flops *= 0.5
        # (BATCH, H, N_CTX, D_HEAD, causal)(**config)
        self.fwd_program = _mha_fwd(batch_size, num_heads, seq_len, head_dim,
                                    causal)(**self.fwd_config)
        # self.fwd_kernel = tilelang.compile(self.fwd_program, out_idx=[4, 5])
        self.fwd_profiler = self.fwd_program.get_profiler(
            tensor_supply_type=tilelang.TensorSupplyType.Auto)
        self.bwd_program = _mha_bwd(batch_size, num_heads, seq_len, head_dim,
                                    causal)(**self.bwd_config)
        # self.bwd_kernel = tilelang.compile(self.bwd_program)
        self.bwd_profiler = self.bwd_program.get_profiler(
            tensor_supply_type=tilelang.TensorSupplyType.Randn)

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
        q, k, v, do = self.gen_inputs()
        o = self.forward(q, k, v)
        o.backward(do)
        dq, q.grad = q.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dv, v.grad = v.grad.clone(), None
        o_ref = self.ref_program(q, k, v)
        o_ref.backward(do)
        dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
        assert torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2), "o does not match reference"
        assert torch.allclose(dq, dq_ref, rtol=1e-2, atol=1e-2), "dq does not match reference"
        assert torch.allclose(dk, dk_ref, rtol=1e-2, atol=1e-2), "dk does not match reference"
        assert torch.allclose(dv, dv_ref, rtol=1e-2, atol=1e-2), "dv does not match reference"
        print("All checks passed! ✅")

    def profile(self, warmup=500):
        q, k, v, do = self.gen_inputs()
        # fwd
        with torch.no_grad():
            if self.fwd_tune_config is None and self.fwd_tune:
                self.fwd_autotune()
            if self.fwd_tune_config:
                self.fwd_program = _mha_fwd(self.batch_size, self.num_heads, self.seq_len,
                                            self.head_dim, self.causal)(**self.fwd_config)
                # self.fwd_kernel = tilelang.compile(self.fwd_program, out_idx=[4, 5])
                self.fwd_profiler = self.fwd_program.get_profiler(
                    tensor_supply_type=tilelang.TensorSupplyType.Auto)
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
                                        self.head_dim, self.causal)(**self.bwd_config)
            self.bwd_profiler = self.bwd_program.get_profiler(
                tensor_supply_type=tilelang.TensorSupplyType.Auto)
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


def get_configs_decode():
    block_M = [32, 64, 128]
    block_N = [32, 64, 128]
    num_split = [2, 4, 8]
    num_stages = [1, 2]
    threads = [128]
    _configs = list(itertools.product(block_M, block_N, num_split, num_stages, threads))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'num_split': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def _mha_decode(batch, heads, seqlen_q, seqlen_kv, dim, tune=False):
    """This kernel is directly adapted from tilelang/examples/example_mha_inference.py. """
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, seqlen_q, heads, dim]
    shape_kv = [batch, seqlen_kv, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    def _mha_decode_func(block_M, block_N, num_split, num_stages, threads):
        part_shape = [batch, seqlen_q, heads, num_split, dim]

        @T.macro
        def MMA0(
            K: T.Tensor(shape_kv, dtype),  # type: ignore
            Q_shared: T.SharedBuffer([block_M, dim], dtype),  # type: ignore
            K_shared: T.SharedBuffer([block_N, dim], dtype),  # type: ignore
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),  # type: ignore
            k: T.int32,
            mid: T.int32,
            hid: T.int32,
            bid: T.int32,
            sid: T.int32,
        ):
            T.copy(
                K[bid, (seqlen_kv // num_split) * sid + k * block_N:(seqlen_kv // num_split) * sid +
                  (k + 1) * block_N, hid, :], K_shared)
            T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(shape_kv, dtype),  # type: ignore
            V_shared: T.SharedBuffer([block_M, dim], dtype),  # type: ignore
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),  # type: ignore
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),  # type: ignore
            k: T.int32,
            hid: T.int32,
            bid: T.int32,
            sid: T.int32,
        ):
            T.copy(
                V[bid, (seqlen_kv // num_split) * sid + k * block_N:(seqlen_kv // num_split) * sid +
                  (k + 1) * block_N, hid, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),  # type: ignore
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),  # type: ignore
                scores_max: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
                logsum: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
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
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),  # type: ignore
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.macro
        def flash_attn_split(
                Q: T.Tensor(shape_q, dtype),  # type: ignore
                K: T.Tensor(shape_kv, dtype),  # type: ignore
                V: T.Tensor(shape_kv, dtype),  # type: ignore
                glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),  # type: ignore
                Output_partial: T.Tensor(part_shape, dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seqlen_q, block_M), heads * batch, num_split,
                    threads=threads) as (bx, by, bz):
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

                mid = bx
                hid = by % heads
                bid = by // heads
                sid = bz

                T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bid, mid * block_M:(mid + 1) * block_M, hid, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # TODO: Handle causal split case
                loop_range = T.ceildiv((seqlen_kv // num_split), block_N)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, mid, hid, bid, sid)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, hid, bid, sid)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, glse[bid, hid, sid, mid * block_M:(mid + 1) * block_M])
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output_partial[bid, mid * block_M:(mid + 1) * block_M, hid,
                                                sid, :])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),  # type: ignore
                Output_partial: T.Tensor(part_shape, dtype),  # type: ignore
                Output: T.Tensor(shape_q, dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seqlen_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                po_local = T.alloc_fragment([block_M, dim], dtype)
                po_shared = T.alloc_shared([block_M, dim], dtype)
                o_accum_local = T.alloc_fragment([block_M, dim], accum_dtype)
                o_shared = T.alloc_shared([block_M, dim], dtype)
                lse_local = T.alloc_fragment([num_split, block_M], dtype)
                lse_local_split = T.alloc_fragment([block_M], accum_dtype)
                lse_logsum_local = T.alloc_fragment([block_M], accum_dtype)
                lse_max_local = T.alloc_fragment([block_M], accum_dtype)
                scale_local = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({
                    o_accum_local:
                        T.Fragment(o_accum_local.shape, forward_thread_fn=lambda i, j: i),
                    lse_local_split:
                        T.Fragment(lse_local_split.shape, forward_thread_fn=lambda i: i),
                    o_shared:
                        tl.layout.make_swizzled_layout(o_shared),
                    po_shared:
                        tl.layout.make_swizzled_layout(po_shared),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                T.copy(glse[
                    bz,
                    by,
                    :,
                    bx * block_M:(bx + 1) * block_M,
                ], lse_local)
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=False)
                for k in T.Pipelined(num_split):
                    T.copy(lse_local[k, :], lse_local_split)
                    for i in T.Parallel(block_M):
                        lse_logsum_local[i] += T.exp2(lse_local_split[i] - lse_max_local[i])
                for i in T.Parallel(block_M):
                    lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]
                for k in T.Pipelined(num_split, num_stages=2):
                    T.copy(Output_partial[bz, bx * block_M:(bx + 1) * block_M, by, k, :], po_shared)
                    T.copy(po_shared, po_local)
                    T.copy(lse_local[k, :], lse_local_split)
                    for i in T.Parallel(block_M):
                        scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
                    for i, j in T.Parallel(block_M, dim):
                        o_accum_local[i, j] += po_local[i, j] * scale_local[i]
                T.copy(o_accum_local, o_shared)
                T.copy(o_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        @T.prim_func
        def _mha_decode_main(
                Q: T.Tensor(shape_q, dtype),  # type: ignore
                K: T.Tensor(shape_kv, dtype),  # type: ignore
                V: T.Tensor(shape_kv, dtype),  # type: ignore
                glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),  # type: ignore
                Output_partial: T.Tensor(part_shape, dtype),  # type: ignore
                Output: T.Tensor(shape_q, dtype),  # type: ignore
        ):
            flash_attn_split(Q, K, V, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return _mha_decode_main

    if tune:

        @autotune(configs=get_configs_decode(), warmup=10, rep=10)
        @tl.jit(out_idx=[5], cache_input_tensors=False)
        def _mha_decode_kernel(block_M=None,
                               block_N=None,
                               num_split=None,
                               num_stages=None,
                               threads=None):
            return _mha_decode_func(block_M, block_N, num_split, num_stages, threads)

        return _mha_decode_kernel()
    else:

        @tilelang.jit(out_idx=[5])
        def _mha_decode_kernel(block_M, block_N, num_split, num_stages, threads):
            return _mha_decode_func(block_M, block_N, num_split, num_stages, threads)

        return _mha_decode_kernel


class _MHA_decode_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, num_split, config):
        BATCH, KV_CTX, H, D_HEAD = k.shape

        mod = _mha_decode(BATCH, H, 1, KV_CTX, D_HEAD)(**config)
        glse = torch.empty((BATCH, H, num_split, 1), dtype=q.dtype, device=q.device)
        Output_partial = torch.empty((BATCH, 1, H, num_split, D_HEAD),
                                     dtype=q.dtype,
                                     device=q.device)
        return mod(q, k, v, glse, Output_partial)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("This kernel is used for decoding only!")


MHA_decode_attention = _MHA_decode_attention.apply


class MHADecodeKernel(nn.Module):

    def __init__(self,
                 batch_size,
                 num_heads,
                 seqlen_kv,
                 head_dim,
                 threads=None,
                 block_M=None,
                 block_N=None,
                 num_split=4,
                 tune=False,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = MHA_decode_attention
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seqlen_kv = seqlen_kv
        self.head_dim = head_dim
        block_M_ = 64
        block_N_ = 64 if head_dim <= 128 else 32
        self.block_M = block_M if block_M is not None else block_M_
        self.block_N = block_N if block_N is not None else block_N_
        self.threads = threads if threads else 128
        self.num_split = num_split
        self.dtype = dtype
        self.device = device
        self.config = {
            "block_M": self.block_M,
            "block_N": self.block_N,
            "num_split": self.num_split,
            "num_stages": 2,
            "threads": self.threads
        }
        self.tune = tune
        self.tune_config = None
        self.program = _mha_decode(self.batch_size, self.num_heads, 1, self.seqlen_kv,
                                   self.head_dim)(**self.config)
        # self.kernel = tilelang.compile(self.program, out_idx=[5])
        self.profiler = self.program.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        flops_per_matmul = 2.0 * batch_size * num_heads * seqlen_kv * head_dim
        self.total_flops = 2 * flops_per_matmul

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        assert Q.dim() == 4 and Q.size(1) == 1, "Q must have shape (bsz, 1, H, D)"
        if self.tune_config is None and self.tune:
            self.autotune()
        config = self.tune_config if self.tune_config else self.config
        o = self.attention(Q, K, V, self.num_split, config)
        return o

    def autotune(self):
        best_result = _mha_decode(
            self.batch_size, self.num_heads, 1, self.seqlen_kv, self.head_dim, tune=True)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best fwd latency: {best_latency}")
        print(f"Best TFlops: {self.total_flops / best_latency * 1e-9}")
        print(f"Best fwd config: {best_config}")
        print(f"Ref latency: {ref_latency}")
        if best_result.config:
            self.tune_config = dict(
                zip(["block_M", "block_N", "num_split", "num_stages", "threads"], list(best_config.values())))
            self.num_split = best_config["num_split"]

    @classmethod
    def ref_program(cls,
                    Q: torch.Tensor,
                    K: torch.Tensor,
                    V: torch.Tensor,
                    glse: torch.Tensor = None,
                    Output_partial: torch.Tensor = None) -> torch.Tensor:
        assert Q.dim() == 4 and Q.size(1) == 1, "Q must have shape (bsz, 1, H, D)"
        dim = Q.size(-1)
        scores = torch.einsum('bqhd,bkhd->bqhk', Q, K)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bqhk,bkhd->bqhd', attention_weights, V)
        return output

    def gen_inputs(self):
        shape_q = self.batch_size, 1, self.num_heads, self.head_dim
        shape_kv = self.batch_size, self.seqlen_kv, self.num_heads, self.head_dim
        Q = torch.randn(shape_q, dtype=self.dtype, device=self.device)
        K = torch.randn(shape_kv, dtype=self.dtype, device=self.device)
        V = torch.randn(shape_kv, dtype=self.dtype, device=self.device)
        return Q, K, V

    def check(self):
        Q, K, V = self.gen_inputs()
        o = self.forward(Q, K, V)
        o_ref = self.ref_program(Q, K, V)
        assert torch.allclose(
            o, o_ref, rtol=1e-2, atol=1e-2), "o does not match reference, max diff: {:.4f}".format(
                torch.max(torch.abs(o - o_ref)))
        print("All checks passed! ✅")

    def profile(self, warmup=500):
        if self.tune_config is None and self.tune:
            self.autotune()
        if self.tune_config:
            self.program = _mha_decode(self.batch_size, self.num_heads, 1, self.seqlen_kv,
                                       self.head_dim)(**self.tune_config)
            # self.kernel = tilelang.compile(self.program, out_idx=[5])
            self.profiler = self.program.get_profiler(
                tensor_supply_type=tilelang.TensorSupplyType.Auto)
        with torch.no_grad():
            ref_latency = self.profiler.do_bench(self.ref_program, warmup=warmup)
            print(f'Reference Latency: {ref_latency:.2f} ms')
            print(f"Reference FLOPs: {self.total_flops / ref_latency * 1e-9:.2f} TFLOPs")

            latency = self.profiler.do_bench(warmup=warmup)
            print(f'Latency: {latency:.2f} ms')
            print(f"FLOPs: {self.total_flops / latency * 1e-9:.2f} TFLOPs")
