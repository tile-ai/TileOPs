import torch
from torch import nn
from torch.nn import functional as F
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import autotune
import itertools
from top.utils import is_hopper


__all__ = ['MHADecodeKernel']


def get_configs_decode():
    block_M = [32, 64, 128]
    block_N = [32, 64, 128]
    num_split = [2, 4, 8]
    num_stages = [1, 2]
    threads = [128, 256]
    _configs = list(itertools.product(block_M, block_N, num_split, num_stages, threads))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'num_split': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def _mha_decode(batch, heads, seqlen_q, seqlen_kv, dim, dtype: str, tune=False):
    """This kernel is directly adapted from tilelang/examples/example_mha_inference.py. """
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, seqlen_q, heads, dim]
    shape_kv = [batch, seqlen_kv, heads, dim]
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

                # To Do: support tma
                for i, j in T.Parallel(block_M, dim):
                    g_row = mid * block_M + i
                    if g_row < seqlen_q:
                        Q_shared[i, j] = Q[bid, g_row, hid, j]
                    else:
                        Q_shared[i, j] = T.cast(0, dtype)

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

                # To Do: support tma
                for i, j in T.Parallel(block_M, dim):
                    g_row = mid * block_M + i
                    if g_row < seqlen_q:
                        Output_partial[bid, g_row, hid, sid, j] = O_shared[i, j]

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

                # To Do: support tma
                for k in T.Pipelined(num_split, num_stages=2):
                    for i, j in T.Parallel(block_M, dim):
                        g_row = bx * block_M + i
                        if g_row < seqlen_q:
                            po_shared[i, j] = Output_partial[bz, g_row, by, k, j]
                        else:
                            po_shared[i, j] = T.cast(0, dtype)
                    T.copy(po_shared, po_local)
                    T.copy(lse_local[k, :], lse_local_split)
                    for i in T.Parallel(block_M):
                        scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
                    for i, j in T.Parallel(block_M, dim):
                        o_accum_local[i, j] += po_local[i, j] * scale_local[i]
                T.copy(o_accum_local, o_shared)

                # To Do: support tma
                for i, j in T.Parallel(block_M, dim):
                    g_row = bx * block_M + i
                    if g_row < seqlen_q:
                        Output[bz, g_row, by, j] = o_shared[i, j]

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

        @autotune(configs=get_configs_decode(), warmup=10, rep=10, cache_input_tensors=False)
        @tl.jit(
            out_idx=[5],
            pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True},
            compile_flags=["-O3", "-DENABLE_BF16"])
        def _mha_decode_kernel(block_M=None,
                               block_N=None,
                               num_split=None,
                               num_stages=None,
                               threads=None):
            return _mha_decode_func(block_M, block_N, num_split, num_stages, threads)

        return _mha_decode_kernel()
    else:

        @tl.jit(
            out_idx=[5],
            pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True},
            compile_flags=["-O3", "-DENABLE_BF16"])
        def _mha_decode_kernel(block_M, block_N, num_split, num_stages, threads):
            return _mha_decode_func(block_M, block_N, num_split, num_stages, threads)

        return _mha_decode_kernel


class _MHA_decode_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, num_split, config):
        BATCH, KV_CTX, H, D_HEAD = k.shape
        dtype = q.dtype
        dtype_str = dtype.__str__().split('.')[-1]

        mod = _mha_decode(BATCH, H, 1, KV_CTX, D_HEAD, dtype_str)(**config)
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
                 block_M=None,
                 block_N=None,
                 num_split=None,
                 num_stages=None,
                 threads=None,
                 tune=False,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = MHA_decode_attention
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seqlen_kv = seqlen_kv
        self.head_dim = head_dim
        
        # Use heuristic params if not specified
        if is_hopper():
            _block_M = 128
            _block_N = 128
            _num_split = 4
            _threads = 256
            _num_stages = 2
        else:  # Ampere
            _block_M = 64
            _block_N = 64 if head_dim <= 128 else 32
            _num_split = 4
            _threads = 128
            _num_stages = 2
        
        self.block_M = block_M if block_M is not None else _block_M
        self.block_N = block_N if block_N is not None else _block_N
        self.num_split = num_split if num_split is not None else _num_split
        self.num_stages = num_stages if num_stages is not None else _num_stages
        self.threads = threads if threads is not None else _threads
        
        assert dtype in [torch.float16, torch.bfloat16], f"dtype must be float16 or bfloat16, got {dtype}"
        self.dtype = dtype
        self.dtype_str = dtype.__str__().split('.')[-1]
        self.device = device
        self.config = {
            "block_M": self.block_M,
            "block_N": self.block_N,
            "num_split": self.num_split,
            "num_stages": self.num_stages,
            "threads": self.threads
        }
        print(f'MHADecodeKernel config: {self.config}')
        self.tune = tune
        self.tune_config = None
        self.program = _mha_decode(self.batch_size, self.num_heads, 1, self.seqlen_kv,
                                   self.head_dim, self.dtype_str)(**self.config)
        # self.kernel = tilelang.compile(self.program, out_idx=[5])
        self.profiler = self.program.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
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
            self.batch_size, self.num_heads, 1, self.seqlen_kv, self.head_dim, self.dtype_str, tune=True)
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
        rtol, atol = {
            torch.float16: (1e-2, 1e-2),
            torch.bfloat16: (2e-2, 2e-2),
        }[self.dtype]
        
        Q, K, V = self.gen_inputs()
        o = self.forward(Q, K, V)
        o_ref = self.ref_program(Q, K, V)
        assert torch.allclose(
            o, o_ref, rtol=rtol, atol=atol), f"o max err: {(o-o_ref).abs().max()}"
        print("All checks passed! ✅")

    def profile(self, warmup=500):
        if self.tune_config is None and self.tune:
            self.autotune()
        if self.tune_config:
            self.program = _mha_decode(self.batch_size, self.num_heads, 1, self.seqlen_kv,
                                       self.head_dim, self.dtype_str)(**self.tune_config)
            # self.kernel = tilelang.compile(self.program, out_idx=[5])
            self.profiler = self.program.get_profiler(
                tensor_supply_type=tl.TensorSupplyType.Auto)
        with torch.no_grad():
            ref_latency = self.profiler.do_bench(self.ref_program, warmup=warmup)
            print(f'Reference Latency: {ref_latency:.2f} ms')
            print(f"Reference FLOPs: {self.total_flops / ref_latency * 1e-9:.2f} TFLOPs")

            latency = self.profiler.do_bench(warmup=warmup)
            print(f'Latency: {latency:.2f} ms')
            print(f"FLOPs: {self.total_flops / latency * 1e-9:.2f} TFLOPs")
