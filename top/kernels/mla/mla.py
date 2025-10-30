import torch
import tilelang
import tilelang.language as T
from top.kernels.kernel import Kernel
from typing import Optional
import itertools

__all__ = ["mla_decode_kernel"]


def _mla_decode_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim):
    scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    assert kv_head_num == 1, "kv_head_num must be 1"

    @tilelang.jit(
            out_idx=[6],
            pass_configs={
                tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            },
            compile_flags=["-O3", "-DENABLE_BF16"])
    def _mla_decode_func(block_H, block_N, num_split, num_stages, threads=128):
        
        VALID_BLOCK_H = min(block_H, kv_group_num)

        @T.macro
        def _mla_no_split(
                Q: T.Tensor([batch, heads, dim], dtype),
                Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(batch, heads // min(block_H, kv_group_num), threads) as (bx, by):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared = T.alloc_shared([block_N, dim], dtype)
                K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared = T.alloc_shared([block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                cur_kv_head = by // (kv_group_num // block_H)
                T.use_swizzle(10)
                T.annotate_layout({
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                })

                T.copy(Q[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_shared)
                T.copy(Q_pe[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(seqlen_kv, block_N)
                for k in T.Pipelined(loop_range, num_stages):
                    T.copy(KV[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], KV_shared)
                    T.copy(K_pe[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(
                        Q_pe_shared,
                        K_pe_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, S_shared)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :])

        @T.macro
        def _mla_split(
                Q: T.Tensor([batch, heads, dim], dtype),
                Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        ):
            with T.Kernel(
                    batch, heads // min(block_H, kv_group_num), num_split, threads=256) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared = T.alloc_shared([block_N, dim], dtype)
                K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared = T.alloc_shared([block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                cur_kv_head = by // (kv_group_num // block_H)
                T.use_swizzle(10)
                T.annotate_layout({
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                    S_shared: tilelang.layout.make_swizzled_layout(S_shared),
                })

                T.copy(Q[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_shared)
                T.copy(Q_pe[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                for k in T.Pipelined(loop_range, num_stages=2):
                    kv_start = (seqlen_kv // num_split) * bz + k * block_N
                    kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                    T.copy(KV[bx, kv_start:kv_end, cur_kv_head, :], KV_shared)
                    T.copy(K_pe[bx, kv_start:kv_end, cur_kv_head, :], K_pe_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(
                        Q_pe_shared,
                        K_pe_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, S_shared)
                    T.copy(S_shared, acc_s_cast)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, glse[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, bz])
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output_partial[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, bz, :])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_local([1], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local: T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                lse_max_local[0] = -T.infinity(accum_dtype)
                for k in T.serial(num_split):
                    lse_max_local[0] = T.max(lse_max_local[0], glse[bz, by, k])
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
        def main_split(
                Q: T.Tensor([batch, heads, dim], dtype),
                Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            _mla_split(Q, Q_pe, KV, K_pe, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def main_no_split(
                Q: T.Tensor([batch, heads, dim], dtype),
                Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            _mla_no_split(Q, Q_pe, KV, K_pe, Output)

        if num_split > 1:
            return main_split
        else:
            return main_no_split
        
    return _mla_decode_func


@torch.library.custom_op("top::mla_decode_wrapped_kernel", mutates_args=())
def _mla_decode_wrapped_kernel(
    batch: int,
    heads: int,
    kv_head_num: int,
    seqlen_kv: int,
    dim: int,
    pe_dim: int,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    num_split: int,
    Q: torch.Tensor,
    Q_pe: torch.Tensor,
    Kv: torch.Tensor,
    K_pe: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor
) -> torch.Tensor:  
    return _mla_decode_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim)(
        block_H, block_N, num_split, num_stages, threads)(
            Q, Q_pe, Kv, K_pe, glse, Output_partial)


@_mla_decode_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    kv_head_num: int,
    seqlen_kv: int,
    dim: int,
    pe_dim: int,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    num_split: int,
    Q: torch.Tensor,
    Q_pe: torch.Tensor,
    Kv: torch.Tensor,
    K_pe: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor
) -> torch.Tensor:
    return torch.empty((batch, heads, dim), dtype=Q.dtype, device=Q.device)


class mla_decode_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 kv_head_num,
                 seqlen_kv,
                 dim,
                 pe_dim,
                 dtype="float16",
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim
        self.dtype = dtype

        self.kernel = _mla_decode_kernel(self.batch, self.heads, self.kv_head_num, self.seqlen_kv,
                                        self.dim, self.pe_dim)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_H": min(64, self.heads // self.kv_head_num),
            "block_N": 64,
            "num_split": 1,
            "num_stages": 2,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128]
        block_N = [32, 64, 128]
        num_split = [2, 4, 8]
        num_stages = [1, 2, 3]
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

    def forward(self,
                Q: torch.Tensor,
                Q_pe: torch.Tensor,
                K: torch.Tensor,
                K_pe: torch.Tensor):
        glse = torch.empty((self.batch, self.heads, self.num_split), dtype=self.dtype, device=Q.device)
        Output_partial = torch.empty((self.batch, self.heads, self.num_split, self.dim),
                                     dtype=self.dtype, device=Q.device)
        return _mla_decode_wrapped_kernel(
            self.batch, self.heads, self.kv_head_num, self.seqlen_kv,
            self.dim, self.pe_dim, self.config["block_H"],
            self.config["block_N"], self.config["num_stages"],
            self.config["threads"], self.config["num_split"],
            Q, Q_pe, K, K_pe, glse, Output_partial)
    
