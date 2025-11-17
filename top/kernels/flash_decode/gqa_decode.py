import torch
import tilelang
import tilelang.language as T
from top.kernels.kernel import Kernel
from typing import Optional
import itertools

__all__ = [
    "gqa_decode_kernel"
]


def _gqa_decode_kernel(batch, heads, groups, seqlen_kv, dim):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_decode_func(block_H, block_N, num_split, num_stages, threads):

        shape_q = [batch, heads, dim]
        shape_k = [batch, seqlen_kv, groups, dim]
        shape_v = [batch, seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        kv_group_num = heads // groups

        part_shape = [batch, heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)
        valid_block_N = min(block_N, seqlen_kv // num_split)

        @T.macro
        def _gqa_decode_no_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
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
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
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
        def _gqa_decode_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([valid_block_N, dim], dtype)
                V_shared = T.alloc_shared([valid_block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, valid_block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, valid_block_N], dtype)
                mask_local = T.alloc_fragment([valid_block_N], "uint8")
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

                loop_range = T.ceildiv((seqlen_kv // num_split), valid_block_N)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        K[bid, (seqlen_kv // num_split) * sid +
                        k * valid_block_N:(seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                        cur_kv_head, :], K_shared)
                    T.copy(
                        mask[bid, (seqlen_kv // num_split) * sid +
                            k * valid_block_N:(seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                            cur_kv_head], mask_local)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, valid_block_N):
                        acc_s[i,
                            j] = T.if_then_else((mask_local[j] != 0) & (j < seqlen_kv // num_split),
                                                acc_s[i, j], -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, valid_block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[bid, (seqlen_kv // num_split) * sid +
                        k * valid_block_N:(seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                        cur_kv_head, :], V_shared)
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
                # 1) 读 glse 到一个 1D fragment，再做 reduce_max
                glse_vec = T.alloc_fragment([num_split], dtype)
                for k in T.Parallel(num_split):
                    glse_vec[k] = glse[bz, by, k]
                lse_max = T.alloc_fragment([1], accum_dtype)
                T.fill(lse_max, -T.infinity(accum_dtype))
                T.reduce_max(glse_vec, lse_max, dim=0, clear=False)

                # 2) 计算 logsum（串行或小批并行累加）
                lse_logsum = T.alloc_local([1], accum_dtype)
                lse_logsum[0] = 0
                for k in T.serial(num_split):
                    lse_logsum[0] += T.exp2(glse[bz, by, k] - lse_max[0])
                lse_logsum[0] = T.log2(lse_logsum[0]) + lse_max[0]

                # 3) 按权重合并 partial 输出
                o_accum = T.alloc_fragment([dim], accum_dtype)
                T.clear(o_accum)
                for k in T.serial(num_split):
                    w = T.exp2(glse[bz, by, k] - lse_logsum[0])
                    for i in T.Parallel(dim):
                        o_accum[i] += Output_partial[bz, by, k, i] * w
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum[i]

        @T.prim_func
        def gqa_decode_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_split(Q, K, V, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def gqa_decode_no_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_no_split(Q, K, V, mask, Output)

        if num_split > 1:
            return gqa_decode_split
        else:
            return gqa_decode_no_split

    return _gqa_decode_func


@torch.library.custom_op("top::gqa_decode_wrapped_kernel", mutates_args=())
def _gqa_decode_wrapped_kernel(
    batch: int,
    heads: int,
    groups: int,
    seqlen_kv: int,
    dim: int,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    num_split: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor
) -> torch.Tensor:  
    return _gqa_decode_kernel(batch, heads, groups, seqlen_kv, dim)(
        block_H, block_N, num_split, num_stages, threads)(
            Q, K, V, mask, glse, Output_partial)


@_gqa_decode_wrapped_kernel.register_fake
def _(    
    batch: int,
    heads: int,
    groups: int,
    seqlen_kv: int,
    dim: int,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    num_split: int,
    *inputs
) -> torch.Tensor:
    return torch.empty_like(inputs[0])


class gqa_decode_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 groups,
                 seqlen_kv,
                 dim,
                 dtype="float16",
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.dtype = dtype

        self.kernel = _gqa_decode_kernel(self.batch, self.heads, self.groups, self.seqlen_kv,
                                        self.dim)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_H": 64,
            "block_N": 128,
            "num_split": 16,
            "num_stages": 2,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
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

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                mask: torch.Tensor):
        glse = torch.empty((self.batch, self.heads, self.config["num_split"]), dtype=self.dtype, device=Q.device)
        Output_partial = torch.empty((self.batch, self.heads, self.config["num_split"], self.dim),
                                     dtype=self.dtype, device=Q.device)
        return _gqa_decode_wrapped_kernel(
            self.batch, self.heads, self.groups, self.seqlen_kv,
            self.dim, self.config["block_H"],
            self.config["block_N"], self.config["num_stages"],
            self.config["threads"], self.config["num_split"],
            Q, K, V, mask, glse, Output_partial)

