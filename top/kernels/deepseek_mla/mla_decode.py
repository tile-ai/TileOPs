import torch
import tilelang
import tilelang.language as T
from top.kernels.kernel import Kernel
from typing import Optional
import itertools

__all__ = ["mla_decode_kernel", "mla_decode_ws_kernel"]


def _mla_decode_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype='float16'):
    scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    assert kv_head_num == 1, "kv_head_num must be 1"

    @tilelang.jit(out_idx=[6], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def mla_decode_func(block_H, block_N, num_split, num_stages, threads=128):
        
        VALID_BLOCK_H = min(block_H, kv_group_num)

        @T.macro
        def mla_no_split(
                Q: T.Tensor([batch, heads, dim], dtype),
                Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(batch, heads // VALID_BLOCK_H, threads=threads) as (bx, by):
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
                for k in T.Pipelined(loop_range, num_stages=num_stages):
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
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
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
        def mla_split(
                Q: T.Tensor([batch, heads, dim], dtype),
                Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        ):
            with T.Kernel(
                    batch, heads // min(block_H, kv_group_num), num_split, threads=threads) as (bx, by, bz):
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
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    kv_start = (seqlen_kv // num_split) * bz + k * block_N
                    kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                    T.copy(KV[bx, kv_start:kv_end, cur_kv_head, :], KV_shared)
                    T.copy(K_pe[bx, kv_start:kv_end, cur_kv_head, :], K_pe_shared)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(kv_start + j >= seqlen_kv, 
                                                -T.infinity(acc_s.dtype), 0)
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
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
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
            with T.Kernel(heads, batch, threads=threads) as (by, bz):
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
            mla_split(Q, Q_pe, KV, K_pe, glse, Output_partial)
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
            mla_no_split(Q, Q_pe, KV, K_pe, Output)

        if num_split > 1:
            return main_split
        else:
            return main_no_split
        
    return mla_decode_func


@torch.library.custom_op("top::mla_decode_wrapped_kernel", mutates_args=())
def _mla_decode_wrapped_kernel(
    batch: int,
    heads: int,
    kv_head_num: int,
    seqlen_kv: int,
    dim: int,
    pe_dim: int,
    dtype: str,
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
    return _mla_decode_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype)(
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
    dtype: str,
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
                 dtype,
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
                                        self.dim, self.pe_dim, self.dtype_str)

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
        block_H = [64, 128]
        block_N = [64, 128]
        num_split = [1, 2, 4, 8]
        num_stages = [2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_H, block_N, num_split, num_stages, threads))

        configs = [{
            'block_H': c[0],
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
        glse = torch.empty((self.batch, self.heads, self.config["num_split"]), dtype=self.dtype, device=Q.device)
        Output_partial = torch.empty((self.batch, self.heads, self.config["num_split"], self.dim),
                                     dtype=self.dtype, device=Q.device)
        return _mla_decode_wrapped_kernel(
            self.batch, self.heads, self.kv_head_num, self.seqlen_kv,
            self.dim, self.pe_dim, self.dtype_str, self.config["block_H"],
            self.config["block_N"], self.config["num_stages"],
            self.config["threads"], self.config["num_split"],
            Q, Q_pe, K, K_pe, glse, Output_partial)
    


def _mla_decode_ws_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype='float16'):
    sm_scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    assert kv_head_num == 1, "kv_head_num must be 1"

    @tilelang.jit(
        out_idx=[6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=[
            "-O3", "-Wno-deprecated-declarations", "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "--ptxas-options=-v,--register-usage-level=10", "-DNDEBUG"
        ],
    )
    def mla_decode_ws_func(block_H, block_N, num_split, num_stages, threads=128):

        VALID_BLOCK_H = min(block_H, kv_group_num)
        @T.macro
        def flash_attn(
                Q: T.Tensor([batch, heads, dim], dtype),
                Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(heads // min(block_H, kv_group_num), batch, threads=threads) as (hid, bid):
                Q_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
                Q_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
                Q_tail_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared_0_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_0_r = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_r = T.alloc_shared([block_N, dim // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
                K_tail_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r

                acc_o_l = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                sumexp = T.alloc_fragment([block_H], accum_dtype)
                sum_exp_shared = T.alloc_shared([block_H], accum_dtype)
                sumexp_i = T.alloc_fragment([block_H], accum_dtype)
                alpha_shared = T.alloc_shared([block_H], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([block_H], accum_dtype)
                m_i = T.alloc_fragment([block_H], accum_dtype)
                m_i_prev = T.alloc_fragment([block_H], accum_dtype)

                # TODO: Multi buffer
                bar_q = T.alloc_barrier(arrive_count=384)
                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                cur_kv_head = hid // (kv_group_num // block_H)
                NI = T.ceildiv((seqlen_kv // num_split), block_N)

                tx = T.get_thread_binding()

                T.copy(Q[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, 0:dim // 2], Q_shared_l)
                T.copy(Q[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, dim // 2:dim], Q_shared_r)
                T.copy(Q_pe[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_tail_shared)

                T.barrier_arrive(bar_q)

                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.fill(sumexp, 0)
                    T.fill(m_i, -2**30)  # avoid -inf - inf to cause nan
                    T.fill(acc_o_l, 0)
                    T.barrier_wait(bar_q, 0)

                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                        T.clear(acc_s)
                        T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)

                        T.wait_wgmma(0)

                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        # Buffer 1
                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        T.clear(acc_s)
                        T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=-1)

                        T.wait_wgmma(0)

                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    # Rescale
                    for h_i in T.Parallel(block_H):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(O_shared_l, Output[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H,
                                            0:dim // 2])

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        # Buffer 1
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(NI, 2) - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    # Rescale
                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(O_shared_r, Output[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H,
                                            dim // 2:dim])

                elif tx >= 256:
                    # producer
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            kv_indices = (i_i * 2) * block_N + r * 16 + (tx - 256) // 8
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        KV_shared_0_l[r * 16 + (tx - 256) // 8,
                                                    64 * u + (tx - 256) % 8 * 8 +
                                                    v] = KV[bid, kv_indices, cur_kv_head,
                                                            64 * u + (tx - 256) % 8 * 8 + v]
                                        KV_shared_0_r[r * 16 + (tx - 256) // 8,
                                                    64 * u + (tx - 256) % 8 * 8 +
                                                    v] = KV[bid, kv_indices, cur_kv_head, dim // 2 +
                                                            64 * u + (tx - 256) % 8 * 8 + v]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_0[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 +
                                                    v] = K_pe[bid, kv_indices, cur_kv_head,
                                                            (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        # Buffer 1
                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            kv_indices = (i_i * 2 + 1) * block_N + r * 16 + (tx - 256) // 8
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        KV_shared_1_l[r * 16 + (tx - 256) // 8,
                                                    64 * u + (tx - 256) % 8 * 8 +
                                                    v] = KV[bid, kv_indices, cur_kv_head,
                                                            64 * u + (tx - 256) % 8 * 8 + v]
                                        KV_shared_1_r[r * 16 + (tx - 256) // 8,
                                                    64 * u + (tx - 256) % 8 * 8 +
                                                    v] = KV[bid, kv_indices, cur_kv_head, dim // 2 +
                                                            64 * u + (tx - 256) % 8 * 8 + v]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_1[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 +
                                                    v] = K_pe[bid, kv_indices, cur_kv_head,
                                                            (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        @T.macro
        def flash_attn_split(
                Q: T.Tensor([batch, heads, dim], dtype),
                Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        ):
            with T.Kernel(
                    batch, heads // min(block_H, kv_group_num), num_split,
                    threads=threads) as (bid, hid, bz):
                Q_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
                Q_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
                Q_tail_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared_0_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_0_r = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_r = T.alloc_shared([block_N, dim // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
                K_tail_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r

                acc_o_l = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                sumexp = T.alloc_fragment([block_H], accum_dtype)
                sum_exp_shared = T.alloc_shared([block_H], accum_dtype)
                sumexp_i = T.alloc_fragment([block_H], accum_dtype)
                alpha_shared = T.alloc_shared([block_H], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([block_H], accum_dtype)
                m_i = T.alloc_fragment([block_H], accum_dtype)
                m_i_prev = T.alloc_fragment([block_H], accum_dtype)

                # TODO: Multi buffer
                bar_q = T.alloc_barrier(arrive_count=384)
                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                cur_kv_head = hid // (kv_group_num // block_H)
                NI = T.ceildiv((seqlen_kv // num_split), block_N)

                tx = T.get_thread_binding()

                T.copy(Q[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, 0:dim // 2], Q_shared_l)
                T.copy(Q[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, dim // 2:dim], Q_shared_r)
                T.copy(Q_pe[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_tail_shared)

                T.barrier_arrive(bar_q)

                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.fill(sumexp, 0)
                    T.fill(m_i, -2**30)  # avoid -inf - inf to cause nan
                    T.fill(acc_o_l, 0)
                    T.barrier_wait(bar_q, 0)

                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                        T.clear(acc_s)
                        T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)

                        T.wait_wgmma(0)

                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        # Buffer 1
                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        T.clear(acc_s)
                        T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=-1)

                        T.wait_wgmma(0)

                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    # Rescale
                    for h_i in T.Parallel(block_H):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(
                        O_shared_l, Output_partial[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H,
                                                bz, 0:dim // 2])
                    T.copy(sumexp, glse[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, bz])

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        # Buffer 1
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(NI, 2) - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    # Rescale
                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(
                        O_shared_r, Output_partial[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H,
                                                bz, dim // 2:dim])

                elif tx >= 256:
                    # producer
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            kv_indices = (seqlen_kv // num_split) * bz + (
                                i_i * 2) * block_N + r * 16 + (tx - 256) // 8
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        KV_shared_0_l[r * 16 + (tx - 256) // 8,
                                                    64 * u + (tx - 256) % 8 * 8 +
                                                    v] = KV[bid, kv_indices, cur_kv_head,
                                                            64 * u + (tx - 256) % 8 * 8 + v]
                                        KV_shared_0_r[r * 16 + (tx - 256) // 8,
                                                    64 * u + (tx - 256) % 8 * 8 +
                                                    v] = KV[bid, kv_indices, cur_kv_head, dim // 2 +
                                                            64 * u + (tx - 256) % 8 * 8 + v]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_0[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 +
                                                    v] = K_pe[bid, kv_indices, cur_kv_head,
                                                            (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        # Buffer 1
                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            kv_indices = (seqlen_kv // num_split) * bz + (
                                i_i * 2 + 1) * block_N + r * 16 + (tx - 256) // 8
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        KV_shared_1_l[r * 16 + (tx - 256) // 8,
                                                    64 * u + (tx - 256) % 8 * 8 +
                                                    v] = KV[bid, kv_indices, cur_kv_head,
                                                            64 * u + (tx - 256) % 8 * 8 + v]
                                        KV_shared_1_r[r * 16 + (tx - 256) // 8,
                                                    64 * u + (tx - 256) % 8 * 8 +
                                                    v] = KV[bid, kv_indices, cur_kv_head, dim // 2 +
                                                            64 * u + (tx - 256) % 8 * 8 + v]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_1[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 +
                                                    v] = K_pe[bid, kv_indices, cur_kv_head,
                                                            (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (hid, bz):
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
                    lse_max_local[0] = T.max(lse_max_local[0], glse[bz, hid, k])
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, hid, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, hid, k, i]
                    lse_local_split[0] = glse[bz, hid, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, hid, i] = o_accum_local[i]

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
            flash_attn_split(Q, Q_pe, KV, K_pe, glse, Output_partial)
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
            flash_attn(Q, Q_pe, KV, K_pe, Output)

        if num_split > 1:
            return main_split
        else:
            return main_no_split
        
    return mla_decode_ws_func


@torch.library.custom_op("top::mla_decode_ws_wrapped_kernel", mutates_args=())
def _mla_decode_ws_wrapped_kernel(
    batch: int,
    heads: int,
    kv_head_num: int,
    seqlen_kv: int,
    dim: int,
    pe_dim: int,
    dtype: str,
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
    return _mla_decode_ws_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype)(
        block_H, block_N, num_split, num_stages, threads)(
            Q, Q_pe, Kv, K_pe, glse, Output_partial)


@_mla_decode_ws_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    kv_head_num: int,
    seqlen_kv: int,
    dim: int,
    pe_dim: int,
    dtype: str,
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

class mla_decode_ws_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch,
                 heads,
                 kv_head_num,
                 seqlen_kv,
                 dim,
                 pe_dim,
                 dtype,
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

        self.kernel = _mla_decode_ws_kernel(self.batch, self.heads, self.kv_head_num, self.seqlen_kv,
                                        self.dim, self.pe_dim, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_H": min(64, self.heads // self.kv_head_num),
            "block_N": 64,
            "num_split": 2,
            "num_stages": 1,
            "threads": 384
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_H = [64, 128]
        block_N = [64, 128]
        num_split = [1, 2, 4]
        num_stages = [1, 2, 3]
        threads = [384]
        _configs = list(itertools.product(block_H, block_N, num_split, num_stages, threads))

        configs = [{
            'block_H': c[0],
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
        glse = torch.empty((self.batch, self.heads, self.config["num_split"]), dtype=self.dtype, device=Q.device)
        Output_partial = torch.empty((self.batch, self.heads, self.config["num_split"], self.dim),
                                     dtype=self.dtype, device=Q.device)
        return _mla_decode_ws_wrapped_kernel(
            self.batch, self.heads, self.kv_head_num, self.seqlen_kv,
            self.dim, self.pe_dim, self.dtype_str, self.config["block_H"],
            self.config["block_N"], self.config["num_stages"],
            self.config["threads"], self.config["num_split"],
            Q, Q_pe, K, K_pe, glse, Output_partial)
    