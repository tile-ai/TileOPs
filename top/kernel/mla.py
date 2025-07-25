# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import torch.nn as nn
from einops import rearrange, einsum
from tilelang.utils.tensor import torch_assert_close


@tilelang.jit(out_idx=[6])
def _mla(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H, num_split):
    scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.macro
    def _mla_no_split(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(batch, heads // min(block_H, kv_group_num), threads=128) as (bx, by):
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
            for k in T.Pipelined(loop_range, num_stages=2):
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


class MLAKernel(nn.Module):

    def __init__(self,
                 batch,
                 heads,
                 kv_head_num,
                 seqlen_kv,
                 dim,
                 pe_dim,
                 block_N,
                 block_H,
                 num_split,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.kernel = _mla(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H,
                           num_split)
        self.profiler = self.kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
        self.intermediate_tensor = {}
        self.intermediate_tensor['glse'] = torch.empty(
            batch, heads, num_split, device=device, dtype=dtype)
        self.intermediate_tensor['output_partial'] = torch.empty(
            batch, heads, num_split, dim, device=device, dtype=dtype)

    def forward(self, q, q_pe, kv, k_pe):
        o = self.kernel(q, q_pe, kv, k_pe, self.intermediate_tensor['glse'],
                        self.intermediate_tensor['output_partial'])
        return o

    def profile(self, warmup=500):
        latency = self.profiler.do_bench(warmup=warmup)
        return latency

    def ref_program(self, q, q_pe, kv, k_pe):
        #     """
        #     Inputs:
        #     - q (Tensor): [batch, heads, dim]
        #     - q_pe (Tensor): [batch, heads, pe_dim]
        #     - kv (Tensor): [batch, seqlen_kv, kv_head_num, dim]
        #     - k_pe (Tensor): [batch, seqlen_kv, kv_head_num, pe_dim]
        #     Outputs:
        #     - output (Tensor): [batch, heads, dim]
        #     """
        dim = q.shape[-1]
        pe_dim = q_pe.shape[-1]
        num_head_groups = q.shape[1] // kv.shape[2]
        scale = (dim + pe_dim)**0.5
        q = rearrange(
            q, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

        q_pe = rearrange(
            q_pe, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, pe_dim]

        kv = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

        k_pe = rearrange(k_pe,
                         'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, pe_dim]

        query = torch.concat([q, q_pe], dim=-1)
        key = torch.concat([kv, k_pe], dim=-1)

        scores = einsum(
            query, key,
            'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

        attention = F.softmax(
            scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(attention, kv,
                     'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
        return out

    def check(self, q, q_pe, kv, k_pe):
        o = self.kernel(q, q_pe, kv, k_pe, self.intermediate_tensor['glse'],
                        self.intermediate_tensor['output_partial'])
        o_ref = self.ref_program(q, q_pe, kv, k_pe)
        torch_assert_close(o, o_ref, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.01)
        print("MLA kernel check passed!")
