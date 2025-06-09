import torch.nn.functional as F
import torch
import tilelang
import tilelang.language as T
import torch.nn as nn


@tilelang.jit(out_idx=[4, 5])
def blocksparse_flashattn_fwd(batch,
                              heads,
                              seq_len,
                              dim_qk,
                              dim_v,
                              is_causal,
                              block_M,
                              block_N,
                              groups=1):
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    downsample_q_len = (seq_len + block_M - 1) // block_M
    downsample_kv_len = (seq_len + block_N - 1) // block_N
    block_mask_shape = [batch, heads, downsample_q_len, downsample_kv_len]
    dtype = "float16"
    accum_dtype = "float"
    block_mask_dtype = "bool"

    @T.prim_func
    def flash_fwd(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(k_shape, dtype),  # type: ignore
            V: T.Tensor(v_shape, dtype),  # type: ignore
            BlockSparseMask: T.Tensor(block_mask_shape,
                                      block_mask_dtype),  # type: ignore
            Output: T.Tensor([batch, seq_len, heads, dim_v],
                             dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len],
                          accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch,
                      threads=256) as (bx, by, bz):
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
            block_mask = T.alloc_local([downsample_kv_len], block_mask_dtype)

            T.annotate_layout(
                {Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for vj in T.serial(downsample_kv_len):
                block_mask[vj] = BlockSparseMask[bz, by, bx, vj]

            loop_range = (T.ceildiv((bx + 1) * block_M, block_N)
                          if is_causal else T.ceildiv(seq_len, block_N))
            for k in T.Pipelined(loop_range, num_stages=1):
                if block_mask[k] != 0:
                    T.copy(
                        K[bz, k * block_N:(k + 1) * block_N, by // groups, :],
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
                    T.copy(
                        V[bz, k * block_N:(k + 1) * block_N, by // groups, :],
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

    return flash_fwd


@torch.compile
class _blocksparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_mask, block_M, block_N, causal, groups=1):
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        D_HEAD_V = v.shape[-1]
        assert k.shape == (BATCH, N_CTX, H // groups,
                           D_HEAD_QK), "k shape mismatch"
        assert v.shape == (BATCH, N_CTX, H // groups,
                           D_HEAD_V), "v shape mismatch"
        mod = blocksparse_flashattn_fwd(BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V,
                                        causal, block_M, block_N, groups)
        o, lse = mod(q, k, v, block_mask)
        ctx.save_for_backward(q, k, v, block_mask, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError(
            "Backward pass is not implemented for blocksparse attention.")


blocksparse_attention = _blocksparse_attention.apply


class BlockSparseAttention_kernel(nn.Module):

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
        self.attention = blocksparse_attention
        self.block_M = block_M
        self.block_N = block_N
        self.causal = causal
        self.groups = groups
        flops_per_qk = 2.0 * batch * heads * seq_len * seq_len * dim_qk
        flops_per_v = 2.0 * batch * heads * seq_len * seq_len * dim_v
        self.total_flops = 3 * flops_per_qk + 2 * flops_per_v
        if causal:
            self.total_flops *= 0.5

    def forward(self, q, k, v, block_mask):
        o = self.attention(q, k, v, block_mask, self.block_M, self.block_N,
                           self.causal, self.groups)
        return o

    def backward(self, do, q, k, v):
        raise NotImplementedError(
            "Backward pass is not implemented for BlockSparseAttention_kernel."
        )

    def ref_program(self, q, k, v, block_mask):
        B, T, HQ, D_QK = q.shape
        _, _, HK, _ = k.shape
        _, _, HV, D_V = v.shape
        block_M = self.block_M
        block_N = self.block_N
        causal = self.causal
        groups = self.groups

        assert HQ == HK * groups, f"Q heads {HQ} != K heads {HK} * groups {groups}"
        assert HQ == HV * groups, f"Q heads {HQ} != V heads {HV} * groups {groups}"

        k_expanded = k.repeat_interleave(groups, dim=2)  # [B, T, HQ, D_QK]
        v_expanded = v.repeat_interleave(groups, dim=2)  # [B, T, HQ, D_V]

        attn_scores = torch.einsum('bthd,bshd->bhts', q,
                                   k_expanded)  # [B, H, T, T]
        attn_scores = attn_scores / torch.sqrt(
            torch.tensor(
                D_QK, dtype=attn_scores.dtype, device=attn_scores.device))

        # Apply causal mask if needed
        if causal:
            causal_mask = torch.tril(torch.ones(T, T, device=q.device)).view(
                1, 1, T, T)
            attn_scores = attn_scores.masked_fill(causal_mask == 0,
                                                  float('-inf'))

        # Apply block_mask (simulate sparse attention pattern)
        if block_mask is not None:
            mask_exp = block_mask.repeat_interleave(
                block_M, dim=2).repeat_interleave(block_N, dim=3)
            mask_exp = mask_exp[:, :, :T, :T]  # [B, H, T, T]
            attn_scores = attn_scores.masked_fill(~mask_exp.bool(),
                                                  float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum('bhts,bshd->bthd', attn_weights,
                              v_expanded)  # [B, T, HQ, D_V]

        return output

    def check(self, q, k, v, block_mask):
        o = self.attention(q, k, v, block_mask, self.block_M, self.block_N,
                           self.causal, self.groups)

        o_ref = self.ref_program(q, k, v, block_mask)

        assert torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2)
        print("BlockSparseAttention kernel check passed!")
