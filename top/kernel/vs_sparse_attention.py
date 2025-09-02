from typing import Callable
import torch
import tilelang
import tilelang.language as T
import math
import triton
import triton.language as tl

from .base import KernelBase


@tilelang.jit(out_idx=[3])
def _tl_vs_sparse_flashattn(batch, heads, seq_len, dim, vertical_size, slash_size, block_M, block_N, dtype):
    num_stages = 2
    threads = 128
    scale = (1.0 / dim)**0.5 * 1.44269504
    shape = [batch, heads, seq_len, dim]

    count_shape = [batch, heads, (seq_len + block_M - 1) // block_M]

    offset_shape = count_shape + [slash_size]
    index_shape = count_shape + [vertical_size]

    vertical_size_round, slash_size_round = tilelang.next_power_of_2(
        vertical_size), tilelang.next_power_of_2(slash_size)

    accum_dtype = "float"
    int_dtype = "int32"

    def kernel_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def vs_sparse_flashattn(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
                BlockCount: T.Tensor(count_shape, int_dtype),
                BlockOffset: T.Tensor(offset_shape, int_dtype),
                ColumnCount: T.Tensor(count_shape, int_dtype),
                ColumnIndex: T.Tensor(index_shape, int_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bc, by, bz):

                bx = T.ceildiv(seq_len, block_M) - 1 - bc
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
                block_count = T.alloc_local([1], int_dtype)
                block_offset = T.alloc_shared([slash_size_round], int_dtype, scope="shared")
                column_count = T.alloc_local([1], int_dtype)
                column_index = T.alloc_shared([vertical_size_round], int_dtype, scope="shared")

                K_shared_1 = T.alloc_shared([block_N, dim], dtype)
                V_shared_1 = T.alloc_shared([block_N, dim], dtype)
                K_shared_2 = T.alloc_shared([block_N, dim], dtype)
                V_shared_2 = T.alloc_shared([block_N, dim], dtype)

                block_count[0] = BlockCount[bz, by, bx]
                column_count[0] = ColumnCount[bz, by, bx]

                for vi in T.Parallel(slash_size_round):
                    if vi < slash_size:
                        block_offset[vi] = BlockOffset[bz, by, bx, vi]

                for vi in T.Parallel(vertical_size_round):
                    if vi < vertical_size:
                        column_index[vi] = ColumnIndex[bz, by, bx, vi]

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)

                for bi in T.Pipelined(block_count[0], num_stages=num_stages):
                    k = block_offset[bi]
                    T.copy(K[bz, by, k:k + block_N, :], K_shared)

                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k + j, 0,
                                                     -T.infinity(acc_s.dtype))

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)

                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = acc_o[i, j] * scores_scale[i]

                    T.copy(acc_s, acc_s_cast)
                    T.copy(V[bz, by, k:k + block_N, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                if column_count[0] != 0:
                    with T.attr("default", "async_scope", 1):
                        for i, j in T.Parallel(block_N, dim):
                            K_shared_1[i, j] = T.if_then_else(0 + i < column_count[0],
                                                    K[bz, by, column_index[0 + i], j], 0)
                    with T.attr("default", "async_scope", 1):
                        for i, j in T.Parallel(block_N, dim):
                            V_shared_1[i, j] = T.if_then_else(0 + i < column_count[0],
                                                    V[bz, by, column_index[0 + i], j], 0)
                    T.ptx_commit_group()

                    for bi in T.serial(T.ceildiv(column_count[0], block_N) - 1):
                        k = bi * block_N
                        if bi % 2 == 0:
                            with T.attr("default", "async_scope", 1):
                                for i, j in T.Parallel(block_N, dim):
                                    K_shared_2[i, j] = T.if_then_else(k + block_N + i < column_count[0],
                                                            K[bz, by, column_index[k + block_N + i], j], 0)
                            with T.attr("default", "async_scope", 1):
                                for i, j in T.Parallel(block_N, dim):
                                    V_shared_2[i, j] = T.if_then_else(k + block_N + i < column_count[0],
                                                            V[bz, by, column_index[k + block_N + i], j], 0)
                            T.ptx_commit_group()
                            T.ptx_wait_group(1)
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.if_then_else(k + j < column_count[0], 0,
                                                             -T.infinity(acc_s.dtype))
                            T.gemm(Q_shared, K_shared_1, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(scores_max, scores_max_prev)
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Parallel(block_M):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_M, dim):
                                acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                            T.copy(acc_s, acc_s_cast)
                            T.gemm(acc_s_cast, V_shared_1, acc_o, policy=T.GemmWarpPolicy.FullRow)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Parallel(block_M):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        else:
                            with T.attr("default", "async_scope", 1):
                                for i, j in T.Parallel(block_N, dim):
                                    K_shared_1[i, j] = T.if_then_else(k + block_N + i < column_count[0],
                                                            K[bz, by, column_index[k + block_N + i], j], 0)
                            with T.attr("default", "async_scope", 1):
                                for i, j in T.Parallel(block_N, dim):
                                    V_shared_1[i, j] = T.if_then_else(k + block_N + i < column_count[0],
                                                            V[bz, by, column_index[k + block_N + i], j], 0)
                            T.ptx_commit_group()
                            T.ptx_wait_group(1)
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.if_then_else(k + j < column_count[0], 0,
                                                             -T.infinity(acc_s.dtype))
                            T.gemm(Q_shared, K_shared_2, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(scores_max, scores_max_prev)
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Parallel(block_M):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_M, dim):
                                acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                            T.copy(acc_s, acc_s_cast)
                            T.gemm(acc_s_cast, V_shared_2, acc_o, policy=T.GemmWarpPolicy.FullRow)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Parallel(block_M):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    if T.ceildiv(column_count[0], block_N) % 2 == 0:
                        T.ptx_wait_group(0)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(T.ceildiv(column_count[0], block_N) * block_N - block_N + j < column_count[0], 0,
                                                         -T.infinity(acc_s.dtype))
                        T.gemm(Q_shared, K_shared_2, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                        T.copy(acc_s, acc_s_cast)
                        T.gemm(acc_s_cast, V_shared_2, acc_o, policy=T.GemmWarpPolicy.FullRow)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    else:
                        T.ptx_wait_group(0)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(T.ceildiv(column_count[0], block_N) * block_N - block_N + j < column_count[0], 0,
                                                         -T.infinity(acc_s.dtype))
                        T.gemm(Q_shared, K_shared_1, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                        T.copy(acc_s, acc_s_cast)
                        T.gemm(acc_s_cast, V_shared_1, acc_o, policy=T.GemmWarpPolicy.FullRow)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

        return vs_sparse_flashattn

    return kernel_func(block_M, block_N, num_stages, threads)


@torch.compile
class _vertical_slash_sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_count, block_offset, column_count, column_index, dtype, block_M, block_N):
        BATCH, HEADS, SEQ_LEN, DIM = q.shape
        vertical_size = column_index.shape[-1]
        slash_size = block_offset.shape[-1]
        mod = _tl_vs_sparse_flashattn(BATCH, HEADS, SEQ_LEN, DIM, vertical_size, slash_size, block_M, block_N, dtype)
        o = mod(q, k, v, block_count, block_offset, column_count, column_index)
        return o

    @staticmethod
    def backward(ctx, do):
        pass

vertical_slash_sparse_attention = _vertical_slash_sparse_attention.apply


class VerticalSlashSparseAttentionKernel(KernelBase):
    map_dtype = {torch.float16: "float16", torch.bfloat16: "bfloat16"}

    def __init__(self,
                 batch_size: int,
                 num_heads: int,
                 seq_len: int,
                 head_dim: int,
                 vertical_size: int = 1000,
                 slash_size: int = 200,
                 block_M: int = 64,
                 block_N: int = 64,
                 dtype = torch.float16,
                 device="cuda"):

        
        self.attention = vertical_slash_sparse_attention

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim

        self.vertical_size = vertical_size
        self.slash_size = slash_size

        self.block_M = block_M
        self.block_N = block_N

        assert dtype in self.map_dtype.keys(), f"dtype must be one of {self.map_dtype.keys()}"

        self.dtype = self.map_dtype[dtype]
        self.torch_dtype = dtype

        self._q_shape = [self.batch_size, self.seq_len, self.num_heads, self.head_dim]
        self._k_shape = [self.batch_size, self.seq_len, self.num_heads, self.head_dim]
        self._v_shape = [self.batch_size, self.seq_len, self.num_heads, self.head_dim]

        self._block_count_shape = [self.batch_size, self.num_heads, (self.seq_len + self.block_M - 1) // self.block_M]
        self._block_offset_shape = self._block_count_shape + [self.slash_size]
        self._column_count_shape = self._block_count_shape
        self._column_index_shape = self._block_count_shape + [self.vertical_size]

        assert device == "cuda", "device must be cuda"
        self.device = device

    def forward(self, *args, **kwargs):
        block_count = kwargs.get("block_count")
        block_offset = kwargs.get("block_offset")
        column_count = kwargs.get("column_count")
        column_index = kwargs.get("column_index")
        o = self.attention(*args, block_count, block_offset, column_count, column_index, self.dtype, self.block_M, self.block_N)
        return o

    # TODO: Need pytorch-based reference implementation
    @property
    def ref_program(self) -> Callable:
        seqlens = torch.tensor([self.seq_len], dtype=torch.int32).to(self.device)

        sm_scale = self.head_dim**-0.5

        @triton.jit
        def _triton_mixed_sparse_attn_fwd_kernel(
            Q,
            K,
            V,
            seqlens,
            sm_scale,
            block_count,
            block_offset,
            column_count,
            column_index,
            Out,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_oz,
            stride_oh,
            stride_om,
            stride_ok,
            Z,
            H,
            N_CTX,
            NUM_ROWS,
            NNZ_S,
            NNZ_V,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_DMODEL: tl.constexpr,
            dtype: tl.constexpr,
        ):
            start_m = tl.program_id(0)  # bx
            off_hz = tl.program_id(1)  # by

            seqlen = tl.load(seqlens + off_hz // H)
            if start_m * BLOCK_M >= seqlen:
                return

            # initialize offsets
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, BLOCK_DMODEL)

            qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
            kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

            q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
            k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
            v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
            o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

            num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
            blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
            num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
            cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
            # scale sm_scale by log_2(e) and use
            # 2^x instead of exp in the loop because CSE and LICM
            # don't work as expected with `exp` in the loop
            qk_scale = sm_scale * 1.44269504
            # load q: it will stay in SRAM throughout
            q = tl.load(q_ptrs)
            q = (q * qk_scale).to(dtype)

            # loop over k, v and update accumulator
            m_mask = offs_m[:, None] < seqlen

            for block_index in range(num_blks):
                start_n = tl.load(blks_ptr + block_index)
                cols = start_n + offs_n
                n_mask = cols < seqlen
                # -- load k, v --
                k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
                v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
                # -- compute qk --
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                causal_mask = cols[None, :] <= offs_m[:, None]
                qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
                qk += tl.dot(q, k)
                # -- compute scaling constant --
                m_i_new = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.math.exp2(m_i - m_i_new)
                p = tl.math.exp2(qk - m_i_new[:, None])
                # -- scale and update acc --
                acc_scale = l_i * 0 + alpha  # workaround some compiler bug
                acc *= acc_scale[:, None]
                acc += tl.dot(p.to(dtype), v)
                # -- update m_i and l_i --
                l_i = l_i * alpha + tl.sum(p, 1)
                m_i = m_i_new

            for start_n in range(0, num_cols, BLOCK_N):  #
                # bi * BLOCK_N: bi * BLOCK_N + BLOCK_N
                n_mask = start_n + offs_n < num_cols
                cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
                # -- load k, v --
                k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
                v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
                # -- compute qk --
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                qk = tl.where(m_mask & n_mask, qk, float("-inf"))
                qk += tl.dot(q, k)
                # -- compute scaling constant --
                m_i_new = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.math.exp2(m_i - m_i_new)
                p = tl.math.exp2(qk - m_i_new[:, None])
                # -- scale and update acc --
                acc_scale = l_i * 0 + alpha  # workaround some compiler bug
                acc *= acc_scale[:, None]
                acc += tl.dot(p.to(dtype), v)
                # -- update m_i and l_i --
                l_i = l_i * alpha + tl.sum(p, 1)
                m_i = m_i_new

            # write back O
            acc /= l_i[:, None]
            # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
            tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


        def _triton_mixed_sparse_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            block_count: torch.Tensor,
            block_offset: torch.Tensor,
            column_count: torch.Tensor,
            column_index: torch.Tensor,
            block_size_M: int = 64,
            block_size_N: int = 64,
        ) -> torch.Tensor:
            # shape constraints
            Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
            assert Lq == Lk and Lk == Lv
            assert Lk in {16, 32, 64, 128}
            o = torch.zeros_like(q)
            grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
            dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
            _triton_mixed_sparse_attn_fwd_kernel[grid](
                q,
                k,
                v,
                seqlens,
                sm_scale,
                block_count,
                block_offset,
                column_count,
                column_index,
                o,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                q.shape[0],
                q.shape[1],
                q.shape[2],
                block_count.shape[-1],
                block_offset.shape[-1],
                column_index.shape[-1],
                BLOCK_M=block_size_M,
                BLOCK_N=block_size_N,
                BLOCK_DMODEL=Lk,
                dtype=dtype,
                num_warps=4,
                num_stages=2,
            )

            return o
        return _triton_mixed_sparse_attention


    # Note: The flop compuatation need to know the run time values of block_count and column_count
    def get_flops(self, *args, **kwargs) -> float:        
        block_count = kwargs.get("block_count")
        column_count = kwargs.get("column_count")
        
        if block_count is None or column_count is None:
            raise ValueError("block_count and column_count must be provided in kwargs")
        block_flops = block_count.sum().item() * self.block_M * self.block_N * 2 * 2 * self.head_dim

        column_flops = column_count.sum().item() * self.head_dim * 2 * 2 * (self.seq_len / self.block_M)

        flops = block_flops + column_flops
        return flops

    # we also need to know the run-time values of block_count and column_count
    def get_memory_footprint(self, *args, **kwargs):
        block_count = kwargs.get("block_count")
        column_count = kwargs.get("column_count")
        
        if block_count is None or column_count is None:
            raise ValueError("block_count and column_count must be provided in kwargs")

        memory_footprint = 0

        memory_footprint += math.prod(self._q_shape) * self.torch_dtype.itemsize

        # read all block_count and column_count tensors
        memory_footprint += (math.prod(self._block_count_shape) + math.prod(self._column_count_shape)) * torch.int32.itemsize

        # read all block_offset and column_index tensors
        memory_footprint += (math.prod(self._block_offset_shape) + math.prod(self._column_index_shape)) * torch.int32.itemsize

        # 1st iteration
        memory_footprint += block_count.sum().item() * self.block_N * self.head_dim * 2 * self.torch_dtype.itemsize

        # 2nd iteration
        memory_footprint += column_count.sum().item() * self.head_dim * 2 * self.torch_dtype.itemsize

        # write 
        memory_footprint += math.prod(self._q_shape) * self.torch_dtype.itemsize

        return memory_footprint
