import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from top.kernels.kernel import Kernel
from typing import Optional
import itertools

__all__ = ["sparse_mla_kernel"]


def _sparse_mla_kernel(batch,
                       seq_len,
                       seq_len_kv,
                       heads,
                       dim,
                       tail_dim,
                       topk,
                       kv_stride,
                       q_start_index_s,
                       kv_group=1,
                       sm_scale=None,
                       is_causal=True,
                       CP0=True,
                       dtype="float16"):
    '''
    This code implements sparse attn
    Note that the first kv_stride - 1 token's out would be nan. since this isn't used, we assume it doesn't matter. (**still, one might have to handle carefully in backward to avoid 'dout * nan' propagated!**)
    It might be OK to set these nan to zero, but we assume it might serve as a reminder of taking care of these out in 'delta = out * dout'.
    The above feature might be replaced with out being undefined if we fix CP0 logic (this logic is currently wrong due to some bug in compiler)
    '''
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal, 'non-causal is not supported'
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // kv_group
    ori_heads = heads
    indices_dtype = "int32"
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        compile_flags=[
            "--use_fast_math", "-O3", "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "--ptxas-options=-v,--register-usage-level=10", "-DNDEBUG"
        ],
    )
    def _sparse_mla_fwd_func(block_I, threads):

        q_shape = [batch, seq_len, ori_heads, dim + tail_dim]
        kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
        o_shape = [batch, seq_len, ori_heads, dim]
        indices_shape = [batch, seq_len, kv_group, topk]

        heads = head_kv
        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != heads:
            assert kv_group == 1, 'here we solve the heads padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)'

        assert topk % block_I == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        assert NI % 2 == 0, 'NI should be a multiple of 2'
        D = dim
        D_tail = tail_dim
        KV_stride = kv_stride

        if head_kv > 64:
            assert head_kv % 64 == 0, 'head_kv should be a multiple of 64'
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1

        H_per_block = padded_H if REPLICATE_H == 1 else 64

        @T.prim_func
        def _sparse_mla_fwd_main(
                Q: T.Tensor(q_shape, dtype),  # type: ignore
                KV: T.Tensor(kv_shape, dtype),  # type: ignore
                Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
                Output: T.Tensor(o_shape, dtype),  # type: ignore
        ):
            with T.Kernel(
                (seq_len - kv_stride + 1 if CP0 else seq_len) * REPLICATE_H,
                    batch,
                    kv_group,
                    threads=threads) as (bx, by, bz):
                Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
                Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
                K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r
                is_kv_valid = T.alloc_shared([BI], "bool", scope="shared")

                acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                indices_local = T.alloc_local([1], indices_dtype)

                # TODO: Multi buffer
                bar_q = T.alloc_barrier(arrive_count=384)
                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                b_i, g_i = by, bz
                s_i = (bx + (KV_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (
                    bx // REPLICATE_H + (KV_stride - 1 if CP0 else 0))
                q_i = q_start_index_s + s_i
                max_kv_i = (q_i + 1 - KV_stride) // KV_stride

                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                H1 = H0 + H_per_block

                tx = T.get_thread_binding()

                T.copy(Q[b_i, s_i, H0:H1, 0:D // 2], Q_shared_l)
                T.copy(Q[b_i, s_i, H0:H1, D // 2:D], Q_shared_r)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
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

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0,
                                                              -T.infinity(acc_s.dtype))
                        T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)

                        T.wait_wgmma(0)

                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i,
                                  bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        # Buffer 1
                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0,
                                                              -T.infinity(acc_s.dtype))
                        T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=-1)

                        T.wait_wgmma(0)

                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i,
                                  bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    # Rescale
                    for h_i in T.Parallel(H_per_block):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0:D // 2])

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        # Buffer 1
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(NI, 2) - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    # Rescale
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2:D])
                elif tx >= 256:
                    # producer
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            indices_local[0] = Indices[b_i, s_i, g_i,
                                                       (i_i * 2) * BI + r * 16 + (tx - 256) // 8]
                            is_kv_valid[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                            if is_kv_valid[r * 16 + (tx - 256) // 8]:
                                with T.attr("default", "async_scope", 1):
                                    for u in T.serial(4):
                                        for v in T.vectorized(8):
                                            KV_shared_0_l[r * 16 + (tx - 256) // 8,
                                                          64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, indices_local[0], g_i,
                                                                  64 * u + (tx - 256) % 8 * 8 + v]
                                            KV_shared_0_r[r * 16 + (tx - 256) // 8,
                                                          64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, indices_local[0], g_i,
                                                                  D // 2 + 64 * u +
                                                                  (tx - 256) % 8 * 8 + v]
                                with T.attr("default", "async_scope", 1):
                                    for v in T.vectorized(8):
                                        K_tail_shared_0[r * 16 + (tx - 256) // 8,
                                                        (tx - 256) % 8 * 8 +
                                                        v] = KV[b_i, indices_local[0], g_i,
                                                                D + (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        # Buffer 1
                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2 + 1) * BI + r * 16 +
                                                       (tx - 256) // 8]
                            is_kv_valid[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                            if is_kv_valid[r * 16 + (tx - 256) // 8]:
                                with T.attr("default", "async_scope", 1):
                                    for u in T.serial(4):
                                        for v in T.vectorized(8):
                                            KV_shared_1_l[r * 16 + (tx - 256) // 8,
                                                          64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, indices_local[0], g_i,
                                                                  64 * u + (tx - 256) % 8 * 8 + v]
                                            KV_shared_1_r[r * 16 + (tx - 256) // 8,
                                                          64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, indices_local[0], g_i,
                                                                  D // 2 + 64 * u +
                                                                  (tx - 256) % 8 * 8 + v]
                                with T.attr("default", "async_scope", 1):
                                    for v in T.vectorized(8):
                                        K_tail_shared_1[r * 16 + (tx - 256) // 8,
                                                        (tx - 256) % 8 * 8 +
                                                        v] = KV[b_i, indices_local[0], g_i,
                                                                D + (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        return _sparse_mla_fwd_main

    return _sparse_mla_fwd_func


@torch.library.custom_op("top::sparse_mla_fwd_wrapped_kernel", mutates_args=())
def _sparse_mla_wrapped_kernel(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_stride: int,
    q_start_index_s: int,
    kv_group: int,
    sm_scale: Optional[float],
    is_causal: bool,
    CP0: bool,
    dtype: str,
    block_I: int,
    threads: int,
    Q: torch.Tensor,
    KV: torch.Tensor,
    Indices: torch.Tensor,
) -> torch.Tensor:
    return _sparse_mla_kernel(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride,
                              q_start_index_s, kv_group, sm_scale, is_causal, CP0,
                              dtype)(block_I, threads)(Q, KV, Indices)


@_sparse_mla_wrapped_kernel.register_fake
def _(batch, seq_len, seq_len_kv, heads, dim, tail_dim, dtype, topk, kv_stride, q_start_index_s, kv_group, sm_scale, is_causal, CP0, block_I, threads, *inputs):
    fake_o = torch.empty([batch, seq_len, heads, dim],
                         device=inputs[0].device,
                         dtype=inputs[0].dtype)
    return fake_o


class sparse_mla_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch,
                 seq_len,
                 seq_len_kv,
                 heads,
                 dim,
                 tail_dim,
                 dtype,
                 topk,
                 kv_stride,
                 q_start_index_s,
                 kv_group=1,
                 sm_scale=None,
                 is_causal=True,
                 CP0=True,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.tail_dim = tail_dim
        self.dtype = dtype
        self.topk = topk
        self.kv_stride = kv_stride
        self.kv_group = kv_group
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.q_start_index_s = q_start_index_s
        self.CP0 = CP0

        self.kernel = _sparse_mla_kernel(self.batch, self.seq_len, self.seq_len_kv, self.heads,
                                         self.dim, self.tail_dim, self.topk, self.kv_stride,
                                         self.q_start_index_s, self.kv_group, self.sm_scale,
                                         self.is_causal, self.CP0, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_I": 64, "threads": 384}

    @property
    def autotune_configs(self) -> list[dict]:
        block_I = [64, 128]
        threads = [384, 512]
        _configs = list(itertools.product(block_I, threads))

        configs = [{
            'block_I': c[0],
            'threads': c[1],
        } for c in _configs]
        return configs

    def forward(self, Q: torch.Tensor, KV: torch.Tensor, Indices: torch.Tensor):
        return _sparse_mla_wrapped_kernel(self.batch, self.seq_len, self.seq_len_kv, self.heads,
                                          self.dim, self.tail_dim, self.topk, self.kv_stride,
                                          self.q_start_index_s, self.kv_group, self.sm_scale,
                                          self.is_causal, self.CP0, self.dtype_str,
                                          self.config["block_I"], self.config["threads"], Q, KV,
                                          Indices)

    # @property
    def supply_prog(self, params=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim + self.tail_dim,
            device='cuda',
            dtype=self.dtype)
        KV = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.dim + self.tail_dim,
            device='cuda',
            dtype=self.dtype)
        Indices = torch.full((self.batch, self.seq_len, self.kv_group, self.topk),
                             self.seq_len_kv,
                             dtype=torch.int32,
                             device='cuda')
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.kv_group):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.kv_stride)),
                            self.seq_len_kv))[:self.topk]
                    Indices[b, t, h, :len(i_i)] = i_i

        return Q, KV, Indices

    def autotune(self, warmup=10, rep=10):  # Removed supply_prog parameter
        if self.autotune_configs is None:
            return  # kernel doesn't support autotuning
        print(f'Start autotuning {self.__class__.__name__}...')

        # Apply autotune decorator to the kernel function
        autotuned_kernel_fn = autotune(
            configs=self.autotune_configs, warmup=warmup, rep=rep, supply_prog=self.supply_prog)(
                self.kernel)

        # Call without config parameters to trigger autotuning, returns the tuned kernel
        tuned_kernel = autotuned_kernel_fn()

        # Extract and store the best config
        self.config = tuned_kernel.config
        print(f'Best config: {self.config}')
