import itertools
from collections.abc import Callable
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import DSADecodeCall, SparseMLADecodeFwdInterface
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import BLOCK_SHARED_BYTES_OPT_IN, LOG2E
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = ["DSADecodeBasicKernel", "DSADecodeKernel", "DSADecodeKernelBase"]


class DSADecodeKernelBase(Kernel, SparseMLADecodeFwdInterface):
    """Shared shape checks, construction and tuning inputs for sparse MLA decode."""

    serves_attn_sink: ClassVar[bool] = False
    # The builder __init__ compiles from, or None for a kernel that builds at launch.
    _build: ClassVar[Callable | None] = None

    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        heads: int,
        dim: int,
        tail_dim: int,
        dtype: torch.dtype,
        topk: int,
        kv_stride: int,
        q_start_index_s: int,
        kv_group: int,
        sm_scale: float | None,
        is_causal: bool,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.tail_dim = tail_dim
        self.dtype = dtype
        self.topk = topk
        self.kv_stride = kv_stride
        self.q_start_index_s = q_start_index_s
        self.kv_group = kv_group
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        if self._build is not None:
            self.kernel = self._build(
                batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride,
                q_start_index_s, kv_group, sm_scale, is_causal, self.dtype_str,
            )  # fmt: skip
            self.init_config()

    @classmethod
    def refusal(cls, call: DSADecodeCall) -> str | None:
        if call.has_attn_sink and not cls.serves_attn_sink:
            return "does not serve attn_sink"
        # A causal query before position kv_stride - 1 sees no key.
        if call.is_causal and not (
            call.q_start_index_s > call.kv_stride
            or (call.q_start_index_s == 0 and call.kv_stride == 1)
        ):
            return "requires q_start_index_s past kv_stride, or 0 with kv_stride 1"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: DSADecodeCall) -> Entry:
        """The call spec is the identity; the kernel is built on its device."""
        sink = {"has_attn_sink": call.has_attn_sink} if cls.serves_attn_sink else {}
        return call, lambda: cls(
            call.batch,
            call.seq_len,
            call.seq_len_kv,
            call.heads,
            call.dim,
            call.tail_dim,
            call.dtype,
            call.topk,
            call.kv_stride,
            call.q_start_index_s,
            call.kv_group,
            call.sm_scale,
            call.is_causal,
            device_index=call.device_index,
            **sink,
        )

    @staticmethod
    def shape_refusal(
        dim: int, tail_dim: int, heads: int, kv_group: int, is_causal: bool
    ) -> str | None:
        """Why neither DSADecodeBasicKernel nor DSADecodeKernel serves this shape, or ``None``;
        their builders ask too."""
        if not is_causal:
            return "requires the causal mask"
        pow2 = tilelang.math.next_power_of_2
        if dim != pow2(dim) or (tail_dim != 0 and tail_dim != pow2(tail_dim)):
            return "requires power-of-two dim and a zero or power-of-two tail_dim"
        group_heads = heads // kv_group
        if group_heads > 64 and group_heads % 64 != 0:
            return "requires at most 64 heads per KV group, or a multiple of 64"
        if kv_group != 1 and max(pow2(group_heads), 16) != group_heads:
            return "requires a power of two of at least 16 heads per KV group when kv_group > 1"
        return None

    def supply_prog(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate query, KV, and valid sparse indices for autotuning."""
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim + self.tail_dim,
            device="cuda",
            dtype=self.dtype,
        )
        kv = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.dim + self.tail_dim,
            device="cuda",
            dtype=self.dtype,
        )
        indices = torch.full(
            (self.batch, self.seq_len, self.kv_group, self.topk),
            self.seq_len_kv,
            dtype=torch.int32,
            device="cuda",
        )
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.kv_group):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.kv_stride)),
                            self.seq_len_kv,
                        )
                    )[: self.topk]
                    indices[b, t, h, : len(i_i)] = i_i

        return q, kv, indices

    @property
    def autotune_supply_prog(self) -> Callable | None:
        # TileLang passes the candidate's JIT parameters, which supply_prog does not read.
        return lambda *args, **kwargs: self.supply_prog()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        attn_sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.program(self._compile)(q, kv, indices)


@target_cache(maxsize=32)
def _dsa_decode_kernel(
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
    sm_scale: float | None,
    is_causal: bool,
    dtype: str,
) -> Callable:
    """Causal sparse MLA on SM90: a KV-gathering producer and two WGMMA consumer warpgroups."""
    reason = DSADecodeKernelBase.shape_refusal(dim, tail_dim, heads, kv_group, is_causal)
    if reason is not None:
        raise ValueError(reason)
    sm_scale = ((1.0 / (dim + tail_dim)) ** 0.5 if sm_scale is None else sm_scale) * LOG2E

    head_kv = heads // kv_group
    ori_heads = heads
    indices_dtype = "int32"
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        compile_flags=[
            "--use_fast_math",
            *CUDA_COMPILE_FLAGS,
            "--ptxas-options=-v,--register-usage-level=10",
        ],
    )
    def _dsa_decode_fwd_func(block_i: int, threads: int):
        q_shape = (batch, seq_len, ori_heads, dim + tail_dim)
        kv_shape = (batch, seq_len_kv, kv_group, dim + tail_dim)
        o_shape = (batch, seq_len, ori_heads, dim)
        indices_shape = (batch, seq_len, kv_group, topk)

        padded_h = max(tilelang.math.next_power_of_2(head_kv), 16)

        if topk % block_i != 0:
            raise ValueError("otherwise will load some index=0 thus causing wrong kv to be loaded")
        i_block = block_i
        n_i = tilelang.cdiv(topk, block_i)
        if n_i % 2 != 0:
            raise ValueError("n_i should be a multiple of 2")
        d = dim
        d_tail = tail_dim
        stride_kv = kv_stride

        # Two 128-thread consumer warpgroups take tx < 256; the rest gather KV,
        # 8 threads to a row, and must tile i_block exactly.
        consumer_threads = 256
        producer_threads = threads - consumer_threads
        if producer_threads <= 0 or producer_threads % 8 != 0:
            raise ValueError(
                f"threads={threads} leaves {producer_threads} producer threads; "
                "the KV gather needs a positive multiple of 8"
            )
        producer_rows = producer_threads // 8
        if i_block % producer_rows != 0:
            raise ValueError(
                f"block_i={i_block} is not a multiple of the {producer_rows} rows one "
                f"gather pass copies with threads={threads}"
            )
        reason = DSADecodeKernel.gather_refusal(d, d_tail)
        if reason is not None:
            raise ValueError(reason)

        replicate_h = head_kv // 64 if head_kv > 64 else 1

        h_per_block = padded_h if replicate_h == 1 else 64

        @T.macro
        def score_block(q_l, q_r, q_tail, kv_l, kv_r, k_tail, is_kv_valid, acc_s, s_shared, acc_o_l, sumexp, sumexp_i, alpha_shared, alpha_local, m_i, m_i_prev, bar_k_ready, bar_k_free, bar_s_ready, bar_s_free, phase, wait_s_free, s_free_parity):  # fmt: skip
            """Consumer 0's step over one KV buffer: score it, fold it into the online softmax,
            hand the weights and the rescale to consumer 1, and accumulate the left value half."""
            T.barrier_wait(bar_k_ready[0], phase)

            for h_i, bi_i in T.Parallel(h_per_block, i_block):
                acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0, -T.infinity(acc_s.dtype))
            T.wgmma_gemm(q_l, kv_l, acc_s, transpose_B=True)
            T.wgmma_gemm(q_r, kv_r, acc_s, transpose_B=True)
            T.wgmma_gemm(q_tail, k_tail, acc_s, transpose_B=True)

            T.wait_wgmma(0)

            if wait_s_free:
                T.barrier_arrive(bar_s_free)
                T.barrier_wait(bar_s_free, s_free_parity)

            T.copy(m_i, m_i_prev)
            T.reduce_max(acc_s, m_i, dim=1, clear=False)
            # m_i_prev becomes the scaled max the weights subtract: 0 while the
            # row has no valid key, so its weights are 0 instead of NaN.
            for h_i in T.Parallel(h_per_block):
                m_sub = T.if_then_else(
                    m_i[h_i] == -T.infinity(accum_dtype), 0.0, m_i[h_i] * sm_scale
                )
                alpha_local[h_i] = T.exp2(m_i_prev[h_i] * sm_scale - m_sub)
                m_i_prev[h_i] = m_sub
            for h_i, bi_i in T.Parallel(h_per_block, i_block):
                acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i_prev[h_i])
            T.reduce_sum(acc_s, sumexp_i, dim=1)
            for h_i in T.Parallel(h_per_block):
                sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
            for h_i, d_i in T.Parallel(h_per_block, d // 2):
                acc_o_l[h_i, d_i] *= alpha_local[h_i]
            T.copy(alpha_local, alpha_shared)

            T.copy(acc_s, s_shared)
            T.gemm(s_shared, kv_l, acc_o_l)

            T.barrier_arrive(bar_s_ready)
            T.barrier_arrive(bar_k_free[0])

        @T.macro
        def gather_block(kv, indices, kv_l, kv_r, k_tail, is_kv_valid, indices_local, bar_k_ready, bar_k_free, b_i, s_i, g_i, block, phase, max_kv_i, tx):  # fmt: skip
            """The producer's fill of one KV buffer with top-k block `block`: each thread copies
            8 columns of row (tx - 256) // 8 of every pass."""
            T.barrier_wait(bar_k_free[0], phase ^ 1)
            for r in T.serial(i_block // producer_rows):
                row = r * producer_rows + (tx - 256) // 8
                col = (tx - 256) % 8 * 8
                indices_local[0] = indices[b_i, s_i, g_i, block * i_block + row]
                is_kv_valid[row] = (indices_local[0] >= 0) & (indices_local[0] <= max_kv_i)
                if is_kv_valid[row]:
                    with T.attr("default", "async_scope", 1):
                        for u in T.serial(d // 128):
                            for v in T.vectorized(8):
                                kv_l[row, 64 * u + col + v] = kv[
                                    b_i, indices_local[0], g_i, 64 * u + col + v
                                ]
                                kv_r[row, 64 * u + col + v] = kv[
                                    b_i, indices_local[0], g_i, d // 2 + 64 * u + col + v
                                ]
                    with T.attr("default", "async_scope", 1):
                        for v in T.vectorized(8):
                            k_tail[row, col + v] = kv[b_i, indices_local[0], g_i, d + col + v]
                else:
                    # Zero, not stale: the row's softmax weight is zero, and zero times a NaN
                    # an earlier tile left in shared memory is NaN.
                    for u in T.serial(d // 128):
                        for v in T.vectorized(8):
                            kv_l[row, 64 * u + col + v] = 0
                            kv_r[row, 64 * u + col + v] = 0
                    for v in T.vectorized(8):
                        k_tail[row, col + v] = 0
            T.cp_async_barrier_noinc(bar_k_ready[0])
            T.barrier_arrive(bar_k_ready[0])

        @T.prim_func
        def _dsa_decode_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            kv: T.Tensor(kv_shape, dtype),  # type: ignore
            indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            output: T.Tensor(o_shape, dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                seq_len * replicate_h,
                batch,
                kv_group,
                threads=threads,
            ) as (bx, by, bz):
                q_shared_l = T.alloc_shared([h_per_block, d // 2], dtype)
                q_shared_r = T.alloc_shared([h_per_block, d // 2], dtype)
                q_tail_shared = T.alloc_shared([h_per_block, d_tail], dtype)
                kv_shared_0_l = T.alloc_shared([i_block, d // 2], dtype)
                kv_shared_0_r = T.alloc_shared([i_block, d // 2], dtype)
                kv_shared_1_l = T.alloc_shared([i_block, d // 2], dtype)
                kv_shared_1_r = T.alloc_shared([i_block, d // 2], dtype)
                k_tail_shared_0 = T.alloc_shared([i_block, d_tail], dtype)
                k_tail_shared_1 = T.alloc_shared([i_block, d_tail], dtype)
                o_shared_l = q_shared_l
                o_shared_r = q_shared_r
                # One mask per KV buffer: the producer fills the next block's mask while a
                # consumer still reads the other's.
                is_kv_valid_0 = T.alloc_shared([i_block], "bool", scope="shared")
                is_kv_valid_1 = T.alloc_shared([i_block], "bool", scope="shared")

                acc_o_l = T.alloc_fragment([h_per_block, d // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([h_per_block, d // 2], accum_dtype)
                acc_s = T.alloc_fragment([h_per_block, i_block], accum_dtype)
                s_shared = T.alloc_shared([h_per_block, i_block], dtype)
                sumexp = T.alloc_fragment([h_per_block], accum_dtype)
                sum_exp_shared = T.alloc_shared([h_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([h_per_block], accum_dtype)
                alpha_shared = T.alloc_shared([h_per_block], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([h_per_block], accum_dtype)
                m_i = T.alloc_fragment([h_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([h_per_block], accum_dtype)
                indices_local = T.alloc_local([1], indices_dtype)

                # Each producer thread arrives twice: once from its cp.async copies, and once
                # with release semantics for the masks and zero-filled rows it stores directly,
                # which the cp.async arrival does not order.
                bar_k_0_ready = T.alloc_barrier(arrive_count=2 * producer_threads)
                bar_k_1_ready = T.alloc_barrier(arrive_count=2 * producer_threads)
                bar_k_0_free = T.alloc_barrier(arrive_count=consumer_threads)
                bar_k_1_free = T.alloc_barrier(arrive_count=consumer_threads)
                bar_s_scale_and_s_ready = T.alloc_barrier(arrive_count=consumer_threads)
                bar_s_scale_and_s_free = T.alloc_barrier(arrive_count=consumer_threads)
                # Consumer 0's warpgroup publishes the row sums consumer 1 divides by.
                bar_sum_exp_ready = T.alloc_barrier(arrive_count=128)

                b_i, g_i = by, bz
                s_i = bx if replicate_h == 1 else bx // replicate_h
                q_i = q_start_index_s + s_i
                # The causal limit, clamped to the rows kv holds: a padded
                # top-k slot must not address kv past its last row.
                max_kv_i = T.min((q_i + 1 - stride_kv) // stride_kv, seq_len_kv - 1)

                h0 = g_i * padded_h + (0 if replicate_h == 1 else (bx % replicate_h) * 64)
                h1 = h0 + h_per_block

                tx = T.get_thread_binding()

                # Q -> shared copies stay inside tx < 128, the bounds of the WGMMA that reads
                # them: TileLang's layout inference finds no layout for a shared buffer a
                # wider copy writes and a 128-thread WGMMA reads.
                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.copy(q[b_i, s_i, h0:h1, 0 : d // 2], q_shared_l)
                    T.copy(q[b_i, s_i, h0:h1, d // 2 : d], q_shared_r)
                    T.copy(q[b_i, s_i, h0:h1, d:], q_tail_shared)
                    T.fill(sumexp, 0)
                    T.fill(m_i, -T.infinity(accum_dtype))
                    T.fill(acc_o_l, 0)

                    for i_i in T.serial(T.ceildiv(n_i, 2)):
                        score_block(
                            q_shared_l, q_shared_r, q_tail_shared, kv_shared_0_l, kv_shared_0_r,
                            k_tail_shared_0, is_kv_valid_0, acc_s, s_shared, acc_o_l, sumexp,
                            sumexp_i, alpha_shared, alpha_local, m_i, m_i_prev, bar_k_0_ready,
                            bar_k_0_free, bar_s_scale_and_s_ready, bar_s_scale_and_s_free,
                            i_i & 1, i_i != 0, 1,
                        )  # fmt: skip
                        score_block(
                            q_shared_l, q_shared_r, q_tail_shared, kv_shared_1_l, kv_shared_1_r,
                            k_tail_shared_1, is_kv_valid_1, acc_s, s_shared, acc_o_l, sumexp,
                            sumexp_i, alpha_shared, alpha_local, m_i, m_i_prev, bar_k_1_ready,
                            bar_k_1_free, bar_s_scale_and_s_ready, bar_s_scale_and_s_free,
                            i_i & 1, True, 0,
                        )  # fmt: skip

                    # Only a row with no valid key sums to 0; its acc_o is 0, and so is its output.
                    for h_i in T.Parallel(h_per_block):
                        sumexp[h_i] = T.if_then_else(sumexp[h_i] == 0, 1.0, sumexp[h_i])
                        sum_exp_shared[h_i] = sumexp[h_i]
                    T.barrier_arrive(bar_sum_exp_ready)
                    for h_i, d_i in T.Parallel(h_per_block, d // 2):
                        acc_o_l[h_i, d_i] /= sumexp[h_i]
                    T.copy(acc_o_l, o_shared_l)
                    T.copy(o_shared_l, output[b_i, s_i, h0:h1, 0 : d // 2])

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(n_i, 2)):
                        T.barrier_arrive(bar_s_scale_and_s_ready)
                        T.barrier_wait(bar_s_scale_and_s_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(h_per_block, d // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(s_shared, kv_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_s_scale_and_s_free)

                        T.barrier_arrive(bar_s_scale_and_s_ready)
                        T.barrier_wait(bar_s_scale_and_s_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(h_per_block, d // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(s_shared, kv_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(n_i, 2) - 1:
                            T.barrier_arrive(bar_s_scale_and_s_free)

                    T.barrier_wait(bar_sum_exp_ready, 0)
                    for h_i, d_i in T.Parallel(h_per_block, d // 2):
                        acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, o_shared_r)
                    T.copy(o_shared_r, output[b_i, s_i, h0:h1, d // 2 : d])

                elif tx >= 256:
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(n_i, 2)):
                        gather_block(
                            kv, indices, kv_shared_0_l, kv_shared_0_r, k_tail_shared_0,
                            is_kv_valid_0, indices_local, bar_k_0_ready, bar_k_0_free, b_i, s_i,
                            g_i, i_i * 2, i_i & 1, max_kv_i, tx,
                        )  # fmt: skip
                        gather_block(
                            kv, indices, kv_shared_1_l, kv_shared_1_r, k_tail_shared_1,
                            is_kv_valid_1, indices_local, bar_k_1_ready, bar_k_1_free, b_i, s_i,
                            g_i, i_i * 2 + 1, i_i & 1, max_kv_i, tx,
                        )  # fmt: skip

        return _dsa_decode_fwd_main

    return _dsa_decode_fwd_func


@target_cache(maxsize=32)
def _dsa_decode_basic_kernel(
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
    sm_scale: float | None,
    is_causal: bool,
    dtype: str,
) -> Callable:
    """Causal sparse MLA on any architecture: ``_dsa_decode_kernel``'s online softmax over the
    gathered top-k KV rows, with one thread group, ``T.Pipelined`` stages and per-row
    ``T.copy`` gathers in place of warp specialization and WGMMA."""
    reason = DSADecodeKernelBase.shape_refusal(dim, tail_dim, heads, kv_group, is_causal)
    if reason is not None:
        raise ValueError(reason)
    sm_scale = ((1.0 / (dim + tail_dim)) ** 0.5 if sm_scale is None else sm_scale) * LOG2E

    head_kv = heads // kv_group
    ori_heads = heads
    indices_dtype = "int32"
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        compile_flags=[
            "--use_fast_math",
            *CUDA_COMPILE_FLAGS,
            "--ptxas-options=-v,--register-usage-level=10",
        ],
    )
    def _dsa_decode_basic_fwd_func(block_i: int, threads: int, num_stages: int, block_h: int):
        if topk % block_i != 0:
            raise ValueError("otherwise will load some index=0 thus causing wrong kv to be loaded")
        i_block = block_i
        n_i = tilelang.cdiv(topk, block_i)

        d = dim
        d_tail = tail_dim
        stride_kv = kv_stride

        # A group wider than block_h is covered by several blocks; loads past the heads read
        # zero and stores there are dropped.
        replicate_h = tilelang.cdiv(head_kv, block_h) if head_kv > block_h else 1

        padded_h = max(tilelang.math.next_power_of_2(head_kv), 16)

        h_per_block = DSADecodeBasicKernel.heads_per_block(head_kv, block_h)
        # Width of the tail buffers; with no tail they are allocated but never read.
        tail_cols = d_tail if d_tail > 0 else 16

        q_shape = (batch, seq_len, ori_heads, dim + tail_dim)
        kv_shape = (batch, seq_len_kv, kv_group, dim + tail_dim)
        o_shape = (batch, seq_len, ori_heads, dim)
        indices_shape = (batch, seq_len, kv_group, topk)

        @T.prim_func
        def _dsa_decode_basic_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            kv: T.Tensor(kv_shape, dtype),  # type: ignore
            indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            output: T.Tensor(o_shape, dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                seq_len * replicate_h,
                batch,
                kv_group,
                threads=threads,
            ) as (bx, by, bz):
                # Q/KV are split at [.., :d] / [.., d:] so the PV gemm can
                # consume kv_shared directly: V is the first `dim` columns of
                # the fused KV cache (v = kv[..., :dim]).
                q_shared = T.alloc_shared([h_per_block, d], dtype)
                q_tail_shared = T.alloc_shared([h_per_block, tail_cols], dtype)
                kv_shared = T.alloc_shared([i_block, d], dtype)
                kv_tail_shared = T.alloc_shared([i_block, tail_cols], dtype)
                s_shared = T.alloc_shared([h_per_block, i_block], dtype)
                # Q is dead once the last QK^T gemm has been issued, so the
                # output staging reuses its shared buffer.
                o_shared = q_shared

                acc_s = T.alloc_fragment([h_per_block, i_block], accum_dtype)
                acc_o = T.alloc_fragment([h_per_block, d], accum_dtype)
                sumexp = T.alloc_fragment([h_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([h_per_block], accum_dtype)
                alpha_local = T.alloc_fragment([h_per_block], accum_dtype)
                m_i = T.alloc_fragment([h_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([h_per_block], accum_dtype)

                b_i, g_i = by, bz
                s_i = bx if replicate_h == 1 else bx // replicate_h
                q_i = q_start_index_s + s_i
                # The causal limit, clamped to the rows kv holds: a padded
                # top-k slot must not address kv past its last row.
                max_kv_i = T.min((q_i + 1 - stride_kv) // stride_kv, seq_len_kv - 1)

                h0 = g_i * padded_h + (0 if replicate_h == 1 else (bx % replicate_h) * block_h)
                h1 = h0 + h_per_block

                T.copy(q[b_i, s_i, h0:h1, :d], q_shared)
                if d_tail > 0:
                    T.copy(q[b_i, s_i, h0:h1, d:], q_tail_shared)
                T.fill(sumexp, 0)
                T.fill(m_i, -T.infinity(accum_dtype))
                T.fill(acc_o, 0)

                for i_i in T.Pipelined(n_i, num_stages=num_stages):
                    # Gather the selected KV rows; zero the rest, so that the
                    # zero weight the mask below gives them cannot meet a NaN
                    # an earlier tile left in shared memory.
                    for r in T.serial(i_block):
                        kv_idx = indices[b_i, s_i, g_i, i_i * i_block + r]
                        if (kv_idx >= 0) & (kv_idx <= max_kv_i):
                            T.copy(kv[b_i, kv_idx, g_i, :d], kv_shared[r, :])
                            if d_tail > 0:
                                T.copy(kv[b_i, kv_idx, g_i, d:], kv_tail_shared[r, :])
                        else:
                            T.clear(kv_shared[r, :])
                            if d_tail > 0:
                                T.clear(kv_tail_shared[r, :])

                    # acc_s starts at 0 for valid rows / -inf for invalid
                    # ones; the gemms below accumulate onto it.
                    for h_i, bi_i in T.Parallel(h_per_block, i_block):
                        mask_idx = indices[b_i, s_i, g_i, i_i * i_block + bi_i]
                        acc_s[h_i, bi_i] = T.if_then_else(
                            (mask_idx >= 0) & (mask_idx <= max_kv_i),
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                    T.gemm(
                        q_shared,
                        kv_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    if d_tail > 0:
                        T.gemm(
                            q_tail_shared,
                            kv_tail_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    # m_i_prev becomes the scaled max the weights subtract: 0 while the row
                    # has no valid key, so its weights are 0 instead of NaN.
                    for h_i in T.Parallel(h_per_block):
                        m_sub = T.if_then_else(
                            m_i[h_i] == -T.infinity(accum_dtype), 0.0, m_i[h_i] * sm_scale
                        )
                        alpha_local[h_i] = T.exp2(m_i_prev[h_i] * sm_scale - m_sub)
                        m_i_prev[h_i] = m_sub
                    for h_i, bi_i in T.Parallel(h_per_block, i_block):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i_prev[h_i])
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(h_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(h_per_block, d):
                        acc_o[h_i, d_i] *= alpha_local[h_i]

                    # O += P @ V (V is kv[..., :dim]).
                    T.copy(acc_s, s_shared)
                    T.gemm(s_shared, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

                # Only a row with no valid key sums to 0; its acc_o is 0, and so is its output.
                for h_i in T.Parallel(h_per_block):
                    sumexp[h_i] = T.if_then_else(sumexp[h_i] == 0, 1.0, sumexp[h_i])
                for h_i, d_i in T.Parallel(h_per_block, d):
                    acc_o[h_i, d_i] /= sumexp[h_i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, output[b_i, s_i, h0:h1, :d])

        return _dsa_decode_basic_fwd_main

    return _dsa_decode_basic_fwd_func


class DSADecodeBasicKernel(DSADecodeKernelBase):
    """Causal sparse MLA with pipelined GEMM and shared-memory KV gathers."""

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True
    _build = staticmethod(_dsa_decode_basic_kernel)

    @classmethod
    def refusal(cls, call: DSADecodeCall) -> str | None:
        """The shared shape region, where the default config fits the block's shared memory."""
        reason = cls.shape_refusal(
            call.dim, call.tail_dim, call.heads, call.kv_group, call.is_causal
        )
        limit = BLOCK_SHARED_BYTES_OPT_IN.get(call.arch)
        if reason is not None:
            return reason
        if limit is not None:
            head_kv = call.heads // call.kv_group
            itemsize = call.dtype.itemsize
            config = cls._default_config_for(
                call.arch, head_kv, call.dim, call.tail_dim, itemsize, call.topk
            )
            need = cls._shared_bytes(
                cls.heads_per_block(head_kv, config["block_h"]),
                call.dim,
                call.tail_dim,
                itemsize,
                config["block_i"],
                config["threads"],
                call.topk,
            )
            if need > limit:
                return (
                    f"needs {need} bytes of shared memory per block, over the {limit} bytes "
                    f"sm{call.arch} allows"
                )
        return super().refusal(call)

    @staticmethod
    def heads_per_block(head_kv: int, block_h: int) -> int:
        """Query heads a block holds: the padded group, at most *block_h*; the builder asks too."""
        return block_h if head_kv > block_h else max(tilelang.math.next_power_of_2(head_kv), 16)

    @classmethod
    def _default_config_for(
        cls, arch: int, head_kv: int, dim: int, tail_dim: int, itemsize: int, topk: int
    ) -> dict:
        """The config this kernel builds on *arch*, which its region is sized for."""
        block_i = 64 if arch >= 90 else 32
        threads = 128
        limit = BLOCK_SHARED_BYTES_OPT_IN.get(arch)
        block_h = next(
            (
                h
                for h in (64, 32)
                if limit is None
                or cls._shared_bytes(
                    cls.heads_per_block(head_kv, h),
                    dim,
                    tail_dim,
                    itemsize,
                    block_i,
                    threads,
                    topk,
                )
                <= limit
            ),
            16,
        )
        return {"block_i": block_i, "threads": threads, "num_stages": 2, "block_h": block_h}

    @staticmethod
    def _shared_bytes(
        h_per_block: int,
        dim: int,
        tail_dim: int,
        itemsize: int,
        block_i: int,
        threads: int,
        topk: int,
    ) -> int:
        """Shared memory TileLang gives one block; the serial KV gather adds no stage buffers.
        Over several KV tiles q, kv, their tails, s and the two reductions' workspaces each keep
        their own space; over one, only q and its tail, kv, and the larger of kv's tail and s."""
        q = h_per_block * (dim + tail_dim) * itemsize
        kv = block_i * dim * itemsize
        kv_tail, s = block_i * tail_dim * itemsize, h_per_block * block_i * itemsize
        if topk <= block_i:
            return q + kv + max(kv_tail, s)
        return q + kv + kv_tail + s + 2 * threads * 4

    @property
    def default_config(self) -> dict:
        return self._default_config_for(
            self.build_target.arch,
            self.heads // self.kv_group,
            self.dim,
            self.tail_dim,
            self.dtype.itemsize,
            self.topk,
        )

    @property
    def autotune_configs(self) -> list[dict]:
        # The autotuner drops a candidate that fails to compile.
        block_h = self.default_config["block_h"]
        return [
            {"block_i": block_i, "threads": threads, "num_stages": 2, "block_h": block_h}
            for block_i, threads in itertools.product((32, 64), (128, 256))
        ]

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(cfg["block_i"], cfg["threads"], cfg["num_stages"], cfg["block_h"])


class DSADecodeKernel(DSADecodeKernelBase):
    """Causal sparse MLA with SM90 WGMMA and manual warp specialization."""

    supported_archs: list[int] = [90]
    _build = staticmethod(_dsa_decode_kernel)

    @staticmethod
    def gather_refusal(dim: int, tail_dim: int) -> str | None:
        """Why the warp-specialized KV gather cannot copy these widths; its builder asks too."""
        if dim % 128 != 0 or tail_dim != 64:
            return "requires dim a multiple of 128 and tail_dim 64"
        return None

    @classmethod
    def refusal(cls, call: DSADecodeCall) -> str | None:
        """The shared shape region, narrowed to what the warp-specialized gather covers."""
        reason = cls.shape_refusal(
            call.dim, call.tail_dim, call.heads, call.kv_group, call.is_causal
        ) or cls.gather_refusal(call.dim, call.tail_dim)
        if reason is not None:
            return reason
        # The default block_i of 64, taken an even number of times.
        if call.topk % 128 != 0:
            return "requires topk a multiple of 128"
        return super().refusal(call)

    @property
    def default_config(self) -> dict:
        return {"block_i": 64, "threads": 384}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_i": block_i, "threads": threads}
            for block_i, threads in itertools.product((64, 128), (384, 512))
        ]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        super().autotune(warmup=warmup, rep=rep)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(cfg["block_i"], cfg["threads"])
