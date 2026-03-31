import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.online_softmax import make_log2e_scale, make_online_softmax, make_rescale

__all__ = [
    "MHADecodePagedKernel",
    "MHADecodePagedSplitKVKernel",
    "MHADecodePagedCombineKernel",
]

# ---------------------------------------------------------------------------
# JIT kernel: no-split variant (paged)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _mha_decode_no_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
                                dtype):
    scale = make_log2e_scale(dim)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _func(block_M, block_N, num_stages, threads):

        shape_q = [batch, seqlen_q, heads, dim]
        shape_kv = [seqlen_kv, heads, dim]

        online_softmax = make_online_softmax(scale, accum_dtype, block_M, block_N)
        rescale = make_rescale(block_M, dim)

        @T.prim_func
        def mha_decode_no_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_kv, dtype),
                V: T.Tensor(shape_kv, dtype),
                real_seqlen_kv: T.Tensor([batch], T.int32),
                block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
                Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seqlen_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                seqlen_kv = real_seqlen_kv[bz]

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

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv((bx + 1) *
                              block_M, block_N) if is_causal else T.ceildiv(seqlen_kv, block_N))

                num_blockn_in_page = page_size // block_N

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    page_idx = k // num_blockn_in_page
                    block_idx_in_page = k % num_blockn_in_page
                    blockn_num_offset = block_table[
                        bz, page_idx] * num_blockn_in_page + block_idx_in_page

                    T.copy(K[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N, by, :],
                           K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                         -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(k * block_N + j < seqlen_kv, 0,
                                                         -T.infinity(acc_s.dtype))
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.copy(V[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N, by, :],
                           V_shared)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale

        return mha_decode_no_split

    return _func


# ---------------------------------------------------------------------------
# JIT kernel: split-only variant (paged, no combine)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _mha_decode_split_only_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
                                  dtype):
    scale = make_log2e_scale(dim)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[5, 6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _func(block_M, block_N, num_split, num_stages, threads):

        shape_q = [batch, seqlen_q, heads, dim]
        shape_kv = [seqlen_kv, heads, dim]
        part_shape = [batch, seqlen_q, heads, num_split, dim]

        rescale = make_rescale(block_M, dim)

        @T.macro
        def MMA0(
            K: T.Tensor(shape_kv, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k_global: T.int32,
            k: T.int32,
            mid: T.int32,
            hid: T.int32,
            bid: T.int32,
            sid: T.int32,
        ):
            seqlen_kv = real_seqlen_kv[bid]
            T.copy(K[k * block_N:(k + 1) * block_N, hid, :], K_shared)
            # TODO: Handle causal split case
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(mid * block_M + i >= k_global * block_N + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(k_global * block_N + j < seqlen_kv, 0,
                                                 -T.infinity(acc_s.dtype))
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(shape_kv, dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            real_seqlen_kv: T.int32,
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k_global: T.int32,
            k: T.int32,
            hid: T.int32,
            bid: T.int32,
            sid: T.int32,
        ):
            T.copy(V[k * block_N:(k + 1) * block_N, hid, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_scale[i] = T.if_then_else(
                    scores_max[i] == -T.infinity(accum_dtype), 0,
                    T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale))
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.prim_func
        def mha_decode_split_only(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_kv, dtype),
                V: T.Tensor(shape_kv, dtype),
                real_seqlen_kv: T.Tensor([batch], T.int32),
                block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
                glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                split_length: T.Tensor([batch, num_split], "int32"),
        ):
            with T.Kernel(
                    T.ceildiv(seqlen_q, block_M), heads * batch, num_split,
                    threads=128) as (bx, by, bz):

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
                # =======================================
                split_length_shared = T.alloc_shared([num_split], "int32")
                T.copy(split_length[bid, :], split_length_shared, disable_tma=True)
                # ========================================
                # NOTE(wt): tma barrier has some problems with padded dimensions (seq_q here) currently
                # disable relevant tma copy and use SIMT as fallback for now
                T.copy(
                    Q[bid, mid * block_M:(mid + 1) * block_M, hid, :], Q_shared, disable_tma=True)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                real_seqlen_kv_b = real_seqlen_kv[bid]
                # Per-batch loop_range: only iterate blocks within real_seqlen_kv[bid],
                # so shorter batches don't run empty splits -> all -inf -> NaN in combine.
                start_block_sid = T.if_then_else(sid > 0, split_length_shared[sid - 1] // block_N,
                                                 0)
                end_block_valid = T.ceildiv(real_seqlen_kv_b, block_N)
                blocks_valid_this_split = end_block_valid - start_block_sid
                blocks_in_split = T.if_then_else(
                    sid > 0,
                    T.ceildiv(split_length_shared[sid] - split_length_shared[sid - 1], block_N),
                    T.ceildiv(split_length_shared[0], block_N),
                )
                loop_range = T.if_then_else(
                    blocks_valid_this_split <= 0,
                    0,
                    T.if_then_else(
                        blocks_valid_this_split <= blocks_in_split,
                        blocks_valid_this_split,
                        blocks_in_split,
                    ),
                )
                num_blockn_in_page = page_size // block_N
                loop_range = blocks_in_split
                offset = 0 if sid == 0 else split_length_shared[sid - 1] // block_N

                for k in T.Pipelined(loop_range, num_stages=2):
                    k_global = k
                    k_global += offset

                    page_idx = k_global // num_blockn_in_page
                    block_idx_in_page = k_global % num_blockn_in_page
                    blockn_num_offset = block_table[
                        bid, page_idx] * num_blockn_in_page + block_idx_in_page

                    MMA0(K, Q_shared, K_shared, real_seqlen_kv, acc_s, k_global, blockn_num_offset,
                         mid, hid, bid, sid)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, real_seqlen_kv, acc_s_cast, acc_o, k_global,
                         blockn_num_offset, hid, bid, sid)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                for i in T.Parallel(block_M):
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale
                T.copy(logsum, glse[bid, hid, sid, mid * block_M:(mid + 1) * block_M])
                T.copy(acc_o, O_shared)
                T.copy(
                    O_shared,
                    Output_partial[bid, mid * block_M:(mid + 1) * block_M, hid, sid, :],
                    disable_tma=True)

        return mha_decode_split_only

    return _func


# ---------------------------------------------------------------------------
# JIT kernel: combine-only variant
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _mha_decode_paged_combine_kernel(batch, heads, seqlen_q, num_split, dim, dtype):
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _func(block_M, threads):
        shape_q = [batch, seqlen_q, heads, dim]
        part_shape = [batch, seqlen_q, heads, num_split, dim]

        @T.prim_func
        def mha_decode_combine(
                glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(T.ceildiv(seqlen_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
                    o_shared:
                        tilelang.layout.make_swizzled_layout(o_shared),
                    po_shared:
                        tilelang.layout.make_swizzled_layout(po_shared),
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
                    T.copy(
                        Output_partial[bz, bx * block_M:(bx + 1) * block_M, by, k, :],
                        po_shared,
                        disable_tma=True)
                    T.copy(po_shared, po_local)
                    for i in T.Parallel(block_M):
                        lse_local_split[i] = lse_local[k, i]
                    for i in T.Parallel(block_M):
                        scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
                    for i, j in T.Parallel(block_M, dim):
                        o_accum_local[i, j] += po_local[i, j] * scale_local[i]
                T.copy(o_accum_local, o_shared)
                T.copy(
                    o_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :], disable_tma=True)

        return mha_decode_combine

    return _func


# ---------------------------------------------------------------------------
# Custom ops (torch.compile compatible wrappers)
# ---------------------------------------------------------------------------


# Use distinct op names so paged and non-paged (mha_decode.py) do not overwrite
# each other in torch.library; otherwise the first-registered impl "changes" the
# registry and later parametrized tests can hit the wrong implementation.
@torch.library.custom_op("top::mha_decode_paged_no_split_op", mutates_args=())
def _mha_decode_paged_no_split_op(
        batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int, page_size: int,
        is_causal: bool, dtype: str, block_M: int, block_N: int, num_stages: int, threads: int,
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor) -> torch.Tensor:
    return _mha_decode_no_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size,
                                       is_causal,
                                       dtype)(block_M, block_N, num_stages,
                                              threads)(Q, K, V, real_seqlen_kv, block_table)


@_mha_decode_paged_no_split_op.register_fake
def _(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int, page_size: int,
      is_causal: bool, dtype: str, block_M: int, block_N: int, num_stages: int, threads: int,
      Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, real_seqlen_kv: torch.Tensor,
      block_table: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(Q)


@torch.library.custom_op("top::mha_decode_paged_split_only_op", mutates_args=())
def _mha_decode_paged_split_only_op(
        batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int, page_size: int,
        is_causal: bool, dtype: str, block_M: int, block_N: int, num_stages: int, threads: int,
        num_split: int, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        real_seqlen_kv: torch.Tensor, block_table: torch.Tensor,
        split_length: torch.Tensor) -> list[torch.Tensor]:
    return _mha_decode_split_only_kernel(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
        dtype)(block_M, block_N, num_split, num_stages,
               threads)(Q, K, V, real_seqlen_kv, block_table, split_length)


@_mha_decode_paged_split_only_op.register_fake
def _(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int, page_size: int,
      is_causal: bool, dtype: str, block_M: int, block_N: int, num_stages: int, threads: int,
      num_split: int, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
      real_seqlen_kv: torch.Tensor, block_table: torch.Tensor,
      split_length: torch.Tensor) -> list[torch.Tensor]:
    torch_dtype = getattr(torch, dtype)
    glse = torch.empty(batch, heads, num_split, seqlen_q, dtype=torch_dtype, device=Q.device)
    Output_partial = torch.empty(
        batch, seqlen_q, heads, num_split, dim, dtype=torch_dtype, device=Q.device)
    return [glse, Output_partial]


@torch.library.custom_op("top::mha_decode_paged_combine_op", mutates_args=())
def _mha_decode_paged_combine_op(batch: int, heads: int, seqlen_q: int, num_split: int, dim: int,
                                  dtype: str, block_M: int, threads: int, glse: torch.Tensor,
                                  Output_partial: torch.Tensor) -> torch.Tensor:
    return _mha_decode_paged_combine_kernel(batch, heads, seqlen_q, num_split, dim,
                                             dtype)(block_M, threads)(glse, Output_partial)


@_mha_decode_paged_combine_op.register_fake
def _(batch: int, heads: int, seqlen_q: int, num_split: int, dim: int, dtype: str, block_M: int,
      threads: int, glse: torch.Tensor, Output_partial: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        batch, seqlen_q, heads, dim, dtype=glse.dtype, device=glse.device)


# ---------------------------------------------------------------------------
# Kernel classes
# ---------------------------------------------------------------------------


class MHADecodePagedKernel(Kernel):
    """No-split paged MHA decode kernel for short sequences."""
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 seqlen_q,
                 seqlen_kv,
                 dim,
                 page_size,
                 is_causal=False,
                 dtype="float16",
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_decode_no_split_kernel(
            self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim, self.page_size,
            self.is_causal, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        block_N = min(64 if self.dim <= 128 else 32, self.page_size)
        return {"block_M": 128, "block_N": block_N, "num_stages": 2, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [64, 128]
        block_N = [64, 128]
        num_stages = [1, 2, 3]
        threads = [128]
        _configs = list(itertools.product(block_M, block_N, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs if c[1] <= self.page_size]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor):
        block_M = self.config["block_M"]
        block_N = self.config["block_N"]
        num_stages = self.config["num_stages"]
        threads = self.config["threads"]

        return _mha_decode_paged_no_split_op(
            self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim, self.page_size,
            self.is_causal, self.dtype_str, block_M, block_N, num_stages, threads, Q, K, V,
            real_seqlen_kv, block_table)


class MHADecodePagedSplitKVKernel(Kernel):
    """Split-only paged MHA decode kernel (no combine). Outputs glse and Output_partial."""
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 seqlen_q,
                 seqlen_kv,
                 dim,
                 page_size,
                 is_causal=False,
                 dtype="float16",
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_decode_split_only_kernel(
            self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim, self.page_size,
            self.is_causal, self.dtype_str)
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    def _make_supply_prog(self):
        """Create a supply_prog that handles int32 tensor parameters for paged attention."""
        from tilelang.utils.tensor import get_tensor_supply as _get_tensor_supply

        default_supply = _get_tensor_supply(tilelang.TensorSupplyType.Auto)
        seqlen_kv = self.seqlen_kv
        batch = self.batch
        page_size = self.page_size
        num_pages = seqlen_kv // page_size

        def supply_prog(params):
            inputs = []
            for param in params:
                if str(param.dtype) == "int32":
                    shape = param.shape
                    if len(shape) == 1 and shape[0] == batch:
                        # real_seqlen_kv: [batch]
                        inputs.append(
                            torch.full((batch,), seqlen_kv, dtype=torch.int32, device="cuda"))
                    elif len(shape) == 2 and shape[1] == num_pages:
                        # block_table: [batch, num_pages] — sequential page indices
                        t = torch.arange(
                            num_pages, dtype=torch.int32,
                            device="cuda").unsqueeze(0).expand(batch, -1).contiguous()
                        inputs.append(t)
                    elif len(shape) == 2:
                        # acc_split_length (cumulative): [batch, num_split]
                        num_split = shape[1]
                        base = seqlen_kv // num_split
                        t = torch.full(shape, base, dtype=torch.int32, device="cuda")
                        t[:, -1] += seqlen_kv % num_split
                        t = torch.cumsum(t, dim=1).to(torch.int32)
                        inputs.append(t)
                    else:
                        inputs.append(default_supply(param))
                else:
                    inputs.append(default_supply(param))
            return inputs

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    @property
    def default_config(self) -> dict:
        block_N = min(64 if self.dim <= 128 else 32, self.page_size)
        return {
            "block_M": 128,
            "block_N": block_N,
            "num_split": 4,
            "num_stages": 2,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [64, 128]
        block_N = [64, 128]
        num_split = [2, 4, 8]
        num_stages = [2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_split, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_split': c[2],
            'num_stages': c[3],
            'threads': c[4]
        } for c in _configs if c[1] <= self.page_size]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor,
                split_length: torch.Tensor):
        block_M = self.config["block_M"]
        block_N = self.config["block_N"]
        num_split = self.config["num_split"]
        num_stages = self.config["num_stages"]
        threads = self.config["threads"]

        return _mha_decode_paged_split_only_op(
            self.batch, self.heads, self.seqlen_q, self.seqlen_kv, self.dim, self.page_size,
            self.is_causal, self.dtype_str, block_M, block_N, num_stages, threads, num_split, Q, K,
            V, real_seqlen_kv, block_table, split_length)


class MHADecodePagedCombineKernel(Kernel):
    """Combine-only kernel: reduces partial outputs from split kernel into final output."""
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 seqlen_q,
                 num_split,
                 dim,
                 dtype="float16",
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.num_split = num_split
        self.dim = dim
        self.dtype = dtype

        self.kernel = _mha_decode_paged_combine_kernel(
            self.batch, self.heads, self.seqlen_q, self.num_split, self.dim, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_M": 128, "threads": 128}

    def forward(self, glse: torch.Tensor, Output_partial: torch.Tensor):
        block_M = self.config["block_M"]
        threads = self.config["threads"]

        return _mha_decode_paged_combine_op(self.batch, self.heads, self.seqlen_q, self.num_split,
                                             self.dim, self.dtype_str, block_M, threads, glse,
                                             Output_partial)


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

# Keep old name as an alias for the split kernel (which was the primary autotune target)
mha_decode_paged_kernel = MHADecodePagedSplitKVKernel
