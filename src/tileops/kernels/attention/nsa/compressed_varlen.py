import functools
from typing import Callable, Optional, Tuple

import tilelang
import torch
from tilelang import language as T
from tilelang.layout import make_swizzled_layout

from tileops.kernels.attention.call_spec import NSACall, NSACompressedFwdInterface
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel


@functools.lru_cache(maxsize=32)
def _nsa_compressed_fwd_varlen_kernel(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    group: int,
    scale: float,
    bc: int,
    bs: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    scale_log2 = scale * LOG2E
    head_kv = heads // group
    # The shared-memory tiles span the whole head dimension: a narrower tile
    # would silently truncate the QK contraction and the O write-back.
    bk = dim_k
    bv = dim_v

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _nsa_compressed_fwd_varlen_func(threads: int):
        @T.prim_func
        def _parallel_nsa_compressed_fwd_varlen_main(
            q: T.Tensor((c_seq_len, heads, dim_k), dtype),
            k_cmp: T.Tensor((chunk_num, head_kv, dim_k), dtype),
            v_cmp: T.Tensor((chunk_num, head_kv, dim_v), dtype),
            offsets: T.Tensor((seq_num + 1,), T.int32),
            chunk_offsets: T.Tensor((seq_num + 1,), T.int32),
            token_indices: T.Tensor((c_seq_len, 2), T.int32),
            output: T.Tensor((c_seq_len, heads, dim_v), dtype),
            temp_lse: T.Tensor((c_seq_len, heads), dtype),
        ):
            with T.Kernel(c_seq_len, head_kv, threads=threads) as (bx, by):
                q_shared = T.alloc_shared([group, bk], dtype)
                k_shared = T.alloc_shared([bc, bk], dtype)
                v_shared = T.alloc_shared([bc, bv], dtype)

                i_c, i_h = bx, by
                i_n, i_t = token_indices[i_c, 0], token_indices[i_c, 1]

                bos = offsets[i_n]
                boc = chunk_offsets[i_n]
                nc = (i_t + 1) // bs

                T.copy(q[bos + i_t, i_h * group : (i_h + 1) * group, :bk], q_shared)

                b_o = T.alloc_fragment([group, bv], accum_dtype)
                b_lse = T.alloc_fragment([group], dtype)
                acc_s = T.alloc_fragment([group, bc], accum_dtype)
                acc_s_cast = T.alloc_fragment([group, bc], dtype)
                scores_max = T.alloc_fragment([group], accum_dtype)
                scores_max_prev = T.alloc_fragment([group], accum_dtype)
                scores_scale = T.alloc_fragment([group], accum_dtype)
                scores_sum = T.alloc_fragment([group], accum_dtype)
                logsum = T.alloc_fragment([group], accum_dtype)

                T.fill(b_o, 0.0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.fill(logsum, 0.0)

                for i_loop in T.serial(T.ceildiv(nc, bc)):
                    curr_bc = T.min(bc, nc - i_loop * bc)
                    # Initialize shared memory in each iteration to avoid stale data
                    T.fill(k_shared, 0.0)
                    T.fill(v_shared, 0.0)
                    T.copy(
                        k_cmp[boc + i_loop * bc : boc + i_loop * bc + curr_bc, i_h, :bk],
                        k_shared[:curr_bc, :bk],
                    )
                    T.copy(
                        v_cmp[boc + i_loop * bc : boc + i_loop * bc + curr_bc, i_h, :bv],
                        v_shared[:curr_bc, :bv],
                    )

                    for g_m, c_m in T.Parallel(group, bc):
                        acc_s[g_m, c_m] = T.if_then_else(
                            c_m < curr_bc, 0.0, -T.infinity(accum_dtype)
                        )

                    T.gemm(
                        q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for i in T.Parallel(group):
                        scores_scale[i] = T.if_then_else(
                            scores_max[i] > -T.infinity(accum_dtype),
                            T.exp2(scores_max_prev[i] * scale_log2 - scores_max[i] * scale_log2),
                            0.0,
                        )

                    for i, j in T.Parallel(group, bc):
                        acc_s[i, j] = T.if_then_else(
                            acc_s[i, j] > -T.infinity(accum_dtype),
                            T.exp2(acc_s[i, j] * scale_log2 - scores_max[i] * scale_log2),
                            0.0,
                        )

                    for i, k_idx in T.Parallel(group, bv):
                        b_o[i, k_idx] *= scores_scale[i]

                    T.copy(acc_s, acc_s_cast)

                    T.gemm(acc_s_cast, v_shared, b_o, policy=T.GemmWarpPolicy.FullRow)

                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(group):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, k_idx in T.Parallel(group, bv):
                    if nc > 0 and logsum[i] > 0:
                        b_o[i, k_idx] /= logsum[i]

                for i in T.Parallel(group):
                    if nc == 0 or logsum[i] <= 0:
                        b_lse[i] = 0.0
                    else:
                        b_lse[i] = (scores_max[i] * scale_log2 + T.log2(logsum[i])) / LOG2E

                T.copy(b_o, output[bos + i_t, i_h * group : (i_h + 1) * group, :dim_v])
                T.copy(b_lse, temp_lse[bos + i_t, i_h * group : (i_h + 1) * group])

        return _parallel_nsa_compressed_fwd_varlen_main

    return _nsa_compressed_fwd_varlen_func


class NSACompressedFwdVarlenKernel(Kernel, NSACompressedFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]
    # Chunks one tile holds.
    _BC = 32

    @classmethod
    def entry_for(cls, call: NSACall) -> Entry:
        return call, lambda: cls(
            seq_num=call.batch,
            c_seq_len=call.c_seq_len,
            heads=call.heads,
            dim_k=call.dim,
            dim_v=call.dim_v,
            chunk_num=call.chunk_num,
            group=call.heads // call.heads_kv,
            scale=call.scale,
            bc=cls._BC,
            bs=call.block_size,
            dtype=call.dtype,
            accum_dtype=torch.float32,
        )

    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_num: int,
        group: int,
        scale: float,
        bc: int,
        bs: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_num = chunk_num
        self.group = group
        self.scale = scale
        self.bc = bc
        self.bs = bs
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.accum_dtype_str = self.dtype_to_str(self.accum_dtype)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {
            "threads": 32,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        threads = [32]
        return [{"threads": t} for t in threads]

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _nsa_compressed_fwd_varlen_kernel(
            self.seq_num,
            self.c_seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_num,
            self.group,
            self.scale,
            self.bc,
            self.bs,
            self.dtype_str,
            self.accum_dtype_str,
        )(self.config["threads"])(
            q.to(self.dtype),
            k_cmp.to(self.dtype),
            v_cmp.to(self.dtype),
            offsets.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )


# Rows one m64 WGMMA computes: the query rows of a tile, tile_t tokens of one KV head.
_TILE_ROWS = 64
# Columns of one 128-byte swizzle block of 16-bit elements: Q, K, V and O move in halves.
_HALF = 64
# Keys one n=16 QK WGMMA scores; a chunk block is two of them.
_KEY_HALF = 16
# Query tiles in flight per CTA.
_STAGES = 2
# CTAs resident per SM: 48 KB of shared memory and 128 threads each.
_CTAS_PER_SM = 4


@functools.lru_cache(maxsize=32)
def _nsa_compressed_fwd_varlen_tma_kernel(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    group: int,
    scale: float,
    bs: int,
    workers: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    """Persistent SM90 compression attention: each CTA walks a run of token tiles of one KV head.

    A tile is the 64 query rows of tile_t tokens of one KV head. One warpgroup per CTA;
    its elected thread issues the TMAs. The next tile's queries load into a two-stage
    ring while this tile computes, as two 64-column halves. K and V of the tile's chunk
    block stay in shared memory until the tile's request or block changes. QK is two
    16-key halves of n=16 SS WGMMAs over both DK halves; P feeds n=64 RS WGMMAs over
    both DV halves. The normalized O halves are staged in the query stage QK has
    finished reading and leave by TMA, with the log-sum-exp beside them.

    A tile that straddles a request boundary makes one pass per request, and a row
    scores only its own request's closed chunks; passes and chunk blocks past the first
    rescale the running output (online softmax).
    """
    scale_log2 = scale * LOG2E
    head_kv = heads // group
    tile_t = _TILE_ROWS // group
    m = _TILE_ROWS
    half = _HALF
    kh = _KEY_HALF
    bc = 2 * kh
    stages = _STAGES
    n_tiles = (c_seq_len + tile_t - 1) // tile_t
    per_worker = (n_tiles + workers - 1) // workers
    # A finite start: a fully masked block then weighs exp2(-inf) = 0, never NaN.
    max_init = -1.0e30

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _func(threads: int):
        @T.prim_func
        def _main(
            q: T.Tensor([c_seq_len, heads, dim_k], dtype),
            k_cmp: T.Tensor([chunk_num, head_kv, dim_k], dtype),
            v_cmp: T.Tensor([chunk_num, head_kv, dim_v], dtype),
            offsets: T.Tensor([seq_num + 1], T.int32),
            chunk_offsets: T.Tensor([seq_num + 1], T.int32),
            token_indices: T.Tensor([c_seq_len, 2], T.int32),
            output: T.Tensor([c_seq_len, heads, dim_v], dtype),
            temp_lse: T.Tensor([c_seq_len, heads], dtype),
        ):
            with T.Kernel(workers, head_kv, threads=threads) as (i_w, i_h):
                q_lo = T.alloc_shared([stages, m, half], dtype)
                q_hi = T.alloc_shared([stages, m, half], dtype)
                # One chunk block of K and V, each [key half, key, column].
                k_lo = T.alloc_shared([2, kh, half], dtype)
                k_hi = T.alloc_shared([2, kh, half], dtype)
                v_lo = T.alloc_shared([2, kh, half], dtype)
                v_hi = T.alloc_shared([2, kh, half], dtype)
                lse_s = T.alloc_shared([m], dtype)
                T.annotate_layout(
                    {buf: make_swizzled_layout(buf) for buf in (q_lo, q_hi, k_lo, k_hi, v_lo, v_hi)}
                )
                q_full = T.alloc_barrier([threads] * stages)
                kv_full = T.alloc_barrier([threads])

                s0 = T.alloc_fragment([m, kh], accum_dtype)
                s1 = T.alloc_fragment([m, kh], accum_dtype)
                p0 = T.alloc_fragment([m, kh], dtype)
                p1 = T.alloc_fragment([m, kh], dtype)
                o_lo = T.alloc_fragment([m, half], accum_dtype)
                o_hi = T.alloc_fragment([m, half], accum_dtype)
                scores_max = T.alloc_fragment([m], accum_dtype)
                scores_max_prev = T.alloc_fragment([m], accum_dtype)
                scores_scale = T.alloc_fragment([m], accum_dtype)
                scores_sum = T.alloc_fragment([m], accum_dtype)
                logsum = T.alloc_fragment([m], accum_dtype)
                # Chunks each row's token has closed in the request of this pass.
                closed = T.alloc_fragment([m], T.int32)
                # The (request, chunk block) K/V hold, the cached request
                # (id, bos, eos, boc), the K/V loads issued, whether acc_o holds a block.
                loaded = T.alloc_local([2], T.int32)
                req_cache = T.alloc_local([4], T.int32)
                kv_loads = T.alloc_local([1], T.int32)
                first = T.alloc_local([1], T.int32)

                j0 = i_w * per_worker
                j1 = T.min(j0 + per_worker, n_tiles)
                h0 = i_h * group
                loaded[0] = -1
                loaded[1] = -1
                req_cache[0] = -1
                kv_loads[0] = 0
                q_lo_tiles = T.view(q_lo, [stages, tile_t, group, half])
                q_hi_tiles = T.view(q_hi, [stages, tile_t, group, half])
                for pf in T.serial(T.min(stages - 1, j1 - j0)):
                    tp = (j0 + pf) * tile_t
                    T.tma_copy(
                        q[tp : tp + tile_t, h0 : h0 + group, 0:half],
                        q_lo_tiles[pf, :, :, :],
                        barrier=q_full[pf],
                    )
                    T.tma_copy(
                        q[tp : tp + tile_t, h0 : h0 + group, half:dim_k],
                        q_hi_tiles[pf, :, :, :],
                        barrier=q_full[pf],
                    )
                    T.mbarrier_arrive(q_full[pf])

                for jj in T.serial(j1 - j0):
                    t0 = (j0 + jj) * tile_t
                    t_last = T.min(t0 + tile_t, c_seq_len) - 1
                    st = jj % stages
                    jn = jj + stages - 1
                    if jn < j1 - j0:
                        sn = jn % stages
                        tn = (j0 + jn) * tile_t
                        # The stage was last read by this warpgroup's WGMMAs and TMA store.
                        T.fence_proxy_async()
                        T.tma_copy(
                            q[tn : tn + tile_t, h0 : h0 + group, 0:half],
                            q_lo_tiles[sn, :, :, :],
                            barrier=q_full[sn],
                        )
                        T.tma_copy(
                            q[tn : tn + tile_t, h0 : h0 + group, half:dim_k],
                            q_hi_tiles[sn, :, :, :],
                            barrier=q_full[sn],
                        )
                        T.mbarrier_arrive(q_full[sn])
                    T.fill(logsum, 0)
                    T.fill(scores_max, max_init)
                    first[0] = 1
                    T.mbarrier_wait_parity(q_full[st], (jj // stages) % 2)
                    r0 = token_indices[t0, 0]
                    r1 = token_indices[t_last, 0]
                    for rr in T.serial(r1 - r0 + 1):
                        req = r0 + rr
                        if req_cache[0] != req:
                            req_cache[0] = req
                            req_cache[1] = offsets[req]
                            req_cache[2] = offsets[req + 1]
                            req_cache[3] = chunk_offsets[req]
                        bos = req_cache[1]
                        eos = req_cache[2]
                        # Chunks the tile's last token in this request has closed.
                        n_chunk = (T.min(t_last, eos - 1) - bos + 1) // bs
                        for i in T.Parallel(m):
                            tok = t0 + i // group
                            closed[i] = T.if_then_else(
                                (tok >= bos) & (tok < eos), (tok - bos + 1) // bs, 0
                            )
                        for cb in T.serial(T.ceildiv(n_chunk, bc)):
                            c0 = cb * bc
                            if (loaded[0] != req) | (loaded[1] != cb):
                                row = req_cache[3] + c0
                                # Rows past the tensor load as zeros; rows of the next
                                # request are masked below.
                                T.fence_proxy_async()
                                T.tma_copy(
                                    k_cmp[row : row + bc, i_h, 0:half],
                                    T.view(k_lo, [bc, half]),
                                    barrier=kv_full[0],
                                )
                                T.tma_copy(
                                    k_cmp[row : row + bc, i_h, half:dim_k],
                                    T.view(k_hi, [bc, half]),
                                    barrier=kv_full[0],
                                )
                                T.tma_copy(
                                    v_cmp[row : row + bc, i_h, 0:half],
                                    T.view(v_lo, [bc, half]),
                                    barrier=kv_full[0],
                                )
                                T.tma_copy(
                                    v_cmp[row : row + bc, i_h, half:dim_v],
                                    T.view(v_hi, [bc, half]),
                                    barrier=kv_full[0],
                                )
                                T.mbarrier_arrive(kv_full[0])
                                T.mbarrier_wait_parity(kv_full[0], kv_loads[0] % 2)
                                kv_loads[0] = kv_loads[0] + 1
                                loaded[0] = req
                                loaded[1] = cb
                            T.wgmma_gemm(
                                q_lo[st, :, :],
                                k_lo[0, :, :],
                                s0,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.wgmma_gemm(q_hi[st, :, :], k_hi[0, :, :], s0, transpose_B=True)
                            T.wgmma_gemm(
                                q_lo[st, :, :],
                                k_lo[1, :, :],
                                s1,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.wgmma_gemm(q_hi[st, :, :], k_hi[1, :, :], s1, transpose_B=True)
                            T.wait_wgmma(0)
                            for i, c in T.Parallel(m, kh):
                                s0[i, c] = T.if_then_else(
                                    c0 + c < closed[i],
                                    s0[i, c] * scale_log2,
                                    -T.infinity(accum_dtype),
                                )
                                s1[i, c] = T.if_then_else(
                                    c0 + kh + c < closed[i],
                                    s1[i, c] * scale_log2,
                                    -T.infinity(accum_dtype),
                                )
                            T.copy(scores_max, scores_max_prev)
                            T.reduce_max(s0, scores_max, dim=1, clear=False)
                            T.reduce_max(s1, scores_max, dim=1, clear=False)
                            for i in T.Parallel(m):
                                scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                            for i, c in T.Parallel(m, kh):
                                s0[i, c] = T.exp2(s0[i, c] - scores_max[i])
                                s1[i, c] = T.exp2(s1[i, c] - scores_max[i])
                            T.copy(s0, p0)
                            T.copy(s1, p1)
                            if first[0] == 1:
                                T.wgmma_gemm(p0, v_lo[0, :, :], o_lo, clear_accum=True)
                                T.wgmma_gemm(p1, v_lo[1, :, :], o_lo)
                                T.wgmma_gemm(p0, v_hi[0, :, :], o_hi, clear_accum=True)
                                T.wgmma_gemm(p1, v_hi[1, :, :], o_hi)
                            else:
                                for i, d in T.Parallel(m, half):
                                    o_lo[i, d] *= scores_scale[i]
                                    o_hi[i, d] *= scores_scale[i]
                                T.wgmma_gemm(p0, v_lo[0, :, :], o_lo)
                                T.wgmma_gemm(p1, v_lo[1, :, :], o_lo)
                                T.wgmma_gemm(p0, v_hi[0, :, :], o_hi)
                                T.wgmma_gemm(p1, v_hi[1, :, :], o_hi)
                            T.reduce_sum(s0, scores_sum, dim=1)
                            for i in T.Parallel(m):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            T.reduce_sum(s1, scores_sum, dim=1)
                            for i in T.Parallel(m):
                                logsum[i] = logsum[i] + scores_sum[i]
                            T.wait_wgmma(0)
                            first[0] = 0

                    # A row no closed chunk reaches writes zeros to both outputs.
                    for i in T.Parallel(m):
                        scores_scale[i] = T.if_then_else(logsum[i] > 0, 1.0 / logsum[i], 0)
                    if first[0] == 1:
                        T.clear(o_lo)
                        T.clear(o_hi)
                    for i, d in T.Parallel(m, half):
                        o_lo[i, d] *= scores_scale[i]
                        o_hi[i, d] *= scores_scale[i]
                    T.copy(o_lo, q_lo[st, :, :])
                    T.copy(o_hi, q_hi[st, :, :])
                    for i in T.Parallel(m):
                        lse_s[i] = T.if_then_else(
                            logsum[i] > 0, (scores_max[i] + T.log2(logsum[i])) / LOG2E, 0
                        )
                    T.copy(
                        q_lo_tiles[st, :, :, :], output[t0 : t0 + tile_t, h0 : h0 + group, 0:half]
                    )
                    T.copy(
                        q_hi_tiles[st, :, :, :],
                        output[t0 : t0 + tile_t, h0 : h0 + group, half:dim_v],
                    )
                    T.copy(
                        T.view(lse_s, [tile_t, group]), temp_lse[t0 : t0 + tile_t, h0 : h0 + group]
                    )

        return _main

    return _func


class NSACompressedFwdVarlenTMAKernel(NSACompressedFwdVarlenKernel):
    """SM90 compression attention: persistent token tiles, TMA rings and WGMMA."""

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"nsa_compressed_fwd_varlen_kernel"})

    @classmethod
    def refusal(cls, call: NSACall) -> Optional[str]:
        """Two 64-column halves per head and one m64 row block per tile."""
        if call.dim != 2 * _HALF or call.dim_v != 2 * _HALF:
            return f"head dims ({call.dim}, {call.dim_v}) are not two {_HALF}-column halves"
        group = call.heads // call.heads_kv
        if _TILE_ROWS % group != 0:
            return f"a group of {group} heads does not tile the {_TILE_ROWS}-row WGMMA block"
        return None

    @classmethod
    def entry_for(cls, call: NSACall) -> Entry:
        group = call.heads // call.heads_kv
        tile_t = _TILE_ROWS // group
        n_tiles = -(-call.c_seq_len // tile_t)
        # Spread the tiles over the CTAs one wave holds, with equal runs per worker.
        most = max(1, call.sm_count * _CTAS_PER_SM // call.heads_kv)
        per_worker = -(-n_tiles // most)
        workers = -(-n_tiles // per_worker)
        return call, lambda: cls(
            seq_num=call.batch,
            c_seq_len=call.c_seq_len,
            heads=call.heads,
            dim_k=call.dim,
            dim_v=call.dim_v,
            chunk_num=call.chunk_num,
            group=group,
            scale=call.scale,
            bc=2 * _KEY_HALF,
            bs=call.block_size,
            dtype=call.dtype,
            accum_dtype=torch.float32,
            workers=workers,
        )

    def __init__(self, *args, workers: int, **kwargs) -> None:
        self.workers = workers
        super().__init__(*args, **kwargs)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": 128}]

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attend each token to its request's closed chunks through the SM90 program."""
        return _nsa_compressed_fwd_varlen_tma_kernel(
            self.seq_num,
            self.c_seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_num,
            self.group,
            self.scale,
            self.bs,
            self.workers,
            self.dtype_str,
            self.accum_dtype_str,
        )(self.config["threads"])(
            q.to(self.dtype),
            k_cmp.to(self.dtype),
            v_cmp.to(self.dtype),
            offsets.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )
