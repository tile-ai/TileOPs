import functools
from typing import Callable, Optional

import tilelang
import torch
from tilelang import language as T
from tilelang.layout import make_swizzled_layout

from tileops.kernels.attention.call_spec import NSACall, NSAFwdInterface
from tileops.kernels.attention.online_softmax import make_online_softmax, make_rescale
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel


@functools.lru_cache(maxsize=32)
def _nsa_fwd_varlen_kernel(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    if scale is None:
        scale = (1.0 / dim) ** 0.5
    scale *= LOG2E

    @tilelang.jit(
        out_idx=[3],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _nsa_fwd_varlen_func(threads: int):
        head_kv = heads // groups
        q_shape = [c_seq_len, heads, dim]
        kv_shape = [c_seq_len, head_kv, dim]
        o_slc_shape = [c_seq_len, heads, dim]
        block_indices_shape = [c_seq_len, head_kv, selected_blocks]
        block_counts_shape = [c_seq_len, head_kv]
        offsets_shape = [batch + 1]
        token_indices_shape = [c_seq_len, 2]
        block_indices_dtype = T.int32
        block_counts_dtype = T.int32
        offsets_dtype = T.int32
        token_indices_dtype = T.int32

        block_s = block_size
        block_t = min(128, tilelang.math.next_power_of_2(dim))

        nk = tilelang.cdiv(dim, block_t)
        nv = tilelang.cdiv(dim, block_t)
        if nk != 1:
            raise ValueError("The key dimension can not be larger than 128")

        g = groups
        bs = block_s
        bk = bv = block_t
        num_stages = 4

        online_softmax = make_online_softmax(scale, accum_dtype, g, bs)
        rescale = make_rescale(g, bv)

        @T.prim_func
        def _nsa_fwd_varlen_main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            o_slc: T.Tensor(o_slc_shape, dtype),
            block_indices: T.Tensor(block_indices_shape, block_indices_dtype),
            block_counts: T.Tensor(block_counts_shape, block_counts_dtype),
            offsets: T.Tensor(offsets_shape, offsets_dtype),
            token_indices: T.Tensor(token_indices_shape, token_indices_dtype),
        ):
            with T.Kernel(c_seq_len, nv, head_kv, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([g, bk], dtype)
                k_shared = T.alloc_shared([bs, bk], dtype)
                v_shared = T.alloc_shared([bs, bv], dtype)
                o_shared = T.alloc_shared([g, bv], dtype)

                acc_s = T.alloc_fragment([g, bs], accum_dtype)
                acc_s_cast = T.alloc_fragment([g, bs], dtype)
                acc_o = T.alloc_fragment([g, bv], accum_dtype)
                scores_max = T.alloc_fragment([g], accum_dtype)
                scores_max_prev = T.alloc_fragment([g], accum_dtype)
                scores_scale = T.alloc_fragment([g], accum_dtype)
                scores_sum = T.alloc_fragment([g], accum_dtype)
                logsum = T.alloc_fragment([g], accum_dtype)

                i_c, i_v, i_bh = bx, by, bz
                _, i_h = i_bh // head_kv, i_bh % head_kv

                i_n, i_t = token_indices[i_c, 0], token_indices[i_c, 1]

                bos = offsets[i_n]
                # Causal: keys up to the token. Otherwise: up to the sequence end.
                limit = i_t + 1 if is_causal else offsets[i_n + 1] - bos

                ns = block_counts[bos + i_t, i_h]
                T.copy(q[bos + i_t, i_h * g : (i_h + 1) * g, :bk], q_shared)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for i in T.Pipelined(ns, num_stages=num_stages):
                    i_s = block_indices[bos + i_t, i_h, i] * bs
                    if i_s < limit and i_s >= 0:
                        # Rows past the bound load as zeros: a non-finite value survives the mask.
                        if i_s + bs <= limit:
                            T.copy(k[bos + i_s : bos + i_s + bs, i_h, :bk], k_shared)
                            T.copy(v[bos + i_s : bos + i_s + bs, i_h, :bv], v_shared)
                        else:
                            for j, d in T.Parallel(bs, bk):
                                if (i_s + j < limit) & (d < dim):
                                    k_shared[j, d] = k[bos + i_s + j, i_h, d]
                                    v_shared[j, d] = v[bos + i_s + j, i_h, d]
                                else:
                                    k_shared[j, d] = T.cast(0, dtype)
                                    v_shared[j, d] = T.cast(0, dtype)

                        for i, j in T.Parallel(g, bs):
                            acc_s[i, j] = T.if_then_else(
                                i_s + j < limit, 0, -T.infinity(acc_s.dtype)
                            )

                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                        online_softmax(
                            acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                        )
                        T.copy(acc_s, acc_s_cast)

                        rescale(acc_o, scores_scale)

                        # V * softmax(Q * K)
                        T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                # A token no selected block gives a key outputs zeros.
                for i, j in T.Parallel(g, bv):
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                T.copy(acc_o, o_shared)
                T.copy(
                    o_shared, o_slc[bos + i_t, i_h * g : (i_h + 1) * g, i_v * bv : (i_v + 1) * bv]
                )

        return _nsa_fwd_varlen_main

    return _nsa_fwd_varlen_func


@functools.lru_cache(maxsize=32)
def _nsa_fwd_varlen_tma_kernel(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    """One warp per (token, KV head) on SM90: tiles move by TMA, Q stays in registers.

    The live slots of a token form a bitmask walked lowest-first by two cursors: the
    load cursor runs ``stages - 1`` slots ahead of the compute cursor, so with more than
    one selected block the next K/V block is in flight while this one is attended. With
    one selected block the ring is a single stage: a second one would only hold smem.
    TMA zero-fills rows past the packed tensor; keys past the token's limit are masked.
    The output is staged in the Q tile, whose last reader is the register copy of Q.
    """
    if scale is None:
        scale = (1.0 / dim) ** 0.5
    scale *= LOG2E
    head_kv = heads // groups
    g = groups
    bs = block_size
    stages = min(2, selected_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _nsa_fwd_varlen_tma_func(threads: int):
        @T.prim_func
        def _nsa_fwd_varlen_tma_main(
            q: T.Tensor([c_seq_len, heads, dim], dtype),
            k: T.Tensor([c_seq_len, head_kv, dim], dtype),
            v: T.Tensor([c_seq_len, head_kv, dim], dtype),
            block_indices: T.Tensor([c_seq_len, head_kv, selected_blocks], T.int32),
            block_counts: T.Tensor([c_seq_len, head_kv], T.int32),
            offsets: T.Tensor([batch + 1], T.int32),
            token_indices: T.Tensor([c_seq_len, 2], T.int32),
            o_slc: T.Tensor([c_seq_len, heads, dim], dtype),
        ):
            with T.Kernel(c_seq_len, head_kv, threads=threads) as (i_c, i_h):
                q_shared = T.alloc_shared([g, dim], dtype)
                k_shared = T.alloc_shared([stages, bs, dim], dtype)
                v_shared = T.alloc_shared([stages, bs, dim], dtype)
                T.annotate_layout(
                    {
                        q_shared: make_swizzled_layout(q_shared),
                        k_shared: make_swizzled_layout(k_shared),
                        v_shared: make_swizzled_layout(v_shared),
                    }
                )
                q_ready = T.alloc_barrier([threads])
                kv_ready = T.alloc_barrier([threads] * stages)

                q_local = T.alloc_fragment([g, dim], dtype)
                acc_s = T.alloc_fragment([g, bs], accum_dtype)
                acc_s_cast = T.alloc_fragment([g, bs], dtype)
                acc_o = T.alloc_fragment([g, dim], accum_dtype)
                scores_max = T.alloc_fragment([g], accum_dtype)
                scores_max_prev = T.alloc_fragment([g], accum_dtype)
                scores_scale = T.alloc_fragment([g], accum_dtype)
                scores_sum = T.alloc_fragment([g], accum_dtype)
                logsum = T.alloc_fragment([g], accum_dtype)
                inv_logsum = T.alloc_fragment([g], accum_dtype)
                # Live slots not yet loaded, live slots not yet attended, and their count.
                to_load = T.alloc_local([1], T.int32)
                to_attend = T.alloc_local([1], T.int32)
                n_live = T.alloc_local([1], T.int32)

                i_t = token_indices[i_c, 1]
                # A packed token sits at its request's start plus its position.
                bos = i_c - i_t
                # Causal: keys up to the token. Otherwise: up to the request end.
                limit = i_t + 1 if is_causal else offsets[token_indices[i_c, 0] + 1] - bos
                ns = block_counts[i_c, i_h]

                T.tma_copy(q[i_c, i_h * g : (i_h + 1) * g, :], q_shared, barrier=q_ready[0])
                T.mbarrier_arrive(q_ready[0])

                to_load[0] = 0
                n_live[0] = 0
                for i in T.unroll(selected_blocks):
                    i_s = block_indices[i_c, i_h, i] * bs
                    if i < ns and i_s >= 0 and i_s < limit:
                        to_load[0] = to_load[0] | (1 << i)
                        n_live[0] = n_live[0] + 1
                to_attend[0] = to_load[0]

                for p in T.unroll(stages - 1):
                    if to_load[0] != 0:
                        row = bos + block_indices[i_c, i_h, T.__ffs(to_load[0]) - 1] * bs
                        to_load[0] = to_load[0] & (to_load[0] - 1)
                        T.tma_copy(
                            k[row : row + bs, i_h, :], k_shared[p, :, :], barrier=kv_ready[p]
                        )
                        T.tma_copy(
                            v[row : row + bs, i_h, :], v_shared[p, :, :], barrier=kv_ready[p]
                        )
                        T.mbarrier_arrive(kv_ready[p])

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.mbarrier_wait_parity(q_ready[0], 0)
                T.copy(q_shared, q_local)

                for j in T.serial(n_live[0]):
                    if to_load[0] != 0:
                        st_next = (j + stages - 1) % stages
                        row = bos + block_indices[i_c, i_h, T.__ffs(to_load[0]) - 1] * bs
                        to_load[0] = to_load[0] & (to_load[0] - 1)
                        # The stage was last read by this warp's own MMAs; order those
                        # generic-proxy reads before the TMA write lands.
                        T.fence_proxy_async()
                        T.tma_copy(
                            k[row : row + bs, i_h, :],
                            k_shared[st_next, :, :],
                            barrier=kv_ready[st_next],
                        )
                        T.tma_copy(
                            v[row : row + bs, i_h, :],
                            v_shared[st_next, :, :],
                            barrier=kv_ready[st_next],
                        )
                        T.mbarrier_arrive(kv_ready[st_next])
                    st = j % stages
                    i_s = block_indices[i_c, i_h, T.__ffs(to_attend[0]) - 1] * bs
                    to_attend[0] = to_attend[0] & (to_attend[0] - 1)
                    T.mbarrier_wait_parity(kv_ready[st], (j // stages) % 2)

                    T.clear(acc_s)
                    T.gemm(
                        q_local,
                        k_shared[st, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, s in T.Parallel(g, bs):
                        acc_s[i, s] = T.if_then_else(
                            i_s + s < limit, acc_s[i, s] * scale, -T.infinity(accum_dtype)
                        )
                    # A live block holds the key at its start, so every row's max is finite.
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(g):
                        scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                    for i, s in T.Parallel(g, bs):
                        acc_s[i, s] = T.exp2(acc_s[i, s] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(g):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, d in T.Parallel(g, dim):
                        acc_o[i, d] *= scores_scale[i]
                    T.gemm(acc_s_cast, v_shared[st, :, :], acc_o, policy=T.GemmWarpPolicy.FullRow)

                # A token no selected block gives a key outputs zeros.
                for i in T.Parallel(g):
                    inv_logsum[i] = T.if_then_else(logsum[i] > 0, 1.0 / logsum[i], 0)
                for i, d in T.Parallel(g, dim):
                    acc_o[i, d] = acc_o[i, d] * inv_logsum[i]
                T.copy(acc_o, q_shared)
                T.copy(q_shared, o_slc[i_c, i_h * g : (i_h + 1) * g, :])

        return _nsa_fwd_varlen_tma_main

    return _nsa_fwd_varlen_tma_func


class NSAFwdVarlenKernel(Kernel, NSAFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: NSACall) -> Entry:
        return call, lambda: cls(
            batch=call.batch,
            heads=call.heads,
            c_seq_len=call.c_seq_len,
            dim=call.dim,
            is_causal=call.is_causal,
            scale=call.scale,
            block_size=call.block_size,
            groups=call.heads // call.heads_kv,
            selected_blocks=call.selected_blocks,
            dtype=call.dtype,
            accum_dtype=torch.float32,
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        c_seq_len: int,
        dim: int,
        is_causal: bool,
        scale: float,
        block_size: int,
        groups: int,
        selected_blocks: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.c_seq_len = c_seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.groups = groups
        self.selected_blocks = selected_blocks
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
        threads = [
            32,
        ]
        return [{"threads": t} for t in threads]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        return _nsa_fwd_varlen_kernel(
            self.batch,
            self.heads,
            self.c_seq_len,
            self.dim,
            self.is_causal,
            self.scale,
            self.block_size,
            self.groups,
            self.selected_blocks,
            self.dtype_str,
            self.accum_dtype_str,
        )(self.config["threads"])(q, k, v, block_indices, block_counts, offsets, token_indices)


class NSAFwdVarlenTMAKernel(NSAFwdVarlenKernel):
    """SM90 NSA forward: TMA tiles, Q held in registers, a K/V ring over live slots."""

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"nsa_fwd_varlen_kernel"})

    @classmethod
    def refusal(cls, call: NSACall) -> Optional[str]:
        """Head dims the MMA K step tiles, and a slot mask that fits one int32."""
        if call.dim % 16 != 0:
            return f"head dim {call.dim} is not a multiple of the MMA K step 16"
        if call.selected_blocks >= 32:
            return f"{call.selected_blocks} selected blocks overflow the int32 slot mask"
        return None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Attend each token to its selected blocks through the SM90 TMA program."""
        return _nsa_fwd_varlen_tma_kernel(
            self.batch,
            self.heads,
            self.c_seq_len,
            self.dim,
            self.is_causal,
            self.scale,
            self.block_size,
            self.groups,
            self.selected_blocks,
            self.dtype_str,
            self.accum_dtype_str,
        )(self.config["threads"])(q, k, v, block_indices, block_counts, offsets, token_indices)
