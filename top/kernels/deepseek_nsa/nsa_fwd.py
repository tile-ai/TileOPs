# ruff: noqa
import torch
from typing import Optional, Any, Callable

import tilelang
from tilelang import language as T

from top.kernels.kernel import Kernel


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
        scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    @tilelang.jit(
        out_idx=[3],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
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
        assert nk == 1, "The key dimension can not be larger than 256"

        g = groups
        bs = block_s
        bk = bv = block_t
        num_stages = 4

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
            with T.Kernel(c_seq_len, nv, batch * head_kv, threads=threads) as (bx, by, bz):
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

                ns = block_counts[bos + i_t, i_h]
                T.copy(q[bos + i_t, i_h * g:(i_h + 1) * g, :bk], q_shared)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for i in T.Pipelined(ns, num_stages=num_stages):
                    i_s = block_indices[bos + i_t, i_h, i] * bs
                    if i_s <= i_t and i_s >= 0:
                        # [BS, BK]
                        # Lei: may have some padding issues
                        # we should learn from mha varlen templates to handle this
                        T.copy(k[bos + i_s:bos + i_s + bs, i_h, :bk], k_shared)

                        if is_causal:
                            for i, j in T.Parallel(g, bs):
                                acc_s[i, j] = T.if_then_else(i_t >= (i_s + j), 0,
                                                             -T.infinity(acc_s.dtype))
                        else:
                            T.clear(acc_s)

                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow)

                        # Softmax
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                        for i in T.Parallel(g):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                                     scores_max[i] * scale)
                        for i, j in T.Parallel(g, bs):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(g):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)

                        # Rescale
                        for i, j in T.Parallel(g, bv):
                            acc_o[i, j] *= scores_scale[i]

                        # V * softmax(Q * K)
                        T.copy(v[bos + i_s:bos + i_s + bs, i_h, i_v * bv:(i_v + 1) * bv], v_shared)
                        T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(g, bv):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o_slc[bos + i_t, i_h * g:(i_h + 1) * g, i_v * bv:(i_v + 1) * bv])

        return _nsa_fwd_varlen_main

    return _nsa_fwd_varlen_func


@torch.library.custom_op("top::nsa_fwd_varlen_wrapped_kernel", mutates_args=())
def _nsa_fwd_varlen_wrapped_kernel(
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
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    offsets: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    return _nsa_fwd_varlen_kernel(batch, heads, c_seq_len, dim, is_causal, scale, block_size,
                                  groups, selected_blocks, dtype,
                                  accum_dtype)(threads)(q, k, v, block_indices, block_counts,
                                                        offsets, token_indices)


@_nsa_fwd_varlen_wrapped_kernel.register_fake
def _(
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
    threads: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    # attention output shape [c_seq_len, heads, dim]
    _ = (batch, is_causal, scale, block_size, groups, selected_blocks, dtype, accum_dtype, threads)
    return torch.empty([c_seq_len, heads, dim], dtype=inputs[0].dtype, device=inputs[0].device)


class NSAFwdVarlenKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
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
                 tune: bool = False) -> None:

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
        self.dtype_name = str(dtype).split('.')[-1]
        self.accum_dtype_name = str(accum_dtype).split('.')[-1]

        self.init_config(config, tune)

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                block_indices: torch.Tensor, block_counts: torch.Tensor, offsets: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        return _nsa_fwd_varlen_wrapped_kernel(self.batch, self.heads, self.c_seq_len, self.dim,
                                              self.is_causal, self.scale, self.block_size,
                                              self.groups, self.selected_blocks, self.dtype_name,
                                              self.accum_dtype_name, self.config["threads"], q, k,
                                              v, block_indices, block_counts, offsets,
                                              token_indices)
