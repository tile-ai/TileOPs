import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .online_softmax import (
    LOG2E,
    make_apply_softcap,
    make_online_softmax,
    make_rescale,
)

__all__ = ["GQADecodeKernel"]


_SPLIT_CANDIDATES = (1, 2, 4, 8, 16, 32)


def _effective_num_split(num_split: int, block_N: int, real_seqlen_kv: int) -> int:
    """Split count the runtime can use for a KV extent of *real_seqlen_kv*.

    The tuned or default ``num_split`` is only a ceiling: shrink it until every
    split keeps at least one full KV tile. ``1`` means the sequence is too
    short to split and the no-split kernel should run. Gating dispatch on the
    tuned value itself (the old ``real_seqlen_kv < num_split * block_N`` test)
    let a large tuned ``num_split`` push execution into the never-tuned
    no-split kernel.
    """
    return max(1, min(num_split, real_seqlen_kv // block_N))


def _effective_dense_num_split(num_split: int, block_N: int, real_seqlen_kv: int) -> int:
    """Map Dense decode to one of its finite autotune split candidates."""
    limit = _effective_num_split(num_split, block_N, real_seqlen_kv)
    return max(candidate for candidate in _SPLIT_CANDIDATES if candidate <= limit)


# JIT kernel: no-split variant


@functools.lru_cache(maxsize=32)
def _gqa_decode_no_split_kernel(
    batch,
    heads,
    groups,
    dim,
    sm_scale,
    softcap,
    dtype,
    fuse_rope=False,
    max_position=1,
    rotary_dim=0,
    rope_layout="neox",
):
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_H, block_N, num_stages, threads):
        seqlen_kv = T.dynamic("seqlen_kv")
        shape_q = [batch, heads, dim]
        shape_k = [batch, seqlen_kv, groups, dim]
        shape_v = [batch, seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        rope_shape = [max_position, rotary_dim // 2]
        kv_group_num = heads // groups

        valid_block_H = min(block_H, kv_group_num)

        online_softmax = make_online_softmax(scale, accum_dtype, block_H, block_N)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_H, block_N)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_H, dim)

        @T.macro
        def compute(Q, K, V, rope_cos, rope_sin, Output):
            with T.Kernel(batch, heads // valid_block_H, 1, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                if fuse_rope:
                    for i, j in T.Parallel(block_H, dim):
                        if i < valid_block_H:
                            if j < rotary_dim:
                                if rope_layout == "neox":
                                    freq = T.if_then_else(
                                        j < rotary_dim // 2, j, j - rotary_dim // 2
                                    )
                                    partner = T.if_then_else(
                                        j < rotary_dim // 2,
                                        j + rotary_dim // 2,
                                        j - rotary_dim // 2,
                                    )
                                    sign = T.if_then_else(j < rotary_dim // 2, -1.0, 1.0)
                                else:
                                    freq = j // 2
                                    partner = T.if_then_else(j % 2 == 0, j + 1, j - 1)
                                    sign = T.if_then_else(j % 2 == 0, -1.0, 1.0)
                                x = T.cast(Q[bid, hid * valid_block_H + i, j], "float")
                                x_partner = T.cast(
                                    Q[bid, hid * valid_block_H + i, partner], "float"
                                )
                                cos = T.cast(rope_cos[seqlen_kv - 1, freq], "float")
                                sin = T.cast(rope_sin[seqlen_kv - 1, freq], "float")
                                Q_shared[i, j] = T.cast(x * cos + sign * x_partner * sin, dtype)
                            else:
                                Q_shared[i, j] = Q[bid, hid * valid_block_H + i, j]
                        else:
                            Q_shared[i, j] = 0
                    T.sync_threads(3, threads)
                else:
                    T.copy(
                        Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :],
                        Q_shared,
                    )
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(seqlen_kv, block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if fuse_rope:
                        for i, j in T.Parallel(block_N, dim):
                            position = k * block_N + i
                            if position < seqlen_kv:
                                if j < rotary_dim:
                                    if rope_layout == "neox":
                                        freq = T.if_then_else(
                                            j < rotary_dim // 2, j, j - rotary_dim // 2
                                        )
                                        partner = T.if_then_else(
                                            j < rotary_dim // 2,
                                            j + rotary_dim // 2,
                                            j - rotary_dim // 2,
                                        )
                                        sign = T.if_then_else(j < rotary_dim // 2, -1.0, 1.0)
                                    else:
                                        freq = j // 2
                                        partner = T.if_then_else(j % 2 == 0, j + 1, j - 1)
                                        sign = T.if_then_else(j % 2 == 0, -1.0, 1.0)
                                    x = T.cast(K[bid, position, cur_kv_head, j], "float")
                                    x_partner = T.cast(
                                        K[bid, position, cur_kv_head, partner], "float"
                                    )
                                    cos = T.cast(rope_cos[position, freq], "float")
                                    sin = T.cast(rope_sin[position, freq], "float")
                                    K_shared[i, j] = T.cast(x * cos + sign * x_partner * sin, dtype)
                                else:
                                    K_shared[i, j] = K[bid, position, cur_kv_head, j]
                            else:
                                K_shared[i, j] = 0
                        T.sync_threads(3, threads)
                    else:
                        T.copy(
                            K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :],
                            K_shared,
                        )
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (k * block_N + j < seqlen_kv),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )
                    if use_softcap:
                        apply_softcap(acc_s)
                    online_softmax(
                        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.copy(V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

        if fuse_rope:

            @T.prim_func
            def gqa_decode_no_split_rope(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                rope_cos: T.Tensor(rope_shape, dtype),
                rope_sin: T.Tensor(rope_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
            ):
                compute(Q, K, V, rope_cos, rope_sin, Output)

            return gqa_decode_no_split_rope

        @T.prim_func
        def gqa_decode_no_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            compute(Q, K, V, Q, Q, Output)

        return gqa_decode_no_split

    return _func


# JIT kernel: split variant (split + combine)


@functools.lru_cache(maxsize=32)
def _gqa_decode_split_kernel(batch, heads, groups, dim, sm_scale, softcap, dtype):
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_H, block_N, num_split, num_stages, threads):
        seqlen_kv = T.dynamic("seqlen_kv")
        shape_q = [batch, heads, dim]
        shape_k = [batch, seqlen_kv, groups, dim]
        shape_v = [batch, seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        kv_group_num = heads // groups

        part_shape = [batch, heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)
        valid_block_N = block_N

        online_softmax_split = make_online_softmax(scale, accum_dtype, block_H, valid_block_N)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_H, valid_block_N)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_H, dim)

        @T.macro
        def _gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (
                bx,
                by,
                bz,
            ):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([valid_block_N, dim], dtype)
                V_shared = T.alloc_shared([valid_block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, valid_block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, valid_block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                # Partition whole KV tiles as evenly as possible (recipe from
                # gqa_decode_bs1): every CTA derives its own tile range from
                # the runtime sequence extent, so the autotuner times the same
                # distribution forward runs and no split is left empty.
                num_tiles = T.ceildiv(seqlen_kv, block_N)
                base_tiles = num_tiles // num_split
                extra_tiles = num_tiles % num_split
                tiles_this_split = base_tiles + T.if_then_else(sid < extra_tiles, 1, 0)
                tile_begin = sid * base_tiles + T.min(sid, extra_tiles)
                base = tile_begin * block_N
                this_len = T.max(T.min(seqlen_kv - base, tiles_this_split * block_N), 0)

                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(this_len, block_N)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        K[
                            bid,
                            base + k * valid_block_N : base + (k + 1) * valid_block_N,
                            cur_kv_head,
                            :,
                        ],
                        K_shared,
                    )
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i, j in T.Parallel(block_H, valid_block_N):
                        acc_s[i, j] = T.if_then_else(
                            (base + k * block_N + j < seqlen_kv),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )
                    if use_softcap:
                        apply_softcap(acc_s)
                    online_softmax_split(
                        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.copy(
                        V[
                            bid,
                            base + k * valid_block_N : base + (k + 1) * valid_block_N,
                            cur_kv_head,
                            :,
                        ],
                        V_shared,
                    )
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    # An empty split (this_len == 0) must not poison the combine
                    # with 0/0; its glse is -inf so it weighs nothing there.
                    acc_o[i, j] /= T.if_then_else(this_len > 0, logsum[i], 1.0)
                for i in T.Parallel(block_H):
                    logsum[i] = T.if_then_else(
                        this_len > 0,
                        T.log2(logsum[i]) + scores_max[i] * scale,
                        -T.infinity(accum_dtype),
                    )

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(
                    O_shared,
                    Output_partial[bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid, :],
                )

        @T.macro
        def combine(
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                #
                glse_vec = T.alloc_fragment([num_split], dtype)
                for k in T.Parallel(num_split):
                    glse_vec[k] = glse[bz, by, k]
                lse_max = T.alloc_fragment([1], accum_dtype)
                T.fill(lse_max, -T.infinity(accum_dtype))
                T.reduce_max(glse_vec, lse_max, dim=0, clear=False)

                #
                lse_logsum = T.alloc_local([1], accum_dtype)
                lse_logsum[0] = 0
                for k in T.serial(num_split):
                    lse_logsum[0] += T.exp2(glse[bz, by, k] - lse_max[0])
                lse_logsum[0] = T.log2(lse_logsum[0]) + lse_max[0]

                #
                o_accum = T.alloc_fragment([dim], accum_dtype)
                T.clear(o_accum)
                for k in T.serial(num_split):
                    w = T.exp2(glse[bz, by, k] - lse_logsum[0])
                    for i in T.Parallel(dim):
                        o_accum[i] += Output_partial[bz, by, k, i] * w
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum[i]

        @T.prim_func
        def gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_split(Q, K, V, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return gqa_decode_split

    return _func


# Custom ops (torch.compile compatible wrappers)


@torch.library.custom_op("tileops::gqa_decode_no_split_op", mutates_args=())
def _gqa_decode_no_split_op(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    return _gqa_decode_no_split_kernel(batch, heads, groups, dim, sm_scale, softcap, dtype)(
        block_H, block_N, num_stages, threads
    )(Q, K, V)


@_gqa_decode_no_split_op.register_fake
def _(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(Q)


@torch.library.custom_op("tileops::gqa_decode_no_split_rope_op", mutates_args=())
def _gqa_decode_no_split_rope_op(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    return _gqa_decode_no_split_kernel(
        batch,
        heads,
        groups,
        dim,
        sm_scale,
        softcap,
        dtype,
        True,
        max_position,
        rotary_dim,
        rope_layout,
    )(block_H, block_N, num_stages, threads)(Q, K, V, rope_cos, rope_sin)


@_gqa_decode_no_split_rope_op.register_fake
def _(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(Q)


@torch.library.custom_op("tileops::gqa_decode_split_op", mutates_args=())
def _gqa_decode_split_op(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    num_split: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
) -> torch.Tensor:
    return _gqa_decode_split_kernel(batch, heads, groups, dim, sm_scale, softcap, dtype)(
        block_H, block_N, num_split, num_stages, threads
    )(Q, K, V, glse, Output_partial)


@_gqa_decode_split_op.register_fake
def _(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_H: int,
    block_N: int,
    num_stages: int,
    threads: int,
    num_split: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(Q)


class GQADecodeKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]
    # The implementation behind the specialised ones for this key.
    general: bool = True

    @classmethod
    def applies(cls, call) -> bool:
        # The broad region: every contiguous decode call. The batch-1 kernel
        # states the narrower one it serves and wins wherever it applies.
        return True

    def __init__(
        self,
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        dtype="float16",
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune=False,
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.groups = heads_kv
        self.seqlen_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.fuse_rope = fuse_rope
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        from tileops.utils import get_sm_version

        arch = get_sm_version(device_index)
        if fuse_rope and arch != 90:
            raise ValueError("fused RoPE decode currently requires SM90")
        self.use_ws_rope = fuse_rope
        if self.groups <= 0:
            raise ValueError("heads_kv must be positive")
        if self.heads % self.groups != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if self.seqlen_kv <= 0:
            raise ValueError("seq_len_kv must be positive")

        self.no_split_jit = _gqa_decode_no_split_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
        )
        self.split_jit = _gqa_decode_split_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
        )
        # autotune targets the split kernel; forward shrinks the tuned
        # num_split to the runtime KV extent instead of gating dispatch on it
        self.kernel = self.split_jit
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    def _make_supply_prog(self):
        """Supply a representative value for the dynamic KV sequence extent."""
        from tilelang.utils.tensor import get_tensor_supply as _get_tensor_supply

        default_supply = _get_tensor_supply(tilelang.TensorSupplyType.Auto)
        seqlen_kv = self.seqlen_kv

        def supply_prog(params):
            inputs = []
            for param in params:
                if param.is_scalar():
                    inputs.append(seqlen_kv)
                else:
                    inputs.append(default_supply(param))
            return inputs

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    @property
    def default_config(self) -> dict:
        return {
            "block_H": 64,
            "block_N": 128,
            "num_split": 16,
            "num_stages": 2,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_N = [64, 128]
        block_H = [64]
        num_split = [1, 2, 4, 8, 16, 32]
        num_stages = [1, 2, 3]
        threads = [128]
        _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

        # Every split keeps at least one full KV tile, so the autotuner never
        # times a distribution the runtime cannot run; num_split=1 lets it
        # compare the unsplit strategy, which degenerates to no-split.
        configs = [
            {
                "block_N": c[0],
                "block_H": c[1],
                "num_split": c[2],
                "num_stages": c[3],
                "threads": c[4],
            }
            for c in _configs
            if c[2] <= max(1, self.seqlen_kv // c[0])
        ]
        return configs

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_cuda(q=q, k=k, v=v)
        del q_scale, k_scale, v_scale
        if self.fuse_rope and (rope_cos is None or rope_sin is None):
            raise ValueError("fused RoPE requires rope_cos and rope_sin")
        Q = q.squeeze(1)
        K = k
        V = v
        real_seqlen_kv = k.shape[1]
        block_H = self.config["block_H"]
        block_N = self.config["block_N"]
        num_stages = self.config["num_stages"]
        threads = self.config["threads"]
        # The tuned num_split is a ceiling: shrink it until every split keeps
        # one full KV tile. 1 means the sequence is too short to split.
        num_split = _effective_dense_num_split(self.config["num_split"], block_N, real_seqlen_kv)

        # Dispatch: no-split for sequences too short to give each split a tile
        if num_split == 1:
            if self.fuse_rope:
                output = _gqa_decode_no_split_rope_op(
                    self.batch,
                    self.heads,
                    self.groups,
                    self.dim,
                    self.sm_scale,
                    self.softcap,
                    self.dtype_str,
                    self.max_position,
                    self.rotary_dim,
                    self.rope_layout,
                    block_H,
                    block_N,
                    num_stages,
                    threads,
                    Q,
                    K,
                    V,
                    rope_cos,
                    rope_sin,
                )
                return output.unsqueeze(1)
            output = _gqa_decode_no_split_op(
                self.batch,
                self.heads,
                self.groups,
                self.dim,
                self.sm_scale,
                self.softcap,
                self.dtype_str,
                block_H,
                block_N,
                num_stages,
                threads,
                Q,
                K,
                V,
            )
            return output.unsqueeze(1)

        if self.use_ws_rope:
            # The Hopper producer/consumer kernel supports arbitrary batch
            # sizes; use it here so RoPE stays fused without replacing TMA and
            # WGMMA with scalar global-memory loads.
            from .gqa_decode_bs1 import _gqa_decode_bs1_ctx_op

            glse = torch.empty(
                (self.batch, self.heads, num_split), dtype=torch.float32, device=Q.device
            )
            Output_partial = torch.empty(
                (self.batch, self.heads, num_split, self.dim),
                dtype=torch.float32,
                device=Q.device,
            )
            output = _gqa_decode_bs1_ctx_op(
                self.batch,
                self.heads,
                self.groups,
                self.dim,
                self.sm_scale,
                self.softcap,
                self.dtype_str,
                True,
                self.max_position,
                self.rotary_dim,
                self.rope_layout,
                64,
                block_N,
                num_split,
                160,
                Q,
                K,
                V,
                rope_cos,
                rope_sin,
                glse,
                Output_partial,
            )
            return output.unsqueeze(1)

        # Split path: the kernel partitions KV tiles from the runtime extent
        glse = torch.empty((self.batch, self.heads, num_split), dtype=self.dtype, device=Q.device)
        Output_partial = torch.empty(
            (self.batch, self.heads, num_split, self.dim), dtype=self.dtype, device=Q.device
        )

        output = _gqa_decode_split_op(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            block_H,
            block_N,
            num_stages,
            threads,
            num_split,
            Q,
            K,
            V,
            glse,
            Output_partial,
        )
        return output.unsqueeze(1)


class GQADecodeLongContextKernel(GQADecodeKernel):
    """Dense decode specialization with the measured long-context defaults."""

    @property
    def default_config(self) -> dict:
        return {
            "block_H": 64,
            "block_N": 64,
            "num_split": 32,
            "num_stages": 2,
            "threads": 128,
        }
