import copy
import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import AttentionCall, GQADenseFwdInterface
from tileops.kernels.attention.online_softmax import (
    make_apply_softcap,
    make_online_softmax,
    make_rescale,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin

__all__ = ["GQADecodeKernel"]


# JIT kernel: no-split variant


@functools.lru_cache(maxsize=32)
def gqa_decode_no_split_kernel(
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
        blocks_per_kv = -(-kv_group_num // valid_block_H)
        tail_rows = kv_group_num - (blocks_per_kv - 1) * valid_block_H

        online_softmax = make_online_softmax(scale, accum_dtype, block_H, block_N)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_H, block_N)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_H, dim)

        @T.macro
        def compute(Q, K, V, rope_cos, rope_sin, Output):
            # Let TileLang synchronize shared RoPE writes after warp specialization;
            # a fixed named barrier can alias the producer's generated barrier.
            with T.Kernel(batch, groups * blocks_per_kv, 1, threads=threads) as (bx, by, bz):
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
                cur_kv_head = hid // blocks_per_kv
                head_base = cur_kv_head * kv_group_num + hid % blocks_per_kv * valid_block_H
                rows = (
                    valid_block_H
                    if tail_rows == valid_block_H
                    else T.if_then_else(
                        hid % blocks_per_kv == blocks_per_kv - 1, tail_rows, valid_block_H
                    )
                )

                if fuse_rope:
                    for i, j in T.Parallel(block_H, dim):
                        if i < rows:
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
                                x = T.cast(Q[bid, head_base + i, j], "float")
                                x_partner = T.cast(Q[bid, head_base + i, partner], "float")
                                cos = T.cast(rope_cos[seqlen_kv - 1, freq], "float")
                                sin = T.cast(rope_sin[seqlen_kv - 1, freq], "float")
                                Q_shared[i, j] = T.cast(x * cos + sign * x_partner * sin, dtype)
                            else:
                                Q_shared[i, j] = Q[bid, head_base + i, j]
                        else:
                            Q_shared[i, j] = 0
                elif tail_rows == valid_block_H:
                    T.copy(Q[bid, head_base : head_base + block_H, :], Q_shared)
                else:
                    for i, j in T.Parallel(block_H, dim):
                        if i < rows:
                            Q_shared[i, j] = Q[bid, head_base + i, j]
                        else:
                            Q_shared[i, j] = 0
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
                if tail_rows == valid_block_H:
                    T.copy(O_shared, Output[bid, head_base : head_base + valid_block_H, :])
                else:
                    for i, j in T.Parallel(valid_block_H, dim):
                        if i < rows:
                            Output[bid, head_base + i, j] = O_shared[i, j]

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
        blocks_per_kv = -(-kv_group_num // valid_block_H)
        tail_rows = kv_group_num - (blocks_per_kv - 1) * valid_block_H
        valid_block_N = block_N

        online_softmax_split = make_online_softmax(scale, accum_dtype, block_H, valid_block_N)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_H, valid_block_N)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_H, dim)

        # Each split's log-sum-exp is float32: bfloat16 steps by 0.5 between 64 and 128,
        # which would misweigh the split in the combine. The partial outputs keep the
        # storage dtype, half the traffic of float32, and the combine sums them in float32.
        @T.macro
        def _gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(batch, groups * blocks_per_kv, num_split, threads=threads) as (
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
                cur_kv_head = hid // blocks_per_kv
                head_base = cur_kv_head * kv_group_num + hid % blocks_per_kv * valid_block_H
                rows = (
                    valid_block_H
                    if tail_rows == valid_block_H
                    else T.if_then_else(
                        hid % blocks_per_kv == blocks_per_kv - 1, tail_rows, valid_block_H
                    )
                )

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

                if tail_rows == valid_block_H:
                    T.copy(Q[bid, head_base : head_base + block_H, :], Q_shared)
                else:
                    for i, j in T.Parallel(block_H, dim):
                        if i < rows:
                            Q_shared[i, j] = Q[bid, head_base + i, j]
                        else:
                            Q_shared[i, j] = 0
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
                    if i < rows:
                        glse[bid, head_base + i, sid] = logsum[i]
                T.copy(acc_o[:valid_block_H, :], O_shared)
                if tail_rows == valid_block_H:
                    T.copy(
                        O_shared,
                        Output_partial[bid, head_base : head_base + valid_block_H, sid, :],
                    )
                else:
                    for i, j in T.Parallel(valid_block_H, dim):
                        if i < rows:
                            Output_partial[bid, head_base + i, sid, j] = O_shared[i, j]

        @T.macro
        def combine(
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                #
                glse_vec = T.alloc_fragment([num_split], accum_dtype)
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
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_split(Q, K, V, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return gqa_decode_split

    return _func


def gqa_decode_no_split_run(
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
    return gqa_decode_no_split_kernel(batch, heads, groups, dim, sm_scale, softcap, dtype)(
        block_H, block_N, num_stages, threads
    )(Q, K, V)


def gqa_decode_split_run(
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


class GQADecodeKernel(Kernel, GQADenseFwdInterface):
    supported_archs: list[int] = [80, 89, 90]
    # The implementation behind the specialised ones for this key.
    general: bool = True
    # The split counts the autotune sweep offers.
    _SPLIT_CANDIDATES = (1, 2, 4, 8, 16, 32)

    @classmethod
    def effective_num_split(cls, num_split: int, block_N: int, real_seqlen_kv: int) -> int:
        """The split candidate the runtime uses for a KV extent of *real_seqlen_kv*.

        The tuned or default ``num_split`` is only a ceiling: it shrinks until every split
        keeps at least one full KV tile, then to a candidate. ``1`` means the sequence is too
        short to fill two tiles.
        """
        limit = max(1, min(num_split, real_seqlen_kv // block_N))
        return max(candidate for candidate in cls._SPLIT_CANDIDATES if candidate <= limit)

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """The contiguous decode region and the limits the program builds within."""
        if not call.dense_decode_region:
            return "does not serve this call"
        if call.fuse_rope and call.arch != 90:
            return "fuses RoPE on SM90 only"
        return cls._limit_refusal(call.dim, call.seqlen_kv, call.arch)

    @staticmethod
    def _limit_refusal(dim: int, seq_len_kv: int, arch: int) -> Optional[str]:
        """Why the program cannot build this shape; read by the region and the constructor,
        which a caller can reach directly. D256 uses the narrow, one-stage tile.
        """
        if seq_len_kv <= 0:
            return "requires a non-empty KV cache"
        if dim == 256 and arch in (80, 89, 90):
            return None
        if dim % 16 != 0 or not 16 <= dim <= 128:
            return "requires head dimension a multiple of 16 in [16, 128], or 256"
        return None

    @classmethod
    def split_tier(cls, call) -> tuple:
        """The cache-length tier this program compiles a split count for."""
        full_tiles = max(1, call.seqlen_kv // 64)
        return (min(32, 1 << (full_tiles.bit_length() - 1)),)

    @staticmethod
    def construction_args(call: AttentionCall) -> dict:
        """The constructor arguments of the contiguous decode kernels, batch 1 included."""
        return dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            seq_len_kv=call.seqlen_kv,
            dim=call.dim,
            dtype=call.dtype,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            **call.rope_args,
            device_index=call.device.index if call.device is not None else None,
        )

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The cache length is taken at runtime: what it compiles for is its split tier."""
        args = cls.construction_args(call)
        identity = (*(v for k, v in args.items() if k != "seq_len_kv"), *cls.split_tier(call))
        return identity, lambda: cls(**args)

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
        self._is_sm90 = arch == 90
        self._shared_memory_optin = get_shared_memory_optin(device_index)
        if fuse_rope and arch != 90:
            raise ValueError("fused RoPE decode currently requires SM90")
        self.use_ws_rope = fuse_rope
        if self.groups <= 0:
            raise ValueError("heads_kv must be positive")
        if self.heads % self.groups != 0:
            raise ValueError("heads must be divisible by heads_kv")
        reason = self._limit_refusal(dim, seq_len_kv, arch)
        if reason is not None:
            raise ValueError(f"{type(self).__name__} {reason}")
        self._use_batched_config = (
            self._is_sm90
            and self.batch > 1
            and self.dim == 128
            and self.heads // self.groups <= 8
            and not self.fuse_rope
        )

        # Without RoPE, the arguments match gqa_decode_no_split_run's, so both share one
        # cached builder.
        rope = (True, max_position, rotary_dim, rope_layout) if fuse_rope else ()
        self.no_split_jit = gqa_decode_no_split_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            *rope,
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
        self.ws_rope_jit = None
        if self.use_ws_rope:
            # decode_bs1 imports this module.
            from tileops.kernels.attention.gqa.decode_bs1 import gqa_decode_bs1_ctx_kernel

            self.ws_rope_jit = gqa_decode_bs1_ctx_kernel(
                self.batch,
                self.heads,
                self.groups,
                self.dim,
                self.sm_scale,
                self.softcap,
                self.dtype_str,
                True,
                max_position,
                rotary_dim,
                rope_layout,
            )
        # Forward shrinks the tuned num_split to the runtime KV extent instead of gating
        # dispatch on it; autotune times each candidate on the program forward launches.
        self.kernel = self.split_jit
        self._supply_prog = self._make_supply_prog()
        self.init_config(config)

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
                    continue
                # K and V carry the symbolic extent in their shape; give it the same value.
                static = copy.copy(param)
                static.shape = [d if isinstance(d, int) else seqlen_kv for d in param.shape]
                inputs.append(default_supply(static))
            return inputs

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    @property
    def default_config(self) -> dict:
        if self.dim == 256:
            # The split program's query rows and one K and V tile of 64 keys: 98304 bytes.
            # If the no-split output rows exceed the device limit, forward uses two splits.
            return {"block_H": 64, "block_N": 64, "num_split": 16, "num_stages": 1, "threads": 128}
        if self._use_batched_config:
            head_ctas = self.batch * self.groups
            splits = max(1, (256 + head_ctas - 1) // head_ctas)
            return {
                "block_H": 64,
                "block_N": 64,
                "num_split": min(16, 1 << (splits - 1).bit_length()),
                "num_stages": 2,
                "threads": 128,
            }
        wide = {"block_H": 64, "block_N": 128, "num_split": 16, "num_stages": 2, "threads": 128}
        # One stage of 64 keys where shared memory cannot hold two of 128 (SM89 holds 99 KB).
        narrow = {**wide, "block_N": 64, "num_stages": 1}
        return wide if self._fits(wide, split=False) else narrow

    def _shared_bytes(self, config: dict, *, split: bool) -> int:
        """Shared memory one program allocates for *config*: the query rows, a K and a V
        tile per stage and, in the no-split program, the output rows.

        Both programs stage their output rows. TileLang 0.1.12 places them in the query
        rows' space, which the loop no longer reads, in the split program with two or more
        splits and, for SM90, in the no-split program too; forward launches the split
        program only with two or more.
        """
        rows = config["block_H"] + 2 * config["num_stages"] * config["block_N"]
        if not split:
            rows += min(config["block_H"], self.heads // self.groups)
        return rows * self.dim * getattr(torch, self.dtype_str).itemsize

    def _fits(self, config: dict, *, split: bool) -> bool:
        return self._shared_bytes(config, split=split) <= self._shared_memory_optin

    def _launch_split(self, config: dict, seqlen_kv: int) -> int:
        """The split count forward launches for *config* over *seqlen_kv* keys; ``1`` runs
        the no-split program.

        Where the no-split program's output rows do not fit, two splits run instead, the
        second possibly empty.
        """
        num_split = self.effective_num_split(config["num_split"], config["block_N"], seqlen_kv)
        if num_split == 1 and not self._fits(config, split=False):
            return 2
        return num_split

    def _launched_program(self, config: dict, num_split: int) -> tuple[str, dict]:
        """The program forward launches for *config* at *num_split* splits, and the
        parameters it takes."""
        if num_split == 1:
            return "no_split", {key: value for key, value in config.items() if key != "num_split"}
        if self.use_ws_rope:
            # The warp-specialized program runs one warpgroup of 64 head rows and a
            # producer warp; it takes only the tile length and the split count.
            params = {"block_M": 64, "block_N": config["block_N"], "ctx_splits": num_split}
            return "ws_rope", {**params, "threads": 160}
        return "split", {**config, "num_split": num_split}

    def _config_of(self, program: str, params: dict) -> dict:
        """The kernel config whose launch runs *program* with *params*."""
        if program == "no_split":
            return {**params, "num_split": 1}
        if program == "ws_rope":
            return {**self.config, "block_N": params["block_N"], "num_split": params["ctx_splits"]}
        return params

    @property
    def autotune_configs(self) -> list[dict]:
        block_N = [64, 128]
        block_H = [64]
        num_split = [1, 2, 4, 8, 16, 32]
        num_stages = [1, 2, 3]
        threads = [128]
        _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

        # Every split keeps at least one full KV tile, and every candidate fits the
        # shared memory of the program forward launches for it.
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
        return [
            config
            for config in configs
            if self._fits(config, split=self._launch_split(config, self.seqlen_kv) > 1)
        ]

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        """Time each candidate on the program forward launches for it, and keep the fastest.

        A candidate whose split count falls to one runs the no-split program; every other
        candidate runs the split program, or with fused RoPE the warp-specialized one, at
        the split count forward launches.
        """
        print(f"Start autotuning {self.__class__.__name__}...")
        jits = {"no_split": self.no_split_jit, "split": self.split_jit, "ws_rope": self.ws_rope_jit}
        programs: dict[str, list[dict]] = {program: [] for program in jits}
        for config in self.tuning_candidates(self.autotune_configs):
            program, params = self._launched_program(
                config, self._launch_split(config, self.seqlen_kv)
            )
            if params not in programs[program]:
                programs[program].append(params)

        best_latency, best_config = float("inf"), None
        for program, candidates in programs.items():
            if not candidates:
                continue
            tuned = self.tune_jit_kernel(
                jits[program], candidates, warmup=warmup, rep=rep, seed_config=candidates[0]
            )
            if tuned.latency < best_latency:
                best_latency = tuned.latency
                best_config = self._config_of(program, tuned.config)
        if best_config is not None:
            self.config = best_config
        print(f"Best config: {self.config}")

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
        self._require_devices(q=q, k=k, v=v)
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
        num_split = self._launch_split(self.config, real_seqlen_kv)

        # Dispatch: no-split for sequences too short to give each split a tile
        if num_split == 1:
            if self.fuse_rope:
                output = self.no_split_jit(block_H, block_N, num_stages, threads)(
                    Q, K, V, rope_cos, rope_sin
                )
                return output.unsqueeze(1)
            output = gqa_decode_no_split_run(
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
            # The SM90 producer/consumer kernel supports arbitrary batch
            # sizes; use it here so RoPE stays fused without replacing TMA and
            # WGMMA with scalar global-memory loads.
            glse = torch.empty(
                (self.batch, self.heads, num_split), dtype=torch.float32, device=Q.device
            )
            Output_partial = torch.empty(
                (self.batch, self.heads, num_split, self.dim),
                dtype=torch.float32,
                device=Q.device,
            )
            _, params = self._launched_program(self.config, num_split)
            output = self.ws_rope_jit(**params)(Q, K, V, rope_cos, rope_sin, glse, Output_partial)
            return output.unsqueeze(1)

        # Split path: the kernel partitions KV tiles from the runtime extent
        glse = torch.empty(
            (self.batch, self.heads, num_split), dtype=torch.float32, device=Q.device
        )
        Output_partial = torch.empty(
            (self.batch, self.heads, num_split, self.dim), dtype=self.dtype, device=Q.device
        )

        output = gqa_decode_split_run(
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

    general: bool = False
    preferred_over = frozenset({"gqa_dense_decode_bs1"})

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """The one decode shape the long-context split serves."""
        served = (
            call.dense_decode_region
            and not call.fuse_rope
            and call.seqlen_kv >= 1024
            and call.batch == 1
            and call.heads == 32
            and call.heads_kv == 4
            and call.dim == 128
            and call.dtype == torch.float16
            and call.softcap == 0.0
        )
        return None if served else "does not serve this call"

    @classmethod
    def split_tier(cls, call) -> tuple:
        """As the general decode tier, plus the SM90 tile tier it also compiles."""
        tier = super().split_tier(call)
        if call.arch == 90:
            return (*tier, cls.sequence_bucket(call.seqlen_kv))
        return tier

    @staticmethod
    def sequence_bucket(seq_len_kv: int) -> int:
        """Tile-size tier for long-context decode."""
        return int(seq_len_kv > 131072)

    @property
    def default_config(self) -> dict:
        return {
            "block_H": 64,
            "block_N": 128 if self._is_sm90 and self.sequence_bucket(self.seqlen_kv) else 64,
            "num_split": 32,
            "num_stages": 2,
            "threads": 128,
        }
