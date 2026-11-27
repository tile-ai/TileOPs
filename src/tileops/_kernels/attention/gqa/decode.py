import itertools

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import AttentionCall, GQADenseFwdInterface
from tileops._kernels.attention.gqa.decode_bs1_common import (
    allocate_partials,
    balanced_split,
    make_gqa_decode_bs1_combine,
    rope_partner,
)
from tileops._kernels.attention.online_softmax import (
    make_apply_softcap,
    make_dense_sink_scale,
    make_online_softmax,
    make_rescale,
)
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import LOG2E
from tileops._kernels.kernel_base import Entry, Kernel, dynamic_extent_supply

__all__ = ["GQADecodeKernel", "GQADecodeNarrowGroupKernel"]


@target_cache(maxsize=32)
@tilelang.jit(out_idx=[3])
def _narrow_group_decode(groups: int, parts: int, score_scale: float, keys_per_part: int):
    head_dim = 128
    heads_per_group = 8
    heads = groups * heads_per_group
    cache_length = parts * keys_per_part
    dtype = "float16"

    @T.prim_func
    def attend(
        query: T.Tensor((1, 1, heads, head_dim), dtype),
        key: T.Tensor((1, cache_length, groups, head_dim), dtype),
        value: T.Tensor((1, cache_length, groups, head_dim), dtype),
        output: T.Tensor((1, 1, heads, head_dim), dtype),
    ):
        partial = T.alloc_global((heads, parts, head_dim), "float32")
        stats = T.alloc_global((heads, parts, 2), "float32")
        with T.Kernel(parts, groups, threads=64) as (part, group):
            q = T.alloc_shared((heads_per_group, head_dim), dtype)
            k = T.alloc_shared((keys_per_part, head_dim), dtype)
            v = T.alloc_shared((keys_per_part, head_dim), dtype)
            scores = T.alloc_fragment((keys_per_part, heads_per_group), "float32")
            probabilities = T.alloc_shared((keys_per_part, heads_per_group), dtype)
            product = T.alloc_fragment((head_dim, heads_per_group), "float32")
            peaks = T.alloc_fragment((heads_per_group,), "float32")
            sums = T.alloc_fragment((heads_per_group,), "float32")
            for h, d in T.Parallel(heads_per_group, head_dim):
                q[h, d] = query[0, 0, group * heads_per_group + h, d]
            for i, d in T.Parallel(keys_per_part, head_dim):
                k[i, d] = key[0, part * keys_per_part + i, group, d]
                v[i, d] = value[0, part * keys_per_part + i, group, d]
            # Put the eight query heads on the narrow axis of both MMA products.
            T.gemm(k, q, scores, transpose_B=True, clear_accum=True)
            for i, h in T.Parallel(keys_per_part, heads_per_group):
                scores[i, h] *= score_scale
            T.reduce_max(scores, peaks, dim=0)
            for i, h in T.Parallel(keys_per_part, heads_per_group):
                scores[i, h] = T.exp2(scores[i, h] - peaks[h])
                probabilities[i, h] = T.cast(scores[i, h], dtype)
            T.reduce_sum(scores, sums, dim=0)
            T.gemm(v, probabilities, product, transpose_A=True, clear_accum=True)
            for d, h in T.Parallel(head_dim, heads_per_group):
                partial[group * heads_per_group + h, part, d] = product[d, h]
            for h in T.Parallel(heads_per_group):
                stats[group * heads_per_group + h, part, 0] = peaks[h]
                stats[group * heads_per_group + h, part, 1] = sums[h]

        with T.Kernel(heads, threads=128) as head:
            peaks = T.alloc_fragment((parts,), "float32")
            sums = T.alloc_fragment((parts,), "float32")
            peak = T.alloc_fragment((1,), "float32")
            denom = T.alloc_fragment((1,), "float32")
            acc = T.alloc_fragment((head_dim,), "float32")
            weights = T.alloc_shared((parts,), "float32")
            for part in T.Parallel(parts):
                peaks[part] = stats[head, part, 0]
            T.reduce_max(peaks, peak, dim=0)
            for part in T.Parallel(parts):
                sums[part] = T.exp2(peaks[part] - peak[0])
                weights[part] = sums[part]
                sums[part] *= stats[head, part, 1]
            T.reduce_sum(sums, denom, dim=0)
            T.clear(acc)
            for part in T.serial(parts):
                for d in T.Parallel(head_dim):
                    acc[d] += weights[part] * partial[head, part, d]
            for d in T.Parallel(head_dim):
                output[0, 0, head, d] = T.cast(acc[d] / denom[0], dtype)

    return attend


class GQADecodeNarrowGroupKernel(Kernel, GQADenseFwdInterface):
    """Decode eight FP16 heads per KV head with transposed score tiles."""

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_dense_decode_split"})

    @classmethod
    def refusal(cls, call: AttentionCall) -> str | None:
        """Serve single-query decode when the KV tiles fill half to one SM90 wave."""
        if call.dense_decode_program != "split" or call.return_lse or call.batch != 1:
            return "requires single-request split decode with output only"
        if call.dtype != torch.float16 or call.dim != 128 or call.heads != 8 * call.heads_kv:
            return "requires eight FP16 query heads of width 128 per KV head"
        if (
            call.fuse_rope
            or call.has_sinks
            or call.softcap
            or call.sm_scale not in (None, 128**-0.5)
        ):
            return "serves plain attention at the default scale"
        parts, tail = divmod(call.seqlen_kv, cls._keys_per_part(call.seqlen_kv))
        if (
            tail
            or parts not in (16, 32)
            or not call.sm_count // 2 <= parts * call.heads_kv <= call.sm_count
        ):
            return "requires a half to one wave of whole short context tiles"
        return super().refusal(call)

    @staticmethod
    def _keys_per_part(seqlen_kv: int) -> int:
        """Keys a CTA of one KV head attends to: 128 from 2048 keys, 32 below."""
        return 128 if seqlen_kv >= 2048 else 32

    def __init__(self, call: AttentionCall) -> None:
        super().__init__(device_index=call.device_index)
        self.groups = call.heads_kv
        self.keys_per_part = self._keys_per_part(call.seqlen_kv)
        self.parts = call.seqlen_kv // self.keys_per_part
        self.scale = call.dim**-0.5 * LOG2E

    def _compile(self, _config: dict):
        return _narrow_group_decode(self.groups, self.parts, self.scale, self.keys_per_part)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, k=k, v=v)
        return self.program(self._compile)(q, k, v)


@target_cache(maxsize=32)
def gqa_decode_kernel(
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
    has_sinks=False,
    split=False,
):
    """The decode builder. With *split*, each CTA scans one of ``num_split`` balanced parts of
    the keys into ``glse`` and ``Output_partial``, and a second launch combines the parts;
    RoPE is fused into the unsplit program only."""
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else sm_scale * LOG2E
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def _func(block_H, block_N, num_stages, threads, num_split=1):
        seqlen_kv = T.dynamic("seqlen_kv")
        shape_q = [batch, heads, dim]
        shape_k = [batch, seqlen_kv, groups, dim]
        shape_v = [batch, seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        part_shape = [batch, heads, num_split, dim]
        sinks_shape = [heads] if has_sinks else shape_q
        sinks_dtype = "float32" if has_sinks else dtype
        rope_shape = [max_position, rotary_dim // 2]
        kv_group_num = heads // groups

        valid_block_H = min(block_H, kv_group_num)
        blocks_per_kv = -(-kv_group_num // valid_block_H)
        tail_rows = kv_group_num - (blocks_per_kv - 1) * valid_block_H

        online_softmax = make_online_softmax(scale, accum_dtype, block_H, block_N)
        apply_softcap = (
            make_apply_softcap(sm_scale, softcap, accum_dtype, block_H, block_N)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_H, dim)
        sink_scale = make_dense_sink_scale(scale, block_H)

        # Each split's log-sum-exp is float32: bfloat16 steps by 0.5 between 64 and 128,
        # which would misweigh the split in the combine. The partial outputs keep the
        # storage dtype, half the traffic of float32, and the combine sums them in float32.
        @T.macro
        def attend(Q, K, V, rope_cos, rope_sin, sinks, Output, glse):
            # Let TileLang synchronize shared RoPE writes after warp specialization;
            # a fixed named barrier can alias the producer's generated barrier.
            with T.Kernel(batch, groups * blocks_per_kv, num_split, threads=threads) as (
                bx,
                by,
                bz,
            ):
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

                if split:
                    # Every CTA derives its tile range from the runtime sequence extent, so
                    # the autotuner times the same distribution forward runs.
                    base, this_len = balanced_split(seqlen_kv, block_N, num_split, sid)
                else:
                    base = 0

                if fuse_rope:
                    for i, j in T.Parallel(block_H, dim):
                        if i < rows:
                            if j < rotary_dim:
                                freq, partner, sign = rope_partner(rope_layout, rotary_dim, j)
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

                loop_range = T.ceildiv(this_len if split else seqlen_kv, block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    keys = slice(base + k * block_N, base + (k + 1) * block_N)
                    if fuse_rope:
                        for i, j in T.Parallel(block_N, dim):
                            position = base + k * block_N + i
                            if position < seqlen_kv:
                                if j < rotary_dim:
                                    freq, partner, sign = rope_partner(rope_layout, rotary_dim, j)
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
                        T.copy(K[bid, keys, cur_kv_head, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (base + k * block_N + j < seqlen_kv),
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
                    T.copy(V[bid, keys, cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if split:
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
                else:
                    if has_sinks:
                        sink_scale(logsum, scores_max, sinks, scores_scale, head_base, 1, rows)
                        for i, j in T.Parallel(block_H, dim):
                            acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                T.copy(acc_o[:valid_block_H, :], O_shared)
                if tail_rows == valid_block_H:
                    if split:
                        T.copy(O_shared, Output[bid, head_base : head_base + valid_block_H, sid, :])
                    else:
                        T.copy(O_shared, Output[bid, head_base : head_base + valid_block_H, :])
                else:
                    for i, j in T.Parallel(valid_block_H, dim):
                        if i < rows:
                            if split:
                                Output[bid, head_base + i, sid, j] = O_shared[i, j]
                            else:
                                Output[bid, head_base + i, j] = O_shared[i, j]

        if split:
            combine = make_gqa_decode_bs1_combine(
                batch, heads, num_split, dim, dtype, accum_dtype, has_sinks=has_sinks
            )

            @T.prim_func
            def gqa_decode_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                sinks: T.Tensor(sinks_shape, sinks_dtype),
                Output: T.Tensor(shape_o, dtype),
            ):
                attend(Q, K, V, Q, Q, sinks, Output_partial, glse)
                combine(glse, Output_partial, Output, sinks)

            return gqa_decode_split

        if fuse_rope:

            @T.prim_func
            def gqa_decode_no_split_rope(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                rope_cos: T.Tensor(rope_shape, dtype),
                rope_sin: T.Tensor(rope_shape, dtype),
                sinks: T.Tensor(sinks_shape, sinks_dtype),
                Output: T.Tensor(shape_o, dtype),
            ):
                attend(Q, K, V, rope_cos, rope_sin, sinks, Output, None)

            return gqa_decode_no_split_rope

        @T.prim_func
        def gqa_decode_no_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            sinks: T.Tensor(sinks_shape, sinks_dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            attend(Q, K, V, Q, Q, sinks, Output, None)

        return gqa_decode_no_split

    return _func


class GQADecodeKernel(Kernel, GQADenseFwdInterface):
    supported_archs: list[int] = [80, 89, 90]
    # The implementation behind the specialised ones for this key.
    general: bool = True
    # The split counts the autotune sweep offers.
    _SPLIT_CANDIDATES = (1, 2, 4, 8, 16, 32)
    # The architectures whose programs fold attention sinks into the softmax.
    _SINK_ARCHS = (89, 90)

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
    def refusal(cls, call: AttentionCall) -> str | None:
        """The contiguous decode region and the limits the program builds within."""
        if call.return_lse:
            return "does not return the log-sum-exp"
        if not call.dense_decode_region:
            return "does not serve this call"
        if call.has_sinks and call.arch not in cls._SINK_ARCHS:
            return "attention sinks require SM89 or SM90"
        if call.fuse_rope and call.arch != 90:
            return "fuses RoPE on SM90 only"
        return cls._limit_refusal(call.dim, call.seqlen_kv, call.arch)

    @staticmethod
    def _limit_refusal(dim: int, seq_len_kv: int, arch: int) -> str | None:
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
            has_sinks=call.has_sinks,
            **call.rope_args,
            device_index=call.device_index,
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
        sm_scale: float | None = None,
        softcap: float = 0.0,
        config: dict | None = None,
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        has_sinks: bool = False,
        device_index: int | None = None,
    ):
        super().__init__(device_index=device_index)
        self.has_sinks = has_sinks
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
        arch = self.build_target.arch
        if has_sinks and arch not in self._SINK_ARCHS:
            raise ValueError("attention sinks require SM89 or SM90")
        self._is_sm90 = arch == 90
        self._shared_memory_optin = self.build_target.smem_budget
        if fuse_rope and arch != 90:
            raise ValueError("fused RoPE decode requires SM90")
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

        # Without RoPE, the arguments match the build in ``_compile``, so both share one
        # cached builder.
        rope = (True, max_position, rotary_dim, rope_layout) if fuse_rope else ()
        self.no_split_jit = gqa_decode_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            *rope,
            has_sinks=self.has_sinks,
        )
        self.split_jit = gqa_decode_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            has_sinks=self.has_sinks,
            split=True,
        )
        self.ws_rope_jit = None
        if fuse_rope:
            # decode_bs1 imports this module.
            from tileops._kernels.attention.gqa.decode_bs1 import gqa_decode_bs1_ctx_kernel

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
                has_sinks=self.has_sinks,
            )
        # Forward shrinks the tuned num_split to the runtime KV extent instead of gating
        # dispatch on it; autotune times each candidate on the program forward launches.
        self.kernel = self.split_jit
        self.init_config(config)

    @property
    def autotune_supply_prog(self):
        return dynamic_extent_supply(self.seqlen_kv)

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

        The split program stages its output rows in the query rows' space; the no-split count
        includes them, so it never undercounts.
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
        if self.fuse_rope:
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
        num_stages = [1, 2, 3]
        threads = [128]
        _configs = list(
            itertools.product(block_N, block_H, self._SPLIT_CANDIDATES, num_stages, threads)
        )

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
            self._publish_tuned(best_config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, k=k, v=v)
        del q_scale, k_scale, v_scale
        if self.fuse_rope and (rope_cos is None or rope_sin is None):
            raise ValueError("fused RoPE requires rope_cos and rope_sin")
        Q = q.squeeze(1)
        num_split = self._launch_split(self.config, k.shape[1])
        program = self.program(self._compile, num_split, sinks is not None)

        # Dispatch: no-split for sequences too short to give each split a tile
        if num_split == 1:
            if self.fuse_rope:
                output = program(Q, k, v, rope_cos, rope_sin, sinks if self.has_sinks else Q)
            else:
                output = program(Q, k, v, sinks if sinks is not None else Q)
            return output.unsqueeze(1)

        if self.fuse_rope:
            # The SM90 producer/consumer kernel serves any batch, so RoPE stays fused
            # without replacing TMA and WGMMA with scalar global-memory loads.
            glse, Output_partial = allocate_partials(Q, self.batch, self.heads, self.dim, num_split)
            output = program(
                Q, k, v, rope_cos, rope_sin, glse, Output_partial, sinks if self.has_sinks else Q
            )
            return output.unsqueeze(1)

        # Split path: the kernel partitions KV tiles from the runtime extent
        glse = torch.empty(
            (self.batch, self.heads, num_split), dtype=torch.float32, device=Q.device
        )
        Output_partial = torch.empty(
            (self.batch, self.heads, num_split, self.dim), dtype=self.dtype, device=Q.device
        )
        output = program(Q, k, v, glse, Output_partial, sinks if sinks is not None else Q)
        return output.unsqueeze(1)

    def _compile(self, cfg: dict, num_split: int, with_sinks: bool):
        """The program forward launches for *cfg* at *num_split* splits; *with_sinks* says
        whether the call passes sinks, which the programs without RoPE are built for."""
        program, params = self._launched_program(cfg, num_split)
        if program == "ws_rope":
            return self.ws_rope_jit(**params)
        launch = (cfg["block_H"], cfg["block_N"], cfg["num_stages"], cfg["threads"], num_split)
        if self.fuse_rope:
            return self.no_split_jit(*launch)
        return gqa_decode_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            has_sinks=with_sinks,
            split=num_split > 1,
        )(*launch)


class GQADecodeSplitKernel(GQADecodeKernel):
    """The general decode program with its keys in 32 parts, for the calls
    :attr:`AttentionCall.dense_decode_program` names ``"split"``."""

    general: bool = False
    # The region is fitted to SM90.
    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_dense_decode_ws"})
    # The program's 64-row block of a KV head's group, its base key tile and its key parts,
    # which the region in ``AttentionCall.dense_decode_program`` counts CTAs by.
    HEAD_BLOCK = 64
    KEY_TILE = 64
    PARTS = 32

    @classmethod
    def refusal(cls, call: AttentionCall) -> str | None:
        """The split's region, within what the general decode program serves."""
        if call.return_lse:
            return "does not return the log-sum-exp"
        if call.dense_decode_program != "split":
            return "does not serve this call"
        return super().refusal(call)

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
            "block_H": self.HEAD_BLOCK,
            "block_N": 128
            if self._is_sm90 and self.sequence_bucket(self.seqlen_kv)
            else self.KEY_TILE,
            "num_split": self.PARTS,
            "num_stages": 2,
            "threads": 128,
        }
