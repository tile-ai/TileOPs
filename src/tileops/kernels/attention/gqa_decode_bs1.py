"""Warp-specialized batch=1 GQA decode kernel (Hopper), context-split.

``GQADecodeBs1Kernel`` dispatches on the runtime K/V sequence extent.  Full-dimensional
RoPE always uses the context-only warp-specialized split; plain and partial-RoPE calls
retain the generic single-kernel path below their measured or established crossovers.
The split path has a TMA producer feeding a four-warp WGMMA consumer, exp2-domain online
softmax, and FP32 partial reduction through a combine kernel.  Full-dimensional RoPE
expands the producer into a warpgroup so lookup-table loads and K rotation overlap the
consumer.  Hopper-only, low-level ``tma_copy`` / ``mbarrier`` / ``wgmma_gemm``.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.gqa_decode import (
    _gqa_decode_no_split_op,
    _gqa_decode_no_split_rope_op,
)
from tileops.kernels.kernel_base import Kernel

from .call_spec import decode_bs1_region
from .gqa_decode_bs1_common import (
    COMPILE_FLAGS,
    RING_DEPTH,
    make_gqa_decode_bs1_combine,
)
from .online_softmax import LOG2E

__all__ = ["GQADecodeBs1Kernel"]

CONSUMER_THREADS = 128
TMA_THREADS = 32
ROPE_PIPELINE_THREADS = 256


def _make_dense_decode_split(
    batch: int,
    groups: int,
    block_m: int,
    block_n: int,
    dim: int,
    dtype: str,
    scale: float,
    kv_group_num: int,
    ctx_splits: int,
    threads: int,
    accum_dtype: str,
    real_seqlen_is_buffer: bool,
    load_kv,
    fuse_rope: bool = False,
    rotary_dim: int = 0,
    rope_layout: str = "neox",
):
    """Create the Dense context-split schedule."""
    ring_depth = RING_DEPTH

    @T.macro
    def consumer(
        Q,
        bid,
        hid,
        sid,
        this_len,
        base,
        seqlen_kv,
        loop_range,
        rope_cos,
        rope_sin,
        Qs,
        Ks,
        Vs,
        Ps,
        ready,
        free,
        acc_s,
        acc_o,
        sm,
        smp,
        alpha,
        ss,
        logsum,
        glse,
        Output_partial,
    ):
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(sm, -T.infinity(accum_dtype))
        if fuse_rope:
            for i, j in T.Parallel(block_m, dim):
                if i < kv_group_num:
                    if j < rotary_dim:
                        if rope_layout == "neox":
                            freq = T.if_then_else(j < rotary_dim // 2, j, j - rotary_dim // 2)
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
                        x = Q[bid, hid * kv_group_num + i, j]
                        x_partner = Q[bid, hid * kv_group_num + i, partner]
                        cos = rope_cos[seqlen_kv - 1, freq]
                        sin = rope_sin[seqlen_kv - 1, freq]
                        Qs[i, j] = x * cos + sign * x_partner * sin
                    else:
                        Qs[i, j] = Q[bid, hid * kv_group_num + i, j]
                else:
                    Qs[i, j] = 0
            T.sync_threads(3, CONSUMER_THREADS)
        else:
            T.copy(
                Q[bid, hid * kv_group_num : hid * kv_group_num + kv_group_num, :],
                Qs[0:kv_group_num, :],
            )
        for k in T.serial(loop_range):
            T.mbarrier_wait_parity(ready[k % ring_depth], (k // ring_depth) % ring_depth)
            if fuse_rope and rotary_dim != dim:
                for i, freq in T.Parallel(block_n, rotary_dim // 2):
                    if i < this_len - k * block_n:
                        d0 = freq if rope_layout == "neox" else 2 * freq
                        d1 = freq + rotary_dim // 2 if rope_layout == "neox" else 2 * freq + 1
                        position = base + k * block_n + i
                        x0 = Ks[k % ring_depth, i, d0]
                        x1 = Ks[k % ring_depth, i, d1]
                        cos = rope_cos[position, freq]
                        sin = rope_sin[position, freq]
                        Ks[k % ring_depth, i, d0] = x0 * cos - x1 * sin
                        Ks[k % ring_depth, i, d1] = x1 * cos + x0 * sin
                T.sync_threads(3, CONSUMER_THREADS)
            T.wgmma_gemm(
                Qs,
                Ks[k % ring_depth, :, :],
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.if_then_else(
                    k * block_n + j < this_len,
                    acc_s[i, j],
                    -T.infinity(accum_dtype),
                )
            T.copy(sm, smp)
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            for i in T.Parallel(block_m):
                alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
            T.reduce_sum(acc_s, ss, dim=1)
            for i in T.Parallel(block_m):
                logsum[i] = logsum[i] * alpha[i] + ss[i]
            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] *= alpha[i]
            T.copy(acc_s, Ps)
            T.wgmma_gemm(
                Ps,
                Vs[k % ring_depth, :, :],
                acc_o,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=False,
            )
            T.wait_wgmma(0)
            T.mbarrier_arrive(free[k % ring_depth])
        for i, j in T.Parallel(block_m, dim):
            acc_o[i, j] /= T.if_then_else(this_len > 0, logsum[i], 1.0)
        for i in T.Parallel(block_m):
            if i < kv_group_num:
                glse[bid, hid * kv_group_num + i, sid] = T.if_then_else(
                    this_len > 0,
                    T.log2(logsum[i]) + sm[i] * scale,
                    -T.infinity(accum_dtype),
                )
        for i, j in T.Parallel(block_m, dim):
            if i < kv_group_num:
                Output_partial[bid, hid * kv_group_num + i, sid, j] = T.if_then_else(
                    this_len > 0,
                    acc_o[i, j],
                    0.0,
                )

    @T.macro
    def split(
        Q,
        K,
        V,
        kv_layout,
        real_seqlen_kv,
        rope_cos,
        rope_sin,
        glse,
        output_partial,
    ):
        with T.Kernel(batch, groups, ctx_splits, threads=threads) as (bid, hid, sid):
            qs = T.alloc_shared([block_m, dim], dtype)
            ks = T.alloc_shared([RING_DEPTH, block_n, dim], dtype)
            vs = T.alloc_shared([RING_DEPTH, block_n, dim], dtype)
            ps = T.alloc_shared([block_m, block_n], dtype)
            # Full-dimensional RoPE has enough work to justify a producer
            # warpgroup.  TMA stages the lookup tables with K/V, then all 128
            # producer threads rotate K while the consumer computes the other
            # ring slot.  Partial RoPE keeps the smaller 32-thread TMA producer.
            tma_rope_pipeline = fuse_rope and rotary_dim == dim
            if tma_rope_pipeline:
                rope_cos_s = T.alloc_shared([block_n, rotary_dim // 2], dtype)
                rope_sin_s = T.alloc_shared([block_n, rotary_dim // 2], dtype)
            T.annotate_layout(
                {
                    qs: tilelang.layout.make_swizzled_layout(qs),
                    ks: tilelang.layout.make_swizzled_layout(ks),
                    vs: tilelang.layout.make_swizzled_layout(vs),
                    ps: tilelang.layout.make_swizzled_layout(ps),
                }
            )
            producer_threads = threads - CONSUMER_THREADS
            ready = T.alloc_barrier([producer_threads] * RING_DEPTH)
            free = T.alloc_barrier([CONSUMER_THREADS] * RING_DEPTH)
            if tma_rope_pipeline:
                loaded = T.alloc_barrier([TMA_THREADS] * RING_DEPTH)
            load_ready = loaded if tma_rope_pipeline else ready
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            sm = T.alloc_fragment([block_m], accum_dtype)
            smp = T.alloc_fragment([block_m], accum_dtype)
            alpha = T.alloc_fragment([block_m], accum_dtype)
            ss = T.alloc_fragment([block_m], accum_dtype)
            logsum = T.alloc_fragment([block_m], accum_dtype)

            seqlen_kv_b = real_seqlen_kv[bid] if real_seqlen_is_buffer else real_seqlen_kv
            # Partition whole KV tiles as evenly as possible.  The old policy
            # rounded every non-final split down, then assigned the entire
            # remainder to the final CTA; non-divisible lengths could therefore
            # leave one split with several times more work than its peers.
            num_tiles = T.ceildiv(seqlen_kv_b, block_n)
            base_tiles = num_tiles // ctx_splits
            extra_tiles = num_tiles % ctx_splits
            tiles_this_split = base_tiles + T.if_then_else(sid < extra_tiles, 1, 0)
            tile_begin = sid * base_tiles + T.min(sid, extra_tiles)
            base = tile_begin * block_n
            this_len = T.max(
                T.min(seqlen_kv_b - base, tiles_this_split * block_n),
                0,
            )
            loop_range = T.ceildiv(this_len, block_n)
            tx = T.get_thread_binding()

            if tx >= CONSUMER_THREADS:
                for k in T.serial(loop_range):
                    T.mbarrier_wait_parity(
                        free[k % RING_DEPTH],
                        ((k // RING_DEPTH) % RING_DEPTH) ^ 1,
                    )
                    if tx < CONSUMER_THREADS + TMA_THREADS:
                        load_kv(
                            K,
                            V,
                            kv_layout,
                            bid,
                            hid,
                            base,
                            k,
                            ks,
                            vs,
                            load_ready,
                        )
                        if tma_rope_pipeline:
                            T.tma_copy(
                                rope_cos[
                                    base + k * block_n : base + (k + 1) * block_n,
                                    :,
                                ],
                                rope_cos_s,
                                barrier=loaded[k % RING_DEPTH],
                            )
                            T.tma_copy(
                                rope_sin[
                                    base + k * block_n : base + (k + 1) * block_n,
                                    :,
                                ],
                                rope_sin_s,
                                barrier=loaded[k % RING_DEPTH],
                            )
                        T.mbarrier_arrive(load_ready[k % RING_DEPTH])
                    if tma_rope_pipeline:
                        T.mbarrier_wait_parity(
                            loaded[k % RING_DEPTH],
                            (k // RING_DEPTH) % RING_DEPTH,
                        )
                        for i, freq in T.Parallel(block_n, rotary_dim // 2):
                            if i < this_len - k * block_n:
                                d0 = freq if rope_layout == "neox" else 2 * freq
                                d1 = (
                                    freq + rotary_dim // 2
                                    if rope_layout == "neox"
                                    else 2 * freq + 1
                                )
                                x0 = ks[k % RING_DEPTH, i, d0]
                                x1 = ks[k % RING_DEPTH, i, d1]
                                cos = rope_cos_s[i, freq]
                                sin = rope_sin_s[i, freq]
                                ks[k % RING_DEPTH, i, d0] = x0 * cos - x1 * sin
                                ks[k % RING_DEPTH, i, d1] = x1 * cos + x0 * sin
                        T.mbarrier_arrive(ready[k % RING_DEPTH])
            else:
                consumer(
                    Q,
                    bid,
                    hid,
                    sid,
                    this_len,
                    base,
                    seqlen_kv_b,
                    loop_range,
                    rope_cos,
                    rope_sin,
                    qs,
                    ks,
                    vs,
                    ps,
                    ready,
                    free,
                    acc_s,
                    acc_o,
                    sm,
                    smp,
                    alpha,
                    ss,
                    logsum,
                    glse,
                    output_partial,
                )

    return split


@functools.lru_cache(maxsize=32)
def _gqa_decode_bs1_ctx_kernel(
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
    scale = score_scale * LOG2E
    accum_dtype = "float"
    kv_group_num = heads // groups

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=COMPILE_FLAGS,
    )
    def _func(block_M, block_N, ctx_splits, threads):
        seqlen_kv = T.dynamic("seqlen_kv")
        shape_q = [batch, heads, dim]
        shape_k = [batch, seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        rope_shape = [max_position, rotary_dim // 2]
        part_shape = [batch, heads, ctx_splits, dim]
        lse_shape = [batch, heads, ctx_splits]

        @T.macro
        def load_kv(K, V, unused_layout, bid, hid, base, k, Ks, Vs, ready):
            T.tma_copy(
                K[bid, base + k * block_N : base + (k + 1) * block_N, hid, :],
                Ks[k % RING_DEPTH, :, :],
                barrier=ready[k % RING_DEPTH],
            )
            T.tma_copy(
                V[bid, base + k * block_N : base + (k + 1) * block_N, hid, :],
                Vs[k % RING_DEPTH, :, :],
                barrier=ready[k % RING_DEPTH],
            )

        split = _make_dense_decode_split(
            batch,
            groups,
            block_M,
            block_N,
            dim,
            dtype,
            scale,
            kv_group_num,
            ctx_splits,
            threads,
            accum_dtype,
            False,
            load_kv,
            fuse_rope,
            rotary_dim,
            rope_layout,
        )
        combine = make_gqa_decode_bs1_combine(
            batch,
            heads,
            ctx_splits,
            dim,
            dtype,
            accum_dtype,
        )

        if fuse_rope:

            @T.prim_func
            def gqa_decode_bs1_ctx_rope(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_k, dtype),
                rope_cos: T.Tensor(rope_shape, dtype),
                rope_sin: T.Tensor(rope_shape, dtype),
                glse: T.Tensor(lse_shape, accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
            ):
                split(
                    Q,
                    K,
                    V,
                    K,
                    seqlen_kv,
                    rope_cos,
                    rope_sin,
                    glse,
                    Output_partial,
                )
                combine(glse, Output_partial, Output)

            return gqa_decode_bs1_ctx_rope

        @T.prim_func
        def gqa_decode_bs1_ctx(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_k, dtype),
            glse: T.Tensor(lse_shape, accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            split(Q, K, V, K, seqlen_kv, Q, Q, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return gqa_decode_bs1_ctx

    return _func


@torch.library.custom_op("tileops::gqa_decode_bs1_ctx_op", mutates_args=())
def _gqa_decode_bs1_ctx_op(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    block_M: int,
    block_N: int,
    ctx_splits: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
) -> torch.Tensor:
    kernel = _gqa_decode_bs1_ctx_kernel(
        batch,
        heads,
        groups,
        dim,
        sm_scale,
        softcap,
        dtype,
        fuse_rope,
        max_position,
        rotary_dim,
        rope_layout,
    )(block_M, block_N, ctx_splits, threads)
    if fuse_rope:
        return kernel(Q, K, V, rope_cos, rope_sin, glse, Output_partial)
    return kernel(Q, K, V, glse, Output_partial)


@_gqa_decode_bs1_ctx_op.register_fake
def _(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    block_M: int,
    block_N: int,
    ctx_splits: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(Q)


class GQADecodeBs1Kernel(Kernel):
    """Hopper warp-specialized batch=1 GQA decode kernel with a context-length switch.

    ``forward`` always uses the context pipeline for full-dimensional RoPE.  Plain calls
    below 640 and partial-RoPE calls below 1024 retain the generic single-kernel path.
    """

    supported_archs: list[int] = [90]
    _MIN_CTX = 1024
    _PLAIN_CTX_MIN = 640
    _TARGET_PARTIAL_CTAS = 128
    _CTX_SPLIT_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)

    def _dense_ctx_splits_for(
        self,
        real_seqlen_kv: int,
        *,
        allow_empty: bool = True,
    ) -> int:
        block_n = self.config["block_N"]
        num_tiles = (real_seqlen_kv + block_n - 1) // block_n
        split_capacity = max(
            1,
            min(self._CTX_SPLIT_CANDIDATES[-1], self._TARGET_PARTIAL_CTAS // self.groups),
        )
        selected = 1
        for ctx_splits in self._CTX_SPLIT_CANDIDATES:
            if ctx_splits > split_capacity:
                break
            if not allow_empty and ctx_splits > num_tiles:
                break
            selected = ctx_splits
            if allow_empty and ctx_splits >= num_tiles:
                break
        return selected

    def _allocate_partials(
        self,
        q: torch.Tensor,
        ctx_splits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        glse = torch.empty(
            (self.batch, self.heads, ctx_splits),
            dtype=torch.float32,
            device=q.device,
        )
        output_partial = torch.empty(
            (self.batch, self.heads, ctx_splits, self.dim),
            dtype=torch.float32,
            device=q.device,
        )
        return glse, output_partial

    @classmethod
    def applies(cls, call) -> bool:
        return decode_bs1_region(call)

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
        if self.groups <= 0:
            raise ValueError("heads_kv must be positive")
        if self.heads % self.groups != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if self.seqlen_kv <= 0:
            raise ValueError("seq_len_kv must be positive")
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_M": 64, "block_N": 128, "threads": 160}

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
        c = self.config
        full_rope_pipeline = self.fuse_rope and self.rotary_dim == self.dim
        use_ctx_pipeline = full_rope_pipeline or real_seqlen_kv >= (
            self._MIN_CTX if self.fuse_rope else self._PLAIN_CTX_MIN
        )
        if not use_ctx_pipeline:
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
                    64,
                    128,
                    2,
                    128,
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
                64,
                128,
                2,
                128,
                Q,
                K,
                V,
            )
            return output.unsqueeze(1)

        ctx_splits = self._dense_ctx_splits_for(
            real_seqlen_kv,
            allow_empty=not self.fuse_rope or full_rope_pipeline,
        )
        if self.fuse_rope:
            glse, Output_partial = self._allocate_partials(Q, ctx_splits)
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
                c["block_M"],
                c["block_N"],
                ctx_splits,
                ROPE_PIPELINE_THREADS if self.rotary_dim == self.dim else c["threads"],
                Q,
                K,
                V,
                rope_cos,
                rope_sin,
                glse,
                Output_partial,
            )
            return output.unsqueeze(1)

        glse, Output_partial = self._allocate_partials(Q, ctx_splits)
        output = _gqa_decode_bs1_ctx_op(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            False,
            1,
            0,
            "neox",
            c["block_M"],
            c["block_N"],
            ctx_splits,
            c["threads"],
            Q,
            K,
            V,
            Q,
            Q,
            glse,
            Output_partial,
        )
        return output.unsqueeze(1)
