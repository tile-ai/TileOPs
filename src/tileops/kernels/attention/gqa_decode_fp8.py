"""Native-FP8 Dense GQA decode kernels for Hopper."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from ..kernel_base import Entry, Kernel
from .call_spec import dense_fp8_decode_region
from .dense_entry import dense_fp8_decode_entry
from .gqa_decode_bs1_common import COMPILE_FLAGS
from .gqa_fwd_fp8 import _validate_fa3_gqa_descales
from .online_softmax import LOG2E

__all__ = ["GQADenseFP8DecodeKernel"]


@functools.lru_cache(maxsize=32)
def _gqa_dense_fp8_decode_ctx_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    out_dtype: str,
    sm_scale: float,
    softcap: float,
):
    """Build a GQA-head-packed, context-split FP8 decode program."""
    q_per_kv = heads // heads_kv
    fp8_dtype = "float8_e4m3fn"
    accum_dtype = "float"
    use_softcap = softcap > 0.0

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=COMPILE_FLAGS,
    )
    def build(block_m: int, block_n: int, ctx_splits: int, threads: int):
        seq_len_kv = T.dynamic("seq_len_kv")
        q_shape = [batch, 1, heads, dim]
        kv_shape = [batch, seq_len_kv, heads_kv, dim]
        scale_shape = [batch, heads_kv]
        lse_shape = [batch, heads, ctx_splits]
        partial_shape = [batch, heads, ctx_splits, dim]
        output_shape = [batch, 1, heads, dim]
        ring_depth = 2

        @T.macro
        def split(
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            glse,
            output_partial,
        ):
            with T.Kernel(batch, heads_kv, ctx_splits, threads=threads) as (bid, hid, sid):
                qs = T.alloc_shared([block_m, dim], fp8_dtype)
                ks = T.alloc_shared([ring_depth, block_n, dim], fp8_dtype)
                vs = T.alloc_shared([ring_depth, block_n, dim], fp8_dtype)
                ps = T.alloc_shared([block_m, block_n], fp8_dtype)
                T.annotate_layout(
                    {
                        qs: tilelang.layout.make_swizzled_layout(qs),
                        ks: tilelang.layout.make_swizzled_layout(ks),
                        vs: tilelang.layout.make_swizzled_layout(vs),
                        ps: tilelang.layout.make_swizzled_layout(ps),
                    }
                )
                ready = T.alloc_barrier([32] * ring_depth)
                free = T.alloc_barrier([128] * ring_depth)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                num_tiles = T.ceildiv(seq_len_kv, block_n)
                base_tiles = num_tiles // ctx_splits
                extra_tiles = num_tiles % ctx_splits
                tiles_this_split = base_tiles + T.if_then_else(sid < extra_tiles, 1, 0)
                tile_begin = sid * base_tiles + T.min(sid, extra_tiles)
                base = tile_begin * block_n
                this_len = T.max(
                    T.min(seq_len_kv - base, tiles_this_split * block_n),
                    0,
                )
                loop_range = T.ceildiv(this_len, block_n)
                tx = T.get_thread_binding()

                if tx >= 128:
                    for tile in T.serial(loop_range):
                        T.mbarrier_wait_parity(
                            free[tile % ring_depth],
                            ((tile // ring_depth) % ring_depth) ^ 1,
                        )
                        T.tma_copy(
                            k[
                                bid,
                                base + tile * block_n : base + (tile + 1) * block_n,
                                hid,
                                :,
                            ],
                            ks[tile % ring_depth, :, :],
                            barrier=ready[tile % ring_depth],
                        )
                        T.tma_copy(
                            v[
                                bid,
                                base + tile * block_n : base + (tile + 1) * block_n,
                                hid,
                                :,
                            ],
                            vs[tile % ring_depth, :, :],
                            barrier=ready[tile % ring_depth],
                        )
                        T.mbarrier_arrive(ready[tile % ring_depth])
                else:
                    T.clear(qs)
                    T.copy(
                        q[bid, 0, hid * q_per_kv : (hid + 1) * q_per_kv, :],
                        qs[0:q_per_kv, :],
                    )
                    T.clear(acc_o)
                    T.clear(logsum)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    qk_scale = (
                        q_descale[bid, hid] * k_descale[bid, hid] * T.cast(sm_scale, accum_dtype)
                    )
                    value_scale = v_descale[bid, hid]

                    for tile in T.serial(loop_range):
                        T.mbarrier_wait_parity(
                            ready[tile % ring_depth],
                            (tile // ring_depth) % ring_depth,
                        )
                        T.gemm(
                            qs,
                            ks[tile % ring_depth, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=True,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            if base + tile * block_n + j < seq_len_kv:
                                if use_softcap:
                                    acc_s[i, j] = T.tanh(
                                        acc_s[i, j] * qk_scale / T.cast(softcap, accum_dtype)
                                    ) * T.cast(softcap, accum_dtype)
                                else:
                                    acc_s[i, j] *= qk_scale
                            else:
                                acc_s[i, j] = -T.infinity(accum_dtype)
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_m):
                            scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * LOG2E)
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.exp2((acc_s[i, j] - scores_max[i]) * LOG2E)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_m):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        for i, j in T.Parallel(block_m, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(acc_s, ps)
                        T.gemm(
                            ps,
                            vs[tile % ring_depth, :, :],
                            acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=False,
                        )
                        T.mbarrier_arrive(free[tile % ring_depth])

                    for i in T.Parallel(q_per_kv):
                        glse[bid, hid * q_per_kv + i, sid] = T.if_then_else(
                            this_len > 0,
                            T.log2(logsum[i]) + scores_max[i] * LOG2E,
                            -T.infinity(accum_dtype),
                        )
                    for i, j in T.Parallel(q_per_kv, dim):
                        output_partial[bid, hid * q_per_kv + i, sid, j] = T.if_then_else(
                            this_len > 0,
                            acc_o[i, j] * value_scale / logsum[i],
                            0.0,
                        )

        @T.macro
        def combine(glse, output_partial, output):
            with T.Kernel(heads, batch, threads=128) as (hq, bid):
                lse_vec = T.alloc_fragment([ctx_splits], accum_dtype)
                lse_max = T.alloc_fragment([1], accum_dtype)
                lse_sum = T.alloc_local([1], accum_dtype)
                o_accum = T.alloc_fragment([dim], accum_dtype)
                for sid in T.Parallel(ctx_splits):
                    lse_vec[sid] = glse[bid, hq, sid]
                T.fill(lse_max, -T.infinity(accum_dtype))
                T.reduce_max(lse_vec, lse_max, dim=0, clear=False)
                lse_sum[0] = 0.0
                for sid in T.serial(ctx_splits):
                    lse_sum[0] += T.exp2(glse[bid, hq, sid] - lse_max[0])
                lse_sum[0] = T.log2(lse_sum[0]) + lse_max[0]
                T.clear(o_accum)
                for sid in T.serial(ctx_splits):
                    weight = T.exp2(glse[bid, hq, sid] - lse_sum[0])
                    for j in T.Parallel(dim):
                        o_accum[j] += output_partial[bid, hq, sid, j] * weight
                for j in T.Parallel(dim):
                    output[bid, 0, hq, j] = T.cast(o_accum[j], out_dtype)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, fp8_dtype),
            k: T.Tensor(kv_shape, fp8_dtype),
            v: T.Tensor(kv_shape, fp8_dtype),
            q_descale: T.Tensor(scale_shape, accum_dtype),
            k_descale: T.Tensor(scale_shape, accum_dtype),
            v_descale: T.Tensor(scale_shape, accum_dtype),
            glse: T.Tensor(lse_shape, accum_dtype),
            output_partial: T.Tensor(partial_shape, accum_dtype),
            output: T.Tensor(output_shape, out_dtype),
        ):
            split(q, k, v, q_descale, k_descale, v_descale, glse, output_partial)
            combine(glse, output_partial, output)

        return main

    return build


def _gqa_dense_fp8_decode_ctx_run(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    out_dtype: str,
    sm_scale: float,
    softcap: float,
    block_m: int,
    block_n: int,
    ctx_splits: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    glse: torch.Tensor,
    output_partial: torch.Tensor,
) -> torch.Tensor:
    kernel = _gqa_dense_fp8_decode_ctx_kernel(
        batch,
        heads,
        heads_kv,
        dim,
        out_dtype,
        sm_scale,
        softcap,
    )(block_m, block_n, ctx_splits, threads)
    return kernel(q, k, v, q_descale, k_descale, v_descale, glse, output_partial)


def _(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    out_dtype: str,
    sm_scale: float,
    softcap: float,
    block_m: int,
    block_n: int,
    ctx_splits: int,
    threads: int,
    q: torch.Tensor,
    *args,
) -> torch.Tensor:
    del heads_kv, sm_scale, softcap, block_m, block_n, ctx_splits, threads, args
    dtype = torch.float16 if out_dtype == "float16" else torch.bfloat16
    return torch.empty((batch, 1, heads, dim), dtype=dtype, device=q.device)


class GQADenseFP8DecodeKernel(Kernel):
    """Context-split native-FP8 Dense decode specialization."""

    supported_archs: list[int] = [90]
    _TARGET_CTAS = 128
    _MAX_SPLITS = 32

    @classmethod
    def applies(cls, call) -> bool:
        return dense_fp8_decode_region(call)

    @classmethod
    def entry_for(cls, call) -> Entry:
        return dense_fp8_decode_entry(cls, call)

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
        **unused,
    ) -> None:
        del unused
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if heads // heads_kv > 16:
            raise ValueError("FP8 Dense decode supports at most 16 query heads per KV head")
        if dim != 128:
            raise ValueError("FP8 Dense decode requires dim == 128")
        self.init_config(config, tune)
        if heads // heads_kv > self.config["block_m"]:
            raise ValueError("block_m must cover every query head sharing one KV head")

    @property
    def default_config(self) -> dict:
        return {"block_m": 16, "block_n": 256, "threads": 160}

    def _ctx_splits_for(self, seq_len_kv: int) -> int:
        num_tiles = (seq_len_kv + self.config["block_n"] - 1) // self.config["block_n"]
        capacity = max(1, min(self._MAX_SPLITS, self._TARGET_CTAS // self.heads_kv))
        selected = 1
        for candidate in (1, 2, 4, 8, 16, 32):
            if candidate > num_tiles or candidate > capacity:
                break
            selected = candidate
        return selected

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
        self._require_cuda(q=q, k=k, v=v, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)
        if rope_cos is not None or rope_sin is not None:
            raise ValueError("GQADenseFP8DecodeKernel does not support RoPE")
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError("FP8 decode requires q_scale, k_scale, and v_scale")
        _validate_fa3_gqa_descales(
            q_scale,
            k_scale,
            v_scale,
            self.batch,
            self.heads_kv,
            q.device,
        )
        ctx_splits = self._ctx_splits_for(k.shape[1])
        glse = torch.empty(
            (self.batch, self.heads, ctx_splits), dtype=torch.float32, device=q.device
        )
        output_partial = torch.empty(
            (self.batch, self.heads, ctx_splits, self.dim),
            dtype=torch.float32,
            device=q.device,
        )
        c = self.config
        return _gqa_dense_fp8_decode_ctx_run(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.dtype_str,
            self.sm_scale,
            self.softcap,
            c["block_m"],
            c["block_n"],
            ctx_splits,
            c["threads"],
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            glse,
            output_partial,
        )
