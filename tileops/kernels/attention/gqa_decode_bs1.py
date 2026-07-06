"""Warp-specialized batch=1 GQA decode kernel (Hopper), context-split.

``GQADecodeBs1Kernel`` dispatches on the runtime ``real_seqlen_kv``: lengths >= 1024
run a context-only warp-specialized split (one TMA producer warp feeding a four-warp
wgmma consumer warpgroup, exp2-domain online softmax, fp32 partial reduce via a combine
kernel); shorter lengths fall back to the generic non-split decode kernel. Hopper-only,
low-level ``tma_copy`` / ``mbarrier`` / ``wgmma_gemm``.
"""
import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.gqa_decode import _gqa_decode_no_split_op
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import LOG2E

__all__ = ["GQADecodeBs1Kernel"]

_RING = 2  # K/V shared-ring depth (double-buffered).

_COMPILE_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DNDEBUG",
]


@functools.lru_cache(maxsize=32)
def _gqa_decode_bs1_ctx_kernel(batch, heads, groups, seqlen_kv, dim, sm_scale, softcap, dtype):
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    scale = score_scale * LOG2E
    accum_dtype = "float"
    kv_group_num = heads // groups
    ns = _RING

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=_COMPILE_FLAGS)
    def _func(block_M, block_N, ctx_splits, threads):
        shape_q = [batch, heads, dim]
        shape_k = [batch, seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        part_shape = [batch, heads, ctx_splits, dim]
        lse_shape = [batch, heads, ctx_splits]
        policy = T.GemmWarpPolicy.FullRow

        @T.macro
        def _split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_k, dtype),
                real_seqlen_kv: T.int32,
                glse: T.Tensor(lse_shape, accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
        ):
            with T.Kernel(batch, groups, ctx_splits, threads=threads) as (bid, hid, sid):
                Qs = T.alloc_shared([block_M, dim], dtype)
                Ks = T.alloc_shared([ns, block_N, dim], dtype)
                Vs = T.alloc_shared([ns, block_N, dim], dtype)
                Ps = T.alloc_shared([block_M, block_N], dtype)
                T.annotate_layout({
                    Qs: tilelang.layout.make_swizzled_layout(Qs),
                    Ks: tilelang.layout.make_swizzled_layout(Ks),
                    Vs: tilelang.layout.make_swizzled_layout(Vs),
                    Ps: tilelang.layout.make_swizzled_layout(Ps),
                })
                ready = T.alloc_barrier([32] * ns)   # producer -> consumer
                free = T.alloc_barrier([128] * ns)   # consumer -> producer
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                sm = T.alloc_fragment([block_M], accum_dtype)
                smp = T.alloc_fragment([block_M], accum_dtype)
                alpha = T.alloc_fragment([block_M], accum_dtype)
                ss = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                # Redistribute over the real length so every split holds a full tile.
                base_len = real_seqlen_kv // (ctx_splits * block_N) * block_N
                this_len = T.if_then_else(sid == ctx_splits - 1,
                                          real_seqlen_kv - (ctx_splits - 1) * base_len, base_len)
                base = base_len * sid
                loop_range = T.ceildiv(this_len, block_N)
                tx = T.get_thread_binding()

                if tx >= 128:   # producer
                    for k in T.serial(loop_range):
                        T.mbarrier_wait_parity(free[k % ns], ((k // ns) % ns) ^ 1)
                        T.tma_copy(K[bid, base + k * block_N:base + (k + 1) * block_N, hid, :],
                                   Ks[k % ns, :, :], barrier=ready[k % ns])
                        T.tma_copy(V[bid, base + k * block_N:base + (k + 1) * block_N, hid, :],
                                   Vs[k % ns, :, :], barrier=ready[k % ns])
                        T.mbarrier_arrive(ready[k % ns])
                else:   # consumer
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(sm, -T.infinity(accum_dtype))
                    T.copy(Q[bid, hid * kv_group_num:hid * kv_group_num + kv_group_num, :],
                           Qs[0:kv_group_num, :])
                    for k in T.serial(loop_range):
                        T.mbarrier_wait_parity(ready[k % ns], (k // ns) % ns)
                        T.wgmma_gemm(Qs, Ks[k % ns, :, :], acc_s, transpose_B=True, policy=policy,
                                     clear_accum=True)
                        T.wait_wgmma(0)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(k * block_N + j < this_len, acc_s[i, j],
                                                         -T.infinity(accum_dtype))
                        T.copy(sm, smp)
                        T.reduce_max(acc_s, sm, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                        T.reduce_sum(acc_s, ss, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * alpha[i] + ss[i]
                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] *= alpha[i]
                        T.copy(acc_s, Ps)
                        T.wgmma_gemm(Ps, Vs[k % ns, :, :], acc_o, policy=policy, clear_accum=False)
                        T.wait_wgmma(0)
                        T.mbarrier_arrive(free[k % ns])
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_M):
                        if i < kv_group_num:
                            glse[bid, hid * kv_group_num + i, sid] = T.log2(logsum[i]) + sm[i] * scale
                    for i, j in T.Parallel(block_M, dim):
                        if i < kv_group_num:
                            Output_partial[bid, hid * kv_group_num + i, sid, j] = acc_o[i, j]

        @T.macro
        def _combine(
                glse: T.Tensor(lse_shape, accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (hq, bid):
                lse_vec = T.alloc_fragment([ctx_splits], accum_dtype)
                lse_max = T.alloc_fragment([1], accum_dtype)
                lse_logsum = T.alloc_local([1], accum_dtype)
                o_accum = T.alloc_fragment([dim], accum_dtype)

                for s in T.Parallel(ctx_splits):
                    lse_vec[s] = glse[bid, hq, s]
                T.fill(lse_max, -T.infinity(accum_dtype))
                T.reduce_max(lse_vec, lse_max, dim=0, clear=False)
                lse_logsum[0] = 0
                for s in T.serial(ctx_splits):
                    lse_logsum[0] += T.exp2(glse[bid, hq, s] - lse_max[0])
                lse_logsum[0] = T.log2(lse_logsum[0]) + lse_max[0]
                T.clear(o_accum)
                for s in T.serial(ctx_splits):
                    w = T.exp2(glse[bid, hq, s] - lse_logsum[0])
                    for j in T.Parallel(dim):
                        o_accum[j] += Output_partial[bid, hq, s, j] * w
                for j in T.Parallel(dim):
                    Output[bid, hq, j] = T.cast(o_accum[j], dtype)

        @T.prim_func
        def gqa_decode_bs1_ctx(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_k, dtype),
                real_seqlen_kv: T.int32,
                glse: T.Tensor(lse_shape, accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            _split(Q, K, V, real_seqlen_kv, glse, Output_partial)
            _combine(glse, Output_partial, Output)

        return gqa_decode_bs1_ctx

    return _func


@torch.library.custom_op("top::gqa_decode_bs1_ctx_op", mutates_args=())
def _gqa_decode_bs1_ctx_op(batch: int, heads: int, groups: int, seqlen_kv: int,
                           real_seqlen_kv: int, dim: int, sm_scale: float, softcap: float,
                           dtype: str, block_M: int, block_N: int, ctx_splits: int, threads: int,
                           Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, glse: torch.Tensor,
                           Output_partial: torch.Tensor) -> torch.Tensor:
    return _gqa_decode_bs1_ctx_kernel(batch, heads, groups, seqlen_kv, dim, sm_scale, softcap,
                                      dtype)(block_M, block_N, ctx_splits, threads)(
                                          Q, K, V, real_seqlen_kv, glse, Output_partial)


@_gqa_decode_bs1_ctx_op.register_fake
def _(batch: int, heads: int, groups: int, seqlen_kv: int, real_seqlen_kv: int, dim: int,
      sm_scale: float, softcap: float, dtype: str, block_M: int, block_N: int, ctx_splits: int,
      threads: int, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, glse: torch.Tensor,
      Output_partial: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(Q)


class GQADecodeBs1Kernel(Kernel):
    """Hopper warp-specialized batch=1 GQA decode kernel with a context-length switch.

    ``forward`` dispatches on the runtime ``real_seqlen_kv``: >= 1024 runs the context-only
    split, shorter lengths run the generic non-split GQA decode kernel.
    """

    supported_archs: list[int] = [90]

    _MIN_CTX = 1024
    # Powers of two TileLang lowers cleanly; the largest dividing the KV length balances slices.
    _CTX_SPLIT_CANDIDATES = (32, 16, 8)

    def __init__(self,
                 batch,
                 heads,
                 groups,
                 seqlen_kv,
                 dim,
                 dtype="float16",
                 sm_scale: Optional[float] = None,
                 softcap: float = 0.0,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        if self.groups <= 0:
            raise ValueError("groups must be positive")
        if self.heads % self.groups != 0:
            raise ValueError("heads must be divisible by groups")
        if self.seqlen_kv <= 0:
            raise ValueError("seqlen_kv must be positive")
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_M": 64, "block_N": 128, "threads": 160}

    def _select_tier(self, real_seqlen_kv: int) -> str:
        return "ctx" if real_seqlen_kv >= self._MIN_CTX else "no_split"

    def _ctx_splits_for(self, real_seqlen_kv: int) -> int:
        block_N = self.config["block_N"]
        for cs in self._CTX_SPLIT_CANDIDATES:
            if real_seqlen_kv % (cs * block_N) == 0:
                return cs
        return self._CTX_SPLIT_CANDIDATES[-1]

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, real_seqlen_kv: int):
        c = self.config
        if real_seqlen_kv < self._MIN_CTX:
            return _gqa_decode_no_split_op(self.batch, self.heads, self.groups, self.seqlen_kv,
                                           real_seqlen_kv, self.dim, self.sm_scale, self.softcap,
                                           self.dtype_str, 64, 128, 2, 128, Q, K, V)

        ctx_splits = self._ctx_splits_for(real_seqlen_kv)
        glse = torch.empty((self.batch, self.heads, ctx_splits),
                           dtype=torch.float32,
                           device=Q.device)
        Output_partial = torch.empty((self.batch, self.heads, ctx_splits, self.dim),
                                     dtype=torch.float32,
                                     device=Q.device)
        return _gqa_decode_bs1_ctx_op(self.batch, self.heads, self.groups, self.seqlen_kv,
                                      real_seqlen_kv, self.dim, self.sm_scale, self.softcap,
                                      self.dtype_str, c["block_M"], c["block_N"], ctx_splits,
                                      c["threads"], Q, K, V, glse, Output_partial)
