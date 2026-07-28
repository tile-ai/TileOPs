"""Warp-specialized batch=1 paged GQA decode kernel (Hopper), context-split.

``GQADecodePagedBs1Kernel`` dispatches on the runtime ``real_seqlen_kv``: lengths >= 1024
run a context-only warp-specialized split (one TMA producer warp feeding a four-warp
wgmma consumer warpgroup, exp2-domain online softmax, fp32 partial reduce via a combine
kernel); shorter lengths fall back to the generic paged non-split decode kernel.
Logical KV tiles are translated through the page table before TMA. Hopper-only, low-level
``tma_copy`` / ``mbarrier`` / ``wgmma_gemm``.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.gqa_decode_paged import _gqa_decode_paged_no_split_op
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import LOG2E

from .gqa_decode_bs1_common import (
    COMPILE_FLAGS,
    RING_DEPTH,
    make_gqa_decode_bs1_combine,
    make_gqa_decode_bs1_consumer,
)

__all__ = ["GQADecodePagedBs1Kernel"]


@functools.lru_cache(maxsize=32)
def _gqa_decode_paged_bs1_ctx_kernel(
    batch, heads, groups, seqlen_kv, dim, page_size, sm_scale, softcap, dtype
):
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    scale = score_scale * LOG2E
    accum_dtype = "float"
    kv_group_num = heads // groups
    ns = RING_DEPTH

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=COMPILE_FLAGS,
    )
    def _func(block_M, block_N, ctx_splits, threads):
        shape_q = [batch, heads, dim]
        shape_k = [seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        part_shape = [batch, heads, ctx_splits, dim]
        lse_shape = [batch, heads, ctx_splits]
        consumer = make_gqa_decode_bs1_consumer(
            block_M,
            block_N,
            dim,
            scale,
            kv_group_num,
            ns,
            accum_dtype,
        )
        combine = make_gqa_decode_bs1_combine(
            batch,
            heads,
            ctx_splits,
            dim,
            dtype,
            accum_dtype,
        )

        @T.macro
        def _split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_k, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
            glse: T.Tensor(lse_shape, accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
        ):
            with T.Kernel(batch, groups, ctx_splits, threads=threads) as (bid, hid, sid):
                Qs = T.alloc_shared([block_M, dim], dtype)
                Ks = T.alloc_shared([ns, block_N, dim], dtype)
                Vs = T.alloc_shared([ns, block_N, dim], dtype)
                Ps = T.alloc_shared([block_M, block_N], dtype)
                T.annotate_layout(
                    {
                        Qs: tilelang.layout.make_swizzled_layout(Qs),
                        Ks: tilelang.layout.make_swizzled_layout(Ks),
                        Vs: tilelang.layout.make_swizzled_layout(Vs),
                        Ps: tilelang.layout.make_swizzled_layout(Ps),
                    }
                )
                ready = T.alloc_barrier([32] * ns)  # producer -> consumer
                free = T.alloc_barrier([128] * ns)  # consumer -> producer
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                sm = T.alloc_fragment([block_M], accum_dtype)
                smp = T.alloc_fragment([block_M], accum_dtype)
                alpha = T.alloc_fragment([block_M], accum_dtype)
                ss = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                # Redistribute over the real length so every split holds a full tile.
                seqlen_kv_b = real_seqlen_kv[bid]
                base_len = seqlen_kv_b // (ctx_splits * block_N) * block_N
                this_len = T.if_then_else(
                    sid == ctx_splits - 1, seqlen_kv_b - (ctx_splits - 1) * base_len, base_len
                )
                base = base_len * sid
                loop_range = T.ceildiv(this_len, block_N)
                tx = T.get_thread_binding()

                if tx >= 128:  # producer
                    for k in T.serial(loop_range):
                        T.mbarrier_wait_parity(free[k % ns], ((k // ns) % ns) ^ 1)
                        logical_block = base // block_N + k
                        blocks_per_page = page_size // block_N
                        page_idx = logical_block // blocks_per_page
                        block_in_page = logical_block % blocks_per_page
                        physical_block = (
                            block_table[bid, page_idx] * blocks_per_page + block_in_page
                        )
                        T.tma_copy(
                            K[physical_block * block_N : (physical_block + 1) * block_N, hid, :],
                            Ks[k % ns, :, :],
                            barrier=ready[k % ns],
                        )
                        T.tma_copy(
                            V[physical_block * block_N : (physical_block + 1) * block_N, hid, :],
                            Vs[k % ns, :, :],
                            barrier=ready[k % ns],
                        )
                        T.mbarrier_arrive(ready[k % ns])
                else:  # consumer
                    consumer(
                        Q,
                        bid,
                        hid,
                        sid,
                        this_len,
                        loop_range,
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
                    )

        @T.prim_func
        def gqa_decode_paged_bs1_ctx(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_k, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
            glse: T.Tensor(lse_shape, accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            _split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return gqa_decode_paged_bs1_ctx

    return _func


@torch.library.custom_op("top::gqa_decode_paged_bs1_ctx_op", mutates_args=())
def _gqa_decode_paged_bs1_ctx_op(
    batch: int,
    heads: int,
    groups: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_M: int,
    block_N: int,
    ctx_splits: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
) -> torch.Tensor:
    return _gqa_decode_paged_bs1_ctx_kernel(
        batch, heads, groups, seqlen_kv, dim, page_size, sm_scale, softcap, dtype
    )(block_M, block_N, ctx_splits, threads)(
        Q, K, V, real_seqlen_kv, block_table, glse, Output_partial
    )


@_gqa_decode_paged_bs1_ctx_op.register_fake
def _(
    batch: int,
    heads: int,
    groups: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_M: int,
    block_N: int,
    ctx_splits: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(Q)


class GQADecodePagedBs1Kernel(Kernel):
    """Hopper warp-specialized batch=1 paged GQA decode kernel.

    ``forward`` dispatches on the runtime ``real_seqlen_kv``: >= 1024 runs the context-only
    split, shorter lengths run the generic non-split paged GQA decode kernel.
    """

    supported_archs: list[int] = [90]

    _MIN_CTX = 1024
    # Powers of two TileLang lowers cleanly; the largest dividing the KV length balances slices.
    _CTX_SPLIT_CANDIDATES = (32, 16, 8)

    @staticmethod
    def block_n_for_page_size(page_size: int) -> Optional[int]:
        """Return a page-contained WGMMA N tile, or None when the fast path is unsafe."""
        if page_size == 64:
            return 64
        if page_size >= 128 and page_size % 128 == 0:
            return 128
        return None

    def __init__(
        self,
        batch,
        heads,
        groups,
        seqlen_kv,
        dim,
        page_size,
        dtype="float16",
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune=False,
    ):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        if self.groups <= 0:
            raise ValueError("groups must be positive")
        if self.heads % self.groups != 0:
            raise ValueError("heads must be divisible by groups")
        if self.seqlen_kv <= 0:
            raise ValueError("seqlen_kv must be positive")
        if self.page_size <= 0 or self.seqlen_kv % self.page_size != 0:
            raise ValueError("page_size must be positive and divide seqlen_kv")
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        block_n = self.block_n_for_page_size(self.page_size)
        if block_n is None:
            raise ValueError(
                "batch=1 paged decode requires page_size=64 or a multiple of 128"
            )
        return {"block_M": 64, "block_N": block_n, "threads": 160}

    def _select_tier(self, real_seqlen_kv: int) -> str:
        return "ctx" if real_seqlen_kv >= self._MIN_CTX else "no_split"

    def _ctx_splits_for(self, real_seqlen_kv: int) -> int:
        block_N = self.config["block_N"]
        for cs in self._CTX_SPLIT_CANDIDATES:
            if real_seqlen_kv % (cs * block_N) == 0:
                return cs
        return self._CTX_SPLIT_CANDIDATES[-1]

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ):
        c = self.config
        real_max = int(real_seqlen_kv.max().item())
        if real_max < self._MIN_CTX:
            return _gqa_decode_paged_no_split_op(
                self.batch,
                self.heads,
                self.groups,
                self.seqlen_kv,
                self.dim,
                self.page_size,
                self.sm_scale,
                self.softcap,
                self.dtype_str,
                64,
                c["block_N"],
                2,
                128,
                Q,
                K,
                V,
                real_seqlen_kv,
                block_table,
            )

        ctx_splits = self._ctx_splits_for(real_max)
        glse = torch.empty(
            (self.batch, self.heads, ctx_splits), dtype=torch.float32, device=Q.device
        )
        Output_partial = torch.empty(
            (self.batch, self.heads, ctx_splits, self.dim), dtype=torch.float32, device=Q.device
        )
        return _gqa_decode_paged_bs1_ctx_op(
            self.batch,
            self.heads,
            self.groups,
            self.seqlen_kv,
            self.dim,
            self.page_size,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            c["block_M"],
            c["block_N"],
            ctx_splits,
            c["threads"],
            Q,
            K,
            V,
            real_seqlen_kv,
            block_table,
            glse,
            Output_partial,
        )
