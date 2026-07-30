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

from tileops.kernels.attention.gqa_decode_paged import (
    _gqa_decode_paged_no_split_op,
    gqa_decode_paged_block_n,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import LOG2E

from .gqa_decode_bs1_common import (
    COMPILE_FLAGS,
    RING_DEPTH,
    GQADecodeBs1KernelMixin,
    make_gqa_decode_bs1_combine,
    make_gqa_decode_bs1_split,
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

        @T.macro
        def load_kv(K, V, block_table, bid, hid, base, k, Ks, Vs, ready):
            logical_block = base // block_N + k
            blocks_per_page = page_size // block_N
            page_idx = logical_block // blocks_per_page
            block_in_page = logical_block % blocks_per_page
            physical_block = block_table[bid, page_idx] * blocks_per_page + block_in_page
            T.tma_copy(
                K[physical_block * block_N : (physical_block + 1) * block_N, hid, :],
                Ks[k % RING_DEPTH, :, :],
                barrier=ready[k % RING_DEPTH],
            )
            T.tma_copy(
                V[physical_block * block_N : (physical_block + 1) * block_N, hid, :],
                Vs[k % RING_DEPTH, :, :],
                barrier=ready[k % RING_DEPTH],
            )

        split = make_gqa_decode_bs1_split(
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
            True,
            load_kv,
        )
        combine = make_gqa_decode_bs1_combine(
            batch,
            heads,
            ctx_splits,
            dim,
            dtype,
            accum_dtype,
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
            split(Q, K, V, block_table, real_seqlen_kv, glse, Output_partial)
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


class GQADecodePagedBs1Kernel(GQADecodeBs1KernelMixin, Kernel):
    """Hopper warp-specialized batch=1 paged GQA decode kernel.

    ``forward`` dispatches on the runtime ``real_seqlen_kv``: >= 1024 runs the context-only
    split, shorter lengths run the generic non-split paged GQA decode kernel.
    """

    supported_archs: list[int] = [90]

    @staticmethod
    def block_n_for_page_size(page_size: int) -> Optional[int]:
        """Return a page-contained WGMMA N tile, or None when the fast path is unsafe."""
        try:
            block_n = gqa_decode_paged_block_n(page_size)
        except ValueError:
            return None
        if page_size == 64 or block_n == 128:
            return block_n
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
        glse, Output_partial = self._allocate_partials(Q, ctx_splits)
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
