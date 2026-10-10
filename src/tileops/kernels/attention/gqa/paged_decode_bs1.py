"""Warp-specialized batch=1 paged GQA decode kernel (SM90), context-split.

``GQADecodePagedBs1Kernel`` dispatches on the static page-table capacity: capacities >= 1024
run a context-only warp-specialized split (one TMA producer warp feeding a four-warp
wgmma consumer warpgroup, exp2-domain online softmax, fp32 partial reduce via a combine
kernel); shorter lengths fall back to the generic paged non-split decode kernel.
Logical KV tiles are translated through the page table before TMA. SM90-only, low-level
``tma_copy`` / ``mbarrier`` / ``wgmma_gemm``.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import (
    AttentionCall,
    GQAPagedFwdInterface,
)
from tileops.kernels.attention.gqa.decode_bs1_common import (
    COMPILE_FLAGS,
    RING_DEPTH,
    GQADecodeBs1KernelMixin,
    make_gqa_decode_bs1_combine,
    make_gqa_decode_bs1_split,
)
from tileops.kernels.attention.gqa.paged_decode import (
    gqa_decode_no_split_paged_kernel,
    gqa_decode_paged_block_ns,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["GQADecodePagedBs1Kernel"]


@functools.lru_cache(maxsize=32)
def _gqa_decode_paged_bs1_ctx_kernel(
    batch,
    heads,
    groups,
    seqlen_kv,
    dim,
    page_size,
    max_pages_per_req,
    sm_scale,
    softcap,
    dtype,
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
            block_table: T.Tensor([batch, max_pages_per_req], T.int32),
            glse: T.Tensor(lse_shape, accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            split(Q, K, V, block_table, real_seqlen_kv, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return gqa_decode_paged_bs1_ctx

    return _func


class GQADecodePagedBs1Kernel(GQADecodeBs1KernelMixin, Kernel, GQAPagedFwdInterface):
    """SM90 warp-specialized batch=1 paged GQA decode kernel.

    ``forward`` specializes from the static page-table capacity; the kernels consume
    live cache lengths on the device, including during CUDA Graph replay.
    """

    supported_archs: list[int] = [90]
    # The batch-1 shape, inside what the packed kernel also serves.
    preferred_over = frozenset({"gqa_paged_varlen_kernel"})

    @classmethod
    def applies(cls, call) -> bool:
        # The page-tile question is asked of this class, so a kernel_map
        # override answers for its own tiling rather than for the shipped one.
        return (
            call.max_seqlen_q == 1
            and call.paged_decode_refusal is None
            and call.decode_bs1_region
            # The warp-specialized softmax reduces raw QK before applying the scale.
            and (call.sm_scale is None or call.sm_scale >= 0.0)
            and cls.block_n_for_page_size(call.page_size) is not None
        )

    @staticmethod
    def block_n_for_page_size(page_size: int) -> Optional[int]:
        """Return a page-contained WGMMA N tile, or None when the fast path is unsafe."""
        try:
            block_n = gqa_decode_paged_block_ns(page_size)[0]
        except ValueError:
            return None
        if page_size == 64 or block_n == 128:
            return block_n
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        # The device index is in the identity: the kernel is compiled for its architecture.
        index = call.device.index if call.device is not None else None
        args = (
            call.batch,
            call.heads,
            call.heads_kv,
            call.seqlen_kv,
            call.dim,
            call.page_size,
            call.max_pages_per_req,
            call.dtype,
        )
        extra = dict(sm_scale=call.sm_scale, softcap=call.softcap)
        identity = (*args, *extra.values(), index)
        return identity, lambda: cls(*args, **extra, device_index=index)

    def __init__(
        self,
        batch,
        heads,
        groups,
        seqlen_kv,
        dim,
        page_size,
        max_pages_per_req,
        dtype="float16",
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune=False,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.max_pages_per_req = max_pages_per_req
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
        if self.max_pages_per_req <= 0:
            raise ValueError("max_pages_per_req must be positive")
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        block_n = self.block_n_for_page_size(self.page_size)
        if block_n is None:
            raise ValueError("batch=1 paged decode requires page_size=64 or a multiple of 128")
        return {"block_M": 64, "block_N": block_n, "threads": 160}

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        """``cu_seqlens_q`` is unread: every request of this region carries one query token."""
        c = self.config
        capacity = self.max_pages_per_req * self.page_size
        if capacity < self._MIN_CTX:
            kernel = gqa_decode_no_split_paged_kernel(
                self.batch,
                self.heads,
                self.groups,
                1,
                self.seqlen_kv,
                self.dim,
                self.page_size,
                self.max_pages_per_req,
                False,
                self.sm_scale,
                self.softcap,
                self.dtype_str,
            )(64, c["block_N"], 2, 128)
            q = Q.view(self.batch, 1, self.heads, self.dim)
            return kernel(q, K, V, real_seqlen_kv, block_table).view(Q.shape)

        ctx_splits = self._ctx_splits_for(capacity)
        glse, Output_partial = self._allocate_partials(Q, ctx_splits)
        return _gqa_decode_paged_bs1_ctx_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.seqlen_kv,
            self.dim,
            self.page_size,
            self.max_pages_per_req,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
        )(c["block_M"], c["block_N"], ctx_splits, c["threads"])(
            Q, K, V, real_seqlen_kv, block_table, glse, Output_partial
        )
