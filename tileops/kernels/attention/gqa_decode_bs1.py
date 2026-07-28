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

from .gqa_decode_bs1_common import (
    COMPILE_FLAGS,
    RING_DEPTH,
    GQADecodeBs1KernelMixin,
    make_gqa_decode_bs1_combine,
    make_gqa_decode_bs1_split,
)

__all__ = ["GQADecodeBs1Kernel"]


@functools.lru_cache(maxsize=32)
def _gqa_decode_bs1_ctx_kernel(batch, heads, groups, seqlen_kv, dim, sm_scale, softcap, dtype):
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    scale = score_scale * LOG2E
    accum_dtype = "float"
    kv_group_num = heads // groups

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=COMPILE_FLAGS)
    def _func(block_M, block_N, ctx_splits, threads):
        shape_q = [batch, heads, dim]
        shape_k = [batch, seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
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
            False,
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
        def gqa_decode_bs1_ctx(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_k, dtype),
                real_seqlen_kv: T.int32,
                glse: T.Tensor(lse_shape, accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            split(Q, K, V, K, real_seqlen_kv, glse, Output_partial)
            combine(glse, Output_partial, Output)

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


class GQADecodeBs1Kernel(GQADecodeBs1KernelMixin, Kernel):
    """Hopper warp-specialized batch=1 GQA decode kernel with a context-length switch.

    ``forward`` dispatches on the runtime ``real_seqlen_kv``: >= 1024 runs the context-only
    split, shorter lengths run the generic non-split GQA decode kernel.
    """

    supported_archs: list[int] = [90]

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

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, real_seqlen_kv: int):
        c = self.config
        if real_seqlen_kv < self._MIN_CTX:
            return _gqa_decode_no_split_op(self.batch, self.heads, self.groups, self.seqlen_kv,
                                           real_seqlen_kv, self.dim, self.sm_scale, self.softcap,
                                           self.dtype_str, 64, 128, 2, 128, Q, K, V)

        ctx_splits = self._ctx_splits_for(real_seqlen_kv)
        glse, Output_partial = self._allocate_partials(Q, ctx_splits)
        return _gqa_decode_bs1_ctx_op(self.batch, self.heads, self.groups, self.seqlen_kv,
                                      real_seqlen_kv, self.dim, self.sm_scale, self.softcap,
                                      self.dtype_str, c["block_M"], c["block_N"], ctx_splits,
                                      c["threads"], Q, K, V, glse, Output_partial)
