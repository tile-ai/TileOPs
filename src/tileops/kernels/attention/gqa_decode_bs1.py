"""Warp-specialized batch=1 GQA decode kernel (Hopper), context-split.

``GQADecodeBs1Kernel`` dispatches on the runtime K/V sequence extent: lengths >= 1024
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

from .call_spec import decode_bs1_region
from .gqa_decode_bs1_common import (
    COMPILE_FLAGS,
    RING_DEPTH,
    GQADecodeBs1KernelMixin,
    make_gqa_decode_bs1_combine,
    make_gqa_decode_bs1_split,
)
from .online_softmax import LOG2E

__all__ = ["GQADecodeBs1Kernel"]


@functools.lru_cache(maxsize=32)
def _gqa_decode_bs1_ctx_kernel(batch, heads, groups, dim, sm_scale, softcap, dtype):
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
            glse: T.Tensor(lse_shape, accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            split(Q, K, V, K, seqlen_kv, glse, Output_partial)
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
    block_M: int,
    block_N: int,
    ctx_splits: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
) -> torch.Tensor:
    return _gqa_decode_bs1_ctx_kernel(batch, heads, groups, dim, sm_scale, softcap, dtype)(
        block_M, block_N, ctx_splits, threads
    )(Q, K, V, glse, Output_partial)


@_gqa_decode_bs1_ctx_op.register_fake
def _(
    batch: int,
    heads: int,
    groups: int,
    dim: int,
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
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(Q)


class GQADecodeBs1Kernel(GQADecodeBs1KernelMixin, Kernel):
    """Hopper warp-specialized batch=1 GQA decode kernel with a context-length switch.

    ``forward`` dispatches on the runtime K/V sequence extent: >= 1024 runs the context-only
    split, shorter lengths run the generic non-split GQA decode kernel.
    """

    supported_archs: list[int] = [90]

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
        del q_scale, k_scale, v_scale, rope_cos, rope_sin
        Q = q.squeeze(1)
        K = k
        V = v
        real_seqlen_kv = k.shape[1]
        c = self.config
        if real_seqlen_kv < self._MIN_CTX:
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

        ctx_splits = self._ctx_splits_for(real_seqlen_kv)
        glse, Output_partial = self._allocate_partials(Q, ctx_splits)
        output = _gqa_decode_bs1_ctx_op(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
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
            glse,
            Output_partial,
        )
        return output.unsqueeze(1)
