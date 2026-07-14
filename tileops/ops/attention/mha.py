from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.attention import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
    GQAFwdWsPersistentCausalKernel,
    MHABwdKernel,
    MHABwdWgmmaPipelinedKernel,
    MHADecodeKernel,
    MHADecodePagedKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.utils import is_hopper

from ..op_base import Op
from .gqa import GroupedQueryAttentionFwdOp, _select_gqa_prefill_fwd_kernel_cls

__all__ = [
    "MultiHeadAttentionBwdOp",
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
    "MultiHeadAttentionDecodeWithKVCacheFwdOp",
    "MultiHeadAttentionFwdOp",
]


class MultiHeadAttentionFwdOp(Op):
    """Layout: BSHD.

    MHA is the heads_kv == heads specialization of GQA, so route the
    maintained forward path through the GQA prefill dispatcher while keeping
    the historical MHA return contract `(output, lse)`.
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self._gqa_op = GroupedQueryAttentionFwdOp(
            batch=batch,
            heads=heads,
            heads_kv=heads,
            seq_len=seq_len,
            dim=dim,
            is_causal=is_causal,
            dtype=dtype,
            kernel_map=self.kernel_map,
            tune=tune,
        )
        # GroupedQueryAttentionFwdOp instantiates its kernel lazily. MHA's
        # torch.compile smoke expects forward to call an already-built custom op,
        # so instantiate the same dense-path choice at construction time.
        self._kernel = (
            self._gqa_op._prefill_op._get_square_dense_kernel()
            if self._gqa_op._prefill_op._uses_square_dense_fast_path()
            else self._gqa_op.kernel
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_fwd_kernel": _select_gqa_prefill_fwd_kernel_cls(
                self.dim,
                self.is_causal,
                self.dtype,
                sm_scale=None,
                softcap=0.0,
                hopper=is_hopper(),
            ),
            "gqa_prefill_square_fwd_kernel": GQAFwdWsPersistentCausalKernel,
        }

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.kernel(q, k, v)


class MultiHeadAttentionBwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.prep_kernel = self.kernel_map["mha_bwd_preprocess_kernel"](batch, heads, seq_len, dim,
                                                                        self.dtype)
        self.kernel = self.kernel_map["mha_bwd_kernel"](
            batch, heads, seq_len, dim, is_causal, self.dtype, tune=tune)
        if not is_hopper():
            self.post_kernel = self.kernel_map["mha_bwd_postprocess_kernel"](batch, heads, seq_len,
                                                                             dim, self.dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "mha_bwd_preprocess_kernel":
                FlashAttnBwdPreprocessKernel,
            "mha_bwd_kernel":
                MHABwdWgmmaPipelinedKernel if is_hopper() else MHABwdKernel,
            "mha_bwd_postprocess_kernel":
                FlashAttnBwdPostprocessKernel if not is_hopper() else None,
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                do: torch.Tensor,
                lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        do = do.contiguous()
        delta = self.prep_kernel(o, do)
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk, dv = self.kernel(q, k, v, do, lse, delta, dq)
        dq = dq.to(self.dtype) if is_hopper() else self.post_kernel(dq)
        return dq, dk, dv


class MultiHeadAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mha_decode_kernel"](
            batch, heads, seqlen_q, seqlen_kv, dim, False, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mha_decode_kernel": MHADecodeKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        real_seqlen_kv = k.shape[1]
        if real_seqlen_kv < self.seqlen_kv:
            k = F.pad(
                k, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)
            v = F.pad(
                v, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)
        return self.kernel(q, k, v, real_seqlen_kv)


class MultiHeadAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged MHA decode with dynamic KV cache. Layout: Q [batch, seqlen_q, heads, dim] (BSHD);
    K, V physical cache [seqlen_kv, heads, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 page_size: int,
                 is_causal: bool = False,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mha_decode_paged_kernel"](
            batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mha_decode_paged_kernel": MHADecodePagedKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k, v, real_seqlen_kv, block_table)
