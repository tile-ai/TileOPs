from typing import Dict, Optional

import torch

from top.kernels.flash_attn import (
    flashattn_bwd_postprocess_kernel,
    flashattn_bwd_preprocess_kernel,
    mha_bwd_kernel,
    mha_bwd_wgmma_pipelined_kernel,
    mha_fwd_kernel,
    mha_fwd_wgmma_pipelined_kernel,
)
from top.kernels.kernel import Kernel
from top.utils import is_hopper

from .op import Op

__all__ = ['MultiHeadAttentionFwdOp', 'MultiHeadAttentionBwdOp']


class MultiHeadAttentionFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
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
        self.kernel = self.kernel_map["mha_fwd_kernel"](batch,
                                                        heads,
                                                        seq_len,
                                                        dim,
                                                        is_causal,
                                                        self.dtype,
                                                        tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mha_fwd_kernel": mha_fwd_wgmma_pipelined_kernel if is_hopper() else mha_fwd_kernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k, v)


class MultiHeadAttentionBwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
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
        self.kernel = self.kernel_map["mha_bwd_kernel"](batch,
                                                        heads,
                                                        seq_len,
                                                        dim,
                                                        is_causal,
                                                        self.dtype,
                                                        tune=tune)
        if not is_hopper():
            self.post_kernel = self.kernel_map["mha_bwd_postprocess_kernel"](batch, heads, seq_len,
                                                                             dim, self.dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "mha_bwd_preprocess_kernel":
                flashattn_bwd_preprocess_kernel,
            "mha_bwd_kernel":
                mha_bwd_wgmma_pipelined_kernel if is_hopper() else mha_bwd_kernel,
            "mha_bwd_postprocess_kernel":
                flashattn_bwd_postprocess_kernel if not is_hopper() else None,
        }

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            o: torch.Tensor,
            do: torch.Tensor,  # noqa: VNE002
            lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        do = do.contiguous()  # noqa: VNE002
        delta = self.prep_kernel(o, do)
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk, dv = self.kernel(q, k, v, do, lse, delta, dq)
        dq = dq.to(self.dtype) if is_hopper() else self.post_kernel(dq)
        return dq, dk, dv
