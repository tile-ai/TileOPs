from typing import Dict, Optional

import torch

from top.kernels.flash_attn import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
    MhaBwdKernel,
    MhaBwdWgmmaPipelinedKernel,
    MhaFwdKernel,
    MhaFwdWgmmaPipelinedKernel,
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
        self.kernel = self.kernel_map["MhaFwdKernel"](
            batch, heads, seq_len, dim, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"MhaFwdKernel": MhaFwdWgmmaPipelinedKernel if is_hopper() else MhaFwdKernel}

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
        self.kernel = self.kernel_map["MhaBwdKernel"](
            batch, heads, seq_len, dim, is_causal, self.dtype, tune=tune)
        if not is_hopper():
            self.post_kernel = self.kernel_map["mha_bwd_postprocess_kernel"](batch, heads, seq_len,
                                                                             dim, self.dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "mha_bwd_preprocess_kernel":
                FlashAttnBwdPreprocessKernel,
            "MhaBwdKernel":
                MhaBwdWgmmaPipelinedKernel if is_hopper() else MhaBwdKernel,
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
