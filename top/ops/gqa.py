import torch
from .op import Op
from top.kernels import (gqa_fwd_kernel, gqa_fwd_wgmma_pipelined_kernel,
                         flashattn_bwd_preprocess_kernel, gqa_bwd_kernel,
                         gqa_bwd_wgmma_pipelined_kernel, Kernel, flashattn_bwd_postprocess_kernel)
from top.utils import is_hopper
from typing import Optional, Dict

__all__ = ['GroupQueryAttentionFwdOp', 'GroupQueryAttentionBwdOp']


class GroupQueryAttentionFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 heads_kv,
                 seq_len,
                 dim,
                 is_causal,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  #TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_fwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"gqa_fwd_kernel": gqa_fwd_wgmma_pipelined_kernel if is_hopper() else gqa_fwd_kernel}

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.kernel(Q, K, V)


class GroupQueryAttentionBwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 heads_kv,
                 seq_len,
                 dim,
                 is_causal,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  #TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.prep_kernel = self.kernel_map["gqa_bwd_preprocess_kernel"](batch, heads, seq_len, dim,
                                                                        self.dtype)
        self.kernel = self.kernel_map["gqa_bwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype, tune=tune)
        if not is_hopper():
            self.post_kernel = self.kernel_map["gqa_bwd_postprocess_kernel"](batch, heads, seq_len,
                                                                             dim, self.dtype)

    @property
    def default_kernel_map(self):
        return {
            "gqa_bwd_preprocess_kernel":
                flashattn_bwd_preprocess_kernel,
            "gqa_bwd_kernel":
                gqa_bwd_wgmma_pipelined_kernel if is_hopper() else gqa_bwd_kernel,
            "gqa_bwd_postprocess_kernel":
                flashattn_bwd_postprocess_kernel if not is_hopper() else None,
        }

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
                dO: torch.Tensor, lse: torch.Tensor):
        dO = dO.contiguous()
        delta = self.prep_kernel(O, dO)
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.zeros_like(K, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)
        self.kernel(Q, K, V, dO, lse, delta, dQ, dK, dV)
        dQ = dQ.to(self.dtype) if is_hopper() else self.post_kernel(dQ)
        dK, dV = dK.to(self.dtype), dV.to(self.dtype)
        return dQ, dK, dV
