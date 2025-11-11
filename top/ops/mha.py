import torch
from .op import Op
from top.kernels import (mha_fwd_kernel, mha_fwd_wgmma_pipelined_kernel,
                         flashattn_bwd_preprocess_kernel, mha_bwd_kernel,
                         mha_bwd_wgmma_pipelined_kernel, Kernel, flashattn_bwd_postprocess_kernel)
from top.utils import is_hopper
from typing import Optional, Dict

__all__ = ['mha_fwd', 'mha_bwd']


class mha_fwd(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  #TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mha_fwd_kernel"](
            batch, heads, seq_len, dim, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"mha_fwd_kernel": mha_fwd_wgmma_pipelined_kernel if is_hopper() else mha_fwd_kernel}

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.kernel(Q, K, V)


class mha_bwd(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  #TODO: support s_q != s_kv
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
    def default_kernel_map(self):
        return {
            "mha_bwd_preprocess_kernel":
                flashattn_bwd_preprocess_kernel,
            "mha_bwd_kernel":
                mha_bwd_wgmma_pipelined_kernel if is_hopper() else mha_bwd_kernel,
            "mha_bwd_postprocess_kernel":
                flashattn_bwd_postprocess_kernel if not is_hopper() else None,
        }

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
                dO: torch.Tensor, lse: torch.Tensor):
        dO = dO.contiguous()
        delta = self.prep_kernel(O, dO)
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK, dV = self.kernel(Q, K, V, dO, lse, delta, dQ)
        dQ = dQ.to(self.dtype) if is_hopper() else self.post_kernel(dQ)
        return dQ, dK, dV
