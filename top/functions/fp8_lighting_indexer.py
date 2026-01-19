from typing import Dict, Optional, Tuple, Any


import torch
from torch.autograd.function import FunctionCtx

from top.ops import FP8LightingIndexerOp

from top.kernels.kernel import Kernel

from .function import Function

__all__ = ['FP8LightingIndexerFunc']

class FP8LightingindexerCtx(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: torch.Tensor,
                fwd_op: FP8LightingIndexerOp) -> torch.Tensor:
        o = fwd_op(input)
        return o
    
    @staticmethod
    def backward(ctx: FunctionCtx, do: torch.Tensor) -> Any:
        raise RuntimeError("Inference-only op")
    
class FP8LightingIndexerFunc(Function):
    def __init__(self,
                 seq_len,
                 heads,
                 index_dim,
                 seq_len_kv,
                 clean_logits=True,
                 config: Optional[dict] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False) -> None:
        self.seq_len = seq_len
        self.heads = heads  
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.clean_logits = clean_logits
        self.config = config
        self.kernel_map = kernel_map
        self.tune = tune
        self.fwd_op = FP8LightingIndexerOp(
            seq_len, heads, index_dim, seq_len_kv, clean_logits, config, kernel_map, tune=tune)


    def forward(self, index_q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor,
                cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        index_q = index_q.to(torch.float8_e4m3fn)
        return FP8LightingindexerCtx.apply(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ks, self.fwd_op)