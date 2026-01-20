from typing import Optional, Any

import torch
from torch.autograd.function import FunctionCtx

from top.ops import FP8LightingIndexerOp

from .function import Function

__all__ = ['FP8LightingIndexerFunc']


class FP8LightingindexerCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, index_q: torch.Tensor, index_k: torch.Tensor,
                weights: torch.Tensor, cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor,
                fwd_op: FP8LightingIndexerOp) -> torch.Tensor:
        o = fwd_op(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke)
        return fwd_op(o)

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
                 tune=False) -> None:
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.clean_logits = clean_logits
        self.config = config
        self.tune = tune
        self.fwd_op = FP8LightingIndexerOp(
            seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune=tune)

    def forward(self, index_q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor,
                cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        return FP8LightingindexerCtx.apply(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke,
                                           self.fwd_op)
