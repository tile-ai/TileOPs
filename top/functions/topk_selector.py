from typing import Any

import torch
from torch.autograd.function import FunctionCtx

from top.ops import TopkSelectorOp

from .function import Function

__all__ = ['TopkSelectorFunc']


class TopkSelectorCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, index_scores: torch.Tensor, starts: torch.Tensor,
                ends: torch.Tensor, topk_selector_op: TopkSelectorOp) -> torch.Tensor:
        indices = topk_selector_op(index_scores, starts, ends)
        return indices

    @staticmethod
    def backward(ctx: FunctionCtx, indices: torch.Tensor) -> Any:
        raise RuntimeError("Inference-only op")


class TopkSelectorFunc(Function):

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 topk: int,
                 in_dtype: str,
                 out_dtype: str,
                 tune: bool = False):

        self.batch = batch
        self.seq_len = seq_len
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

        self.topk_selector_op = TopkSelectorOp(batch, seq_len, topk, in_dtype, out_dtype)

    def forward(self, index_scores: torch.Tensor, starts: torch.Tensor,
                ends: torch.Tensor) -> torch.Tensor:
        return TopkSelectorCtx.apply(index_scores, starts, ends, self.topk_selector_op)
