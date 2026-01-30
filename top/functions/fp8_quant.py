from typing import Any, Tuple

import torch
from torch.autograd.function import FunctionCtx

from top.ops import Fp8QuantOp

from .function import Function


class Fp8QuantCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, input_tensor: torch.Tensor, fwd_op: Fp8QuantOp) -> Any:
        scale_tensor, output_tensor = fwd_op(input_tensor)
        return scale_tensor, output_tensor

    @staticmethod
    def backward(ctx: FunctionCtx, *args: Any) -> Any:
        raise RuntimeError("Inference-only op")


class Fp8QuantFunc(Function):

    def __init__(self,
                 seq_len_kv: int,
                 index_dim: int,
                 in_dtype: torch.dtype = torch.float16,
                 tune: bool = False):
        self.seq_len_kv = seq_len_kv
        self.index_dim = index_dim
        self.in_dtype = in_dtype

        self.fwd_op = Fp8QuantOp(seq_len_kv, index_dim, in_dtype)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return Fp8QuantCtx.apply(input_tensor, self.fwd_op)
