import torch
from torch import nn
from top.functions.grouped_gemm import grouped_gemm_fn


class GROUPED_GEMM(nn.Module):

    def __init__(self, batch_sum, batch_count, N, K, dtype):
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.fn = grouped_gemm_fn(batch_sum, batch_count, N, K, dtype)

    def forward(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.fn(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
