import torch
from torch import nn

from top.functions import GroupedGemmFunc


class GroupedGemmLayer(nn.Module):

    def __init__(self, batch_sum: int, batch_count: int, n: int, k: int, dtype: str):
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = n
        self.K = k
        self.dtype = dtype
        self.fn = GroupedGemmFunc(batch_sum, batch_count, n, k, dtype)

    def forward(self, a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return self.fn(a, b, batch_sizes, batch_offsets, batch_padded_offsets)
