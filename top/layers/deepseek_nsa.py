import torch
from torch import nn
from top.functions import NativeSparseAttentionFunc


class NativeSparseAttentionLayer(nn.Module):
    def __init__(
        self,
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        scale=None,
        block_size=64,
        groups=1,
        selected_blocks=16,
        tune=False
    ):
        super().__init__()

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.groups = groups
        self.selected_blocks = selected_blocks
        self.tune = tune

        self.fn = NativeSparseAttentionFunc(
            batch, heads, seq_len, dim, is_causal, scale, block_size, groups, selected_blocks, tune=tune)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, BlockIndices: torch.Tensor) -> torch.Tensor:
        return self.fn(Q, K, V, BlockIndices)
