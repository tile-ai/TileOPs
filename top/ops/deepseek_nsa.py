import torch
from top.ops.op import Op
from top.kernels.kernel import Kernel
from top.kernels.deepseek_nsa.nsa_fwd import nsa_fwd_kernel
from top.kernels.deepseek_nsa.mean_pooling_fwd import mean_pooling_fwd_kernel
from typing import Optional, Dict
from fla.ops.common.utils import prepare_chunk_indices

__all__ = ["NativeSparseAttentionForwardOp", "MeanPoolingForwardOp"]


class MeanPoolingForwardOp(Op):

    def __init__(self,
                 batch_size: int,
                 total_seqlen: int,
                 total_chunks: int,
                 heads: int,
                 dim: int,
                 chunk_size: int,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False) -> torch.Tensor:
        self.batch_size = batch_size
        self.total_seqlen = total_seqlen
        self.total_chunks = total_chunks
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.tune = tune

        self.dispatch_kernel(kernel_map)

        self.kernel = self.kernel_map["mean_pooling_fwd_kernel"](
            batch_size=self.batch_size,
            total_seqlen=self.total_seqlen,
            total_chunks=self.total_chunks,
            heads=self.heads,
            dim=self.dim,
            chunk_size=self.chunk_size,
            tune=self.tune)

    @property
    def default_kernel_map(self):
        return {"mean_pooling_fwd_kernel": mean_pooling_fwd_kernel}

    def forward(self, x_unpad: torch.Tensor, cu_seqlens: torch.Tensor, chunk_indices: torch.Tensor):
        return self.kernel(x_unpad, cu_seqlens, chunk_indices)

    # def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, chunk_indices: torch.Tensor):
    #     out = self.kernel(x, cu_seqlens, chunk_indices)
    #     print(self.batch_size)
    #     return out.view(self.batch_size,-1, self.heads, self.dim)


class NativeSparseAttentionForwardOp(Op):

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 scale=None,
                 block_size=64,
                 groups=1,
                 selected_blocks=16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
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

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["nsa_fwd_kernel"](
            self.batch,
            self.heads,
            self.seq_len,
            self.dim,
            self.is_causal,
            self.scale,
            self.block_size,
            self.groups,
            self.selected_blocks,
            tune=self.tune)

    @property
    def default_kernel_map(self):
        return {"nsa_fwd_kernel": nsa_fwd_kernel}

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                BlockIndices: torch.Tensor):
        return self.kernel(Q, K, V, BlockIndices)


def mean_pooling_tilelang(x_unpad, cu_seqlens, chunk_size, block_D=64):
    total_T, H, D = x_unpad.shape
    B = cu_seqlens.shape[0] - 1
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    total_chunks = chunk_indices.shape[0]

    op = MeanPoolingForwardOp(
        batch_size=B,
        total_seqlen=total_T,
        total_chunks=total_chunks,
        heads=H,
        dim=D,
        chunk_size=chunk_size,
        tune=True)
    return op.forward(x_unpad, cu_seqlens, chunk_indices)
