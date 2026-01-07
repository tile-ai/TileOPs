import torch
from top.ops.op import Op
from top.kernels.kernel import Kernel
from top.kernels.deepseek_nsa.nsa_fwd import nsa_fwd_kernel
from top.kernels.deepseek_nsa.mean_pooling_fwd import mean_pooling_fwd_kernel
from top.kernels.deepseek_nsa.nsa_topk import nsa_topk_fwd_kernel
from typing import Optional, Dict

import argparse

__all__ = ["NativeSparseAttentionForwardOp", "MeanPoolingForwardOp", "NsaTopkForwardOp"]


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


class NsaTopkForwardOp(Op):

    def __init__(self,
                 M: int,
                 N: int,
                 topk: int,
                 dtype: str,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.M = M
        self.N = N
        self.topk = topk
        self.dtype = dtype
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["nsa_topk_fwd_kernel"](
            M=self.M, N=self.N, topk=self.topk, dtype=self.dtype, tune=self.tune)

    @property
    def default_kernel_map(self):
        return {"nsa_topk_fwd_kernel": nsa_topk_fwd_kernel}

    def forward(self, logits: torch.Tensor):
        return self.kernel(logits)


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


def ref_program(logits, top_k):
    top_k_gates, top_k_indices = logits.topk(top_k, dim=1)

    return top_k_gates, top_k_indices.to(torch.int32)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=320, help="num_tokens")
    parser.add_argument("--N", type=int, default=128, help="num_experts")
    parser.add_argument("--topk", type=int, default=6, help="topk")
    args = parser.parse_args(argv)
    M, N, topk = args.M, args.N, args.topk

    logits = torch.rand((M, N), device="cuda", dtype=torch.float32)

    op = NsaTopkForwardOp(M=M, N=N, topk=topk, dtype="float32", tune=True)
    tl_gates, tl_indices = op.forward(logits)

    torch_gates, torch_indices = ref_program(logits, topk)

    # test accuracy
    torch.testing.assert_close(tl_gates, torch_gates)
    torch.testing.assert_close(tl_indices, torch_indices)

    assert torch.allclose(
        tl_gates, torch_gates, atol=1e-4, rtol=1e-4), "NsaTopkForwardOp is not accurate"
    assert torch.allclose(
        tl_indices, torch_indices, atol=1e-4, rtol=1e-4), "NsaTopkForwardOp is not accurate"


if __name__ == "__main__":
    main()
