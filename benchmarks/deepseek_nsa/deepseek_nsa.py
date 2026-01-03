from benchmarks.benchmark import Benchmark
from top.ops import NativeSparseAttentionForwardOp
import torch
from torch.nn import functional as f
from top.kernels.deepseek_nsa.nsa_torch import naive_nsa

class NativeSparseAttentionForwardBenchmark(Benchmark):
    op_type = NativeSparseAttentionForwardOp

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
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.groups = groups
        self.selected_blocks = selected_blocks

        self.head_kv = self.heads // self.groups
        self.dtype = torch.float16
        self.tune = tune

    @property
    def total_flops(self):
        B = self.batch
        T = self.seq_len
        HQ = self.heads
        D = self.dim
        S = self.selected_blocks
        BS = self.block_size

        window_size = 0
        total_keys = S * BS + window_size
        flops = 4 * B * T * HQ * D * total_keys
        return flops
    
    @property
    def total_memory(self):
        return (self.batch * self.heads * (2 * self.seq_len) * self.dim * self.dtype.itemsize)


    def gen_inputs(self):
        Q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len, self.head_kv, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len, self.head_kv, self.dim, device='cuda', dtype=self.dtype)
        
        self.g_slc = torch.ones((self.batch, self.seq_len, self.heads), dtype=self.dtype, device="cuda").requires_grad_(True)
        self.g_swa = torch.ones((self.batch, self.seq_len, self.heads), dtype=self.dtype, device="cuda").requires_grad_(True)
        
        block_indices = torch.full((self.batch, self.seq_len, self.head_kv, self.selected_blocks), self.seq_len, dtype=torch.long, device="cuda")
        self.block_counts = torch.zeros((self.batch, self.seq_len, self.head_kv), dtype=torch.long, device="cuda")
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.head_kv): 
                    i_i = torch.randperm(max(1, (t // self.block_size)))[:self.selected_blocks]
                    block_indices[b, t, h, : len(i_i)] = i_i
                    self.block_counts[b, t, h] = (block_indices[b, t, h] != self.seq_len).sum().item()
        block_indices = block_indices.sort(-1)[0].to(torch.int32)
        return Q, K, V, block_indices

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, BlockIndices: torch.Tensor):
        print("running naive nsa ref_program")
        return naive_nsa(
            q=Q,
            k=K,
            v=V,
            g_slc=self.g_slc,
            g_swa=self.g_swa,
            block_indices=BlockIndices.to(torch.long),
            block_counts=self.block_counts,
            block_size=self.block_size,  
            scale=self.scale,
        ) 