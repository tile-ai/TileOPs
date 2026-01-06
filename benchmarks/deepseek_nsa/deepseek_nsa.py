from benchmarks.benchmark import Benchmark
from top.ops import NativeSparseAttentionForwardOp
from top.ops import MeanPoolingForwardOp

import torch
from torch.nn import functional as f

from typing import Tuple, Any, Optional
from native_sparse_attention.ops.naive import naive_nsa
from native_sparse_attention.ops.parallel import parallel_nsa_fwd
from fla.ops.utils import mean_pooling

from fla.ops.common.utils import prepare_chunk_indices


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
        
        self.o_slc = torch.empty((self.batch, self.seq_len, self.heads, self.dim), dtype=self.dtype, device="cuda")
        self.lse_slc = torch.empty((self.batch, self.seq_len, self.heads, self.dim), dtype=torch.float, device="cuda")

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


    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, BlockIndices: torch.Tensor) -> torch.Tensor:
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
    

    def baseline_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, BlockIndices: torch.Tensor)-> torch.Tensor:
        o, lse = parallel_nsa_fwd(
            q=Q,
            k=K,
            v=V,
            block_indices=BlockIndices,
            block_counts=self.block_counts,
            block_size=self.block_size,
            scale=self.scale,
        )
        return o


    def baseline_profile(self, *inputs: Any, warmup: int = 100, rep: int = 100, device: str = "cuda:0") -> Any:
        print("===== Profiling FLA NSA_Fwd backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FLA", warmup=warmup, rep=rep, device=device)
    

class MeanPoolingForwardBenchmark(Benchmark):
    op_type = MeanPoolingForwardOp

    def __init__(
        self,
        batch_size,
        total_seqlen,
        total_chunks,
        heads,
        dim,
        chunk_size,
        tune= True
    ):
        self.batch_size = batch_size
        self.total_seqlen = total_seqlen
        self.total_chunks = total_chunks
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.tune= tune
        self.dtype = torch.float16

    @property
    def total_flops(self):
        flops = self.heads * self.dim * (self.total_seqlen + self.total_chunks)
        return flops
    
    @property
    def total_memory(self):
        return self.heads*self.dim*(self.total_seqlen+self.total_chunks)*self.dtype.itemsize + 16*self.total_chunks
        
    def gen_inputs(self):
        x_unpad = torch.randn(self.total_seqlen, self.heads, self.dim, device='cuda', dtype=self.dtype)
        # fixed length
        b = self.batch_size
        t = self.total_seqlen//b

        cu_seqlens = torch.arange(0, (b + 1) * t, t, dtype=torch.int32, device='cuda') 
        chunk_indices = prepare_chunk_indices(cu_seqlens, self.chunk_size)
        
        return x_unpad, cu_seqlens, chunk_indices


    def ref_program(self, x_unpad:torch.Tensor, cu_seqlens:torch.Tensor, chunk_indices:torch.Tensor) -> torch.Tensor:
        b = self.batch_size
        t = self.total_seqlen//b
        x = x_unpad.view(b, t, self.heads, self.dim)
         
        return mean_pooling(x, chunk_size=self.chunk_size, cu_seqlens=None, head_first=False).view(-1,self.heads, self.dim)
        

    def baseline_program(self, x_unpad:torch.Tensor, cu_seqlens:torch.Tensor, chunk_indices:torch.Tensor) -> torch.Tensor:
        b = self.batch_size
        t = self.total_seqlen//b
        x = x_unpad.view(b, t, self.heads, self.dim)
        return mean_pooling(x, chunk_size=self.chunk_size, cu_seqlens=None, head_first=False).view(-1,self.heads, self.dim)
        


    def baseline_profile(self, *inputs: Any, warmup: int = 100, rep: int = 100, device: str = "cuda:0") -> Any:
        print("===== Profiling Mean Pooling_Fwd backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="Mean Pooling", warmup=warmup, rep=rep, device=device)
    
