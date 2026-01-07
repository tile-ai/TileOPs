from typing import Tuple

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.benchmark import Benchmark
from top.ops import MultiHeadAttentionDecodeWithKVCacheOp


class MultiHeadAttentionDecodeBenchmark(Benchmark):

    op_type = MultiHeadAttentionDecodeWithKVCacheOp

    def __init__(self, batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                 dtype: torch.dtype):
        self.batch = batch
        self.heads = heads
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype

    @property
    def total_flops(self) -> float:
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len_q * self.seq_len_kv * self.dim
        flops = flops_per_matmul * 2
        return flops

    @property
    def total_memory(self) -> float:
        # Q: batch * seq_len_q * heads * dim
        # K, V: batch * seq_len_kv * heads * dim
        # Output: batch * seq_len_q * heads * dim
        return (self.batch * self.heads * (2 * self.seq_len_q + 2 * self.seq_len_kv) * self.dim *
                self.dtype.itemsize)

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(
            self.batch, self.seq_len_q, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.heads, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len_kv, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return Q, K, V

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.transpose(1, 2)  # [B, H, S_q, D]
        k_bhsd = k.transpose(1, 2)  # [B, H, S_kv, D]
        v_bhsd = v.transpose(1, 2)  # [B, H, S_kv, D]
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output
