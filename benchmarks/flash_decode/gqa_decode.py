from benchmarks.benchmark import Benchmark
from top.ops import GroupQueryAttentionDecodeWithKVCacheOp
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Tuple


class GroupQueryAttentionDecodeBenchmark(Benchmark):

    op_type = GroupQueryAttentionDecodeWithKVCacheOp

    def __init__(self, batch: int, heads: int, groups: int, seq_len_kv: int, dim: int,
                 dtype: torch.dtype):
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype

    @property
    def total_flops(self) -> float:
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len_kv * self.dim
        flops = flops_per_matmul * 2
        return flops

    @property
    def total_memory(self) -> float:
        # Q: batch * 1 * heads * dim
        # K, V: batch * seq_len_kv * heads_kv * dim
        # Output: batch * 1 * heads * dim
        return 2 * self.batch * self.dim * self.dtype.itemsize * (
            self.heads + self.groups * self.seq_len_kv)

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.groups, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len_kv, self.groups, self.dim, device='cuda', dtype=self.dtype)
        return Q, K, V

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.unsqueeze(1).transpose(1, 2)  # [B, H, 1, D]
        k_bhsd = k.transpose(1, 2)  # [B, H, S_kv, D]
        v_bhsd = v.transpose(1, 2)  # [B, H, S_kv, D]
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_bhsd = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).squeeze(1).contiguous()
        return output
