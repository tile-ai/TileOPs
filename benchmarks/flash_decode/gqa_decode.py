from benchmarks.benchmark import Benchmark
from top.ops import GroupQueryAttentionDecodeOp
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


class GroupQueryAttentionDecodeBenchmark(Benchmark):

    op_type = GroupQueryAttentionDecodeOp

    def __init__(self, batch, heads, groups, seq_len_kv, dim, dtype):
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype

    @property
    def total_flops(self):
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len_kv * self.dim
        flops = flops_per_matmul * 2
        return flops

    @property
    def total_memory(self):
        # Q: batch * 1 * heads * dim
        # K, V: batch * seq_len_kv * heads_kv * dim
        # Output: batch * 1 * heads * dim
        return 2 * self.batch * self.dim * self.dtype.itemsize * (
            self.heads + self.groups * self.seq_len_kv)

    def gen_inputs(self):
        Q = torch.randn(self.batch, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.groups, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len_kv, self.groups, self.dim, device='cuda', dtype=self.dtype)
        mask = torch.randint(
            0, 2, (self.batch, self.seq_len_kv, self.groups), device='cuda', dtype=torch.uint8)
        return Q, K, V, mask

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
        q_bhsd = Q.unsqueeze(1).transpose(1, 2)  # [B, H, 1, D]
        k_bhsd = K.transpose(1, 2)  # [B, H, S_kv, D]
        v_bhsd = V.transpose(1, 2)  # [B, H, S_kv, D]
        mask = mask.to(torch.bool).transpose(1, 2).unsqueeze(2)  # [B, G, 1, S_kv]
        mask = mask.expand(self.batch, self.groups, self.heads // self.groups,
                           self.seq_len_kv).reshape(self.batch, self.heads,
                                                    self.seq_len_kv).unsqueeze(2)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, attn_mask=mask, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).squeeze(1).contiguous()
        return output
