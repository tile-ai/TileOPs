from typing import Tuple

import pytest
import torch
from einops import einsum, rearrange
from torch.nn import functional as F

from tests.test_base import TestBase, FixtureBase
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp


class MlaDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, head_num_kv, seq_len_kv, dim, dim_pe, dtype, tune", [
            (32, 128, 1, 8192, 512, 64, torch.float16, False),
        ]),
    ]


class MlaDecodeTest(TestBase):

    def __init__(self, batch: int, heads: int, head_num_kv: int, seq_len_kv: int, dim: int,
                 dim_pe: int, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.head_num_kv = head_num_kv
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_pe = dim_pe
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device='cuda', dtype=self.dtype)
        Q_pe = torch.randn(self.batch, self.heads, self.dim_pe, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.head_num_kv,
            self.dim,
            device='cuda',
            dtype=self.dtype)
        K_pe = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.head_num_kv,
            self.dim_pe,
            device='cuda',
            dtype=self.dtype)
        return Q, Q_pe, K, K_pe

    def ref_program(self, q: torch.Tensor, q_pe: torch.Tensor, kv: torch.Tensor,
                    k_pe: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        - q (Tensor): [batch, heads, dim]
        - q_pe (Tensor): [batch, heads, dim_pe]
        - kv (Tensor): [batch, seqlen_kv, head_num_kv, dim]
        - k_pe (Tensor): [batch, seqlen_kv, head_num_kv, dim_pe]
        Outputs:
        - output (Tensor): [batch, heads, dim]
        """
        dim = q.shape[-1]
        dim_pe = q_pe.shape[-1]
        num_head_groups = q.shape[1] // kv.shape[2]
        scale = (dim + dim_pe)**0.5
        Q = rearrange(
            q, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

        Q_pe = rearrange(
            q_pe, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim_pe]

        KV = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

        K_pe = rearrange(k_pe,
                         'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, dim_pe]

        query = torch.concat([Q, Q_pe], dim=-1)
        key = torch.concat([KV, K_pe], dim=-1)

        scores = einsum(
            query, key,
            'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

        attention = F.softmax(
            scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(attention, KV,
                     'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
        return out


@MlaDecodeFixture
def test_mla_decode(batch: int, heads: int, head_num_kv: int, seq_len_kv: int, dim: int,
                    dim_pe: int, dtype: torch.dtype, tune: bool):
    test = MlaDecodeTest(batch, heads, head_num_kv, seq_len_kv, dim, dim_pe, dtype)
    op = MultiHeadLatentAttentionDecodeWithKVCacheOp(
        batch, heads, head_num_kv, seq_len_kv, dim, dim_pe, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
