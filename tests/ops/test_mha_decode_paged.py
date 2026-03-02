"""Test MultiHeadAttentionDecodePagedWithKVCacheOp (paged MHA decode with dynamic KV cache)."""

import math
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import TestBase, FixtureBase
from tileops.ops import MultiHeadAttentionDecodePagedWithKVCacheOp


class MhaDecodePagedFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune", [
            (1, 16, 1, 512, 128, 128, False, torch.float16, False),
            (1, 8, 1, 1024, 64, 256, False, torch.float16, False),
            (2, 8, 1, 1024, 64, 256, False, torch.float16, False),
            (1, 8, 1, 512, 64, 256, False, torch.float16, False),
        ]),
    ]


class MhaDecodePagedTest(TestBase):

    def __init__(self, batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                 page_size: int, is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
            self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = self.seqlen_kv // self.page_size
        real_seqlen_kv = torch.ones(
            (self.batch,), dtype=torch.int32, device="cuda") * self.seqlen_kv
        q = torch.randn(
            self.batch, self.seqlen_q, self.heads, self.dim, device="cuda", dtype=self.dtype)
        k = torch.randn(self.seqlen_kv, self.heads, self.dim, device="cuda", dtype=self.dtype)
        v = torch.randn(self.seqlen_kv, self.heads, self.dim, device="cuda", dtype=self.dtype)
        # Identity block_table: logical page i -> physical page i (contiguous layout)
        block_table = torch.arange(
            num_pages, dtype=torch.int32, device="cuda").unsqueeze(0).expand(self.batch, -1)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_table = block_table.contiguous()
        real_seqlen_kv = real_seqlen_kv.contiguous()

        return q, k, v, real_seqlen_kv, block_table

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Reassemble paged K/V to logical layout per batch, then SDPA."""
        batch, seqlen_q, heads, dim = q.shape
        seqlen_kv = k.shape[0]
        out_list = []
        for i_b in range(batch):
            q_b = q[i_b:i_b + 1, :, :, :]
            k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            num_pages = math.ceil(real_seqlen_kv[i_b].item() / self.page_size)
            for i_paged in range(num_pages):
                start_pos = block_table[i_b, i_paged].item() * self.page_size
                end_pos = min(start_pos + self.page_size, seqlen_kv)
                page_len = end_pos - start_pos
                k_logical[i_paged * self.page_size:i_paged * self.page_size +
                          page_len, :, :] = k[start_pos:end_pos, :, :]
                v_logical[i_paged * self.page_size:i_paged * self.page_size +
                          page_len, :, :] = v[start_pos:end_pos, :, :]
            k_logical = k_logical[:real_seqlen_kv[i_b].item(), :, :]
            v_logical = v_logical[:real_seqlen_kv[i_b].item(), :, :]
            k_b = k_logical.unsqueeze(0)
            v_b = v_logical.unsqueeze(0)
            q_bhsd = q_b.transpose(1, 2)
            k_bhsd = k_b.transpose(1, 2)
            v_bhsd = v_b.transpose(1, 2)
            out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
            out_b = out_b.transpose(1, 2).contiguous()
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)

    def _maxdiff_cosine_compare(self, output: torch.Tensor, output_ref: torch.Tensor, atol: float = 0.001) -> None:
        """Compare using max-diff and cosine similarity."""
        if isinstance(output, (tuple, list)):
            output = output[0]
        max_diff = (output - output_ref).abs().max().item()
        assert max_diff < atol, (
            f"max diff {max_diff} too large (atol={atol})")
        cos_sim = F.cosine_similarity(
            output.reshape(self.batch, -1), output_ref.reshape(self.batch, -1), dim=-1, eps=1e-8)
        assert cos_sim.min() > 0.99, f"cosine similarity {cos_sim.min().item()} too low"


@MhaDecodePagedFixture
def test_mha_decode_paged_op(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MhaDecodePagedTest(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    op = MultiHeadAttentionDecodePagedWithKVCacheOp(
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        is_causal=is_causal,
        dtype=dtype,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), compare=test._maxdiff_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
