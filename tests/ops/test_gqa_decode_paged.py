"""Test GroupQueryAttentionDecodePagedWithKVCacheOp (paged GQA decode with dynamic KV cache)."""

import math
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import TestBase, FixtureBase
from tileops.ops import GroupQueryAttentionDecodePagedWithKVCacheOp


class GqaDecodePagedFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune", [
            (1, 16, 8, 512, 128, 128, torch.float16, False),
            (2, 8, 4, 1024, 64, 256, torch.float16, False),
            (1, 32, 8, 256, 128, 64, torch.float16, False),
            (1, 8, 4, 1024, 64, 256, torch.float16, False),
            (2, 16, 8, 512, 128, 128, torch.float16, False),
            (1, 16, 4, 2048, 128, 512, torch.float16, False),
            (1, 32, 16, 512, 64, 128, torch.float16, False),
        ]),
    ]


class GqaDecodePagedTest(TestBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seqlen_kv: int, dim: int,
                 page_size: int, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.dtype = dtype

    def gen_inputs(
            self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = self.seqlen_kv // self.page_size
        real_seqlen_kv = torch.randint(
            self.page_size, self.seqlen_kv + 1, (self.batch,), dtype=torch.int32, device="cuda")
        real_seqlen_kv = (real_seqlen_kv // self.page_size) * self.page_size
        real_seqlen_kv[0] = min(real_seqlen_kv[0].item(), self.seqlen_kv)

        q = torch.randn(self.batch, self.heads, self.dim, dtype=self.dtype, device="cuda")
        k = torch.randn(self.seqlen_kv, self.heads_kv, self.dim, dtype=self.dtype, device="cuda")
        v = torch.randn(self.seqlen_kv, self.heads_kv, self.dim, dtype=self.dtype, device="cuda")
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
        """Reassemble paged K/V to logical layout per batch, then GQA (expand to heads) + SDPA."""
        batch, _, dim = q.shape
        seqlen_kv, _, _ = k.shape
        kv_group_num = self.heads // self.heads_kv
        out_list = []
        for i_b in range(batch):
            q_b = q[i_b:i_b + 1, :, :]
            k_logical = torch.zeros(seqlen_kv, self.heads_kv, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, self.heads_kv, dim, dtype=q.dtype, device=q.device)
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
            group_id = torch.arange(self.heads, dtype=torch.long, device=q.device) // kv_group_num
            k_bhsd = k_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
            v_bhsd = v_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
            q_bhsd = q_b.unsqueeze(2)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
            out_b = out_b.squeeze(2)
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)

    def check(self, op, *inputs, atol: float = 0.001, rtol: float = 1e-05) -> None:
        """Custom check with max-diff and cosine similarity (matches original test)."""
        outputs_ref = self.ref_program(*inputs)

        with torch.no_grad():
            output = op(*inputs)

        if isinstance(output, (tuple, list)):
            output = output[0]

        max_diff = (output - outputs_ref).abs().max().item()
        assert max_diff < atol, (
            f"max diff {max_diff} too large (atol={atol})")
        cos_sim = F.cosine_similarity(
            output.reshape(self.batch, -1), outputs_ref.reshape(self.batch, -1), dim=-1, eps=1e-8)
        assert cos_sim.min() > 0.99, f"cosine similarity {cos_sim.min().item()} too low"

        print(f"All checks passed for {op.__class__.__name__}.")


@GqaDecodePagedFixture
def test_gqa_decode_paged_op(
    batch: int,
    heads: int,
    heads_kv: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GqaDecodePagedTest(batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype)
    op = GroupQueryAttentionDecodePagedWithKVCacheOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        dtype=dtype,
        tune=tune,
    )
    test.check(op, *test.gen_inputs())


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
