
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek_dsa_decode import DsaDecodeTest as _DsaDecodeTestWorkload


class DsaDecodeTest(_DsaDecodeTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, kv: torch.Tensor,
                    indices: torch.Tensor) -> torch.Tensor:
        q = q.float()
        kv = kv.float()
        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape
        q_start_index_s = self.q_start_index_s
        if self.q_start_index_s is None:
            q_start_index_s = sk * self.stride_kv - sq

        assert kv.shape[-1] == self.dim + self.dim_tail, 'you should assign dim otherwise'
        dim = self.dim
        k = kv
        v = kv[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32,
            device="cuda").view(-1, 1) >= torch.arange(
                self.stride_kv - 1,
                sk * self.stride_kv,
                self.stride_kv,
                dtype=torch.int32,
                device="cuda").view(1, -1)

        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, :self.stride_kv - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        return o.to(torch.float16)


class DsaDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, "
         "q_start_index_s, sm_scale, dtype, tune", [
             pytest.param(
                 1, 128, 1024, 2048, 512, 64, 2048, 1, 1, 1024, None, torch.float16, False,
                 marks=pytest.mark.smoke,
             ),
         ]),
    ]


@DsaDecodeFixture
def test_sparse_mla_decode(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                           dim_tail: int, topk: int, stride_kv: int, heads_kv: int,
                           q_start_index_s: int, sm_scale: float, dtype: torch.dtype,
                           tune: bool) -> None:
    pytest.skip("Temporarily skipping known DeepSeek DSA decode failure under TileLang 5f70374c (#999).")
    test = DsaDecodeTest(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
