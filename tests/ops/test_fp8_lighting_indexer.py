from typing import Optional, Tuple

import torch
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import Fp8LightingIndexerOp


class Fp8LightingIndexerFixture(FixtureBase):
    PARAMS = [
        ("seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune", [
            (4096, 32, 64, 8192, True, None, False),
        ]),
    ]


class Fp8LightingIndexerTest(TestBase):

    def __init__(self, seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                 clean_logits: bool = True, config: Optional[dict] = None):
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.clean_logits = clean_logits
        self.config = config
        self.dtype = torch.float8_e4m3fn
        self.accum_dtype = torch.float32
        self.index_dtype = torch.int32

    def cal_seq_idx_for_q(self, cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor,
                          seq_len: int) -> torch.IntTensor:
        seq_idx_for_q = torch.zeros(seq_len, dtype=torch.int32, device=cu_seqlens_qs.device)
        if len(cu_seqlens_qs) > 1:
            seq_idx_for_q[cu_seqlens_qs[1:]] = 1
        return torch.cumsum(seq_idx_for_q, dim=0, dtype=torch.int32)

    def cal_cu_seqlen_ke_for_q(
        self,
        cu_seqlens_qs: torch.LongTensor,
        cu_seqlens_qe: torch.LongTensor,
        cu_seqlens_ks: torch.LongTensor,
        cu_seqlens_ke: torch.LongTensor,
        q_start_idxs: torch.LongTensor,
        seq_len: int,
        kv_stride: int,
    ) -> torch.IntTensor:
        cu_seqlen_ke_for_each_q = torch.gather(
            input=torch.cat(
                [cu_seqlens_ke,
                 torch.zeros(1, dtype=torch.int32, device=cu_seqlens_qs.device)]),
            dim=0,
            index=self.cal_seq_idx_for_q(
                cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
        )
        casual_cu_seqlen_ke_for_each_q = torch.zeros((seq_len,),
                                                     dtype=torch.int32,
                                                     device=cu_seqlens_qs.device)
        for i in range(len(cu_seqlens_qs)):
            casual_cu_seqlen_ke_for_each_q[cu_seqlens_qs[i]:cu_seqlens_qe[i]] = (torch.arange(
                q_start_idxs[i],
                q_start_idxs[i] + cu_seqlens_qe[i] - cu_seqlens_qs[i],
                dtype=torch.int32,
                device=cu_seqlens_qs.device) + 1) // kv_stride + cu_seqlens_ks[i]
        cu_seqlen_ke_for_each_q = torch.minimum(casual_cu_seqlen_ke_for_each_q,
                                                cu_seqlen_ke_for_each_q)
        return cu_seqlen_ke_for_each_q.int()

    def cal_cu_seqlen_ks_for_q(
        self,
        cu_seqlens_qs: torch.LongTensor,
        cu_seqlens_qe: torch.LongTensor,
        cu_seqlens_ks: torch.LongTensor,
        seq_len: int,
    ) -> torch.IntTensor:
        cu_seqlen_ks_for_each_q = torch.gather(
            input=torch.cat([
                cu_seqlens_ks,
                torch.full((1,),
                           torch.iinfo(torch.int32).max,
                           dtype=torch.int32,
                           device=cu_seqlens_qs.device)
            ]),
            dim=0,
            index=self.cal_seq_idx_for_q(
                cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
        )
        return cu_seqlen_ks_for_each_q.int()

    def generate_random_cu_seqlens(self,
                                   cp_size: int = 4,
                                   cp_rank: int = 3,
                                   kv_stride: int = 1,
                                   average_q_len: int = 512):
        total_seqlen = self.seq_len * cp_size

        cu_seqlens = torch.randint(0, average_q_len * 2,
                                   (total_seqlen // average_q_len * 2,)).cuda()
        last_seq_id = torch.where(cu_seqlens.cumsum(0) >= total_seqlen)[0][0]
        cu_seqlens = cu_seqlens[:last_seq_id]

        if cu_seqlens.sum() < total_seqlen:
            cu_seqlens = torch.cat(
                [cu_seqlens, torch.tensor([total_seqlen - cu_seqlens.sum()]).cuda()])

        cu_seqlens_cumsum = torch.cumsum(cu_seqlens, dim=0)
        cu_seqlens_k_cumsum = torch.cumsum(cu_seqlens // kv_stride, dim=0)
        cu_seqlens_qs = torch.cat([torch.tensor([0]).cuda(), cu_seqlens_cumsum[:-1]])
        cu_seqlens_ks = torch.cat([torch.tensor([0]).cuda(), cu_seqlens_k_cumsum[:-1]])
        cu_seqlens_qe = cu_seqlens_cumsum.clone()
        cu_seqlens_ke = cu_seqlens_k_cumsum.clone()

        cu_seqlens_ks_for_each_q = self.cal_cu_seqlen_ks_for_q(
            cu_seqlens_qs,
            cu_seqlens_qe,
            cu_seqlens_ks,
            total_seqlen,
        )

        cu_seqlens_ke_for_each_q = self.cal_cu_seqlen_ke_for_q(
            cu_seqlens_qs=cu_seqlens_qs,
            cu_seqlens_qe=cu_seqlens_qe,
            cu_seqlens_ks=cu_seqlens_ks,
            cu_seqlens_ke=cu_seqlens_ke,
            q_start_idxs=torch.zeros_like(cu_seqlens_qs),
            seq_len=total_seqlen,
            kv_stride=kv_stride,
        )

        assert self.seq_len % 2 == 0
        per_chunk_seqlen = self.seq_len // 2
        slice_short = slice(cp_rank * per_chunk_seqlen, (cp_rank + 1) * per_chunk_seqlen)
        slice_long = slice(
            total_seqlen - (cp_rank + 1) * per_chunk_seqlen,
            total_seqlen - cp_rank * per_chunk_seqlen,
        )

        ks = torch.cat([
            cu_seqlens_ks_for_each_q[slice_short],
            cu_seqlens_ks_for_each_q[slice_long],
        ])
        ke = torch.cat([
            cu_seqlens_ke_for_each_q[slice_short],
            cu_seqlens_ke_for_each_q[slice_long],
        ])
        assert len(ks) == len(ke) == self.seq_len
        return ks, ke

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
        IndexQ = torch.randn(
            self.seq_len, self.heads, self.index_dim, device='cuda', dtype=torch.bfloat16)
        IndexK = torch.randn(self.seq_len_kv, self.index_dim, device='cuda', dtype=torch.bfloat16)
        Weights = torch.randn(self.seq_len, self.heads, device='cuda', dtype=self.accum_dtype)
        CuSeqLenKS, CuSeqLenKE = self.generate_random_cu_seqlens(
            cp_size=4, cp_rank=3, kv_stride=1, average_q_len=2048)
        return IndexQ, IndexK, Weights, CuSeqLenKS, CuSeqLenKE

    def ref_program(self, q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                    cu_seqlen_ks: torch.Tensor,
                    cu_seqlen_ke: torch.Tensor) -> Tuple[torch.Tensor]:
        k = kv
        q = q.float()
        k = k.float()

        seq_len_kv = kv.shape[0]
        mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("mhd,nd->hmn", q, k)
        logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        logits = logits.masked_fill(~mask, float("-inf"))
        return (logits,)

    @staticmethod
    def _compute_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.data.double(), b.data.double()
        norm_sum = (a * a + b * b).sum()
        return 2 * (a * b).sum() / norm_sum

    @staticmethod
    def _validate_tensor_match(output: torch.Tensor, output_ref: torch.Tensor, tolerance: float = 1e-3) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output_ref, tuple):
            output_ref = output_ref[0]

        a_finite = torch.isfinite(output)
        b_finite = torch.isfinite(output_ref)
        assert torch.all(a_finite == b_finite), "Error: isfinite mask mismatch"
        assert torch.isclose(
            output.masked_fill(a_finite, 0),
            output_ref.masked_fill(b_finite, 0),
            rtol=0,
            atol=0,
            equal_nan=True,
        ).all(), "Error: nonfinite value mismatch"
        output = output.masked_fill(~a_finite, 0)
        output_ref = output_ref.masked_fill(~b_finite, 0)
        correlation = Fp8LightingIndexerTest._compute_correlation(output, output_ref)
        difference = 1.0 - correlation
        assert 0 <= difference <= tolerance, \
            f"outputs is not close to outputs_ref, difference: {difference}"


@Fp8LightingIndexerFixture
def test_indexer(seq_len: int, heads: int, index_dim: int, seq_len_kv: int, clean_logits: bool,
                 config: Optional[dict], tune: bool) -> None:
    test = Fp8LightingIndexerTest(seq_len, heads, index_dim, seq_len_kv, clean_logits, config)
    op = Fp8LightingIndexerOp(
        seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune=tune)
    test.check(op, *test.gen_inputs(), compare=Fp8LightingIndexerTest._validate_tensor_match)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
