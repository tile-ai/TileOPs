
from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import NSACmpFwdVarlenOp
from workloads.nsa_utils import prepare_chunk_offsets
from workloads.ops.deepseek_nsa_cmp_fwd import NsaCmpFwdTest as _NsaCmpFwdTestWorkload


def _parallel_nsa_compression_fwd_pytorch(test, q, k_cmp, v_cmp, block_size, scale, offsets):
    """PyTorch reference implementation on GPU."""
    seq_len, heads, dim_k = q.shape
    _, head_kv, _ = k_cmp.shape
    dim_v = v_cmp.shape[-1]
    group = heads // head_kv
    device = q.device
    num_seq = len(offsets) - 1

    o = torch.zeros((seq_len, heads, dim_v), dtype=torch.float32, device=device)
    lse = torch.full((seq_len, heads), float('-inf'), dtype=torch.float32, device=device)

    chunk_offsets_local = prepare_chunk_offsets(offsets, block_size)

    for i_n in range(num_seq):
        bos, eos = offsets[i_n].item(), offsets[i_n + 1].item()
        boc = chunk_offsets_local[i_n].item()

        for i_t in range(eos - bos):
            nc = (i_t + 1) // block_size
            if nc == 0:
                lse[bos + i_t] = 0.0
                continue

            q_curr = q[bos + i_t].float()
            k_curr = k_cmp[boc:boc + nc].transpose(0, 1).float()
            v_curr = v_cmp[boc:boc + nc].transpose(0, 1).float()

            k_curr = k_curr.unsqueeze(1).expand(-1, group, -1, -1).reshape(heads, nc, dim_k)
            v_curr = v_curr.unsqueeze(1).expand(-1, group, -1, -1).reshape(heads, nc, dim_v)

            scores = torch.matmul(q_curr.unsqueeze(1), k_curr.transpose(-1, -2)).squeeze(1) * scale

            m = torch.max(scores, dim=-1, keepdim=True)[0]
            exp_scores = torch.exp(scores - m)
            sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)

            probs = exp_scores / sum_exp
            out = torch.matmul(probs.unsqueeze(1), v_curr).squeeze(1)

            o[bos + i_t] = out
            lse[bos + i_t] = (m + torch.log(sum_exp)).squeeze(-1)

    return o.to(test.dtype), lse.to(test.dtype)


class NsaCmpFwdTest(_NsaCmpFwdTestWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.LongTensor,
        chunk_offsets: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = chunk_offsets, token_indices
        return _parallel_nsa_compression_fwd_pytorch(self, q, k_cmp, v_cmp, self.bs, self.scale,
                                                      offsets)


class NsaCmpFwdFixture(FixtureBase):
    PARAMS = [
        ("seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bc, bs, bk, bv, "
         "dtype, accum_dtype, tune", [
             pytest.param(
                 9, 8192, 32, 128, 128, 16, 128**-0.5, 32, 32, 128, 128, torch.float16,
                 torch.float32, False, marks=pytest.mark.smoke,
             ),
         ]),
    ]


@NsaCmpFwdFixture
def test_nsa_cmp_fwd_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    group: int,
    scale: float,
    bc: int,
    bs: int,
    bk: int,
    bv: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaCmpFwdTest(seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bc, bs, bk, bv,
                         dtype, accum_dtype)
    inputs = test.gen_inputs()

    op = NSACmpFwdVarlenOp(
        seq_num=seq_num, c_seq_len=c_seq_len, heads=heads, dim_k=dim_k, dim_v=dim_v, group=group,
        scale=scale, bc=bc, bs=bs, bk=bk, bv=bv, dtype=dtype, accum_dtype=accum_dtype, tune=tune,
        chunk_num=test.chunk_num)
    test.check(op, *inputs, atol=4e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
