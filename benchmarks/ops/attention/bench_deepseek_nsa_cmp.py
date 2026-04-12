from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import NSACmpFwdVarlenOp
from workloads.nsa_utils import prepare_chunk_offsets
from workloads.ops.attention.deepseek_nsa_cmp import NsaCmpFwdTest


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


class _NsaCmpFwdTestBaseline(NsaCmpFwdTest):
    """Adds baseline ref_program for benchmark profiling."""

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


class NsaCmpFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return (2 * t.heads * t.dim_k * t.c_seq_len**2) // t.bs

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        q_read = t.heads * t.c_seq_len * t.dim_k * t.dtype.itemsize
        k_read = (t.head_kv * t.dim_k * t.c_seq_len**2 * t.dtype.itemsize) // t.bs
        v_read = (t.head_kv * t.dim_v * t.c_seq_len**2 * t.dtype.itemsize) // t.bs
        return q_read + k_read + v_read


_NSA_CMP_FWD_BENCH_PARAMS = [
    pytest.param(
        9, 8192, 32, 128, 128, 16, 128**-0.5, 32, 32, 128, 128, torch.float16, torch.float32,
        False, id="mainstream-fp16",
    ),
    pytest.param(
        16, 16384, 32, 128, 128, 16, 128**-0.5, 32, 32, 128, 128, torch.float16, torch.float32,
        False, id="long-sequence-fp16",
    ),
]


@pytest.mark.parametrize(
    "seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bc, bs, bk, bv, dtype, accum_dtype, tune",
    _NSA_CMP_FWD_BENCH_PARAMS,
)
def test_nsa_cmp_fwd_bench(seq_num: int, c_seq_len: int, heads: int, dim_k: int, dim_v: int,
                           group: int, scale: float, bc: int, bs: int, bk: int, bv: int,
                           dtype: torch.dtype, accum_dtype: torch.dtype, tune: bool) -> None:
    test = _NsaCmpFwdTestBaseline(seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bc, bs, bk, bv,
                         dtype, accum_dtype)
    bm = NsaCmpFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = NSACmpFwdVarlenOp(
        seq_num=test.seq_num, c_seq_len=test.c_seq_len, heads=test.heads, dim_k=test.dim_k,
        dim_v=test.dim_v, chunk_num=test.chunk_num, group=test.group, scale=test.scale,
        bc=test.bc, bs=test.bs, bk=test.bk, bv=test.bv, dtype=test.dtype,
        accum_dtype=test.accum_dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
