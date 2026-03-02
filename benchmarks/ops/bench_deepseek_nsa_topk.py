from typing import Optional

import pytest
import torch

from tests.ops.test_deepseek_nsa_topk import NsaTopkFixture, NsaTopkTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import NSATopkVarlenOp


class NsaTopkBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Step 1 (LSE) + Step 2 (Scores)
        # Total: c_seq_len * head_kv * 2 * (2 * group * dim * (c_seq_len / (2 * bs)))
        return (2 * t.heads * t.dim * t.c_seq_len**2) // t.bs

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # q: read once, k_cmp: read twice per preceding block per token, block_indices: write once
        q_read = t.heads * t.c_seq_len * t.dim * t.dtype.itemsize
        k_read = (t.head_kv * t.dim * t.c_seq_len**2 * t.dtype.itemsize) // t.bs
        indices_write = t.c_seq_len * t.head_kv * t.selected_block_num * 4
        return q_read + k_read + indices_write


@NsaTopkFixture
def test_nsa_topk_bench(seq_num: int, c_seq_len: int, heads: int, dim: int, group: int,
                        scale: float, selected_block_num: int, bc: int, bs: int, bk: int,
                        dtype: torch.dtype, accum_dtype: torch.dtype, tune: bool) -> None:
    test = NsaTopkTest(seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bc, bs,
                       bk, dtype, accum_dtype)
    bm = NsaTopkBenchmark(test)
    inputs = test.gen_inputs()

    op = NSATopkVarlenOp(
        seq_num=seq_num, c_seq_len=c_seq_len, heads=heads, dim=dim,
        chunk_num=test.chunk_num, group=group, scale=scale,
        selected_block_num=selected_block_num, bc=bc, bs=bs, bk=bk,
        dtype=dtype, accum_dtype=accum_dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("nsa_topk", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("nsa_topk", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
