from typing import Optional

import pytest
import torch

from tests.ops.test_deepseek_nsa_fwd import NsaFwdFixture, NsaFwdTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import NSAFwdVarlenOp


class NsaFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_token = 4 * t.dim * t.selected_blocks * t.block_size
        return flops_per_token * t.c_seq_len * t.heads

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # q, k, v, output, block_indices, block_counts, offsets, token_indices
        # ignore block counts, offsets and token_indices memory
        q_memory = t.heads * t.c_seq_len * t.dim * t.dtype.itemsize
        k_memory = t.head_kv * t.c_seq_len * t.dim * t.dtype.itemsize
        v_memory = t.head_kv * t.c_seq_len * t.dim * t.dtype.itemsize
        output_memory = t.heads * t.c_seq_len * t.dim * t.dtype.itemsize
        block_indices_memory = t.head_kv * t.c_seq_len * t.selected_blocks * 4
        return (q_memory + k_memory + v_memory + output_memory + block_indices_memory)


@NsaFwdFixture
def test_nsa_fwd_bench(batch: int, heads: int, c_seq_len: int, dim: int, is_causal: bool,
                       scale: float, block_size: int, groups: int, selected_blocks: int,
                       dtype: torch.dtype, accum_dtype: torch.dtype, tune: bool) -> None:
    test = NsaFwdTest(batch, heads, c_seq_len, dim, is_causal, scale, block_size, groups,
                      selected_blocks, dtype, accum_dtype)
    bm = NsaFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = NSAFwdVarlenOp(
        batch=test.batch, heads=test.heads, c_seq_len=test.c_seq_len, dim=test.dim,
        is_causal=test.is_causal, scale=test.scale, block_size=test.block_size,
        groups=test.groups, selected_blocks=test.selected_blocks, dtype=test.dtype,
        accum_dtype=test.accum_dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("nsa_fwd", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("nsa_fwd", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
