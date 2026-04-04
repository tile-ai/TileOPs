from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import NSACmpFwdVarlenOp
from workloads.ops.deepseek_nsa_cmp_fwd import NsaCmpFwdTest


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
    test = NsaCmpFwdTest(seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bc, bs, bk, bv,
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
