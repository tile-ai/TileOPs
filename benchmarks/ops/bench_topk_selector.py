from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_topk_selector import TopkSelectorTest
from tileops.ops import TopkSelectorOp


class TopkSelectorBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return None

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        index_score_memory = (t.batch * t.seq_len * t.seq_len_kv * t.kv_group * t.in_dtype.itemsize)
        index_memory = t.batch * t.seq_len * t.topk * t.kv_group * t.out_dtype.itemsize
        starts_memory = t.batch * t.seq_len * t.out_dtype.itemsize
        ends_memory = t.batch * t.seq_len * t.out_dtype.itemsize
        return index_score_memory + index_memory + starts_memory + ends_memory


_TOPK_SELECTOR_BENCH_PARAMS = [
    pytest.param(1, 32 * 1024, 64 * 1024, 1, 1024, torch.float32, torch.int32, True, id="base-topk1024"),
    pytest.param(1, 32 * 1024, 64 * 1024, 1, 2048, torch.float32, torch.int32, True, id="base-topk2048"),
    pytest.param(1, 65535, 128 * 1024, 1, 1024, torch.float32, torch.int32, True,
                 id="large-batch-topk1024"),
    pytest.param(1, 65535, 128 * 1024, 1, 2048, torch.float32, torch.int32, True,
                 id="large-batch-topk2048"),
]


@pytest.mark.parametrize(
    "batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype, tune",
    _TOPK_SELECTOR_BENCH_PARAMS,
)
def test_topk_selector_bench(batch: int, seq_len: int, seq_len_kv: int, kv_group: int, topk: int,
                             in_dtype: torch.dtype, out_dtype: torch.dtype, tune: bool) -> None:
    test = TopkSelectorTest(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)
    bm = TopkSelectorBenchmark(test)
    inputs = test.gen_inputs()

    op = TopkSelectorOp(
        batch=batch,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        topk=topk,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
