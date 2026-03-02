from typing import Optional

import pytest
import torch

from tests.ops.test_topk_selector import TopkSelectorFixture, TopkSelectorTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import TopkSelectorOp
from tileops.utils import str2dtype


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


@TopkSelectorFixture
def test_topk_selector_bench(batch: int, seq_len: int, topk: int, in_dtype_str: str,
                              out_dtype_str: str, tune: bool) -> None:
    in_dtype = str2dtype[in_dtype_str]
    out_dtype = str2dtype[out_dtype_str]
    test = TopkSelectorTest(batch, seq_len, topk, in_dtype, out_dtype)
    bm = TopkSelectorBenchmark(test)
    inputs = test.gen_inputs()

    op = TopkSelectorOp(batch, seq_len, topk, in_dtype, out_dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("topk_selector", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("topk_selector", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
