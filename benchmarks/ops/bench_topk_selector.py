from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_topk_selector import TopkSelectorTest
from tileops.ops import TopkSelectorOp
from tileops.utils import str2dtype


class TopkSelectorBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return None

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        index_score_memory = t.batch * t.seq_len * t.in_dtype.itemsize
        index_memory = t.batch * t.topk * t.out_dtype.itemsize
        starts_memory = t.batch * t.out_dtype.itemsize
        ends_memory = t.batch * t.out_dtype.itemsize
        return index_score_memory + index_memory + starts_memory + ends_memory


_TOPK_SELECTOR_BENCH_PARAMS = [
    pytest.param(64, 32 * 1024, 2048, "float32", "int32", False, marks=pytest.mark.full, id="bench-mid-topk"),
    pytest.param(128, 64 * 1024, 1024, "float32", "int32", False, marks=pytest.mark.full, id="bench-long-seq"),
    pytest.param(128, 64 * 1024, 2048, "float32", "int32", False, marks=pytest.mark.nightly, id="bench-long-seq-high-topk"),
]


@pytest.mark.parametrize(
    "batch, seq_len, topk, in_dtype_str, out_dtype_str, tune",
    _TOPK_SELECTOR_BENCH_PARAMS,
)
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
