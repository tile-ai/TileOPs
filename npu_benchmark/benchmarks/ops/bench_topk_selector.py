"""Benchmark for the top-k selector op on Ascend NPU.

Workload shapes, dtypes, and ``topk`` come from the manifest; roofline
FLOP and byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from manifest import load_workloads
from ops import TopkSelectorOp
from workloads.topk_selector import TopkSelectorWorkload

_TUNE = True
_TOPK_SELECTOR_OP = "TopkSelectorOp"
_TOPK_SELECTOR_PARAMS = workload_field_params(
    load_workloads(_TOPK_SELECTOR_OP),
    ("batch", "seq_len", "seq_len_kv", "kv_group", "topk", "in_dtype", "out_dtype"),
)


class TopkSelectorBenchBaseline(TopkSelectorWorkload):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, index_score: torch.Tensor, starts: torch.Tensor,
                    ends: torch.Tensor) -> torch.Tensor:
        indexes_ref = torch.topk(index_score, self.topk, dim=2)[1]
        return indexes_ref.permute(0, 1, 3, 2)


@pytest.mark.parametrize(
    "batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype",
    _TOPK_SELECTOR_PARAMS,
)
def test_topk_selector_bench(batch: int, seq_len: int, seq_len_kv: int,
                             kv_group: int, topk: int,
                             in_dtype: torch.dtype, out_dtype: torch.dtype) -> None:
    test = TopkSelectorBenchBaseline(batch, seq_len, seq_len_kv, kv_group,
                                     topk, in_dtype, out_dtype)
    inputs = test.gen_inputs()

    op = TopkSelectorOp(topk=topk, tune=_TUNE)
    bm = ManifestBenchmark(_TOPK_SELECTOR_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
