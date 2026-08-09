"""Benchmark for the top-k selector op.

Workload shapes, dtypes, and ``topk`` come from the ops manifest; roofline
FLOP and byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from tileops.manifest import load_workloads
from tileops.ops import TopkSelectorOp
from workloads.topk_selector import TopkSelectorWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


_TOPK_SELECTOR_OP = "TopkSelectorOp"
_TOPK_SELECTOR_PARAMS = workload_field_params(
    load_workloads(_TOPK_SELECTOR_OP),
    ("batch", "seq_len", "seq_len_kv", "kv_group", "topk", "in_dtype", "out_dtype"),
)


@pytest.mark.parametrize(
    "batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype",
    _TOPK_SELECTOR_PARAMS,
)
def test_topk_selector_bench(batch: int, seq_len: int, seq_len_kv: int, kv_group: int, topk: int,
                             in_dtype: torch.dtype, out_dtype: torch.dtype) -> None:
    test = TopkSelectorWorkload(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype,
                                out_dtype)
    inputs = test.gen_inputs()

    op = TopkSelectorOp(topk=topk, tune=_TUNE)
    bm = ManifestBenchmark(_TOPK_SELECTOR_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
