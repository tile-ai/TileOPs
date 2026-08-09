"""Benchmarks for the grouped GEMM op.

Workload shapes, dtypes, and transpose layouts come from the ops manifest;
per-variant roofline FLOP and byte counts come from the op's
``eval_roofline()`` via :class:`ManifestBenchmark`. The composed
forward+backward case keeps a local roofline because it aggregates four
GEMM launches, which no single manifest workload describes.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkBase,
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from tileops.manifest import load_workloads
from tileops.ops import GroupedGemmOp
from workloads.grouped_gemm import (
    GroupedGemmWorkload,
)

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


# Test functions

_GROUPED_GEMM_OP = "GroupedGemmOp"
_GROUPED_GEMM_PARAMS = workload_field_params(
    load_workloads(_GROUPED_GEMM_OP),
    ("batch_sum", "batch_count", "n", "k", "dtype", "transpose_a", "transpose_b"),
)


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b",
    _GROUPED_GEMM_PARAMS,
)
def test_grouped_gemm_bench(batch_sum: int, batch_count: int, N: int, K: int,
                            dtype: torch.dtype, transpose_a: bool,
                            transpose_b: bool) -> None:
    layout = ("T" if transpose_a else "N") + ("T" if transpose_b else "N")
    name = f"grouped_gemm_{layout.lower()}"

    test = GroupedGemmWorkload(batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b)
    inputs = test.gen_inputs()

    op = GroupedGemmOp(transpose_a=transpose_a, transpose_b=transpose_b, tune=_TUNE)
    bm = ManifestBenchmark(_GROUPED_GEMM_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(name, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(name, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
